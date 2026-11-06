"""Frozen RGB and thermal LoRA, plus the shared trainable LoRA3.

Each modality LoRA is injected into qkv and proj in the vision trunk and then frozen.
LoRA3 is attached to the same linear layers. Its parameters are shared by the two
models and are not registered inside SAM3.
"""
from __future__ import annotations
import sys
import torch
import torch.nn as nn
from pathlib import Path
import paths


def _prepare_imports() -> None:
    root = paths.sam3_import_root()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


_prepare_imports()
from sam3.model_builder import build_sam3_image_model
from lora_layers import LoRAConfig, LoRALayer, LoRALinear, apply_lora_to_model, load_lora_weights


LORA3_RANK = 8
LORA3_ALPHA = 16.0
LORA3_DROPOUT = 0.05
LORA3_LAYER_COUNT = 64
_VISION_LORA_TARGETS = ("qkv", "proj")


def _vision_lora_config() -> LoRAConfig:
    return LoRAConfig(
        rank=16,
        alpha=32.0,
        dropout=0.0,
        target_modules=list(_VISION_LORA_TARGETS),
        apply_to_vision_encoder=True,
        apply_to_text_encoder=False,
        apply_to_geometry_encoder=False,
        apply_to_detr_encoder=False,
        apply_to_detr_decoder=False,
        apply_to_mask_decoder=False,
    )


def _linear_shape(layer: LoRALinear) -> tuple[int, int]:
    return (layer.in_features, layer.out_features)


def configure_low_memory_model(model: nn.Module) -> None:
    """Disable DAC. The decoder query count is already 200."""
    decoder = model.transformer.decoder
    decoder.dac = False
    decoder.num_o2m_queries = 0
    model.dac = False


def _load_sam3(checkpoint_path: Path) -> nn.Module:
    paths.require_file(checkpoint_path, "SAM3 checkpoint", paths.WEIGHT_URL, "WEIGHT_URL")
    model = build_sam3_image_model(
        bpe_path=str(paths.bpe_path()),
        device="cpu",
        eval_mode=False,
        checkpoint_path=str(checkpoint_path),
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
    )
    configure_low_memory_model(model)
    return model


def _freeze_model(model: nn.Module, device: torch.device) -> nn.Module:
    for name, param in model.backbone.named_parameters():
        if "lora" not in name:
            param.data = param.data.to(torch.bfloat16)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model.to(device)
    return model


def build_frozen_lora_model(checkpoint_path: Path, lora_path: Path, device: torch.device) -> nn.Module:
    """Load one vision LoRA and freeze it together with SAM3."""
    paths.require_file(lora_path, "LoRA weights", paths.WEIGHT_URL, "WEIGHT_URL")
    model = _load_sam3(checkpoint_path)
    model = apply_lora_to_model(model, _vision_lora_config())
    load_lora_weights(model, str(lora_path))
    return _freeze_model(model, device)


class SharedLoRA(nn.Module):
    """LoRA3 shared by the two vision trunks. Full module names index the LoRA layers."""

    def __init__(self, named_shapes: dict[str, tuple[int, int]]) -> None:
        super().__init__()
        self.index: dict[str, int] = {}
        layers: list[LoRALayer] = []
        for index, (name, (in_features, out_features)) in enumerate(named_shapes.items()):
            self.index[name] = index
            layers.append(
                LoRALayer(
                    in_features=in_features,
                    out_features=out_features,
                    rank=LORA3_RANK,
                    alpha=LORA3_ALPHA,
                    dropout=LORA3_DROPOUT,
                )
            )
        self.layers = nn.ModuleList(layers)

    def delta(self, name: str, x: torch.Tensor) -> torch.Tensor:
        return self.layers[self.index[name]](x)


def _lora3_sites(model: nn.Module) -> dict[str, LoRALinear]:
    """Modality LoRA layers in the vision trunk whose basename is qkv or proj."""
    return {
        name: module
        for name, module in model.named_modules()
        if "vision_backbone" in name
        and name.split(".")[-1] in _VISION_LORA_TARGETS
        and isinstance(module, LoRALinear)
    }


def _bind_shared_lora(layer: LoRALinear, shared: SharedLoRA, name: str) -> None:
    object.__setattr__(layer, "shared_lora", shared)
    object.__setattr__(layer, "shared_name", name)


def build_shared_lora(rgb_model: nn.Module, t_model: nn.Module) -> SharedLoRA:
    """Attach one LoRA3 to two models that already contain modality LoRA. It is not registered in SAM3."""
    rgb_sites = _lora3_sites(rgb_model)
    t_sites = _lora3_sites(t_model)
    if set(rgb_sites) != set(t_sites):
        raise RuntimeError("RGB and thermal LoRA3 insertion sites do not match.")
    if len(rgb_sites) != LORA3_LAYER_COUNT:
        raise RuntimeError(
            f"Each model should have {LORA3_LAYER_COUNT} vision qkv/proj LoRA layers, got {len(rgb_sites)}."
        )

    named_shapes: dict[str, tuple[int, int]] = {}
    for name in sorted(rgb_sites):
        rgb_shape = _linear_shape(rgb_sites[name])
        t_shape = _linear_shape(t_sites[name])
        if rgb_shape != t_shape:
            raise RuntimeError(f"{name} has different linear shapes: {rgb_shape} and {t_shape}.")
        named_shapes[name] = rgb_shape

    shared = SharedLoRA(named_shapes)
    for name in named_shapes:
        _bind_shared_lora(rgb_sites[name], shared, name)
        _bind_shared_lora(t_sites[name], shared, name)
    return shared


def enable_trunk_checkpoint(model: nn.Module) -> None:
    """Enable activation checkpointing in the ViT trunk and disable DropPath."""
    trunk = model.backbone.vision_backbone.trunk
    trunk.train()
    for module in trunk.modules():
        if module.__class__.__name__ == "DropPath":
            module.eval()
