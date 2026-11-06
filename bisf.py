"""Bidirectional interactive spatial fusion for RGB and thermal features.

The module reads the last backbone level from each modality, applies
bidirectional spatial cross-attention, depthwise convolution, and spatial
self-attention, then writes one fused map back into the backbone dictionary.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


M1_DIM = 256
M1_HEADS = 8


def _map_to_tokens(x: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, HW, C)."""
    return x.flatten(2).transpose(1, 2)


def _tokens_to_map(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """(B, HW, C) -> (B, C, H, W)."""
    batch, _, channels = tokens.shape
    return tokens.transpose(1, 2).reshape(batch, channels, height, width)


def _split_heads(tokens: torch.Tensor, num_heads: int) -> torch.Tensor:
    """(B, N, C) -> (B, heads, N, head_dim)."""
    batch, length, channels = tokens.shape
    head_dim = channels // num_heads
    return tokens.view(batch, length, num_heads, head_dim).transpose(1, 2)


def _merge_heads(tokens: torch.Tensor) -> torch.Tensor:
    """(B, heads, N, head_dim) -> (B, N, C)."""
    batch, num_heads, length, head_dim = tokens.shape
    return tokens.transpose(1, 2).reshape(batch, length, num_heads * head_dim)


def _attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int) -> torch.Tensor:
    """q, k, and v are (B, N, C). Returns (B, Nq, C)."""
    out = F.scaled_dot_product_attention(
        _split_heads(q, num_heads),
        _split_heads(k, num_heads),
        _split_heads(v, num_heads),
    )
    return _merge_heads(out)


class BidirectionalCrossAttention(nn.Module):
    """Bidirectional spatial cross-attention. The output replaces the input, with no residual."""

    def __init__(self, dim: int = M1_DIM, num_heads: int = M1_HEADS):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.norm_rgb = nn.LayerNorm(dim)
        self.norm_thermal = nn.LayerNorm(dim)
        self.q_rgb = nn.Linear(dim, dim)
        self.kv_thermal = nn.Linear(dim, dim * 2)
        self.q_thermal = nn.Linear(dim, dim)
        self.kv_rgb = nn.Linear(dim, dim * 2)
        self.out_rgb = nn.Linear(dim, dim)
        self.out_thermal = nn.Linear(dim, dim)

    def _cross(
        self,
        query_tokens: torch.Tensor,
        memory_tokens: torch.Tensor,
        pos_tokens: torch.Tensor,
        q_proj: nn.Linear,
        kv_proj: nn.Linear,
        out_proj: nn.Linear,
    ) -> torch.Tensor:
        q = q_proj(query_tokens)
        k, v = kv_proj(memory_tokens).chunk(2, dim=-1)
        pos = pos_tokens.to(dtype=q.dtype)
        return out_proj(_attend(q + pos, k + pos, v, self.num_heads))

    def forward(
        self,
        rgb: torch.Tensor,
        thermal: torch.Tensor,
        pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = rgb.shape[-2:]
        rgb_tokens = self.norm_rgb(_map_to_tokens(rgb))
        thermal_tokens = self.norm_thermal(_map_to_tokens(thermal))
        pos_tokens = _map_to_tokens(pos)
        rgb_out = self._cross(
            rgb_tokens,
            thermal_tokens,
            pos_tokens,
            self.q_rgb,
            self.kv_thermal,
            self.out_rgb,
        )
        thermal_out = self._cross(
            thermal_tokens,
            rgb_tokens,
            pos_tokens,
            self.q_thermal,
            self.kv_rgb,
            self.out_thermal,
        )
        return _tokens_to_map(rgb_out, height, width), _tokens_to_map(thermal_out, height, width)


class DepthwiseConv2d(nn.Module):
    """3x3 depthwise convolution. The output replaces the input, with no residual."""

    def __init__(self, dim: int = M1_DIM):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dw(x)


class SpatialSelfAttention(nn.Module):
    """Pre-LN spatial self-attention. The output replaces the input, with no residual."""

    def __init__(self, dim: int = M1_DIM, num_heads: int = M1_HEADS):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        tokens = self.norm(_map_to_tokens(x))
        q, k, v = self.qkv(tokens).chunk(3, dim=-1)
        pos_tokens = _map_to_tokens(pos).to(dtype=q.dtype)
        attended = _attend(q + pos_tokens, k + pos_tokens, v, self.num_heads)
        return _tokens_to_map(self.out(attended), height, width)


class M1(nn.Module):
    """Fuse RGB and thermal visual backbone features.

    Arguments:
        rgb_backbone_out: ``backbone.forward_image`` output of the RGB model.
        t_backbone_out: output of the thermal model, with the same structure.

    Values inside each dictionary are tensors, lists, dictionaries, or None.
    Keys include ``vision_features``, ``vision_pos_enc``, ``backbone_fpn``,
    and ``sam2_backbone_out``. Text features do not enter this module.
    Backbone weights are stored in bf16, and the incoming tensors are usually bf16.

    Only the last ``backbone_fpn`` level is changed. After scalp it is 72x72
    with 256 channels. Position encoding comes from the RGB
    ``vision_pos_enc[-1]`` and is added to Q and K.
    Cross-attention, depthwise convolution, and spatial self-attention have no residual.
    Fusion is ``fused = (R0 + T0) / 2 + Conv1x1(cat(R, T))``.
    ``R0`` and ``T0`` are the last-level features before cross-attention.
    The 1x1 weight and bias are initialized to zero.

    Returns a new dictionary with the same structure for ``forward_grounding``.
    ``vision_features`` and ``backbone_fpn[-1]`` are the same fused tensor.
    Earlier FPN levels, position encodings, and ``sam2_backbone_out`` are copied
    from the RGB dictionary. The input dictionaries are not modified.
    The returned tensor is connected to this module's parameters, so training updates M1.
    """

    def __init__(self, dim: int = M1_DIM, num_heads: int = M1_HEADS):
        super().__init__()
        self.dim = dim
        self.cross = BidirectionalCrossAttention(dim, num_heads)
        self.dw_rgb = DepthwiseConv2d(dim)
        self.dw_thermal = DepthwiseConv2d(dim)
        self.self_rgb = SpatialSelfAttention(dim, num_heads)
        self.self_thermal = SpatialSelfAttention(dim, num_heads)
        self.fuse = nn.Conv2d(dim * 2, dim, kernel_size=1)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)

    def forward(self, rgb_backbone_out: dict, t_backbone_out: dict) -> dict:
        rgb_fpn = rgb_backbone_out["backbone_fpn"]
        thermal_fpn = t_backbone_out["backbone_fpn"]
        rgb_pos = rgb_backbone_out["vision_pos_enc"]
        if len(rgb_fpn) == 0 or len(thermal_fpn) == 0 or len(rgb_pos) == 0:
            raise RuntimeError("backbone_fpn and vision_pos_enc must not be empty.")

        rgb = rgb_fpn[-1]
        thermal = thermal_fpn[-1]
        pos = rgb_pos[-1]
        if rgb.shape != thermal.shape:
            raise RuntimeError(
                f"RGB and thermal last-level shapes must match, got {tuple(rgb.shape)} and {tuple(thermal.shape)}"
            )
        if rgb.dim() != 4 or rgb.shape[1] != self.dim:
            raise RuntimeError(f"Last level must be (B, {self.dim}, H, W), got {tuple(rgb.shape)}")
        if pos.shape != rgb.shape:
            raise RuntimeError(
                f"Position encoding shape must match the last level, got {tuple(pos.shape)} and {tuple(rgb.shape)}"
            )

        rgb_aligned, thermal_aligned = self.cross(rgb, thermal, pos)
        rgb_refined = self.self_rgb(self.dw_rgb(rgb_aligned), pos)
        thermal_refined = self.self_thermal(self.dw_thermal(thermal_aligned), pos)
        fused = (rgb + thermal) / 2 + self.fuse(torch.cat((rgb_refined, thermal_refined), dim=1))

        output = dict(rgb_backbone_out)
        fpn = list(rgb_fpn)
        fpn[-1] = fused
        output["backbone_fpn"] = fpn
        output["vision_features"] = fused
        return output
