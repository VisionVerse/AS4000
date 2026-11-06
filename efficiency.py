"""Measure Params(M), GFLOPs, and FPS of RGB LoRA, thermal LoRA, LoRA3, and M1.

Input shapes come from one forward pass with the same path as test.py.
Only these four modules are counted. LoRA3 runs on both RGB and thermal features,
so its compute is counted twice. GFLOPs use multiply-adds divided by 1e9,
and one multiply-add counts as 1.
"""
from __future__ import annotations
import argparse
import torch
import torch.nn as nn
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import NamedTuple
from torch.utils.flop_counter import FlopCounterMode
import paths
from bisf import M1
from cm_lora import build_frozen_lora_model, build_shared_lora
from lora_layers import LoRALayer
from utils import encode_visual_text, forward_sample, make_find_stage


class TensorSpec(NamedTuple):
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


class Part(NamedTuple):
    label: str
    params: int
    calls: tuple[tuple[nn.Module, TensorSpec], ...] = ()
    features: tuple[TensorSpec, TensorSpec, TensorSpec] | None = None


def at_least(minimum: int) -> Callable[[str], int]:
    def parse(value: str) -> int:
        number = int(value)
        if number < minimum:
            raise argparse.ArgumentTypeError(f"must be >= {minimum}")
        return number

    return parse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--sam_checkpoint", type=Path, default=paths.SAM3_CHECKPOINT)
    parser.add_argument("--rgb_lora", type=Path, default=paths.RGB_LORA)
    parser.add_argument("--thermal_lora", type=Path, default=paths.THERMAL_LORA)
    parser.add_argument("--img_size", type=at_least(1), default=1008)
    parser.add_argument("--num_boxes", type=at_least(1), default=1, help="Number of geometric prompt boxes")
    parser.add_argument("--warmup", type=at_least(0), default=10)
    parser.add_argument("--iters", type=at_least(1), default=50)
    return parser.parse_args()


def parameter_count(module: nn.Module, *, contains: str | None = None) -> int:
    return sum(
        param.numel()
        for name, param in module.named_parameters()
        if contains is None or contains in name
    )


def millions(count: int) -> str:
    return f"{count / 1e6:.2f}"


def lora_layers(module: nn.Module | None) -> tuple[nn.Module, ...]:
    if module is None:
        return ()
    return tuple(child for child in module.modules() if isinstance(child, LoRALayer))


def as_spec(tensor: torch.Tensor) -> TensorSpec:
    return TensorSpec(tuple(tensor.shape), tensor.dtype, tensor.device)


def missed_ops(counter: FlopCounterMode) -> list[str]:
    # The attribute name in this PyTorch version omits one letter.
    names = getattr(counter, "unspported_ops", None)
    if names is None:
        names = getattr(counter, "unsupported_ops", ())
    return sorted(map(str, names))


def measure_gflops(run: Callable[[], None]) -> tuple[float, list[str]]:
    """GFLOPs are multiply-adds / 1e9. One multiply-add counts as 1."""
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        run()
    torch.cuda.synchronize()
    gflops = counter.get_total_flops() / 2 / 1e9
    if gflops <= 0:
        raise RuntimeError("GFLOPs are 0. The forward pass was not recorded by the operator counter.")
    return gflops, missed_ops(counter)


def measure_fps(run: Callable[[], None], warmup: int, iters: int) -> tuple[float, float]:
    with torch.no_grad():
        for _ in range(warmup):
            run()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            run()
        end.record()
        torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    return iters * 1e3 / elapsed_ms, elapsed_ms / iters


def record_execution(
    rgb_model: nn.Module,
    thermal_model: nn.Module,
    m1: M1,
    lora3: nn.Module,
    run: Callable[[], None],
) -> tuple[tuple[tuple[nn.Module, TensorSpec], ...], tuple[TensorSpec, TensorSpec, TensorSpec]]:
    """Run one full forward pass and keep the input shapes of LoRA and M1."""
    watched = lora_layers(rgb_model) + lora_layers(thermal_model) + lora_layers(lora3)
    calls: list[tuple[nn.Module, TensorSpec]] = []
    features: list[tuple[TensorSpec, TensorSpec, TensorSpec]] = []

    def remember_lora(module: nn.Module, args: tuple) -> None:
        calls.append((module, as_spec(args[0])))

    def remember_m1(_module: nn.Module, args: tuple) -> None:
        rgb_out, thermal_out = args
        features.append(
            (
                as_spec(rgb_out["backbone_fpn"][-1]),
                as_spec(thermal_out["backbone_fpn"][-1]),
                as_spec(rgb_out["vision_pos_enc"][-1]),
            )
        )

    handles = [module.register_forward_pre_hook(remember_lora) for module in watched]
    handles.append(m1.register_forward_pre_hook(remember_m1))
    try:
        with torch.no_grad():
            run()
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    if len(features) != 1:
        raise RuntimeError(f"M1 should be called once, got {len(features)}.")
    return tuple(calls), features[0]


def replay_of(
    calls: Iterable[tuple[nn.Module, TensorSpec]],
    features: tuple[TensorSpec, TensorSpec, TensorSpec] | None,
    m1: M1 | None,
) -> Callable[[], None]:
    """Reuse one buffer for activations of the same shape. These modules do not edit their inputs in place."""
    cache: dict[TensorSpec, torch.Tensor] = {}

    def buffer(spec: TensorSpec) -> torch.Tensor:
        tensor = cache.get(spec)
        if tensor is None:
            tensor = torch.empty(*spec.shape, dtype=spec.dtype, device=spec.device)
            cache[spec] = tensor
        return tensor

    prepared = [(module, buffer(spec)) for module, spec in calls]
    m1_inputs = None
    if m1 is not None and features is not None:
        rgb, thermal, pos = (buffer(spec) for spec in features)
        m1_inputs = (
            {"backbone_fpn": [rgb], "vision_pos_enc": [pos], "vision_features": rgb},
            {"backbone_fpn": [thermal], "vision_pos_enc": [pos], "vision_features": thermal},
        )

    def replay() -> None:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for module, tensor in prepared:
                module(tensor)
            if m1_inputs is not None:
                m1(*m1_inputs)

    return replay


def calls_of(
    calls: tuple[tuple[nn.Module, TensorSpec], ...], modules: Iterable[nn.Module]
) -> tuple[tuple[nn.Module, TensorSpec], ...]:
    selected = {id(module) for module in modules}
    return tuple((module, spec) for module, spec in calls if id(module) in selected)


def preview(names: list[str], limit: int = 30) -> str:
    shown = names[:limit]
    text = ", ".join(shown)
    rest = len(names) - len(shown)
    if rest:
        text += f" ... and {rest} more"
    return text


def report(
    rows: list[tuple[str, int, float]],
    fps: float,
    latency_ms: float,
    missed: list[str],
) -> None:
    params = sum(count for _, count, _ in rows)
    gflops = sum(value for _, _, value in rows)
    lines = [
        "",
        f"{'Params(M)':>12} {'GFLOPs':>12} {'FPS':>12}",
        f"{params / 1e6:12.2f} {gflops:12.2f} {fps:12.2f}",
        "",
        f"Latency per image: {latency_ms:.2f} ms",
        "GFLOPs are multiply-adds / 1e9, and one multiply-add counts as 1. "
        "The SAM3 trunk, decoder, and text encoder are excluded.",
    ]
    for label, count, value in rows:
        lines.append(f"{label}: {millions(count)} M, {value:.2f} GFLOPs")
    if missed:
        lines.append(f"Operators excluded from GFLOPs: {preview(list(dict.fromkeys(missed)))}")
    print("\n".join(lines))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    paths.require_file(args.sam_checkpoint, "SAM3 checkpoint", paths.WEIGHT_URL, "WEIGHT_URL")
    paths.require_file(args.rgb_lora, "RGB LoRA weights", paths.WEIGHT_URL, "WEIGHT_URL")
    paths.require_file(args.thermal_lora, "thermal LoRA weights", paths.WEIGHT_URL, "WEIGHT_URL")
    device = torch.device("cuda")

    rgb_model = build_frozen_lora_model(args.sam_checkpoint, args.rgb_lora, device)
    thermal_model = build_frozen_lora_model(args.sam_checkpoint, args.thermal_lora, device)
    lora3 = build_shared_lora(rgb_model, thermal_model).to(device)
    m1 = M1().to(device)
    for module in (rgb_model, thermal_model, m1, lora3):
        module.eval()

    text_outputs = encode_visual_text(rgb_model, device)
    find_stage = make_find_stage(device)
    rgb = torch.randn(1, 3, args.img_size, args.img_size, device=device)
    thermal = torch.randn_like(rgb)
    boxes = torch.tensor([[0.5, 0.5, 0.4, 0.4]], device=device).repeat(args.num_boxes, 1)

    def run() -> None:
        forward_sample(
            rgb,
            thermal,
            boxes.clone(),
            rgb_model,
            thermal_model,
            m1,
            text_outputs,
            find_stage,
        )

    print("Scope: RGB LoRA, thermal LoRA, LoRA3, M1")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Input: RGB/T 1x3x{args.img_size}x{args.img_size}, boxes={args.num_boxes}, bf16")
    print("Recording module input shapes from one full forward pass...")
    calls, features = record_execution(rgb_model, thermal_model, m1, lora3, run)
    if not calls:
        raise RuntimeError("LoRA did not run in the forward pass.")

    parts = (
        Part("RGB LoRA", parameter_count(rgb_model, contains="lora"), calls_of(calls, lora_layers(rgb_model))),
        Part("Thermal LoRA", parameter_count(thermal_model, contains="lora"), calls_of(calls, lora_layers(thermal_model))),
        Part("LoRA3", parameter_count(lora3), calls_of(calls, lora_layers(lora3))),
        Part("M1", parameter_count(m1), features=features),
    )
    for part in parts:
        if not part.calls and part.features is None:
            raise RuntimeError(f"{part.label} did not run in the forward pass.")

    print("Counting GFLOPs...")
    rows: list[tuple[str, int, float]] = []
    missed: list[str] = []
    for part in parts:
        value, names = measure_gflops(replay_of(part.calls, part.features, m1 if part.features is not None else None))
        rows.append((part.label, part.params, value))
        missed.extend(names)

    torch.cuda.empty_cache()
    print(f"Measuring FPS, warmup={args.warmup}, iters={args.iters}...")
    fps, latency_ms = measure_fps(replay_of(calls, features, m1), args.warmup, args.iters)
    report(rows, fps, latency_ms, missed)


if __name__ == "__main__":
    main()
