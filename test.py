"""Export test-set probability maps with the fully supervised RGB-T model."""
from __future__ import annotations
import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
import paths
from bisf import M1
from cm_lora import build_frozen_lora_model, build_shared_lora
from data import get_loader
from utils import encode_visual_text, forward_sample, make_find_stage, prediction_probability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=paths.CHECKPOINT, help="Trained M1 and LoRA3 weights")
    parser.add_argument("--sam_checkpoint", type=Path, default=paths.SAM3_CHECKPOINT)
    parser.add_argument("--rgb_lora", type=Path, default=paths.RGB_LORA)
    parser.add_argument("--thermal_lora", type=Path, default=paths.THERMAL_LORA)
    parser.add_argument("--img_size", type=int, default=1008)
    parser.add_argument("--save_dir", type=Path, default=Path("test_maps"))
    parser.add_argument("--test_data_dir", type=Path, default=paths.TEST_DIR, help="Test split with RGB, T, box, and GT")
    return parser.parse_args()


def load_trained_weights(state: dict, rgb_model, t_model, device: torch.device) -> M1:
    m1_state = state.get("m1")
    lora3_state = state.get("lora3")
    if not m1_state:
        raise RuntimeError("Checkpoint does not contain M1 weights.")
    if not lora3_state:
        raise RuntimeError("Checkpoint does not contain shared LoRA3 weights.")
    m1 = M1().to(device)
    m1.load_state_dict(m1_state)
    m1.eval()
    lora3 = build_shared_lora(rgb_model, t_model)
    lora3.load_state_dict(lora3_state)
    lora3.to(device)
    lora3.eval()
    print("Loaded M1 and LoRA3")
    return m1


@torch.no_grad()
def infer_dataset(
    rgb_model,
    t_model,
    m1: M1,
    text_outputs: dict,
    find_stage,
    dataset_root: Path,
    out_dir: Path,
    img_size: int,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = get_loader(
        dataset_root,
        image_size=img_size,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    print(f"    Dataset: {dataset_root.name} | Samples: {len(loader)}")

    for batch in loader:
        rgb = batch["rgb"].unsqueeze(0).cuda(non_blocking=True)
        thermal = batch["t"].unsqueeze(0).cuda(non_blocking=True)
        gt_boxes = batch["boxes"].cuda(non_blocking=True)
        gt_mask = batch["gt_mask"]
        pred_logit = forward_sample(
            rgb,
            thermal,
            gt_boxes,
            rgb_model,
            t_model,
            m1,
            text_outputs,
            find_stage,
        )
        probability = prediction_probability(pred_logit, gt_mask).cpu().numpy()
        image = (probability * 255).round().astype(np.uint8)
        cv2.imwrite(str(out_dir / f"{batch['stem']}.png"), image)
    return len(loader)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    device = torch.device("cuda")
    paths.require_dir(args.test_data_dir, "test set", paths.TEST_URL, "TEST_URL")
    paths.require_file(args.checkpoint, "trained M1 and LoRA3 checkpoint", paths.WEIGHT_URL, "WEIGHT_URL")
    paths.require_file(args.sam_checkpoint, "SAM3 checkpoint", paths.WEIGHT_URL, "WEIGHT_URL")
    paths.require_file(args.rgb_lora, "RGB LoRA weights", paths.WEIGHT_URL, "WEIGHT_URL")
    paths.require_file(args.thermal_lora, "thermal LoRA weights", paths.WEIGHT_URL, "WEIGHT_URL")

    print("\n=== Testing checkpoint ===")
    print(f"Checkpoint: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint must be a dictionary with m1 and lora3 weights.")

    rgb_model = build_frozen_lora_model(args.sam_checkpoint, args.rgb_lora, device)
    t_model = build_frozen_lora_model(args.sam_checkpoint, args.thermal_lora, device)
    m1 = load_trained_weights(state, rgb_model, t_model, device)
    text_outputs = encode_visual_text(rgb_model, device)
    find_stage = make_find_stage(device)

    print("  -> Inferring AS4000 test...")
    count = infer_dataset(
        rgb_model,
        t_model,
        m1,
        text_outputs,
        find_stage,
        args.test_data_dir,
        args.save_dir,
        args.img_size,
    )
    print(f"  -> Saved {count} maps to: {args.save_dir}")
    print("\nAll tasks completed.")


if __name__ == "__main__":
    main()
