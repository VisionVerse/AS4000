"""Evaluate IoU, F1, Pd, and Fa on saved probability maps."""
from __future__ import annotations
import argparse
import json
import os
import cv2
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
from glob import glob
from pathlib import Path
from tqdm import tqdm
import paths
from metrics import AS4000Metrics


METRIC_ORDER = ("IoU", "F1", "Pd", "Fa")


def round_half_up(value, places=5):
    """Round half away from zero and return a fixed-width string."""
    quant = Decimal("1").scaleb(-places)
    number = Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP)
    return f"{number:.{places}f}"


def _pairs(pred_dir, gt_dir):
    pred_files = sorted(glob(os.path.join(pred_dir, "*")))
    gt_files = sorted(glob(os.path.join(gt_dir, "*")))
    if not gt_files:
        raise RuntimeError(f"No ground-truth files found: {gt_dir}")
    gt_map = {os.path.splitext(os.path.basename(path))[0].lower(): path for path in gt_files}
    pairs = []
    for pred_path in pred_files:
        name = os.path.splitext(os.path.basename(pred_path))[0].lower()
        gt_path = gt_map.get(name)
        if gt_path is not None:
            pairs.append((pred_path, gt_path))
    if not pairs:
        raise RuntimeError(f"No matched prediction/GT pairs in {pred_dir} and {gt_dir}")
    return pairs


def _read_pair(pred_path, gt_path):
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if pred is None or gt is None:
        raise RuntimeError(f"Failed to load: {os.path.basename(pred_path)}")
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
    return pred.astype(np.float32) / 255.0, gt


def evaluate_all(pred_root, gt_dir, match_distance=3):
    """IoU, F1, Pd, and Fa. Prediction maps store probabilities as 0-255."""
    metric = AS4000Metrics(match_distance=match_distance)
    for pred_path, gt_path in tqdm(_pairs(pred_root, gt_dir), desc="Computing IoU/F1/Pd/Fa"):
        pred, gt = _read_pair(pred_path, gt_path)
        probability = np.clip(pred, 0.0, 1.0)
        metric.update(probability, gt > 128)
    results = metric.get()
    missing = [name for name in METRIC_ORDER if name not in results]
    if missing:
        raise RuntimeError(f"Missing metrics: {', '.join(missing)}")
    return {name: results[name] for name in METRIC_ORDER}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate IoU / F1 / Pd / Fa")
    parser.add_argument("--pred_root", type=str, default="test_maps")
    parser.add_argument("--gt_dir", type=str, default=str(Path(paths.TEST_DIR) / "GT"))
    parser.add_argument("--match_distance", type=float, default=3)
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    if not gt_dir.is_dir():
        paths.require_dir(paths.TEST_DIR, "test set", paths.TEST_URL, "TEST_URL")
        raise FileNotFoundError(f"Missing ground-truth directory: {gt_dir}")

    results = evaluate_all(args.pred_root, str(gt_dir), match_distance=args.match_distance)
    formatted = {
        "AS4000": {
            metric: round_half_up(value, 8 if metric == "Fa" else 5)
            for metric, value in results.items()
        }
    }
    print(json.dumps(formatted, indent=2))
