"""Fully supervised RGB-T training.

SAM3 and the two modality LoRA modules stay frozen. LoRA3 is built in cm_lora.py
and M1 fusion is built in bisf.py. Both are trained. M1 fuses RGB and thermal
image features, and the ground-truth boxes are passed to the decoder once.
The predicted mask is supervised by the dataset GT with structure loss,
sigmoid focal loss, and Dice loss.

One GPU:

    python train.py

Several GPUs, one image per GPU, with gradients averaged across GPUs:

    torchrun --nproc_per_node=4 train.py
"""
from __future__ import annotations
import argparse
import logging
import torch
import torch.distributed as dist
from datetime import datetime
from pathlib import Path
import paths
from bisf import M1
from cm_lora import build_frozen_lora_model, build_shared_lora, enable_trunk_checkpoint
from data import get_loader, resize_gt, set_loader_epoch
from utils import (
    adjust_lr,
    all_reduce_,
    broadcast_module,
    calc_mae,
    count_trainable,
    dataset_iou,
    dice_loss,
    distributed,
    encode_visual_text,
    format_metric,
    forward_sample,
    init_distributed,
    iou_counts,
    is_main_process,
    make_find_stage,
    prediction_probability,
    require_equal_steps,
    require_trainable,
    save_checkpoint,
    setup_logger,
    sigmoid_focal_loss,
    structure_loss,
    sync_gradients,
    upsample_logits,
)


torch.multiprocessing.set_sharing_strategy("file_system")


def train_one_epoch(
    loader,
    rgb_model,
    t_model,
    m1,
    lora3,
    text_outputs,
    find_stage,
    optimizer,
    epoch: int,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    set_loader_epoch(loader, epoch)

    rgb_model.eval()
    t_model.eval()
    m1.train()
    lora3.train()
    enable_trunk_checkpoint(rgb_model)
    enable_trunk_checkpoint(t_model)

    total_loss = 0.0
    total_step = len(loader)
    trainable = [param for group in optimizer.param_groups for param in group["params"]]
    for step, batch in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        rgb = batch["rgb"].unsqueeze(0).to(device, non_blocking=True)
        thermal = batch["t"].unsqueeze(0).to(device, non_blocking=True)
        gt_boxes = batch["boxes"].to(device, non_blocking=True)
        gt_mask = batch["gt_mask"].to(device, non_blocking=True)

        pred_logit = forward_sample(
            rgb,
            thermal,
            gt_boxes,
            rgb_model,
            t_model,
            m1,
            text_outputs,
            find_stage,
            grad_backbone=True,
        )
        pred_map = upsample_logits(pred_logit, rgb.shape[-2:])
        gt = resize_gt(gt_mask, rgb.shape[-2:])

        str_term = args.str_weight * structure_loss(pred_map, gt)
        focal_term = args.focal_weight * sigmoid_focal_loss(pred_map, gt)
        dice_term = args.dice_weight * dice_loss(pred_map, gt)
        loss = str_term + focal_term + dice_term
        loss.backward()
        sync_gradients(trainable)
        optimizer.step()

        total_loss += float(loss.item())
        if step in (1, total_step) or step % args.log_every == 0:
            stats = torch.stack(
                [str_term.detach(), focal_term.detach(), dice_term.detach(), loss.detach()]
            ).float()
            if distributed():
                all_reduce_(stats, dist.ReduceOp.SUM)
                stats.div_(dist.get_world_size())
            str_value, focal_value, dice_value, loss_value = (float(value) for value in stats.tolist())
            message = (
                f"{datetime.now()} Epoch [{epoch:03d}/{args.epoch:03d}], "
                f"Step [{step:04d}/{total_step:04d}], "
                f"str: {str_value:.6f}, "
                f"focal: {focal_value:.6f}, "
                f"dice: {dice_value:.6f}, "
                f"total: {loss_value:.6f}"
            )
            if is_main_process():
                print(message)
            logging.info("#TRAIN# %s", message)

    totals = torch.tensor([total_loss, float(total_step)], dtype=torch.float64, device=device)
    if distributed():
        all_reduce_(totals, dist.ReduceOp.SUM)
    return float(totals[0].item() / max(totals[1].item(), 1.0))


@torch.no_grad()
def validate(
    loader,
    rgb_model,
    t_model,
    m1,
    lora3,
    text_outputs,
    find_stage,
    device: torch.device,
) -> dict:
    rgb_model.eval()
    t_model.eval()
    m1.eval()
    lora3.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_mae = 0.0
    count = 0
    for batch in loader:
        rgb = batch["rgb"].unsqueeze(0).to(device, non_blocking=True)
        thermal = batch["t"].unsqueeze(0).to(device, non_blocking=True)
        gt_boxes = batch["boxes"].to(device, non_blocking=True)
        gt_mask = batch["gt_mask"]
        orig_h, orig_w = batch["orig_size"]
        if tuple(gt_mask.shape[-2:]) != (orig_h, orig_w):
            raise RuntimeError(
                f"{batch['stem']} GT size {tuple(gt_mask.shape[-2:])} does not match RGB size {(orig_h, orig_w)}"
            )

        pred_logit = forward_sample(
            rgb, thermal, gt_boxes, rgb_model, t_model, m1, text_outputs, find_stage
        )
        probability = prediction_probability(pred_logit, gt_mask)
        tp, fp, fn = iou_counts(probability, gt_mask)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_mae += calc_mae(probability, gt_mask)
        count += 1

    stats = torch.tensor(
        [float(total_tp), float(total_fp), float(total_fn), total_mae, float(count)],
        dtype=torch.float64,
        device=device,
    )
    if distributed():
        all_reduce_(stats, dist.ReduceOp.SUM)
    tp, fp, fn, mae_sum, samples = stats.tolist()
    samples = int(samples)
    scale = max(samples, 1)
    return {"iou": dataset_iou(int(tp), int(fp), int(fn)), "mae": mae_sum / scale, "samples": samples}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fully supervised RGB-T training of M1 and shared LoRA3 on frozen SAM3 and modality LoRA",
        epilog="Multi-GPU example: torchrun --nproc_per_node=4 train.py",
    )
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batchsize", type=int, default=1, help="Must be 1. Each GPU still takes one image.")
    parser.add_argument("--img_size", type=int, default=1008)
    parser.add_argument("--decay_rate", type=float, default=0.1)
    parser.add_argument("--decay_epoch", type=int, default=50)
    parser.add_argument("--str_weight", type=float, default=1.0)
    parser.add_argument("--focal_weight", type=float, default=1.0)
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, default=str(paths.SAM3_CHECKPOINT))
    parser.add_argument("--rgb_lora", type=str, default=str(paths.RGB_LORA))
    parser.add_argument("--thermal_lora", type=str, default=str(paths.THERMAL_LORA))
    parser.add_argument("--train_data_dir", type=str, default=str(paths.TRAIN_DIR))
    parser.add_argument("--val_data_dir", type=str, default=str(paths.VAL_DIR))
    parser.add_argument("--save_path", type=str, default="train_logs")
    return parser.parse_args()


def main() -> None:
    args = get_args()
    rank, world_size, device = init_distributed()
    try:
        if is_main_process():
            setup_logger(args.save_path, "train.log")
            logging.info("Args: %s", args)
            logging.info("GPUs: %d, batch size per GPU: %d", world_size, args.batchsize)
            print("Learning Rate:", args.lr)
            if world_size > 1:
                print(f"Using {world_size} GPUs, one image per GPU, with averaged gradients.")
        else:
            logging.disable(logging.WARNING)

        paths.require_dir(Path(args.train_data_dir), "training set", paths.TRAIN_URL, "TRAIN_URL")
        paths.require_dir(Path(args.val_data_dir), "validation set", paths.VAL_URL, "VAL_URL")
        checkpoint = Path(args.checkpoint)
        paths.require_file(checkpoint, "SAM3 checkpoint", paths.WEIGHT_URL, "WEIGHT_URL")
        paths.require_file(Path(args.rgb_lora), "RGB LoRA weights", paths.WEIGHT_URL, "WEIGHT_URL")
        paths.require_file(Path(args.thermal_lora), "thermal LoRA weights", paths.WEIGHT_URL, "WEIGHT_URL")
        rgb_model = build_frozen_lora_model(checkpoint, Path(args.rgb_lora), device)
        t_model = build_frozen_lora_model(checkpoint, Path(args.thermal_lora), device)
        lora3 = build_shared_lora(rgb_model, t_model).to(device)
        m1 = M1().to(device)
        if distributed():
            broadcast_module(m1)
            broadcast_module(lora3)
            torch.manual_seed(rank)
            torch.cuda.manual_seed(rank)
        m1_trainable = count_trainable(m1)
        lora3_trainable = count_trainable(lora3)
        rgb_trainable = count_trainable(rgb_model)
        t_trainable = count_trainable(t_model)
        logging.info(
            "Trainable parameters m1=%d lora3=%d; frozen rgb=%d thermal=%d",
            m1_trainable,
            lora3_trainable,
            rgb_trainable,
            t_trainable,
        )
        if rgb_trainable or t_trainable:
            raise RuntimeError("SAM3 and both modality LoRA modules must be frozen.")
        params = require_trainable(m1, "M1")
        params.extend(require_trainable(lora3, "LoRA3"))

        multi_gpu = world_size > 1
        train_loader = get_loader(
            args.train_data_dir,
            image_size=args.img_size,
            batch_size=args.batchsize,
            shuffle=True,
            num_workers=args.num_workers,
            distributed=multi_gpu,
            drop_last=multi_gpu,
        )
        val_loader = get_loader(
            args.val_data_dir,
            image_size=args.img_size,
            batch_size=args.batchsize,
            shuffle=False,
            num_workers=args.num_workers,
            distributed=multi_gpu,
        )
        require_equal_steps(train_loader, device)
        if multi_gpu and is_main_process():
            used = len(train_loader) * world_size
            total = len(train_loader.dataset)
            logging.info("Training set has %d images; each epoch uses %d.", total, used)
            if used != total:
                print(f"Training set size {total} is not divisible by {world_size} GPUs; each epoch uses {used}.")
        text_outputs = encode_visual_text(rgb_model, device)
        find_stage = make_find_stage(device)
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

        best_iou = -1.0
        best_epoch = 0
        checkpoint_path = Path(args.save_path) / "best.pth"
        if is_main_process():
            print("Go! Go! Go!")
        for epoch in range(1, args.epoch + 1):
            lr = adjust_lr(optimizer, args.lr, epoch, decay_rate=args.decay_rate, decay_epoch=args.decay_epoch)
            logging.info("Epoch %d lr %.8f", epoch, lr)
            train_loss = train_one_epoch(
                train_loader,
                rgb_model,
                t_model,
                m1,
                lora3,
                text_outputs,
                find_stage,
                optimizer,
                epoch,
                args,
                device,
            )
            logging.info("#TRAIN# epoch=%d mean_loss=%.6f", epoch, train_loss)

            metrics = validate(
                val_loader,
                rgb_model,
                t_model,
                m1,
                lora3,
                text_outputs,
                find_stage,
                device,
            )
            if metrics["samples"] > 0 and metrics["iou"] > best_iou:
                best_iou = metrics["iou"]
                best_epoch = epoch
                if is_main_process():
                    save_checkpoint(checkpoint_path, m1, epoch, metrics["iou"], lora3)
                    logging.info("Saved best checkpoint %s", checkpoint_path)
            if distributed():
                dist.barrier()

            message = (
                f"Epoch: {epoch} IoU: {format_metric(metrics['iou'])} MAE: {metrics['mae']:.6f} "
                f"bestIoU: {format_metric(best_iou)} bestEpoch: {best_epoch} samples: {metrics['samples']}"
            )
            if is_main_process():
                print(message)
            logging.info("#VAL# %s", message)

        logging.info(
            "Training finished. Best validation IoU %s at epoch %d",
            format_metric(best_iou),
            best_epoch,
        )
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
