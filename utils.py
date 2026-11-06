"""Losses, metrics, logging, distributed helpers, and the forward pass."""
from __future__ import annotations
import logging
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any
from bisf import M1
if TYPE_CHECKING:
    from sam3.model.data_misc import FindStage
    from sam3.model.geometry_encoders import Prompt


FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0


def setup_logger(save_path: str, filename: str = "log.txt") -> None:
    log_dir = Path(save_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(log_dir / filename),
        format="[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]",
        level=logging.INFO,
        filemode="a",
        datefmt="%Y-%m-%d %I:%M:%S %p",
        encoding="utf-8",
        force=True,
    )


def adjust_lr(
    optimizer: torch.optim.Optimizer,
    init_lr: float,
    epoch: int,
    *,
    decay_rate: float = 0.1,
    decay_epoch: int = 30,
) -> float:
    step = epoch // decay_epoch
    lr = init_lr * (decay_rate**step)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def clone_tree(obj: Any) -> Any:
    """Copy a nested structure of tensors, lists, and dicts while keeping gradients."""
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, dict):
        return {key: clone_tree(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [clone_tree(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(clone_tree(value) for value in obj)
    return obj


def structure_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Weighted BCE + weighted IoU. pred contains logits and mask is in [0, 1]."""
    weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    weighted_bce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    weighted_bce = (weight * weighted_bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

    pred_prob = torch.sigmoid(pred)
    inter = ((pred_prob * mask) * weight).sum(dim=(2, 3))
    union = ((pred_prob + mask) * weight).sum(dim=(2, 3))
    weighted_iou = 1 - (inter + 1) / (union - inter + 1)
    return (weighted_bce + weighted_iou).mean()


def calc_mae(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Mean absolute error. pred is in [0, 1] and gt is binary."""
    pred = pred.detach().float().reshape(-1)
    gt = gt.detach().bool().to(dtype=pred.dtype, device=pred.device).reshape(-1)
    return float((pred - gt).abs().mean())


def sigmoid_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = FOCAL_ALPHA,
    gamma: float = FOCAL_GAMMA,
) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = prob * target + (1.0 - prob) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    return (alpha_t * (1.0 - p_t).pow(gamma) * ce).mean()


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = tuple(range(1, prob.dim()))
    inter = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def iou_counts(prob: torch.Tensor, mask: torch.Tensor) -> tuple[int, int, int]:
    """Pixel TP / FP / FN after thresholding probabilities at 0.5."""
    binary = prob.detach().reshape(-1) > 0.5
    mask = mask.detach().reshape(-1).to(device=binary.device).bool()
    tp = binary & mask
    fp = binary & ~mask
    fn = ~binary & mask
    return int(tp.sum().item()), int(fp.sum().item()), int(fn.sum().item())


def dataset_iou(tp: int, fp: int, fn: int) -> float:
    """Dataset-level IoU from accumulated pixel counts."""
    union = tp + fp + fn
    return tp / union if union else 0.0


def format_metric(value: float, places: int = 5) -> str:
    """Round half away from zero to the requested number of decimal places."""
    quant = Decimal("1").scaleb(-places)
    number = Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP)
    return f"{number:.{places}f}"


def upsample_logits(logit: torch.Tensor, size: tuple) -> torch.Tensor:
    return F.interpolate(logit.float()[None, None], size=size, mode="bilinear", align_corners=False)


def prediction_probability(logit: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    """Resize the probability map to the GT resolution and clamp it to [0, 1]."""
    probability = torch.sigmoid(logit.float())[None, None]
    probability = F.interpolate(
        probability,
        size=tuple(gt_mask.shape[-2:]),
        mode="bilinear",
        align_corners=False,
    )
    return probability[0, 0].clamp(0.0, 1.0)


def count_trainable(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def is_main_process() -> bool:
    return not distributed() or dist.get_rank() == 0


def _cpu_collectives() -> bool:
    return dist.get_backend() != "nccl"


def all_reduce_(tensor: torch.Tensor, op: dist.ReduceOp) -> None:
    if not _cpu_collectives():
        dist.all_reduce(tensor, op=op)
        return
    cpu = tensor.detach().to("cpu")
    dist.all_reduce(cpu, op=op)
    tensor.copy_(cpu)


def _broadcast_(tensor: torch.Tensor, src: int = 0) -> None:
    if not _cpu_collectives():
        dist.broadcast(tensor, src=src)
        return
    cpu = tensor.detach().to("cpu")
    dist.broadcast(cpu, src=src)
    tensor.copy_(cpu)


def broadcast_module(module: nn.Module) -> None:
    """Broadcast M1 and LoRA3 so every rank starts from the same initial weights."""
    if not distributed():
        return
    for tensor in list(module.parameters()) + list(module.buffers()):
        _broadcast_(tensor.data)


def sync_gradients(params: list[nn.Parameter]) -> None:
    """Average gradients of trainable parameters after each rank has run backward.

    LoRA3 sits inside the frozen SAM3 modules, so the forward pass does not go
    through one DDP module. Gradients are synchronized here.
    """
    if not distributed() or not params:
        return
    flags = torch.tensor(
        [0 if param.grad is None else 1 for param in params],
        device=params[0].device,
        dtype=torch.int32,
    )
    all_reduce_(flags, dist.ReduceOp.SUM)
    grads: list[torch.Tensor] = []
    for param, count in zip(params, flags.tolist()):
        if int(count) == 0:
            continue
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        grads.append(param.grad)
    if not grads:
        return

    world_size = dist.get_world_size()
    buckets: dict[tuple[torch.dtype, str], list[torch.Tensor]] = {}
    for grad in grads:
        buckets.setdefault((grad.dtype, str(grad.device)), []).append(grad)
    for bucket in buckets.values():
        flat = torch.cat([grad.reshape(-1) for grad in bucket])
        all_reduce_(flat, dist.ReduceOp.SUM)
        flat.div_(world_size)
        offset = 0
        for grad in bucket:
            count = grad.numel()
            grad.copy_(flat[offset:offset + count].view_as(grad))
            offset += count


def init_distributed() -> tuple[int, int, torch.device]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required.")
        torch.cuda.set_device(0)
        return 0, 1, torch.device("cuda", 0)

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Launch multiple GPUs with torchrun, for example: torchrun --nproc_per_node=4 train.py")
    local_rank = int(os.environ["LOCAL_RANK"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(f"LOCAL_RANK={local_rank} exceeds the {torch.cuda.device_count()} visible GPUs.")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        backend = "nccl" if dist.is_nccl_available() else "gloo"
        dist.init_process_group(backend=backend)
    return rank, world_size, torch.device("cuda", local_rank)


def require_equal_steps(loader, device: torch.device) -> None:
    if not distributed():
        return
    local = torch.tensor([len(loader)], device=device, dtype=torch.int64)
    minimum = local.clone()
    maximum = local.clone()
    all_reduce_(minimum, dist.ReduceOp.MIN)
    all_reduce_(maximum, dist.ReduceOp.MAX)
    if int(minimum.item()) != int(maximum.item()):
        raise RuntimeError(
            f"Training step counts differ across ranks: min={int(minimum.item())} max={int(maximum.item())}"
        )


def require_trainable(module: nn.Module, name: str) -> list[nn.Parameter]:
    params = [param for param in module.parameters() if param.requires_grad]
    if not params:
        message = f"{name} has no trainable parameters."
        logging.error(message)
        raise RuntimeError(message)
    return params


def make_find_stage(device: torch.device) -> FindStage:
    # Import after cm_lora has put SAM3 on the module search path.
    from sam3.model.data_misc import FindStage

    return FindStage(
        img_ids=torch.tensor([0], device=device, dtype=torch.long),
        text_ids=torch.tensor([0], device=device, dtype=torch.long),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )


def make_geometric_prompt(boxes_cxcywh: torch.Tensor) -> Prompt:
    """Positive boxes in normalized cxcywh with shape (N, 1, 4). Prompt fills labels and the padding mask."""
    from sam3.model.geometry_encoders import Prompt

    if boxes_cxcywh.dim() != 2 or boxes_cxcywh.shape[-1] != 4:
        raise RuntimeError(f"Boxes must have shape (N, 4), got {tuple(boxes_cxcywh.shape)}")
    count = boxes_cxcywh.shape[0]
    if count < 1:
        raise RuntimeError("At least one box is required before calling the decoder.")
    boxes = boxes_cxcywh.to(dtype=torch.float32).view(count, 1, 4)
    return Prompt(box_embeddings=boxes)


def branch_backbone(fused: dict, text_outputs: dict) -> dict:
    """Copy fused features and text. The text is reused for the whole run and must not be edited in place."""
    features = clone_tree(fused)
    for key, value in text_outputs.items():
        features[key] = clone_tree(value)
    return features


def decode(model: nn.Module, features: dict, boxes_cxcywh: torch.Tensor, find_stage: FindStage) -> dict:
    if model.training:
        raise RuntimeError("SAM3 must stay in eval().")
    return model.forward_grounding(
        backbone_out=features,
        find_input=find_stage,
        find_target=None,
        geometric_prompt=make_geometric_prompt(boxes_cxcywh),
    )


def merge_topk_logits(outputs: dict, top_k: int) -> torch.Tensor:
    """Take the top-k queries by presence-adjusted scores and merge mask logits with a pixel-wise maximum."""
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    pred_logits = outputs["pred_logits"]
    pred_masks = outputs["pred_masks"]
    if pred_logits.dim() != 3 or pred_masks.dim() != 4:
        raise RuntimeError(
            "Unexpected decoder output shapes: "
            f"pred_logits {tuple(pred_logits.shape)}, pred_masks {tuple(pred_masks.shape)}"
        )

    scores = pred_logits.sigmoid()
    presence = outputs.get("presence_logit_dec")
    if presence is not None:
        scores = scores * presence.sigmoid().unsqueeze(1)
    query_scores = scores[0].squeeze(-1)
    query_masks = pred_masks[0]
    k = min(int(top_k), int(query_scores.numel()))
    index = torch.topk(query_scores, k=k).indices
    return query_masks.index_select(0, index).max(dim=0).values


def encode_visual_text(model: nn.Module, device: torch.device) -> dict:
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        return model.backbone.forward_text(["visual"], device=device)


def forward_sample(
    rgb: torch.Tensor,
    thermal: torch.Tensor,
    gt_boxes: torch.Tensor,
    rgb_model: nn.Module,
    t_model: nn.Module,
    m1: M1,
    text_outputs: dict,
    find_stage: FindStage,
    grad_backbone: bool = False,
) -> torch.Tensor:
    grad_context = nullcontext() if grad_backbone else torch.no_grad()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        with grad_context:
            rgb_out = rgb_model.backbone.forward_image(rgb)
            t_out = t_model.backbone.forward_image(thermal)
        fused = m1(rgb_out, t_out)
        outputs = decode(rgb_model, branch_backbone(fused, text_outputs), gt_boxes, find_stage)
    return merge_topk_logits(outputs, int(gt_boxes.shape[0]))


def save_checkpoint(path: Path, m1: M1, epoch: int, iou: float, lora3: nn.Module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "m1": m1.state_dict(),
            "lora3": lora3.state_dict(),
            "epoch": epoch,
            "iou": iou,
        },
        path,
    )
