"""RGB-T box-supervised dataset.

Directory layout for every split:

    <split>/RGB/*.jpg
    <split>/T/*.tif
    <split>/box/*.txt
    <split>/GT/*.png

Images are resized to the training resolution and normalized so the frozen
vision LoRA modules see the same input distribution used when they were trained.
Boxes are converted to normalized cxcywh on load. The batch size is fixed at 1.
With multiple GPUs, each GPU still receives one image and the samples are sharded.
"""
from __future__ import annotations
import logging
import torch
import numpy as np
import torch.nn.functional as F
from collections.abc import Iterator
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from torchvision.transforms import v2


RGB_EXTS = frozenset({".jpg"})
THERMAL_EXTS = frozenset({".tif"})
BOX_EXTS = frozenset({".txt"})
GT_EXTS = frozenset({".png"})


def _list_by_stem(root: Path, exts: frozenset[str]) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Missing directory: {root}")

    found: dict[str, Path] = {}
    for path in root.iterdir():
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        stem = path.stem
        if stem in found:
            logging.warning("Duplicate stem %s, keeping %s and ignoring %s", stem, found[stem], path)
            continue
        found[stem] = path
    if not found:
        raise FileNotFoundError(f"No usable files in directory: {root}")
    return found


def _align(maps: dict[str, dict[str, Path]]) -> list[str]:
    stem_sets = {name: set(items) for name, items in maps.items()}
    common = set.intersection(*stem_sets.values())
    if not common:
        names = ", ".join(maps)
        raise ValueError(f"No shared file stems across modalities: {names}")

    for name, stems in stem_sets.items():
        extra = stems - common
        if extra:
            examples = ", ".join(sorted(extra)[:5])
            raise ValueError(
                f"Modality {name} has {len(extra)} file stems that do not match the others, for example: {examples}"
            )
    return sorted(common)


def load_t_image(image_path: Path) -> Image.Image:
    """Normalize a thermal image for the frozen thermal LoRA.

    The arithmetic order is intentional: subtract
    ``min / (max - min) * 255`` from the original values, then round.
    """
    with Image.open(image_path) as t_image:
        t_array = np.array(t_image)
    t_array = (
        t_array.astype(np.float32) - t_array.min() / (t_array.max() - t_array.min()) * 255.0
    ).round().astype(np.uint8)
    return Image.fromarray(t_array, mode="L").convert("RGB")


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    return height, width


def _load_gt_mask(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.array(image.convert("L"))
    return torch.from_numpy(array > 128)


def _discard_box(path: Path, line_no: int, detail: str) -> None:
    message = f"Discard invalid box {path} line {line_no}: {detail}"
    print(message)
    logging.warning(message)


def read_boxes(path: Path, orig_w: int, orig_h: int, image_size: int) -> tuple[tuple[float, float, float, float], ...]:
    """Read pixel xyxy boxes, scale them to image_size, and convert them to normalized cxcywh."""
    boxes = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 4:
                _discard_box(path, line_no, line)
                continue
            try:
                x1, y1, x2, y2 = map(float, parts[:4])
            except ValueError:
                _discard_box(path, line_no, line)
                continue
            if x2 <= x1 or y2 <= y1:
                _discard_box(path, line_no, line)
                continue

            scale_x = image_size / float(orig_w)
            scale_y = image_size / float(orig_h)
            x1_r = min(max(x1 * scale_x, 0.0), float(image_size))
            y1_r = min(max(y1 * scale_y, 0.0), float(image_size))
            x2_r = min(max(x2 * scale_x, 0.0), float(image_size))
            y2_r = min(max(y2 * scale_y, 0.0), float(image_size))
            if x2_r <= x1_r or y2_r <= y1_r:
                _discard_box(path, line_no, f"empty after scaling to {image_size}")
                continue

            boxes.append(
                (
                    ((x1_r + x2_r) / 2.0) / image_size,
                    ((y1_r + y2_r) / 2.0) / image_size,
                    (x2_r - x1_r) / image_size,
                    (y2_r - y1_r) / image_size,
                )
            )

    if not boxes:
        raise ValueError(f"No valid boxes: {path}")
    return tuple(boxes)


class RGBTBoxDataset(Dataset):
    """One RGB/thermal pair and all of its ground-truth boxes and mask."""

    def __init__(self, split_root: str | Path, image_size: int = 1008) -> None:
        super().__init__()
        split_root = Path(split_root)
        self.image_size = image_size

        maps = {
            "rgb": _list_by_stem(split_root / "RGB", RGB_EXTS),
            "t": _list_by_stem(split_root / "T", THERMAL_EXTS),
            "box": _list_by_stem(split_root / "box", BOX_EXTS),
            "gt": _list_by_stem(split_root / "GT", GT_EXTS),
        }
        stems = _align(maps)

        self.samples: list[dict] = []
        for stem in stems:
            rgb_path = maps["rgb"][stem]
            orig_h, orig_w = _image_size(rgb_path)
            self.samples.append(
                {
                    "stem": stem,
                    "rgb": rgb_path,
                    "t": maps["t"][stem],
                    "boxes": read_boxes(maps["box"][stem], orig_w, orig_h, image_size),
                    "orig_size": (orig_h, orig_w),
                    "gt": maps["gt"][stem],
                }
            )

        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _resize(self, image: Image.Image) -> Image.Image:
        return image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        rgb = self.transform(self._resize(_open_rgb(sample["rgb"])))
        thermal = self.transform(self._resize(load_t_image(sample["t"])))
        return {
            "rgb": rgb,
            "t": thermal,
            "boxes": torch.tensor(sample["boxes"], dtype=torch.float32),
            "stem": sample["stem"],
            "orig_size": sample["orig_size"],
            "gt_mask": _load_gt_mask(sample["gt"]),
        }


def _single_item_collate(batch):
    return batch[0]


def resize_gt(gt_mask: torch.Tensor, size: tuple) -> torch.Tensor:
    """Nearest-neighbor resize of an (H, W) GT mask. The result stays binary."""
    return F.interpolate(gt_mask.float()[None, None], size=size, mode="nearest")


def set_loader_epoch(loader: DataLoader, epoch: int) -> None:
    """Reshuffle multi-GPU training samples at the start of each epoch."""
    sampler = loader.sampler
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(epoch)


class IndexShardSampler(Sampler[int]):
    """Validation sampler. Rank r reads dataset[r::world_size], with no padding and no duplicates."""

    def __init__(self, size: int, rank: int, world_size: int) -> None:
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
        self.indices = list(range(rank, size, world_size))

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def get_loader(
    split_root: str | Path,
    *,
    image_size: int,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    distributed: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    if batch_size != 1:
        raise ValueError(f"batch size must be 1, got {batch_size}")

    dataset = RGBTBoxDataset(split_root, image_size=image_size)
    sampler = None
    loader_shuffle = shuffle
    if distributed:
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("distributed=True requires an initialized process group.")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if shuffle:
            if drop_last and len(dataset) < world_size:
                raise RuntimeError(
                    f"Sample count {len(dataset)} is smaller than GPU count {world_size}, so the samples cannot be assigned."
                )
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=drop_last,
            )
        else:
            sampler = IndexShardSampler(len(dataset), rank, world_size)
        loader_shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=loader_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=_single_item_collate,
    )
