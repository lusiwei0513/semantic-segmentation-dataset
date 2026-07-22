"""Dataset and augmentations for semantic segmentation (letterbox + strong aug)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

SizeLike = Union[int, Sequence[int]]


def parse_image_size(image_size: SizeLike) -> Tuple[int, int]:
    """Return (height, width). int -> square; [H,W] or [W] not allowed."""
    if isinstance(image_size, (list, tuple)):
        if len(image_size) != 2:
            raise ValueError(f"image_size list must be [H, W], got {image_size}")
        return int(image_size[0]), int(image_size[1])
    return int(image_size), int(image_size)


def _imread_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def _imread_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("L"))


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: SizeLike, train: bool, strong_aug: bool = True) -> A.Compose:
    """
    Letterbox into target canvas (no stretch):
      - int S  -> S×S
      - [H, W] -> H×W（侧视推荐 [384, 1536]）
    scale = min(W/w0, H/h0), then pad.
    """
    th, tw = parse_image_size(image_size)

    def _fit_box_image(image, **kwargs):
        h0, w0 = image.shape[:2]
        scale = min(tw / float(w0), th / float(h0))
        nh = max(1, int(round(h0 * scale)))
        nw = max(1, int(round(w0 * scale)))
        return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    def _fit_box_mask(mask, **kwargs):
        h0, w0 = mask.shape[:2]
        scale = min(tw / float(w0), th / float(h0))
        nh = max(1, int(round(h0 * scale)))
        nw = max(1, int(round(w0 * scale)))
        return cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

    geo = [
        A.Lambda(name="fit_box", image=_fit_box_image, mask=_fit_box_mask),
        A.PadIfNeeded(
            min_height=th,
            min_width=tw,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
        ),
    ]
    if train:
        aug = [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.08,
                scale_limit=0.25,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                p=0.7 if strong_aug else 0.5,
            ),
            A.PadIfNeeded(
                min_height=th,
                min_width=tw,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=25, val_shift_limit=20, p=1.0),
                    A.CLAHE(p=1.0),
                ],
                p=0.7 if strong_aug else 0.4,
            ),
            A.GaussNoise(std_range=(0.05, 0.2), p=0.35 if strong_aug else 0.2),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(8, 40),
                hole_width_range=(8, 40),
                fill=0,
                fill_mask=0,
                p=0.35 if strong_aug else 0.15,
            ),
        ]
        return A.Compose(geo + aug + [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])
    return A.Compose(geo + [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])


class SegDataset(Dataset):
    def __init__(
        self,
        items: list[tuple[Path, Path]],
        image_size: SizeLike,
        train: bool,
        strong_aug: bool = True,
    ):
        self.items = items
        self.image_size = image_size
        self.image_hw = parse_image_size(image_size)
        self.tf = build_transforms(image_size, train=train, strong_aug=strong_aug)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.items[idx]
        image = _imread_rgb(img_path)
        mask = _imread_mask(mask_path)
        h0, w0 = image.shape[:2]
        out = self.tf(image=image, mask=mask)
        return {
            "image": out["image"].float(),
            "mask": out["mask"].long(),
            "stem": img_path.stem,
            "image_h": h0,
            "image_w": w0,
            "img_path": str(img_path),
            "mask_path": str(mask_path),
        }


def list_pairs(prepared_dir: Path) -> list[tuple[Path, Path]]:
    images_dir = prepared_dir / "images"
    masks_dir = prepared_dir / "masks"
    pairs = []
    for mask_path in sorted(masks_dir.glob("*.png")):
        stem = mask_path.stem
        candidates = list(images_dir.glob(f"{stem}.*"))
        if not candidates:
            continue
        pairs.append((candidates[0], mask_path))
    return pairs


def split_pairs(
    pairs: list[tuple[Path, Path]], val_ratio: float, seed: int
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    from sklearn.model_selection import train_test_split

    if len(pairs) < 2:
        return pairs, []
    train_pairs, val_pairs = train_test_split(
        pairs, test_size=val_ratio, random_state=seed, shuffle=True
    )
    return train_pairs, val_pairs


def split_pairs_three_way(
    pairs: list[tuple[Path, Path]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list, list, list]:
    """Fixed train/val/test once (not reshuffled each epoch)."""
    from sklearn.model_selection import train_test_split

    if len(pairs) < 3:
        tr, va = split_pairs(pairs, val_ratio, seed)
        return tr, va, []
    trainval, test = train_test_split(pairs, test_size=test_ratio, random_state=seed, shuffle=True)
    rel_val = val_ratio / max(1e-6, (1.0 - test_ratio))
    train, val = train_test_split(trainval, test_size=rel_val, random_state=seed + 1, shuffle=True)
    return train, val, test


def load_fold_pairs(
    prepared_dir: Path,
    fold_json: Path,
    split: str = "train",
) -> list[tuple[Path, Path]]:
    """Load pairs using data/splits/fold_*.json sample_id lists."""
    fold = json.loads(fold_json.read_text(encoding="utf-8"))
    ids = set(fold.get(split, []))
    all_pairs = list_pairs(prepared_dir)
    return [(i, m) for i, m in all_pairs if i.stem in ids]
