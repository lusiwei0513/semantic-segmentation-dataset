"""Seg + tip heatmap dataset (front / joint). Tip via keypoints JSON or mask centroid."""

from __future__ import annotations

import json
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from src.keypoint import load_nose_tip_xy, make_gaussian_heatmap

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLS_BG, CLS_BODY, CLS_WIN, CLS_TIP = 0, 1, 2, 3


def _imread_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def _imread_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("L"))


def load_tip_xy_kp_json(path: Path | None) -> tuple[float, float] | None:
    if path is None or not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("visible") is False:
        return None
    if "x" in data and "y" in data:
        return float(data["x"]), float(data["y"])
    return None


def tip_from_mask(mask: np.ndarray, tip_id: int = CLS_TIP) -> tuple[float, float] | None:
    ys, xs = np.where(mask == tip_id)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def remap_mask_front(mask: np.ndarray) -> np.ndarray:
    """Drop tip class: tip pixels -> body. Classes stay 0/1/2."""
    out = mask.copy()
    out[out == CLS_TIP] = CLS_BODY
    return out


def remap_mask_joint(mask: np.ndarray) -> np.ndarray:
    """
    Joint full ids: 0 bg, 1 body, 2 win, 3 tip, 4 bogie, 5 door
    -> train seg (5-cls): tip->body, bogie 4->3, door 5->4
    """
    out = mask.copy()
    out[out == CLS_TIP] = CLS_BODY
    out[out == 4] = 3
    out[out == 5] = 4
    return out


def expand_joint_pred(seg5: np.ndarray) -> np.ndarray:
    """Map 5-class pred back to 6-class canvas (tip slot empty = body until painted)."""
    out = seg5.copy()
    out[seg5 == 4] = 5  # door
    out[seg5 == 3] = 4  # bogie
    return out


def build_transforms(image_size: int, train: bool, strong_aug: bool = True) -> A.Compose:
    geo = [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
        ),
    ]
    ops = list(geo)
    if train:
        ops += [
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
            A.OneOf(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=25, val_shift_limit=20, p=1.0),
                    A.CLAHE(p=1.0),
                ],
                p=0.7 if strong_aug else 0.4,
            ),
            A.GaussNoise(std_range=(0.05, 0.2), p=0.35 if strong_aug else 0.2),
            # No CoarseDropout for tip KP: holes hide tip while heatmap GT still peaks.
        ]
    ops += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(
        ops,
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


class SegTipDataset(Dataset):
    def __init__(
        self,
        items: list[tuple[Path, Path, Path | None]],
        image_size: int,
        train: bool,
        sigma: float = 8.0,
        strong_aug: bool = True,
        mode: str = "front",
    ):
        self.items = items
        self.image_size = image_size
        self.train = train
        self.sigma = sigma
        self.mode = mode
        self.tf = build_transforms(image_size, train=train, strong_aug=strong_aug)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, mask_path, tip_src = self.items[idx]
        image = _imread_rgb(img_path)
        mask_full = _imread_mask(mask_path)
        h0, w0 = mask_full.shape[:2]

        tip = None
        if tip_src is not None:
            if tip_src.suffix.lower() == ".json":
                # team keypoints json OR labelme
                tip = load_tip_xy_kp_json(tip_src)
                if tip is None:
                    tip = load_nose_tip_xy(tip_src)
        if tip is None:
            tip = tip_from_mask(mask_full, CLS_TIP)

        keypoints = [tip] if tip is not None else []
        if self.mode == "joint":
            mask_seg = remap_mask_joint(mask_full)
        else:
            mask_seg = remap_mask_front(mask_full)

        out = self.tf(image=image, mask=mask_seg, keypoints=keypoints)

        heat = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        has_tip = 0.0
        tip_xy = np.array([-1.0, -1.0], dtype=np.float32)
        if out["keypoints"]:
            x, y = out["keypoints"][0]
            if -8 <= x < self.image_size + 8 and -8 <= y < self.image_size + 8:
                x = float(np.clip(x, 0, self.image_size - 1))
                y = float(np.clip(y, 0, self.image_size - 1))
                heat = make_gaussian_heatmap(self.image_size, self.image_size, x, y, self.sigma)
                has_tip = 1.0
                tip_xy = np.array([x, y], dtype=np.float32)

        return {
            "image": out["image"].float(),
            "mask": out["mask"].long(),
            "heat": torch.from_numpy(heat).unsqueeze(0),
            "has_tip": torch.tensor(has_tip, dtype=torch.float32),
            "tip_xy": torch.from_numpy(tip_xy),
            "image_h": torch.tensor(float(self.image_size), dtype=torch.float32),
            "stem": img_path.stem,
            "orig_hw": torch.tensor([h0, w0], dtype=torch.int32),
        }


def list_pairs_with_tip(
    prepared_dir: Path,
    keypoints_dir: Path | None = None,
    labelme_dir: Path | None = None,
) -> list[tuple[Path, Path, Path | None]]:
    images_dir = prepared_dir / "images"
    masks_dir = prepared_dir / "masks"
    items = []
    for mask_path in sorted(masks_dir.glob("*.png")):
        stem = mask_path.stem
        cands = list(images_dir.glob(f"{stem}.*"))
        if not cands:
            continue
        tip_src = None
        if keypoints_dir is not None:
            kp = keypoints_dir / f"{stem}.json"
            if kp.exists():
                tip_src = kp
        if tip_src is None and labelme_dir is not None:
            # stem like front_c0210454 -> try hash match in labelme
            hash_id = stem.split("_", 1)[-1] if "_" in stem else stem
            matches = list(labelme_dir.glob(f"{hash_id}*.json"))
            if matches:
                tip_src = matches[0]
        items.append((cands[0], mask_path, tip_src))
    return items


# backward-compatible alias
def list_pairs_with_json(prepared_dir: Path, labelme_dir: Path):
    return list_pairs_with_tip(prepared_dir, labelme_dir=labelme_dir)


def load_fold_items(
    items: list[tuple[Path, Path, Path | None]],
    fold_json: Path,
    split: str,
) -> list[tuple[Path, Path, Path | None]]:
    fold = json.loads(fold_json.read_text(encoding="utf-8"))
    ids = set(fold.get(split, []))
    return [it for it in items if it[0].stem in ids]


def split_pairs(items, val_ratio: float, seed: int):
    from sklearn.model_selection import train_test_split

    if len(items) < 2:
        return items, []
    return train_test_split(items, test_size=val_ratio, random_state=seed, shuffle=True)
