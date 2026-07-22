#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.datasets.heatmap import make_gaussian_heatmap
from src.datasets.transforms import letterbox, maybe_augment, normalize_imagenet

SEG_CHANNELS = ["body", "windshield", "bogie", "door"]


class RailVehicleDataset(Dataset):
    def __init__(
        self,
        metadata_csv: str | Path,
        processed_root: str | Path,
        sample_ids: Optional[List[str]] = None,
        view: Optional[str] = None,
        front_size: Tuple[int, int] = (640, 640),
        side_size: Tuple[int, int] = (512, 1024),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        heatmap_sigma: float = 4.0,
        train: bool = False,
        seed: int = 42,
        keypoint_out_stride: int = 4,
    ):
        self.root = Path(processed_root)
        df = pd.read_csv(metadata_csv)
        if sample_ids is not None:
            df = df[df["sample_id"].isin(sample_ids)].reset_index(drop=True)
        if view is not None:
            df = df[df["view"] == view].reset_index(drop=True)
        self.df = df
        self.front_size = tuple(front_size)  # (H, W)
        self.side_size = tuple(side_size)
        self.mean = mean
        self.std = std
        self.heatmap_sigma = heatmap_sigma
        self.train = train
        self.rng = np.random.RandomState(seed)
        self.keypoint_out_stride = int(keypoint_out_stride)

    def __len__(self) -> int:
        return len(self.df)

    def _load_mask(self, rel: str, h: int, w: int) -> np.ndarray:
        if rel is None or (isinstance(rel, float) and np.isnan(rel)) or str(rel).strip() == "":
            return np.zeros((h, w), dtype=np.uint8)
        arr = np.array(Image.open(self.root / rel))
        return (arr > 0).astype(np.uint8)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        view = str(row["view"])
        image = np.array(Image.open(self.root / row["image_path"]).convert("RGB"))
        h, w = image.shape[:2]

        masks = []
        valid_seg = []
        for ch, col, vcol in [
            ("body", "body_mask", "valid_body"),
            ("windshield", "windshield_mask", "valid_windshield"),
            ("bogie", "bogie_mask", "valid_bogie"),
            ("door", "door_mask", "valid_door"),
        ]:
            valid = int(row[vcol]) == 1
            valid_seg.append(1.0 if valid else 0.0)
            if valid:
                masks.append(self._load_mask(row[col], h, w))
            else:
                # 占位全零，但 valid=0，损失侧必须屏蔽
                masks.append(np.zeros((h, w), dtype=np.uint8))

        kp_xy = None
        valid_tip = int(row["valid_nose_tip"]) == 1
        if valid_tip:
            tip = json.loads((self.root / row["keypoint_path"]).read_text(encoding="utf-8"))
            kp_xy = (float(tip["x"]), float(tip["y"]))

        image, masks, kp_xy = maybe_augment(
            image, masks, kp_xy, train=self.train, rng=self.rng
        )

        target_size = self.front_size if view == "front" else self.side_size
        packed = letterbox(image, masks, kp_xy, target_size=target_size)
        image = packed["image"]
        masks = packed["masks"]
        kp_xy = packed["keypoint_xy"]
        meta = packed["meta"]

        th, tw = target_size
        # 关键点热图用 1/stride 分辨率，显著降低显存与计算
        stride = max(1, self.keypoint_out_stride)
        hk, wk = th // stride, tw // stride
        heatmap = np.zeros((hk, wk), dtype=np.float32)
        if valid_tip and kp_xy is not None:
            kp_s = (kp_xy[0] / stride, kp_xy[1] / stride)
            heatmap = make_gaussian_heatmap(
                hk, wk, kp_s, sigma=max(1.0, self.heatmap_sigma / stride)
            )

        image_t = torch.from_numpy(normalize_imagenet(image, self.mean, self.std))
        masks_t = torch.from_numpy(np.stack(masks, axis=0).astype(np.float32))
        valid_seg_t = torch.tensor(valid_seg, dtype=torch.float32)
        heatmap_t = torch.from_numpy(heatmap[None, ...])
        valid_tip_t = torch.tensor([1.0 if valid_tip else 0.0], dtype=torch.float32)
        # tip xy in heatmap coords (for soft-argmax L1); invalid -> zeros
        if valid_tip and kp_xy is not None:
            tip_xy_t = torch.tensor(
                [kp_xy[0] / stride, kp_xy[1] / stride], dtype=torch.float32
            )
        else:
            tip_xy_t = torch.zeros(2, dtype=torch.float32)

        return {
            "image": image_t,
            "segmentation": masks_t,
            "valid_seg_tasks": valid_seg_t,
            "nose_tip_heatmap": heatmap_t,
            "valid_nose_tip": valid_tip_t,
            "nose_tip_xy": tip_xy_t,
            "view": view,
            "sample_id": row["sample_id"],
            "vehicle_id": row["vehicle_id"],
            "letterbox_meta": meta,
        }


def collate_same_view(batch: List[Dict]) -> Dict:
    views = {b["view"] for b in batch}
    if len(views) != 1:
        raise ValueError(f"同一 batch 必须同 view，收到 {views}")
    out = {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "segmentation": torch.stack([b["segmentation"] for b in batch], dim=0),
        "valid_seg_tasks": torch.stack([b["valid_seg_tasks"] for b in batch], dim=0),
        "nose_tip_heatmap": torch.stack([b["nose_tip_heatmap"] for b in batch], dim=0),
        "valid_nose_tip": torch.stack([b["valid_nose_tip"] for b in batch], dim=0),
        "nose_tip_xy": torch.stack([b["nose_tip_xy"] for b in batch], dim=0),
        "view": batch[0]["view"],
        "sample_id": [b["sample_id"] for b in batch],
        "vehicle_id": [b["vehicle_id"] for b in batch],
        "letterbox_meta": [b["letterbox_meta"] for b in batch],
    }
    return out
