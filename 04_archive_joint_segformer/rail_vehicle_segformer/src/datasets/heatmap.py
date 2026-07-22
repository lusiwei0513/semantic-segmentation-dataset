#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从互斥 ID 掩码中提取 nose_tip 质心。"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def disk_centroid(mask: np.ndarray, class_id: int = 3) -> Optional[Dict]:
    ys, xs = np.where(mask == class_id)
    if len(xs) == 0:
        return {
            "x": None,
            "y": None,
            "visible": False,
            "source": "disk_centroid",
            "n_pixels": 0,
        }
    x = float(xs.mean())
    y = float(ys.mean())
    return {
        "x": x,
        "y": y,
        "visible": True,
        "source": "disk_centroid",
        "n_pixels": int(len(xs)),
    }


def make_gaussian_heatmap(
    height: int,
    width: int,
    center_xy: Tuple[float, float],
    sigma: float = 4.0,
) -> np.ndarray:
    """生成单通道高斯热图，峰值约 1.0。"""
    cx, cy = center_xy
    ys = np.arange(height, dtype=np.float32)[:, None]
    xs = np.arange(width, dtype=np.float32)[None, :]
    heatmap = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma * sigma))
    return heatmap.astype(np.float32)
