#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def denormalize(image_chw: torch.Tensor, mean, std) -> np.ndarray:
    x = image_chw.detach().cpu().float().numpy()
    mean = np.array(mean)[:, None, None]
    std = np.array(std)[:, None, None]
    x = x * std + mean
    x = np.clip(x.transpose(1, 2, 0), 0, 1)
    return x


def save_prediction_visualization(
    image_chw: torch.Tensor,
    seg_logits: torch.Tensor,
    tip_logits: Optional[torch.Tensor],
    out_path: Path,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    title: str = "",
) -> None:
    img = denormalize(image_chw, mean, std)
    probs = torch.sigmoid(seg_logits).detach().cpu().numpy()
    colors = np.array(
        [
            [0.2, 0.6, 1.0],
            [1.0, 0.85, 0.2],
            [0.2, 0.9, 0.4],
            [1.0, 0.3, 0.3],
        ],
        dtype=np.float32,
    )
    overlay = img.copy()
    for i in range(min(4, probs.shape[0])):
        m = probs[i] >= 0.5
        overlay[m] = overlay[m] * 0.55 + colors[i] * 0.45

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img)
    axes[0].set_title("input")
    axes[0].axis("off")
    axes[1].imshow(np.clip(overlay, 0, 1))
    axes[1].set_title(title or "pred overlay")
    if tip_logits is not None:
        hm = torch.sigmoid(tip_logits[0]).detach().cpu().numpy()
        yy, xx = np.unravel_index(hm.argmax(), hm.shape)
        # 热图可能是 1/4 分辨率，映射回原图坐标
        scale_y = img.shape[0] / hm.shape[0]
        scale_x = img.shape[1] / hm.shape[1]
        axes[1].scatter([xx * scale_x], [yy * scale_y], c="cyan", marker="x", s=50)
    axes[1].axis("off")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
