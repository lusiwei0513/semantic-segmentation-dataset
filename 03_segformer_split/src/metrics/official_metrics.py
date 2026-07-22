#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""与 UNet-KP 对齐的官方评测：椭圆 tip IoU + fg-mIoU + tip MAE/PCK。"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def ellipse_mask(h: int, w: int, x: float, y: float, rx: int = 24, ry: int = 12) -> np.ndarray:
    out = np.zeros((h, w), dtype=np.uint8)
    try:
        import cv2

        cv2.ellipse(
            out,
            (int(round(x)), int(round(y))),
            (int(rx), int(ry)),
            0.0,
            0.0,
            360.0,
            1,
            thickness=-1,
        )
    except Exception:
        yy, xx = np.mgrid[0:h, 0:w]
        out[((xx - x) / max(rx, 1)) ** 2 + ((yy - y) / max(ry, 1)) ** 2 <= 1.0] = 1
    return out


def heatmap_peak_xy(heat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """heat [B,1,H,W] -> x,y in that heatmap resolution."""
    b, _, h, w = heat.shape
    flat = heat.view(b, -1)
    idx = flat.argmax(dim=1)
    y = (idx // w).float()
    x = (idx % w).float()
    return x, y


def _binary_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return float("nan")
    return float(inter / union)


@torch.no_grad()
def official_front_metrics(
    seg_logits: torch.Tensor,
    seg_targets: torch.Tensor,
    valid_seg: torch.Tensor,
    tip_logits: torch.Tensor,
    tip_heat_tgt: torch.Tensor,
    valid_tip: torch.Tensor,
    tip_out_stride: int = 4,
    tip_rx: int = 24,
    tip_ry: int = 12,
    image_hw: Optional[Tuple[int, int]] = None,
    pck_thresholds: Tuple[int, ...] = (5, 10, 20),
) -> Dict[str, float]:
    """
    Front official protocol (align UNet KP):
      - body / windshield IoU from multilabel sigmoid@0.5
      - nose_tip IoU from predicted vs GT ellipse (rx,ry) at peak coords
      - official_miou = mean of available {body, windshield, nose_tip}
      - tip_mae_px / tip_pck@T in letterbox pixel space
    """
    probs = torch.sigmoid(seg_logits)
    pred = probs >= 0.5
    names = ["body", "windshield"]
    out: Dict[str, float] = {}
    ious: List[float] = []

    for i, name in enumerate(names):
        m = valid_seg[:, i] > 0.5
        if not torch.any(m):
            continue
        tp = (pred[m, i] & (seg_targets[m, i] > 0.5)).sum().float()
        fp = (pred[m, i] & (seg_targets[m, i] <= 0.5)).sum().float()
        fn = ((~pred[m, i]) & (seg_targets[m, i] > 0.5)).sum().float()
        iou = float((tp / (tp + fp + fn + 1e-6)).item())
        out[f"{name}_iou"] = iou
        ious.append(iou)

    # tip
    tip_m = valid_tip.view(-1) > 0.5
    tip_ious: List[float] = []
    dists: List[float] = []
    heights: List[float] = []
    if torch.any(tip_m) and tip_logits is not None:
        pred_h = torch.sigmoid(tip_logits[tip_m])
        tgt_h = tip_heat_tgt[tip_m]
        if pred_h.shape[-2:] != tgt_h.shape[-2:]:
            pred_h = F.interpolate(pred_h, size=tgt_h.shape[-2:], mode="bilinear", align_corners=False)
        px, py = heatmap_peak_xy(pred_h)
        gx, gy = heatmap_peak_xy(tgt_h)
        stride = float(max(1, tip_out_stride))
        # map to letterbox full-res
        px_f, py_f = px * stride, py * stride
        gx_f, gy_f = gx * stride, gy * stride
        if image_hw is None:
            H = int(seg_logits.shape[-2])
            W = int(seg_logits.shape[-1])
        else:
            H, W = image_hw

        for i in range(px_f.shape[0]):
            pred_e = ellipse_mask(H, W, float(px_f[i]), float(py_f[i]), rx=tip_rx, ry=tip_ry) > 0
            gt_e = ellipse_mask(H, W, float(gx_f[i]), float(gy_f[i]), rx=tip_rx, ry=tip_ry) > 0
            tip_ious.append(_binary_iou(pred_e, gt_e))
            d = float(torch.sqrt((px_f[i] - gx_f[i]) ** 2 + (py_f[i] - gy_f[i]) ** 2).item())
            dists.append(d)
            heights.append(float(H))

        if tip_ious:
            tip_iou = float(np.nanmean(tip_ious))
            out["nose_tip_iou"] = tip_iou
            ious.append(tip_iou)
        if dists:
            out["tip_mae_px"] = float(np.mean(dists))
            out["tip_mae_norm_h"] = float(np.mean([d / max(h, 1.0) for d, h in zip(dists, heights)]))
            out["tip_n"] = float(len(dists))
            for t in pck_thresholds:
                out[f"tip_pck@{t}"] = float(np.mean([d <= t for d in dists]))

    if ious:
        out["official_miou"] = float(sum(ious) / len(ious))
        out["macro_miou"] = out["official_miou"]
    return out


@torch.no_grad()
def official_side_metrics(
    seg_logits: torch.Tensor,
    seg_targets: torch.Tensor,
    valid_seg: torch.Tensor,
) -> Dict[str, float]:
    """Side fg-mIoU over body/windshield/bogie/door (no background)."""
    probs = torch.sigmoid(seg_logits)
    pred = probs >= 0.5
    names = ["body", "windshield", "bogie", "door"]
    out: Dict[str, float] = {}
    ious: List[float] = []
    for i, name in enumerate(names):
        m = valid_seg[:, i] > 0.5
        if not torch.any(m):
            continue
        tp = (pred[m, i] & (seg_targets[m, i] > 0.5)).sum().float()
        fp = (pred[m, i] & (seg_targets[m, i] <= 0.5)).sum().float()
        fn = ((~pred[m, i]) & (seg_targets[m, i] > 0.5)).sum().float()
        iou = float((tp / (tp + fp + fn + 1e-6)).item())
        out[f"{name}_iou"] = iou
        ious.append(iou)
    if ious:
        out["official_miou"] = float(sum(ious) / len(ious))
        out["macro_miou"] = out["official_miou"]
        out["fg_miou"] = out["official_miou"]
    return out
