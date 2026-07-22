#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_argmax_xy(heat: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Differentiable peak. heat: (B,1,H,W) in [0,1] -> (B,2) as (x,y) in heatmap coords."""
    if heat.ndim == 3:
        heat = heat.unsqueeze(1)
    b, _, h, w = heat.shape
    logits = heat.view(b, -1) / max(float(temperature), 1e-4)
    prob = torch.softmax(logits, dim=1)
    ys = torch.arange(h, device=heat.device, dtype=heat.dtype)
    xs = torch.arange(w, device=heat.device, dtype=heat.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=0)  # 2,HW
    return torch.matmul(prob, grid.t())  # B,2


def heatmap_mse_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    peak_boost: float = 0.0,
    tip_xy: torch.Tensor | None = None,
    tip_coord_weight: float = 0.0,
    soft_argmax_temp: float = 0.1,
) -> torch.Tensor:
    """
    Tip heatmap loss aligned with UNet-KP when peak_boost > 0:
      spatial_w = 1 + peak_boost * target
      BCE-with-logits weighted by spatial_w
    Optional soft-argmax L1 when tip_xy is provided and tip_coord_weight > 0.

    Fallback (peak_boost <= 0): legacy sigmoid + MSE (uniform).
    logits/targets: [B,1,H,W]; valid: [B,1] or [B]
    tip_xy: [B,2] in heatmap (x,y) coords, only rows with valid tip used.
    """
    if valid.ndim == 1:
        valid = valid[:, None]
    m = valid.view(-1) > 0.5
    if not torch.any(m):
        return logits.sum() * 0.0

    logits_v = logits[m]
    tgt = targets[m]
    if logits_v.shape[-2:] != tgt.shape[-2:]:
        logits_v = F.interpolate(logits_v, size=tgt.shape[-2:], mode="bilinear", align_corners=False)

    if peak_boost and float(peak_boost) > 0:
        spatial_w = 1.0 + float(peak_boost) * tgt
        bce = F.binary_cross_entropy_with_logits(logits_v, tgt, reduction="none")
        loss = (bce * spatial_w).sum() / spatial_w.sum().clamp(min=1.0)
    else:
        pred = torch.sigmoid(logits_v)
        loss = F.mse_loss(pred, tgt)

    if tip_xy is not None and float(tip_coord_weight) > 0:
        tip_v = tip_xy[m]
        pred_xy = soft_argmax_xy(torch.sigmoid(logits_v), temperature=float(soft_argmax_temp))
        loss = loss + float(tip_coord_weight) * F.l1_loss(pred_xy, tip_v)

    return loss
