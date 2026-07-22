#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def heatmap_to_coord(heatmap: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    heatmap: [B,1,H,W] probs
    returns x,y [B]
    """
    b, _, h, w = heatmap.shape
    flat = heatmap.view(b, -1)
    idx = flat.argmax(dim=1)
    y = (idx // w).float()
    x = (idx % w).float()
    return x, y


@torch.no_grad()
def keypoint_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    body_masks: Optional[torch.Tensor] = None,
    thresholds=(0.02, 0.05),
) -> Dict[str, float]:
    """
    logits/targets: [B,1,H,W]
    valid: [B,1]
    body_masks: [B,1,H,W] optional for normalized error
    """
    m = valid.view(-1) > 0.5
    if not torch.any(m):
        return {}
    pred = torch.sigmoid(logits[m])
    tgt = targets[m]
    if pred.shape[-2:] != tgt.shape[-2:]:
        pred = F.interpolate(pred, size=tgt.shape[-2:], mode="bilinear", align_corners=False)
    px, py = heatmap_to_coord(pred)
    gx, gy = heatmap_to_coord(tgt)
    dist = torch.sqrt((px - gx) ** 2 + (py - gy) ** 2)
    out = {
        "tip_mean_px_error": float(dist.mean().item()),
        "tip_median_px_error": float(dist.median().item()),
    }

    if body_masks is not None:
        bm = body_masks[m, 0]
        norms = []
        for i in range(bm.shape[0]):
            ys, xs = torch.where(bm[i] > 0.5)
            if len(xs) == 0:
                diag = float(max(bm.shape[-1], bm.shape[-2]))
            else:
                w = (xs.max() - xs.min()).float() + 1
                h = (ys.max() - ys.min()).float() + 1
                diag = torch.sqrt(w * w + h * h).item()
            norms.append(dist[i].item() / max(diag, 1e-6))
        norms_t = torch.tensor(norms)
        out["tip_normalized_error"] = float(norms_t.mean().item())
        for t in thresholds:
            out[f"tip_pck@{t}"] = float((norms_t <= t).float().mean().item())
    return out
