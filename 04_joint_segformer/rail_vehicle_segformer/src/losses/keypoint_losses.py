#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def heatmap_mse_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """
    logits/targets: [B,1,H,W]
    valid: [B,1] or [B]
    """
    if valid.ndim == 1:
        valid = valid[:, None]
    m = valid.view(-1) > 0.5
    if not torch.any(m):
        return logits.sum() * 0.0
    pred = torch.sigmoid(logits[m])
    tgt = targets[m]
    if pred.shape[-2:] != tgt.shape[-2:]:
        pred = F.interpolate(pred, size=tgt.shape[-2:], mode="bilinear", align_corners=False)
    return F.mse_loss(pred, tgt)
