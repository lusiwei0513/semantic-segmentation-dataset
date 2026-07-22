#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        if logits.ndim == 3:
            probs = probs.unsqueeze(1)
            targets = targets.unsqueeze(1)
        intersection = (probs * targets).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))
        dice = (2 * intersection + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()


def masked_bce_dice(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    pos_weight: Optional[float] = None,
) -> torch.Tensor:
    """
    logits/targets: [B, C, H, W]
    valid_mask: [B, C]  1=有效
    pos_weight: 可选，抬高正样本（小目标）权重
    """
    assert logits.shape == targets.shape
    losses = []
    dice = DiceLoss()
    for ci in range(logits.shape[1]):
        m = valid_mask[:, ci] > 0.5
        if not torch.any(m):
            continue
        li = logits[m, ci : ci + 1]
        ti = targets[m, ci : ci + 1]
        if pos_weight is not None and pos_weight > 0:
            pw = torch.tensor([pos_weight], device=li.device, dtype=li.dtype)
            bce = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="mean")
        else:
            bce = nn.BCEWithLogitsLoss(reduction="mean")
        losses.append(bce_weight * bce(li, ti) + dice_weight * dice(li, ti))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()
