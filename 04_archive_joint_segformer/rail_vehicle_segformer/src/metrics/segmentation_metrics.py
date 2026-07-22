#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Optional

import torch


@torch.no_grad()
def _confusion(pred: torch.Tensor, target: torch.Tensor):
    pred = pred.bool()
    target = target.bool()
    tp = (pred & target).sum().float()
    fp = (pred & ~target).sum().float()
    fn = (~pred & target).sum().float()
    return tp, fp, fn


@torch.no_grad()
def segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    logits/targets [B,C,H,W], valid_mask [B,C]
    """
    probs = torch.sigmoid(logits)
    pred = probs >= threshold
    names = ["body", "windshield", "bogie", "door"]
    out = {}
    ious = []
    dices = []
    n_ch = min(logits.shape[1], len(names))
    for i in range(n_ch):
        name = names[i]
        m = valid_mask[:, i] > 0.5
        if not torch.any(m):
            continue
        tp, fp, fn = _confusion(pred[m, i], targets[m, i] > 0.5)
        iou = (tp / (tp + fp + fn + 1e-6)).item()
        dice = (2 * tp / (2 * tp + fp + fn + 1e-6)).item()
        prec = (tp / (tp + fp + 1e-6)).item()
        rec = (tp / (tp + fn + 1e-6)).item()
        out[f"{name}_iou"] = iou
        out[f"{name}_dice"] = dice
        out[f"{name}_precision"] = prec
        out[f"{name}_recall"] = rec
        ious.append(iou)
        dices.append(dice)
    if ious:
        out["macro_miou"] = float(sum(ious) / len(ious))
        out["macro_dice"] = float(sum(dices) / len(dices))
    return out
