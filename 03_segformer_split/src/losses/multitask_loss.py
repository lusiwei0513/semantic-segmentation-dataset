#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from src.losses.keypoint_losses import heatmap_mse_loss
from src.losses.segmentation_losses import masked_bce_dice

CHANNEL_NAMES = ["body", "windshield", "bogie", "door"]


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        loss_weights: Dict[str, float] | None = None,
        pos_weights: Dict[str, float] | None = None,
        heat_peak_boost: float = 0.0,
        tip_coord_weight: float = 0.0,
        soft_argmax_temp: float = 0.1,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.loss_weights = loss_weights or {
            "body": 1.0,
            "windshield": 1.0,
            "bogie": 1.0,
            "door": 1.0,
            "nose_tip": 0.5,
        }
        self.pos_weights = pos_weights or {}
        self.heat_peak_boost = float(heat_peak_boost)
        self.tip_coord_weight = float(tip_coord_weight)
        self.soft_argmax_temp = float(soft_argmax_temp)

    def forward(self, outputs: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        logits = outputs["segmentation_logits"]
        targets = batch["segmentation"]
        valid = batch["valid_seg_tasks"]

        per = {}
        total = logits.sum() * 0.0
        for i, name in enumerate(CHANNEL_NAMES):
            w = float(self.loss_weights.get(name, 1.0))
            pw = self.pos_weights.get(name)
            pw_f: Optional[float] = float(pw) if pw is not None else None
            li = masked_bce_dice(
                logits[:, i : i + 1],
                targets[:, i : i + 1],
                valid[:, i : i + 1],
                bce_weight=self.bce_weight,
                dice_weight=self.dice_weight,
                pos_weight=pw_f,
            )
            per[f"loss_{name}"] = li
            total = total + w * li

        tip_w = float(self.loss_weights.get("nose_tip", 0.5))
        tip_loss = heatmap_mse_loss(
            outputs["nose_tip_heatmap_logits"],
            batch["nose_tip_heatmap"],
            batch["valid_nose_tip"],
            peak_boost=self.heat_peak_boost,
            tip_xy=batch.get("nose_tip_xy"),
            tip_coord_weight=self.tip_coord_weight,
            soft_argmax_temp=self.soft_argmax_temp,
        )
        per["loss_nose_tip"] = tip_loss
        total = total + tip_w * tip_loss
        per["loss_total"] = total
        return per
