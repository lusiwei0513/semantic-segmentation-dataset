#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleSegHead(nn.Module):
    """将 MiT 多尺度特征融合为 4 通道分割 logits。"""

    def __init__(self, in_channels, decoder_channels: int = 256, num_classes: int = 4):
        super().__init__()
        self.projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c, decoder_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(decoder_channels),
                    nn.ReLU(inplace=True),
                )
                for c in in_channels
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_channels * len(in_channels), decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
        )

    def forward(self, features, output_size):
        # features: list of [B,C,H,W] low->high resolution or as returned by backbone
        target = features[0].shape[-2:]
        # unify to highest-res among inputs (usually first is highest for HF? check)
        # HuggingFace SegformerModel returns hidden_states from low to high resolution
        # actually: stage0 highest res, stage3 lowest. Verify in model file.
        ups = []
        ref_h, ref_w = features[0].shape[-2:]
        for feat, proj in zip(features, self.projs):
            x = proj(feat)
            if x.shape[-2:] != (ref_h, ref_w):
                x = F.interpolate(x, size=(ref_h, ref_w), mode="bilinear", align_corners=False)
            ups.append(x)
        x = torch.cat(ups, dim=1)
        logits = self.fuse(x)
        if logits.shape[-2:] != output_size:
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits
