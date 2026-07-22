#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class KeypointHeatmapHead(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
        )

    def forward(self, feat, output_size):
        logits = self.net(feat)
        if logits.shape[-2:] != output_size:
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits
