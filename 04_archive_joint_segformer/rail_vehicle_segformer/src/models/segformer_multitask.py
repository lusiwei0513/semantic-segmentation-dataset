#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SegFormer-B0 / MiT-B0 多任务模型。"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import SegformerConfig, SegformerModel

from src.models.keypoint_head import KeypointHeatmapHead
from src.models.segmentation_head import MultiScaleSegHead

SEGMENTATION_CHANNELS = ["body", "windshield", "bogie", "door"]


class SegFormerMultiTask(nn.Module):
    def __init__(
        self,
        backbone_name: str = "nvidia/mit-b0",
        pretrained: bool = True,
        num_segmentation_channels: int = 4,
        decoder_channels: int = 256,
        keypoint_head: bool = True,
        keypoint_out_stride: int = 4,
    ):
        super().__init__()
        if pretrained:
            self.backbone = SegformerModel.from_pretrained(backbone_name)
            config = self.backbone.config
        else:
            # 单元测试用：构造小型 MiT 配置，避免依赖网络权重
            config = SegformerConfig(
                num_channels=3,
                num_encoder_blocks=4,
                depths=[1, 1, 1, 1],
                sr_ratios=[8, 4, 2, 1],
                hidden_sizes=[32, 64, 96, 128],
                patch_sizes=[7, 3, 3, 3],
                strides=[4, 2, 2, 2],
                num_attention_heads=[1, 2, 4, 8],
                mlp_ratios=[4, 4, 4, 4],
            )
            self.backbone = SegformerModel(config)

        in_channels: List[int] = list(config.hidden_sizes)
        self.seg_head = MultiScaleSegHead(
            in_channels=in_channels,
            decoder_channels=decoder_channels,
            num_classes=num_segmentation_channels,
        )
        self.use_keypoint_head = keypoint_head
        self.keypoint_out_stride = int(keypoint_out_stride)
        if keypoint_head:
            self.kp_head = KeypointHeatmapHead(
                in_channels=in_channels[0], mid_channels=decoder_channels // 2
            )

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        pixel_values: [B,3,H,W] 已归一化
        """
        outputs = self.backbone(pixel_values=pixel_values, output_hidden_states=True)
        hidden_states = list(outputs.hidden_states)
        n_stages = len(self.backbone.config.hidden_sizes)
        features = hidden_states[-n_stages:]
        _, _, h, w = pixel_values.shape
        seg_logits = self.seg_head(features, output_size=(h, w))

        result = {
            "segmentation_logits": seg_logits,
        }
        if self.use_keypoint_head:
            # 输出到约 1/stride 分辨率（与标签热图对齐），避免全分辨率热图开销
            hk, wk = h // self.keypoint_out_stride, w // self.keypoint_out_stride
            kp_logits = self.kp_head(features[0], output_size=(hk, wk))
            result["nose_tip_heatmap_logits"] = kp_logits
        return result

    def backbone_parameters(self):
        return self.backbone.parameters()

    def head_parameters(self):
        params = list(self.seg_head.parameters())
        if self.use_keypoint_head:
            params += list(self.kp_head.parameters())
        return params
