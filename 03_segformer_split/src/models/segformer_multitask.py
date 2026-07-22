#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SegFormer-B0 / MiT-B0 多任务模型。

pretrained:
  - True / "hf": 从 HuggingFace nvidia/mit-b0 加载（可用 HF_ENDPOINT 镜像）
  - False / "random_b0": 使用真实 MiT-B0 结构随机初始化（不访问网络）
  - 本地目录路径: from_pretrained(local_path)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers import SegformerConfig, SegformerModel

from src.models.keypoint_head import KeypointHeatmapHead
from src.models.segmentation_head import MultiScaleSegHead

SEGMENTATION_CHANNELS = ["body", "windshield", "bogie", "door"]


def mit_b0_config() -> SegformerConfig:
    return SegformerConfig(
        num_channels=3,
        num_encoder_blocks=4,
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        hidden_sizes=[32, 64, 160, 256],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        num_attention_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
    )


class SegFormerMultiTask(nn.Module):
    def __init__(
        self,
        backbone_name: str = "nvidia/mit-b0",
        pretrained: Union[bool, str] = True,
        num_segmentation_channels: int = 4,
        decoder_channels: int = 256,
        keypoint_head: bool = True,
        keypoint_out_stride: int = 4,
    ):
        super().__init__()
        self.backbone, config = self._build_backbone(backbone_name, pretrained)
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

    @staticmethod
    def _build_backbone(backbone_name: str, pretrained: Union[bool, str]):
        if pretrained is True or pretrained == "hf":
            backbone = SegformerModel.from_pretrained(backbone_name)
            return backbone, backbone.config

        if isinstance(pretrained, str) and pretrained not in ("random_b0", "false", "False", "0"):
            # local folder or hub id
            backbone = SegformerModel.from_pretrained(pretrained)
            return backbone, backbone.config

        # real MiT-B0 architecture, random init (no network)
        config = mit_b0_config()
        return SegformerModel(config), config

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.backbone(pixel_values=pixel_values, output_hidden_states=True)
        hidden_states = list(outputs.hidden_states)
        n_stages = len(self.backbone.config.hidden_sizes)
        features = hidden_states[-n_stages:]
        _, _, h, w = pixel_values.shape
        seg_logits = self.seg_head(features, output_size=(h, w))

        result = {"segmentation_logits": seg_logits}
        if self.use_keypoint_head:
            hk, wk = h // self.keypoint_out_stride, w // self.keypoint_out_stride
            result["nose_tip_heatmap_logits"] = self.kp_head(
                features[0], output_size=(hk, wk)
            )
        else:
            # keep key for loss compatibility; zeros with grad path via logits.sum()*0
            result["nose_tip_heatmap_logits"] = seg_logits[:, :1] * 0.0
        return result

    def backbone_parameters(self):
        return self.backbone.parameters()

    def head_parameters(self):
        params = list(self.seg_head.parameters())
        if self.use_keypoint_head:
            params += list(self.kp_head.parameters())
        return params
