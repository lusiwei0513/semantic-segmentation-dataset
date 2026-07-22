#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

from src.datasets.heatmap import make_gaussian_heatmap
from src.datasets.rail_vehicle_dataset import RailVehicleDataset, collate_same_view
from src.datasets.transforms import hflip, letterbox
from src.losses.multitask_loss import MultiTaskLoss
from src.losses.segmentation_losses import DiceLoss, masked_bce_dice
from src.metrics.keypoint_metrics import keypoint_metrics
from src.metrics.segmentation_metrics import segmentation_metrics
from src.models.segformer_multitask import SegFormerMultiTask


@pytest.fixture(scope="module")
def meta_paths():
    meta = ROOT / "data" / "processed" / "metadata.csv"
    root = ROOT / "data" / "processed"
    if not meta.exists():
        pytest.skip("metadata 尚未生成")
    return meta, root


def test_front_task_mask(meta_paths):
    meta, root = meta_paths
    ds = RailVehicleDataset(meta, root, view="front", front_size=(128, 128), side_size=(128, 256), train=False)
    item = ds[0]
    assert item["view"] == "front"
    assert item["valid_seg_tasks"][2] == 0  # bogie
    assert item["valid_seg_tasks"][3] == 0  # door
    assert item["segmentation"].shape[0] == 4
    assert item["image"].shape[1:] == (128, 128)


def test_side_task_mask(meta_paths):
    meta, root = meta_paths
    ds = RailVehicleDataset(meta, root, view="side", front_size=(128, 128), side_size=(128, 256), train=False)
    item = ds[0]
    assert item["view"] == "side"
    assert float(item["valid_nose_tip"][0]) == 0.0
    assert item["image"].shape[1:] == (128, 256)


def test_hflip_keypoint():
    img = np.zeros((40, 80, 3), dtype=np.uint8)
    masks = [np.zeros((40, 80), dtype=np.uint8)]
    kp = (10.0, 20.0)
    img2, masks2, kp2 = hflip(img, masks, kp)
    assert kp2[0] == pytest.approx(80 - 1 - 10)
    assert kp2[1] == pytest.approx(20)


def test_letterbox_keypoint_mapping():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    masks = [np.zeros((100, 200), dtype=np.uint8)]
    kp = (100.0, 50.0)
    out = letterbox(img, masks, kp, target_size=(64, 64))
    assert out["image"].shape == (64, 64, 3)
    assert out["keypoint_xy"] is not None
    x, y = out["keypoint_xy"]
    assert 0 <= x < 64 and 0 <= y < 64


def test_invalid_task_no_loss_contribution():
    logits = torch.randn(2, 4, 32, 32)
    targets = torch.zeros(2, 4, 32, 32)
    valid = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.float32)
    loss = masked_bce_dice(logits, targets, valid)
    assert torch.isfinite(loss)


def test_perfect_dice_near_zero():
    logits = torch.full((1, 1, 16, 16), 20.0)
    targets = torch.ones(1, 1, 16, 16)
    d = DiceLoss()(logits, targets)
    assert float(d) < 1e-3


def test_iou_perfect_and_zero():
    logits = torch.full((1, 1, 8, 8), 10.0)
    targets = torch.ones(1, 1, 8, 8)
    valid = torch.ones(1, 1)
    m = segmentation_metrics(logits, targets, valid)
    assert m["body_iou"] == pytest.approx(1.0, abs=1e-5)

    logits2 = torch.full((1, 1, 8, 8), -10.0)
    m2 = segmentation_metrics(logits2, targets, valid)
    assert m2["body_iou"] == pytest.approx(0.0, abs=1e-5)


def test_keypoint_error_zero():
    hm = make_gaussian_heatmap(32, 32, (10, 12), sigma=2)
    t = torch.from_numpy(hm)[None, None]
    # logits large positive at same peak approx via inverse sigmoid of hm clipped
    logits = torch.logit(t.clamp(1e-4, 1 - 1e-4))
    valid = torch.ones(1, 1)
    body = torch.ones(1, 1, 32, 32)
    m = keypoint_metrics(logits, t, valid, body_masks=body)
    assert m["tip_mean_px_error"] == pytest.approx(0.0, abs=1e-5)


def test_model_forward_cpu_shapes():
    model = SegFormerMultiTask(pretrained=False, decoder_channels=64)
    model.eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out["segmentation_logits"].shape == (1, 4, 128, 128)
    assert out["nose_tip_heatmap_logits"].shape == (1, 1, 32, 32)


def test_multitask_loss_backward():
    model = SegFormerMultiTask(pretrained=False, decoder_channels=64)
    crit = MultiTaskLoss()
    x = torch.randn(1, 3, 64, 128, requires_grad=False)
    batch = {
        "segmentation": torch.zeros(1, 4, 64, 128),
        "valid_seg_tasks": torch.tensor([[1.0, 1.0, 1.0, 1.0]]),
        "nose_tip_heatmap": torch.zeros(1, 1, 64, 128),
        "valid_nose_tip": torch.tensor([[0.0]]),
    }
    out = model(x)
    losses = crit(out, batch)
    losses["loss_total"].backward()
    assert any(p.grad is not None for p in model.parameters())


def test_collate_rejects_mixed_view(meta_paths):
    meta, root = meta_paths
    front = RailVehicleDataset(meta, root, view="front", front_size=(64, 64), side_size=(64, 128), train=False)[0]
    side = RailVehicleDataset(meta, root, view="side", front_size=(64, 64), side_size=(64, 128), train=False)[0]
    with pytest.raises(ValueError):
        collate_same_view([front, side])
