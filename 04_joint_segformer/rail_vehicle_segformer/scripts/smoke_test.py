#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单 batch 冒烟测试：前向 / 损失 / 反向 / 优化器 / 可视化 / 显存。"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.rail_vehicle_dataset import RailVehicleDataset, collate_same_view
from src.losses.multitask_loss import MultiTaskLoss
from src.models.segformer_multitask import SegFormerMultiTask
from src.utils.io import load_yaml, resolve_path
from src.utils.logger import SimpleLogger
from src.utils.seed import set_seed
from src.utils.visualization import save_prediction_visualization
from src.train import build_optimizer


def make_loader(cfg, view: str, sample_ids=None):
    data_cfg = cfg["data"]
    ds = RailVehicleDataset(
        metadata_csv=resolve_path(data_cfg["metadata"]),
        processed_root=resolve_path(data_cfg["root"]),
        sample_ids=sample_ids,
        view=view,
        front_size=tuple(data_cfg["front_size"]),
        side_size=tuple(data_cfg["side_size"]),
        mean=data_cfg["mean"],
        std=data_cfg["std"],
        heatmap_sigma=float(cfg["loss"]["heatmap_sigma"]),
        train=True,
        seed=int(cfg["training"]["seed"]),
    )
    assert len(ds) > 0, f"{view} 数据集为空"
    return DataLoader(
        ds,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(data_cfg["num_workers"]),
        collate_fn=collate_same_view,
    )


def run_one(model, criterion, optimizer, batch, device, amp: bool, scaler=None):
    batch_dev = {
        k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
    }
    optimizer.zero_grad(set_to_none=True)
    if amp and device.type == "cuda":
        with torch.cuda.amp.autocast():
            outputs = model(batch_dev["image"])
            losses = criterion(outputs, batch_dev)
            loss = losses["loss_total"]
        assert torch.isfinite(loss), f"loss 非有限值: {loss}"
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        outputs = model(batch_dev["image"])
        losses = criterion(outputs, batch_dev)
        loss = losses["loss_total"]
        assert torch.isfinite(loss), f"loss 非有限值: {loss}"
        loss.backward()
        optimizer.step()

    # 检查关键参数有梯度
    grad_ok = False
    for p in model.parameters():
        if p.grad is not None:
            grad_ok = True
            break
    assert grad_ok, "没有参数获得梯度"
    return outputs, {k: float(v.detach().cpu()) for k, v in losses.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_smoke.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["training"]["seed"]))
    out_dir = resolve_path(cfg["training"].get("output_dir", "./outputs"))
    log = SimpleLogger(out_dir / "logs" / "smoke_test.log")
    log.log("=== smoke_test start ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.log(f"device={device}")

    # 优先使用 fold_0 train ids；若不存在则用全量
    splits_dir = resolve_path(cfg["data"]["splits_dir"])
    sample_ids = None
    fold_path = splits_dir / "fold_0.json"
    if fold_path.exists():
        sample_ids = json.loads(fold_path.read_text(encoding="utf-8"))["train"]
        log.log(f"使用 fold_0 train ids: {len(sample_ids)}")

    front_loader = make_loader(cfg, "front", sample_ids)
    side_loader = make_loader(cfg, "side", sample_ids)
    front_batch = next(iter(front_loader))
    side_batch = next(iter(side_loader))
    log.log(
        f"front batch image={tuple(front_batch['image'].shape)} "
        f"side batch image={tuple(side_batch['image'].shape)}"
    )

    model_cfg = cfg["model"]
    model = SegFormerMultiTask(
        backbone_name=model_cfg.get("backbone", "nvidia/mit-b0"),
        pretrained=bool(model_cfg.get("pretrained", True)),
        num_segmentation_channels=4,
        decoder_channels=int(model_cfg.get("decoder_channels", 128)),
        keypoint_head=bool(model_cfg.get("keypoint_head", True)),
    ).to(device)
    criterion = MultiTaskLoss(
        bce_weight=float(cfg["loss"]["bce_weight"]),
        dice_weight=float(cfg["loss"]["dice_weight"]),
        loss_weights=cfg["loss"]["loss_weights"],
    )
    optimizer = build_optimizer(model, cfg)
    amp = bool(cfg["training"].get("amp", False))
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")

    mem_records = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    results = {}
    for name, batch in [("front", front_batch), ("side", side_batch)]:
        log.log(f"--- run {name} ---")
        outputs, loss_dict = run_one(model, criterion, optimizer, batch, device, amp, scaler)
        results[name] = loss_dict
        log.log_json(f"{name}_losses", loss_dict)
        assert outputs["segmentation_logits"].shape[1] == 4
        assert outputs["nose_tip_heatmap_logits"].shape[1] == 1

        vis_path = out_dir / "visualizations" / "smoke" / f"{name}_pred.png"
        save_prediction_visualization(
            batch["image"][0],
            outputs["segmentation_logits"][0].detach().cpu(),
            outputs["nose_tip_heatmap_logits"][0].detach().cpu(),
            vis_path,
            mean=cfg["data"]["mean"],
            std=cfg["data"]["std"],
            title=f"smoke {name}",
        )
        log.log(f"saved vis: {vis_path}")

        if device.type == "cuda":
            mem_records[name] = {
                "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
                "max_reserved_mb": torch.cuda.max_memory_reserved() / (1024**2),
            }
            log.log_json(f"{name}_cuda_mem", mem_records[name])

    summary = {
        "device": str(device),
        "losses": results,
        "cuda_mem": mem_records,
        "status": "PASS",
    }
    summary_path = out_dir / "logs" / "smoke_test_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log.log(f"summary -> {summary_path}")
    log.log("=== smoke_test PASS ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
