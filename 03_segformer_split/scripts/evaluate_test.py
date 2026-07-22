#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在 fold test 集上导出与 UNet/DeepLab 对齐的 test_report.json。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.rail_vehicle_dataset import RailVehicleDataset, collate_same_view
from src.metrics.official_metrics import official_front_metrics, official_side_metrics
from src.models.segformer_multitask import SegFormerMultiTask
from src.utils.checkpoint import load_checkpoint
from src.utils.io import ensure_dir, load_yaml, resolve_path
from src.utils.visualization import save_prediction_visualization


def load_fold_ids(splits_dir: Path, fold: int, split: str):
    return json.loads((splits_dir / f"fold_{fold}.json").read_text(encoding="utf-8"))[split]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--view", type=str, default=None, choices=["front", "side"])
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--num-vis", type=int, default=12)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    view = str(args.view or cfg["training"].get("view") or cfg["data"].get("view"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = ensure_dir(
        resolve_path(args.output_dir)
        if args.output_dir
        else resolve_path("outputs") / "eval" / f"{view}_fold{args.fold}_{args.split}"
    )
    vis_dir = ensure_dir(out_dir / "compare")

    ids = load_fold_ids(resolve_path(cfg["data"]["splits_dir"]), args.fold, args.split)
    ds = RailVehicleDataset(
        metadata_csv=resolve_path(cfg["data"]["metadata"]),
        processed_root=resolve_path(cfg["data"]["root"]),
        sample_ids=ids,
        view=view,
        front_size=tuple(cfg["data"]["front_size"]),
        side_size=tuple(cfg["data"]["side_size"]),
        mean=cfg["data"]["mean"],
        std=cfg["data"]["std"],
        heatmap_sigma=float(cfg["loss"]["heatmap_sigma"]),
        train=False,
        seed=int(cfg["training"]["seed"]),
        keypoint_out_stride=int(cfg["model"].get("keypoint_out_stride", 4)),
    )
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_same_view,
    )

    model = SegFormerMultiTask(
        backbone_name=cfg["model"].get("backbone", "nvidia/mit-b0"),
        pretrained=False,
        num_segmentation_channels=4,
        decoder_channels=int(cfg["model"].get("decoder_channels", 256)),
        keypoint_head=bool(cfg["model"].get("keypoint_head", view == "front")),
        keypoint_out_stride=int(cfg["model"].get("keypoint_out_stride", 4)),
    ).to(device)
    # force real B0 backbone if pretrained False used tiny historically
    load_checkpoint(resolve_path(args.checkpoint), model, optimizer=None, map_location=str(device))
    model.eval()

    tip_rx = int(cfg.get("eval", {}).get("tip_ellipse_rx", 24))
    tip_ry = int(cfg.get("eval", {}).get("tip_ellipse_ry", 12))
    stride = int(cfg["model"].get("keypoint_out_stride", 4))

    acc: Dict[str, List[float]] = {}
    stems = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch_d = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(batch_d["image"])
            if view == "front":
                m = official_front_metrics(
                    out["segmentation_logits"],
                    batch_d["segmentation"],
                    batch_d["valid_seg_tasks"],
                    out.get("nose_tip_heatmap_logits"),
                    batch_d["nose_tip_heatmap"],
                    batch_d["valid_nose_tip"],
                    tip_out_stride=stride,
                    tip_rx=tip_rx,
                    tip_ry=tip_ry,
                    image_hw=tuple(batch_d["image"].shape[-2:]),
                )
            else:
                m = official_side_metrics(
                    out["segmentation_logits"],
                    batch_d["segmentation"],
                    batch_d["valid_seg_tasks"],
                )
            for k, v in m.items():
                if isinstance(v, float) and v == v:
                    acc.setdefault(k, []).append(v)
            sid = batch["sample_id"][0]
            stems.append(sid)
            if i < args.num_vis:
                tip = out.get("nose_tip_heatmap_logits")
                save_prediction_visualization(
                    batch_d["image"][0],
                    out["segmentation_logits"][0].cpu(),
                    tip[0].cpu() if tip is not None else None,
                    vis_dir / f"{sid}.png",
                    mean=cfg["data"]["mean"],
                    std=cfg["data"]["std"],
                    title=f"{sid}",
                    valid_seg_tasks=batch_d["valid_seg_tasks"][0].cpu(),
                )

    report = {
        "view": view,
        "fold": args.fold,
        "split": args.split,
        "n": len(stems),
        "checkpoint": str(resolve_path(args.checkpoint)),
        "test_official_miou": float(sum(acc["official_miou"]) / len(acc["official_miou"]))
        if "official_miou" in acc
        else None,
        "test_iou": {k.replace("_iou", ""): float(sum(vs) / len(vs)) for k, vs in acc.items() if k.endswith("_iou")},
        "stems": stems,
    }
    if "tip_mae_px" in acc:
        report["test_tip"] = {
            "tip_mae_px": float(sum(acc["tip_mae_px"]) / len(acc["tip_mae_px"])),
            "tip_mae_norm_h": float(sum(acc["tip_mae_norm_h"]) / len(acc["tip_mae_norm_h"]))
            if "tip_mae_norm_h" in acc
            else None,
            "tip_n": int(sum(acc.get("tip_n", [0]))),
            **{
                k: float(sum(acc[k]) / len(acc[k]))
                for k in acc
                if k.startswith("tip_pck")
            },
        }
    (out_dir / "test_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
