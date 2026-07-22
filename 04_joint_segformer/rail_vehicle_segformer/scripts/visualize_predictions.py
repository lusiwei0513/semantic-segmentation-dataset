#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用最佳权重生成正/侧视预测可视化。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# 权重由 checkpoint 覆盖；离线优先走本地 HF 缓存，避免卡住
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.rail_vehicle_dataset import RailVehicleDataset, collate_same_view
from src.models.segformer_multitask import SegFormerMultiTask
from src.utils.checkpoint import load_checkpoint
from src.utils.io import ensure_dir, load_yaml, resolve_path
from src.utils.visualization import denormalize

COLORS = {
    "body": np.array([0.2, 0.55, 1.0]),
    "windshield": np.array([1.0, 0.85, 0.15]),
    "bogie": np.array([0.15, 0.9, 0.35]),
    "door": np.array([1.0, 0.25, 0.25]),
}
NAMES = ["body", "windshield", "bogie", "door"]


def overlay(img, masks, valid=None, thr=0.5):
    out = img.copy()
    for i, name in enumerate(NAMES):
        if valid is not None and float(valid[i]) < 0.5:
            continue
        m = masks[i] >= thr
        if not np.any(m):
            continue
        out[m] = out[m] * 0.45 + COLORS[name] * 0.55
    return np.clip(out, 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_boost.yaml")
    parser.add_argument(
        "--checkpoint",
        default="../experiments/joint_fold0_boost/checkpoints/best.pt",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--num-front", type=int, default=6)
    parser.add_argument("--num-side", type=int, default=6)
    parser.add_argument("--out-dir", default="outputs/visualizations/best_preds")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = json.loads(
        (resolve_path(cfg["data"]["splits_dir"]) / f"fold_{args.fold}.json").read_text(
            encoding="utf-8"
        )
    )
    sample_ids = splits[args.split]
    out_dir = ensure_dir(resolve_path(args.out_dir))

    # 优先本地 mit-b0（结构用），避免 huggingface.co 超时；权重随后被 checkpoint 覆盖
    backbone_name = cfg["model"].get("backbone", "nvidia/mit-b0")
    local_bb = ROOT.parents[1] / "03_segformer_split" / "pretrained" / "mit-b0"
    if local_bb.is_dir() and (local_bb / "config.json").exists():
        backbone_name = str(local_bb)

    model = SegFormerMultiTask(
        backbone_name=backbone_name,
        pretrained=True,  # 仅取 MiT-B0 结构；参数随即被 checkpoint 覆盖
        decoder_channels=int(cfg["model"].get("decoder_channels", 256)),
        keypoint_head=True,
        keypoint_out_stride=int(cfg["model"].get("keypoint_out_stride", 4)),
    ).to(device)
    load_checkpoint(resolve_path(args.checkpoint), model, map_location=str(device))
    model.eval()

    mean, std = cfg["data"]["mean"], cfg["data"]["std"]

    def make_loader(view, n):
        ds = RailVehicleDataset(
            metadata_csv=resolve_path(cfg["data"]["metadata"]),
            processed_root=resolve_path(cfg["data"]["root"]),
            sample_ids=sample_ids,
            view=view,
            front_size=tuple(cfg["data"]["front_size"]),
            side_size=tuple(cfg["data"]["side_size"]),
            mean=mean,
            std=std,
            heatmap_sigma=float(cfg["loss"]["heatmap_sigma"]),
            train=False,
            keypoint_out_stride=int(cfg["model"].get("keypoint_out_stride", 4)),
        )
        n = min(n, len(ds))
        loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_same_view)
        return loader, n

    saved = []
    for view, n_want in [("front", args.num_front), ("side", args.num_side)]:
        loader, n = make_loader(view, n_want)
        for i, batch in enumerate(loader):
            if i >= n:
                break
            img = batch["image"].to(device)
            with torch.no_grad():
                out = model(img)
            probs = torch.sigmoid(out["segmentation_logits"][0]).cpu().numpy()
            gt = batch["segmentation"][0].numpy()
            valid = batch["valid_seg_tasks"][0].numpy()
            rgb = denormalize(batch["image"][0], mean, std)
            pred_vis = overlay(rgb, probs, valid)
            gt_vis = overlay(rgb, gt, valid, thr=0.5)

            tip_xy = None
            if float(batch["valid_nose_tip"][0, 0]) > 0.5:
                hm = torch.sigmoid(out["nose_tip_heatmap_logits"][0, 0]).cpu().numpy()
                yy, xx = np.unravel_index(hm.argmax(), hm.shape)
                tip_xy = (xx * rgb.shape[1] / hm.shape[1], yy * rgb.shape[0] / hm.shape[0])

            fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
            axes[0].imshow(rgb)
            axes[0].set_title(f"{batch['sample_id'][0]}\ninput")
            axes[1].imshow(gt_vis)
            axes[1].set_title("GT overlay\nB=blue W=yellow G=green D=red")
            axes[2].imshow(pred_vis)
            axes[2].set_title("Pred overlay")
            if tip_xy is not None:
                axes[2].scatter([tip_xy[0]], [tip_xy[1]], c="cyan", marker="x", s=60)
            for ax in axes:
                ax.axis("off")
            fig.tight_layout()
            path = out_dir / f"{view}_{i:02d}_{batch['sample_id'][0]}.png"
            fig.savefig(path, dpi=140)
            plt.close(fig)
            saved.append(path)
            print("saved", path)

    # legend strip
    legend = out_dir / "legend.txt"
    legend.write_text(
        "颜色说明:\n"
        "body=蓝色, windshield=黄色, bogie=绿色, door=红色\n"
        "青色 X = 预测 nose_tip\n"
        f"checkpoint={args.checkpoint}\n"
        f"split={args.split} fold={args.fold}\n",
        encoding="utf-8",
    )
    print(f"共 {len(saved)} 张 -> {out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
