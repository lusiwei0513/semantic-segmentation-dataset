#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""可视化若干样本的掩码叠加与关键点。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir, load_yaml, resolve_path  # noqa: E402

COLORS = {
    "body": (0.2, 0.6, 1.0),
    "windshield": (1.0, 0.85, 0.2),
    "bogie": (0.2, 0.9, 0.4),
    "door": (1.0, 0.3, 0.3),
}


def overlay(image: np.ndarray, masks: dict) -> np.ndarray:
    out = image.astype(np.float32) / 255.0
    for name, m in masks.items():
        if m is None:
            continue
        color = np.array(COLORS[name], dtype=np.float32)
        alpha = 0.45
        sel = m > 0
        out[sel] = out[sel] * (1 - alpha) + color * alpha
    return np.clip(out, 0, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--num-samples", type=int, default=20)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    root = resolve_path(cfg["processed_root"])
    df = pd.read_csv(resolve_path(cfg["metadata_csv"]))

    out_dir = resolve_path("outputs/visualizations/data_check")
    ensure_dir(out_dir)

    front = df[df.view == "front"].sample(n=min(args.num_samples // 2, (df.view == "front").sum()), random_state=42)
    side = df[df.view == "side"].sample(n=min(args.num_samples - len(front), (df.view == "side").sum()), random_state=42)
    sample_df = pd.concat([front, side], ignore_index=True)

    for _, row in sample_df.iterrows():
        img = np.array(Image.open(root / row["image_path"]).convert("RGB"))
        masks = {}
        for task, col, vcol in [
            ("body", "body_mask", "valid_body"),
            ("windshield", "windshield_mask", "valid_windshield"),
            ("bogie", "bogie_mask", "valid_bogie"),
            ("door", "door_mask", "valid_door"),
        ]:
            if int(row[vcol]) == 1:
                masks[task] = np.array(Image.open(root / row[col]))
            else:
                masks[task] = None

        vis = overlay(img, masks)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(img)
        axes[0].set_title(f"{row['sample_id']} raw")
        axes[0].axis("off")
        axes[1].imshow(vis)
        axes[1].set_title(f"{row['view']} overlay")
        if int(row["valid_nose_tip"]) == 1:
            tip = json.loads((root / row["keypoint_path"]).read_text(encoding="utf-8"))
            axes[1].scatter([tip["x"]], [tip["y"]], c="cyan", s=40, marker="x")
        axes[1].axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"{row['sample_id']}.png", dpi=120)
        plt.close(fig)

    print(f"保存 {len(sample_df)} 张到 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
