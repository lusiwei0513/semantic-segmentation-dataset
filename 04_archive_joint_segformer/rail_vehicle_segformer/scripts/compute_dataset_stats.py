#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据集统计：尺寸、面积比、RGB（抽样）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir, load_yaml, resolve_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--rgb-max-samples", type=int, default=80)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    root = resolve_path(cfg["processed_root"])
    df = pd.read_csv(resolve_path(cfg["metadata_csv"]))

    stats = {
        "n_samples": int(len(df)),
        "n_front": int((df.view == "front").sum()),
        "n_side": int((df.view == "side").sum()),
        "n_vehicles": int(df.vehicle_id.nunique()),
        "valid_task_counts": {
            "body": int(df.valid_body.sum()),
            "windshield": int(df.valid_windshield.sum()),
            "bogie": int(df.valid_bogie.sum()),
            "door": int(df.valid_door.sum()),
            "nose_tip": int(df.valid_nose_tip.sum()),
        },
        "size_by_view": {},
        "mask_area_ratio": {},
        "samples_per_vehicle": {
            "mean": float(df.groupby("vehicle_id").size().mean()),
            "min": int(df.groupby("vehicle_id").size().min()),
            "max": int(df.groupby("vehicle_id").size().max()),
        },
    }

    for view, g in df.groupby("view"):
        stats["size_by_view"][view] = {
            "width_mean": float(g.width.mean()),
            "height_mean": float(g.height.mean()),
            "aspect_mean": float((g.width / g.height).mean()),
        }

    for task, col, vcol in [
        ("body", "body_mask", "valid_body"),
        ("windshield", "windshield_mask", "valid_windshield"),
        ("bogie", "bogie_mask", "valid_bogie"),
        ("door", "door_mask", "valid_door"),
    ]:
        ratios = []
        sub = df[df[vcol] == 1]
        for _, row in sub.iterrows():
            arr = np.array(Image.open(root / row[col]))
            ratios.append(float((arr > 0).mean()))
        stats["mask_area_ratio"][task] = {
            "mean": float(np.mean(ratios)) if ratios else None,
            "median": float(np.median(ratios)) if ratios else None,
            "n": len(ratios),
        }

    # RGB sample
    rng = np.random.RandomState(42)
    idx = rng.choice(len(df), size=min(args.rgb_max_samples, len(df)), replace=False)
    pixels = []
    for i in tqdm(idx, desc="rgb-stats"):
        row = df.iloc[i]
        im = np.array(Image.open(root / row["image_path"]).convert("RGB"), dtype=np.float32)
        # subsample pixels
        flat = im.reshape(-1, 3)
        take = flat[rng.choice(len(flat), size=min(5000, len(flat)), replace=False)]
        pixels.append(take)
    pix = np.concatenate(pixels, axis=0) / 255.0
    stats["rgb"] = {
        "mean": pix.mean(axis=0).tolist(),
        "std": pix.std(axis=0).tolist(),
        "note": "sampled estimate; training still uses ImageNet mean/std",
    }

    out = resolve_path("outputs/dataset_stats.json")
    ensure_dir(out.parent)
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
