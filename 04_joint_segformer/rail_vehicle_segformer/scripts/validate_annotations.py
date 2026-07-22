#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""校验 processed metadata / 掩码 / 关键点。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir, load_yaml, resolve_path  # noqa: E402

SEG_COLS = [
    ("body", "body_mask", "valid_body"),
    ("windshield", "windshield_mask", "valid_windshield"),
    ("bogie", "bogie_mask", "valid_bogie"),
    ("door", "door_mask", "valid_door"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)

    processed_root = resolve_path(cfg["processed_root"])
    meta_path = resolve_path(cfg["metadata_csv"])
    df = pd.read_csv(meta_path)

    errors: List[str] = []
    warnings: List[str] = []

    if df["sample_id"].duplicated().any():
        dups = df.loc[df["sample_id"].duplicated(), "sample_id"].tolist()
        errors.append(f"重复 sample_id: {dups[:10]}")

    if df["image_path"].duplicated().any():
        dups = df.loc[df["image_path"].duplicated(), "image_path"].tolist()
        errors.append(f"重复 image_path: {dups[:10]}")

    area_stats: Dict[str, List[float]] = {k: [] for k, _, _ in SEG_COLS}
    valid_counts = {k: 0 for k, _, _ in SEG_COLS}
    valid_counts["nose_tip"] = 0

    for _, row in df.iterrows():
        sid = row["sample_id"]
        view = row["view"]
        img_path = processed_root / str(row["image_path"])
        if not img_path.exists():
            errors.append(f"图像缺失: {sid} {img_path}")
            continue
        try:
            with Image.open(img_path) as im:
                w, h = im.size
                im.load()
        except Exception as exc:
            errors.append(f"图像不可读: {sid} {exc!r}")
            continue

        if int(row["width"]) != w or int(row["height"]) != h:
            errors.append(
                f"metadata 尺寸不一致: {sid} meta=({row['width']},{row['height']}) real=({w},{h})"
            )

        for task, path_col, valid_col in SEG_COLS:
            path_val = row.get(path_col, "")
            valid = int(row.get(valid_col, 0)) == 1
            path_empty = pd.isna(path_val) or str(path_val).strip() == ""

            # view permanent invalid
            if view == "front" and task in ("bogie", "door"):
                if valid or not path_empty:
                    errors.append(f"front 不应有 {task}: {sid}")
                continue
            if view == "side" and False:
                pass

            if valid:
                if path_empty:
                    errors.append(f"valid={task} 但路径为空: {sid}")
                    continue
                mp = processed_root / str(path_val)
                if not mp.exists():
                    errors.append(f"掩码缺失: {sid} {task} {mp}")
                    continue
                arr = np.array(Image.open(mp))
                if arr.ndim != 2:
                    errors.append(f"掩码非单通道: {sid} {task}")
                    continue
                if arr.shape != (h, w):
                    errors.append(f"掩码尺寸不一致: {sid} {task} {arr.shape} vs {(h, w)}")
                    continue
                uniq = set(np.unique(arr).tolist())
                if not uniq.issubset({0, 1, 255}):
                    errors.append(f"掩码非法像素: {sid} {task} {sorted(uniq)[:10]}")
                    continue
                if uniq <= {0}:
                    errors.append(f"valid 掩码全零: {sid} {task}")
                    continue
                pos = arr > 0
                ratio = float(pos.mean())
                area_stats[task].append(ratio)
                valid_counts[task] += 1
                if ratio < 1e-5 or ratio > 0.95:
                    warnings.append(f"面积异常 {sid}/{task} ratio={ratio:.6f}")
            else:
                if not path_empty:
                    warnings.append(f"invalid 但仍有路径: {sid} {task}")

        # nose tip
        kp_path = row.get("keypoint_path", "")
        valid_tip = int(row.get("valid_nose_tip", 0)) == 1
        kp_empty = pd.isna(kp_path) or str(kp_path).strip() == ""
        if view == "side":
            if valid_tip or not kp_empty:
                errors.append(f"side 不应有 nose_tip: {sid}")
        elif view == "front":
            if valid_tip:
                if kp_empty:
                    errors.append(f"valid nose_tip 但无点路径空: {sid}")
                else:
                    jp = processed_root / str(kp_path)
                    if not jp.exists():
                        errors.append(f"关键点 JSON 缺失: {sid}")
                    else:
                        tip = json.loads(jp.read_text(encoding="utf-8"))
                        if not tip.get("visible", False):
                            errors.append(f"valid tip 但 visible=false: {sid}")
                        else:
                            x, y = float(tip["x"]), float(tip["y"])
                            if not (0 <= x < w and 0 <= y < h):
                                errors.append(f"关键点越界: {sid} ({x},{y})")
                            valid_counts["nose_tip"] += 1
            else:
                if not kp_empty:
                    warnings.append(f"invalid tip 仍有 JSON: {sid}")

        if view == "front" and int(row.get("valid_body", 0)) != 1:
            errors.append(f"front 缺少 body: {sid}")
        if view == "side" and int(row.get("valid_body", 0)) != 1:
            errors.append(f"side 缺少 body: {sid}")

    report = {
        "n_samples": int(len(df)),
        "n_front": int((df["view"] == "front").sum()),
        "n_side": int((df["view"] == "side").sum()),
        "n_vehicles": int(df["vehicle_id"].nunique()),
        "valid_counts": valid_counts,
        "area_ratio_mean": {
            k: (float(np.mean(v)) if v else None) for k, v in area_stats.items()
        },
        "n_errors": len(errors),
        "n_warnings": len(warnings),
        "errors": errors,
        "warnings": warnings[:200],
    }

    out_json = resolve_path("outputs/data_validation_report.json")
    out_md = resolve_path("outputs/data_validation_report.md")
    ensure_dir(out_json.parent)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# 数据校验报告",
        "",
        f"- 总样本: {report['n_samples']}",
        f"- 正视图: {report['n_front']}",
        f"- 侧视图: {report['n_side']}",
        f"- 车辆数: {report['n_vehicles']}",
        f"- 错误数: {report['n_errors']}",
        f"- 警告数: {report['n_warnings']}",
        "",
        "## 有效任务计数",
        "",
        "```json",
        json.dumps(valid_counts, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 错误",
        "",
    ]
    md.extend([f"- {e}" for e in errors] or ["- 无"])
    md.extend(["", "## 警告（最多 200）", ""])
    md.extend([f"- {w}" for w in warnings[:200]] or ["- 无"])
    out_md.write_text("\n".join(md), encoding="utf-8")

    print(f"报告: {out_md}")
    print(f"错误={len(errors)} 警告={len(warnings)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
