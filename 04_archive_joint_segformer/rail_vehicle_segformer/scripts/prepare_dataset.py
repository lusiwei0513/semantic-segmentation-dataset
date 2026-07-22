#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 prepared_front / prepared_side 转换为标准 processed 格式。
不修改原始数据；可重复执行。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.heatmap import disk_centroid  # noqa: E402
from src.utils.io import ensure_dir, load_yaml, resolve_path  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEG_CHANNELS = ["body", "windshield", "bogie", "door"]


def parse_stem(stem: str, view: str) -> Tuple[str, str]:
    suffix = f"_{view}"
    if not stem.endswith(suffix):
        raise ValueError(f"stem 不以 _{view} 结尾: {stem}")
    base = stem[: -len(suffix)]
    uuid, _, rest = base.partition("-")
    vehicle_id = rest if rest else base
    return uuid, vehicle_id


def find_image(images_dir: Path, stem: str) -> Path:
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
        p2 = images_dir / f"{stem}{ext.upper()}"
        if p2.exists():
            return p2
    raise FileNotFoundError(f"找不到图像: {stem} in {images_dir}")


def write_binary_mask(out_path: Path, binary: np.ndarray) -> None:
    ensure_dir(out_path.parent)
    out = (binary.astype(np.uint8) * 255)
    Image.fromarray(out, mode="L").save(out_path)


def process_view(
    view: str,
    raw_dir: Path,
    processed_root: Path,
    class_ids: Dict[str, int],
    view_defaults: Dict[str, bool],
    rows: List[Dict],
    log_lines: List[str],
) -> None:
    images_dir = raw_dir / "images"
    masks_dir = raw_dir / "masks"
    mask_files = sorted(masks_dir.glob("*.png"))
    log_lines.append(f"[{view}] 发现掩码 {len(mask_files)} 张 @ {raw_dir}")

    for mask_path in tqdm(mask_files, desc=f"prepare-{view}"):
        stem = mask_path.stem
        pair_id, vehicle_id = parse_stem(stem, view)
        sample_id = f"{view}_{pair_id}"

        image_src = find_image(images_dir, stem)
        with Image.open(image_src) as im:
            width, height = im.size
            image_rgb = im.convert("RGB")

        mask = np.array(Image.open(mask_path))
        if mask.ndim != 2:
            raise ValueError(f"掩码非单通道: {mask_path}")
        if mask.shape[0] != height or mask.shape[1] != width:
            # PIL size is (W,H), numpy is (H,W)
            raise ValueError(
                f"图掩尺寸不一致: image=({width},{height}) mask={mask.shape} @ {stem}"
            )

        # copy image into processed (do not touch raw)
        rel_image = f"images/{sample_id}{Path(image_src).suffix.lower()}"
        dst_image = processed_root / rel_image
        ensure_dir(dst_image.parent)
        if not dst_image.exists() or dst_image.stat().st_size != image_src.stat().st_size:
            shutil.copy2(image_src, dst_image)

        valid_tasks = {
            "body": False,
            "windshield": False,
            "bogie": False,
            "door": False,
            "nose_tip": False,
        }
        mask_paths = {c: "" for c in SEG_CHANNELS}
        keypoint_path = ""

        # segmentation channels from exclusive ids
        for ch in SEG_CHANNELS:
            # view permanently invalid?
            if not view_defaults.get(ch, False):
                continue
            cid = class_ids.get(ch)
            if cid is None:
                continue
            binary = mask == cid
            n = int(binary.sum())
            if n == 0:
                log_lines.append(f"WARN missing {view}/{sample_id}/{ch}")
                continue
            rel = f"masks/{ch}/{sample_id}.png"
            write_binary_mask(processed_root / rel, binary)
            mask_paths[ch] = rel
            valid_tasks[ch] = True

        # nose_tip for front
        if view_defaults.get("nose_tip", False):
            tip_id = class_ids.get("nose_tip", 3)
            tip = disk_centroid(mask, class_id=tip_id)
            if tip["visible"]:
                # bounds check
                if not (0 <= tip["x"] < width and 0 <= tip["y"] < height):
                    raise ValueError(f"nose_tip 越界: {sample_id} {tip}")
                rel_kp = f"keypoints/{sample_id}.json"
                ensure_dir((processed_root / rel_kp).parent)
                (processed_root / rel_kp).write_text(
                    json.dumps(tip, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                keypoint_path = rel_kp
                valid_tasks["nose_tip"] = True
            else:
                log_lines.append(f"WARN missing {view}/{sample_id}/nose_tip")

        # body 是基础任务：缺失则跳过该样本，不写入 metadata
        if not valid_tasks["body"]:
            log_lines.append(f"SKIP {view}/{sample_id}: body missing")
            continue

        rows.append(
            {
                "sample_id": sample_id,
                "image_path": rel_image,
                "vehicle_id": vehicle_id,
                "pair_id": pair_id,
                "view": view,
                "width": width,
                "height": height,
                "body_mask": mask_paths["body"],
                "windshield_mask": mask_paths["windshield"],
                "bogie_mask": mask_paths["bogie"],
                "door_mask": mask_paths["door"],
                "keypoint_path": keypoint_path,
                "valid_body": int(valid_tasks["body"]),
                "valid_windshield": int(valid_tasks["windshield"]),
                "valid_bogie": int(valid_tasks["bogie"]),
                "valid_door": int(valid_tasks["door"]),
                "valid_nose_tip": int(valid_tasks["nose_tip"]),
                "source_raw": str(raw_dir),
            }
        )
        # keep image_rgb referenced to avoid unused lint in some tooling
        _ = image_rgb


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    processed_root = resolve_path(cfg["processed_root"])
    ensure_dir(processed_root)

    rows: List[Dict] = []
    log_lines: List[str] = [
        f"prepare_dataset start {datetime.now().isoformat(timespec='seconds')}"
    ]

    for view in ["front", "side"]:
        raw_key = f"{view}_dir"
        raw_dir = resolve_path(cfg["raw"][raw_key])
        class_ids = cfg[f"{view}_class_ids"]
        view_defaults = cfg["view_task_defaults"][view]
        process_view(
            view=view,
            raw_dir=raw_dir,
            processed_root=processed_root,
            class_ids=class_ids,
            view_defaults=view_defaults,
            rows=rows,
            log_lines=log_lines,
        )

    df = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    meta_path = resolve_path(cfg["metadata_csv"])
    ensure_dir(meta_path.parent)
    df.to_csv(meta_path, index=False, encoding="utf-8-sig")

    log_path = resolve_path("outputs/prepare_dataset.log")
    ensure_dir(log_path.parent)
    log_lines.append(f"wrote metadata n={len(df)} -> {meta_path}")
    log_lines.append(
        f"front={int((df.view=='front').sum())} side={int((df.view=='side').sum())}"
    )
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"完成: {meta_path} ({len(df)} 行)")
    print(f"日志: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
