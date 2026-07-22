#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
import csv

out_dir = Path(__file__).resolve().parent
meta_path = out_dir / "metadata.csv"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect(view: str, root: Path):
    rows = []
    for p in sorted((root / "images").iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        stem = p.stem
        suffix = "_" + view
        if not stem.endswith(suffix):
            raise ValueError(f"unexpected stem {stem} for view {view}")
        base = stem[: -len(suffix)]
        uuid, _, rest = base.partition("-")
        vehicle_id = rest if rest else base
        rows.append(
            {
                "sample_id": f"{view}_{uuid}",
                "image_path": str(p.resolve()),
                "vehicle_id": vehicle_id,
                "view": view,
                # 无真实设备元数据：高度均为 1080，暂记为同一成像流水线，
                # 避免把 view 人为绑定成不同 device。
                "device_id": "pipeline_1080",
                "source_batch": root.name,
                "pair_id": uuid,
            }
        )
    return rows


def main():
    rows = []
    archive_root = Path(__file__).resolve().parents[1]
    rows += collect("front", archive_root / "data_prepared" / "prepared_front")
    rows += collect("side", archive_root / "data_prepared" / "prepared_side")

    with meta_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "image_path",
                "vehicle_id",
                "view",
                "device_id",
                "source_batch",
                "pair_id",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {meta_path} n={len(rows)}")
    print(
        "front",
        sum(1 for r in rows if r["view"] == "front"),
        "side",
        sum(1 for r in rows if r["view"] == "side"),
    )
    print("unique vehicles", len({r["vehicle_id"] for r in rows}))
    print("unique pairs", len({r["pair_id"] for r in rows}))


if __name__ == "__main__":
    main()
