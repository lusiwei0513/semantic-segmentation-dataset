#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按 vehicle_id 做 StratifiedGroupKFold 五折划分。"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir, load_yaml, resolve_path  # noqa: E402


def vehicle_level_table(df: pd.DataFrame) -> pd.DataFrame:
    """每个 vehicle 一行；分层标签优先用是否含 front（配对车通常 front+side）。"""
    rows = []
    for vid, g in df.groupby("vehicle_id"):
        views = set(g["view"].tolist())
        if views == {"front", "side"}:
            stratum = "both"
        elif views == {"front"}:
            stratum = "front_only"
        else:
            stratum = "side_only"
        rows.append(
            {
                "vehicle_id": vid,
                "stratum": stratum,
                "n_samples": len(g),
                "sample_ids": g["sample_id"].tolist(),
            }
        )
    return pd.DataFrame(rows)


def split_train_val(
    train_vehicles: List[str],
    vehicle_to_samples: Dict[str, List[str]],
    strata: Dict[str, str],
    val_ratio: float = 0.12,
    seed: int = 42,
) -> tuple[List[str], List[str]]:
    rng = np.random.RandomState(seed)
    by_stratum: Dict[str, List[str]] = {}
    for v in train_vehicles:
        by_stratum.setdefault(strata[v], []).append(v)

    val_vehicles: List[str] = []
    remain: List[str] = []
    for _, vids in by_stratum.items():
        vids = list(vids)
        rng.shuffle(vids)
        n_val = max(1, int(round(len(vids) * val_ratio))) if len(vids) >= 5 else max(0, len(vids) // 8)
        val_vehicles.extend(vids[:n_val])
        remain.extend(vids[n_val:])

    if not remain:
        # 兜底：至少留一辆在 train
        remain.append(val_vehicles.pop())

    def expand(vs: List[str]) -> List[str]:
        out = []
        for v in vs:
            out.extend(vehicle_to_samples[v])
        return sorted(out)

    return expand(remain), expand(val_vehicles)


def assert_no_leak(train: List[str], val: List[str], test: List[str], df: pd.DataFrame) -> None:
    id_to_v = dict(zip(df["sample_id"], df["vehicle_id"]))
    sets = {
        "train": set(train),
        "val": set(val),
        "test": set(test),
    }
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        inter = sets[a] & sets[b]
        if inter:
            raise RuntimeError(f"{a}/{b} sample 泄漏: {list(inter)[:5]}")

    vsets = {k: {id_to_v[s] for s in vs} for k, vs in sets.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        inter = vsets[a] & vsets[b]
        if inter:
            raise RuntimeError(f"{a}/{b} vehicle 泄漏: {list(inter)[:5]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.12)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 42))
    df = pd.read_csv(resolve_path(cfg["metadata_csv"]))
    splits_dir = resolve_path(cfg["splits_dir"])
    ensure_dir(splits_dir)

    vtable = vehicle_level_table(df)
    vehicle_to_samples = {
        r["vehicle_id"]: r["sample_ids"] for _, r in vtable.iterrows()
    }
    strata = {r["vehicle_id"]: r["stratum"] for _, r in vtable.iterrows()}

    X = np.arange(len(vtable))
    y = vtable["stratum"].to_numpy()
    groups = vtable["vehicle_id"].to_numpy()

    sgkf = StratifiedGroupKFold(
        n_splits=args.n_splits, shuffle=True, random_state=seed
    )

    summary = []
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
        train_vehicles = vtable.iloc[train_idx]["vehicle_id"].tolist()
        test_vehicles = vtable.iloc[test_idx]["vehicle_id"].tolist()

        train_samples, val_samples = split_train_val(
            train_vehicles,
            vehicle_to_samples,
            strata,
            val_ratio=args.val_ratio,
            seed=seed + fold,
        )
        test_samples = sorted(
            s for v in test_vehicles for s in vehicle_to_samples[v]
        )
        assert_no_leak(train_samples, val_samples, test_samples, df)

        payload = {
            "fold": fold,
            "train": train_samples,
            "val": val_samples,
            "test": test_samples,
        }
        out = splits_dir / f"fold_{fold}.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        def view_count(sids: List[str]) -> Dict[str, int]:
            sub = df[df["sample_id"].isin(sids)]
            return Counter(sub["view"].tolist())

        summary.append(
            {
                "fold": fold,
                "n_train": len(train_samples),
                "n_val": len(val_samples),
                "n_test": len(test_samples),
                "train_views": dict(view_count(train_samples)),
                "val_views": dict(view_count(val_samples)),
                "test_views": dict(view_count(test_samples)),
                "n_train_vehicles": len(train_vehicles) - len({df.set_index("sample_id").loc[s, "vehicle_id"] for s in val_samples}),
                "n_val_vehicles": len({df.set_index("sample_id").loc[s, "vehicle_id"] for s in val_samples}),
                "n_test_vehicles": len(test_vehicles),
            }
        )
        print(f"fold {fold}: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")

    report = {
        "n_splits": args.n_splits,
        "seed": seed,
        "folds": summary,
    }
    report_path = resolve_path("outputs/fold_split_report.json")
    ensure_dir(report_path.parent)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"划分报告: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
