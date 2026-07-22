#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""串行跑 front tip/sigma 消融 A1->A2->A3，并在每组后做 test 评测。"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT.parent / "02_baselines_unet_deeplab" / "front" / ".venv" / "Scripts" / "python.exe"
STATUS = ROOT / "outputs" / "train" / "TIP_ABLATION_STATUS.txt"

JOBS = [
    {
        "name": "A1_w6_s16",
        "config": "configs/train_front_tip_a1_w6_s16.yaml",
        "out": "outputs/train/front_fold0_tip_a1_w6_s16",
        "eval": "outputs/eval/front_fold0_tip_a1_w6_s16_test",
    },
    {
        "name": "A2_w6_s12",
        "config": "configs/train_front_tip_a2_w6_s12.yaml",
        "out": "outputs/train/front_fold0_tip_a2_w6_s12",
        "eval": "outputs/eval/front_fold0_tip_a2_w6_s12_test",
    },
    {
        "name": "A3_w6_s20",
        "config": "configs/train_front_tip_a3_w6_s20.yaml",
        "out": "outputs/train/front_fold0_tip_a3_w6_s20",
        "eval": "outputs/eval/front_fold0_tip_a3_w6_s20_test",
    },
]


def log(msg: str) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with STATUS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> int:
    if not PY.is_file():
        log(f"Python not found: {PY}")
        return 1

    STATUS.write_text("Tip/sigma ablation queue started\n", encoding="utf-8")
    log(f"python={PY}")
    log(f"root={ROOT}")

    py = str(PY)
    for j in JOBS:
        best = ROOT / j["out"] / "checkpoints" / "best.pt"
        if best.is_file():
            log(f"{j['name']}: best.pt exists, skip train")
        else:
            log(f"{j['name']}: TRAIN start -> {j['out']}")
            run(
                [
                    py,
                    "-u",
                    "src/train.py",
                    "--config",
                    j["config"],
                    "--view",
                    "front",
                    "--fold",
                    "0",
                    "--output-dir",
                    j["out"],
                ]
            )
            log(f"{j['name']}: TRAIN done")

        report = ROOT / j["eval"] / "test_report.json"
        if report.is_file():
            log(f"{j['name']}: test report exists, skip eval")
        else:
            log(f"{j['name']}: EVAL test start")
            (ROOT / j["eval"]).mkdir(parents=True, exist_ok=True)
            run(
                [
                    py,
                    "-u",
                    "scripts/evaluate_test.py",
                    "--config",
                    j["config"],
                    "--checkpoint",
                    str(best),
                    "--view",
                    "front",
                    "--split",
                    "test",
                    "--output-dir",
                    j["eval"],
                ]
            )
            log(f"{j['name']}: EVAL done")

    log("ALL DONE — compare tip_mae / PCK / official_mIoU under outputs/eval/front_fold0_tip_*_test/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
