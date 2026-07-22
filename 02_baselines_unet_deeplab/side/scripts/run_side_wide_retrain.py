#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""串行重训侧视 UNet / DeepLab @ 384×1536。"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Prefer same CUDA venv as SegFormer work
PY = (
    ROOT.parent
    / "gpt标注训练正视图-seg_train"
    / ".venv"
    / "Scripts"
    / "python.exe"
)
STATUS = ROOT / "outputs" / "SIDE_WIDE_RETRAIN_STATUS.txt"

JOBS = [
    "config_side_unet_wide_384x1536.yaml",
    "config_side_deeplab_wide_384x1536.yaml",
]


def log(msg: str) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with STATUS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def main() -> int:
    if not PY.is_file():
        log(f"python missing: {PY}")
        return 1
    STATUS.write_text("side 384x1536 retrain queue\n", encoding="utf-8")
    log(f"python={PY}")
    for cfg in JOBS:
        log(f"TRAIN start {cfg}")
        r = subprocess.run(
            [str(PY), "-u", "scripts/train.py", "--config", cfg],
            cwd=str(ROOT),
        )
        if r.returncode != 0:
            log(f"FAILED {cfg} exit={r.returncode}")
            return r.returncode
        log(f"TRAIN done {cfg}")
    log("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
