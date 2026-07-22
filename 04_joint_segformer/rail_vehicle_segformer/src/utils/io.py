#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""共享工具：配置加载与路径解析。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    base = base or PROJECT_ROOT
    return (base / path).resolve()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
