#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "meta": meta or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
