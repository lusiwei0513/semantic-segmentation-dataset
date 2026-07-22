#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""推理入口占位。"""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--image", type=str, default="")
    args = parser.parse_args()
    print(f"infer placeholder ready: config={args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
