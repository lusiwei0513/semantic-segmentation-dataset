#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证入口占位。"""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    print(f"validate placeholder ready: config={args.config} fold={args.fold}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
