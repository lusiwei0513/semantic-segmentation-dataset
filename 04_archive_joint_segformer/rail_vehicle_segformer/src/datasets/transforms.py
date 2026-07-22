#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Letterbox 与同步几何/颜色增强。"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def letterbox(
    image: np.ndarray,
    masks: List[np.ndarray],
    keypoint_xy: Optional[Tuple[float, float]],
    target_size: Tuple[int, int],
    pad_value: int = 0,
) -> Dict:
    """
    target_size: (H, W)
    返回缩放后的 image/masks/keypoint 以及变换元数据，便于映射回原图。
    """
    th, tw = target_size
    h, w = image.shape[:2]
    scale = min(th / h, tw / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    resized_masks = [
        cv2.resize(m.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST)
        for m in masks
    ]

    pad_h = th - nh
    pad_w = tw - nw
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    out_img = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad_value,) * 3
    )
    out_masks = [
        cv2.copyMakeBorder(m, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        for m in resized_masks
    ]

    kp = None
    if keypoint_xy is not None:
        x, y = keypoint_xy
        kp = (x * scale + left, y * scale + top)

    meta = {
        "scale": scale,
        "pad_top": top,
        "pad_left": left,
        "orig_h": h,
        "orig_w": w,
        "target_h": th,
        "target_w": tw,
    }
    return {"image": out_img, "masks": out_masks, "keypoint_xy": kp, "meta": meta}


def hflip(
    image: np.ndarray,
    masks: List[np.ndarray],
    keypoint_xy: Optional[Tuple[float, float]],
) -> Tuple[np.ndarray, List[np.ndarray], Optional[Tuple[float, float]]]:
    image = np.ascontiguousarray(image[:, ::-1])
    masks = [np.ascontiguousarray(m[:, ::-1]) for m in masks]
    if keypoint_xy is not None:
        x, y = keypoint_xy
        keypoint_xy = (image.shape[1] - 1 - x, y)
    return image, masks, keypoint_xy


def small_rotate(
    image: np.ndarray,
    masks: List[np.ndarray],
    keypoint_xy: Optional[Tuple[float, float]],
    angle_deg: float,
) -> Tuple[np.ndarray, List[np.ndarray], Optional[Tuple[float, float]]]:
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    masks = [
        cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0) for m in masks
    ]
    if keypoint_xy is not None:
        x, y = keypoint_xy
        nx = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        ny = M[1, 0] * x + M[1, 1] * y + M[1, 2]
        keypoint_xy = (float(nx), float(ny))
    return image, masks, keypoint_xy


def color_augment(image: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    img = image.astype(np.float32)
    # brightness / contrast
    alpha = 1.0 + rng.uniform(-0.2, 0.2)
    beta = rng.uniform(-20, 20)
    img = img * alpha + beta
    # gamma
    gamma = rng.uniform(0.85, 1.15)
    img = np.clip(img, 0, 255)
    img = 255.0 * ((img / 255.0) ** gamma)
    # slight channel shift
    shift = rng.uniform(-8, 8, size=(1, 1, 3))
    img = img + shift
    return np.clip(img, 0, 255).astype(np.uint8)


def maybe_augment(
    image: np.ndarray,
    masks: List[np.ndarray],
    keypoint_xy: Optional[Tuple[float, float]],
    train: bool,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, List[np.ndarray], Optional[Tuple[float, float]]]:
    if not train:
        return image, masks, keypoint_xy
    rng = rng or np.random.RandomState()
    if rng.rand() < 0.5:
        image, masks, keypoint_xy = hflip(image, masks, keypoint_xy)
    if rng.rand() < 0.5:
        angle = float(rng.uniform(-5, 5))
        image, masks, keypoint_xy = small_rotate(image, masks, keypoint_xy, angle)
    if rng.rand() < 0.8:
        image = color_augment(image, rng)
    return image, masks, keypoint_xy


def normalize_imagenet(image: np.ndarray, mean, std) -> np.ndarray:
    """image RGB uint8 HxWx3 -> float32 CxHxW"""
    x = image.astype(np.float32) / 255.0
    x = (x - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return np.transpose(x, (2, 0, 1))
