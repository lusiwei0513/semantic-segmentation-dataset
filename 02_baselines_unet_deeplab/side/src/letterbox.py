"""Letterbox (LongestMaxSize + PadIfNeeded) forward/inverse helpers."""

from __future__ import annotations

import cv2
import numpy as np


def letterbox_meta(h0: int, w0: int, image_size: int) -> dict:
    """Compute scale and pad that match Albumentations LongestMaxSize + PadIfNeeded(center)."""
    scale = float(image_size) / float(max(h0, w0))
    nh = int(round(h0 * scale))
    nw = int(round(w0 * scale))
    pad_h = image_size - nh
    pad_w = image_size - nw
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return {
        "h0": h0,
        "w0": w0,
        "image_size": image_size,
        "scale": scale,
        "nh": nh,
        "nw": nw,
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
    }


def letterbox_image(rgb: np.ndarray, image_size: int) -> tuple[np.ndarray, dict]:
    h0, w0 = rgb.shape[:2]
    meta = letterbox_meta(h0, w0, image_size)
    resized = cv2.resize(rgb, (meta["nw"], meta["nh"]), interpolation=cv2.INTER_LINEAR)
    out = cv2.copyMakeBorder(
        resized,
        meta["top"],
        meta["bottom"],
        meta["left"],
        meta["right"],
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return out, meta


def letterbox_mask(mask: np.ndarray, image_size: int) -> tuple[np.ndarray, dict]:
    h0, w0 = mask.shape[:2]
    meta = letterbox_meta(h0, w0, image_size)
    resized = cv2.resize(mask, (meta["nw"], meta["nh"]), interpolation=cv2.INTER_NEAREST)
    out = cv2.copyMakeBorder(
        resized,
        meta["top"],
        meta["bottom"],
        meta["left"],
        meta["right"],
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )
    return out, meta


def unletterbox_mask(mask_sq: np.ndarray, meta: dict) -> np.ndarray:
    """Map square letterboxed prediction back to original HxW."""
    top, left = meta["top"], meta["left"]
    nh, nw = meta["nh"], meta["nw"]
    crop = mask_sq[top : top + nh, left : left + nw]
    return cv2.resize(crop, (meta["w0"], meta["h0"]), interpolation=cv2.INTER_NEAREST)


def unletterbox_xy(x: float, y: float, meta: dict) -> tuple[float, float]:
    """Map point from letterboxed canvas to original image coords."""
    x2 = (x - meta["left"]) / meta["scale"]
    y2 = (y - meta["top"]) / meta["scale"]
    return float(x2), float(y2)
