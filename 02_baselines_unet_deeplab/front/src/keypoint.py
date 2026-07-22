"""Nose-tip keypoint helpers: Gaussian heatmap + distance metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_nose_tip_xy(labelme_json: Path) -> tuple[float, float] | None:
    """Prefer point annotation; else compact polygon centroid."""
    if not labelme_json.exists():
        return None
    data = json.loads(labelme_json.read_text(encoding="utf-8"))
    point = None
    poly_c = None
    for shape in data.get("shapes") or []:
        if shape.get("label") != "nose_tip":
            continue
        pts = shape.get("points") or []
        if not pts:
            continue
        st = (shape.get("shape_type") or "polygon").lower()
        if st == "point" or len(pts) == 1:
            point = (float(pts[0][0]), float(pts[0][1]))
            break
        arr = np.asarray(pts, dtype=np.float64)
        bw = float(arr[:, 0].max() - arr[:, 0].min())
        bh = float(arr[:, 1].max() - arr[:, 1].min())
        if bw * bh > 8000:
            continue
        poly_c = (float(arr[:, 0].mean()), float(arr[:, 1].mean()))
    return point if point is not None else poly_c


def make_gaussian_heatmap(
    h: int,
    w: int,
    x: float,
    y: float,
    sigma: float,
) -> np.ndarray:
    """H(x,y)=exp(-((x-x0)^2+(y-y0)^2)/(2σ^2))."""
    if not (np.isfinite(x) and np.isfinite(y)) or sigma <= 0:
        return np.zeros((h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    heat = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma * sigma))
    return heat.astype(np.float32)


def decode_heatmap_peak(heat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """heat: (B,1,H,W) or (B,H,W) -> x,y in pixel coords, shape (B,)."""
    if heat.ndim == 4:
        heat = heat[:, 0]
    b, h, w = heat.shape
    flat = heat.view(b, -1)
    idx = flat.argmax(dim=1)
    y = (idx // w).float()
    x = (idx % w).float()
    return x, y


def circle_mask(h: int, w: int, x: float, y: float, radius: int = 16) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return (((xx - x) ** 2 + (yy - y) ** 2) <= radius * radius).astype(np.uint8)


def ellipse_mask(
    h: int,
    w: int,
    x: float,
    y: float,
    rx: int = 24,
    ry: int = 12,
) -> np.ndarray:
    """Horizontal ellipse mask (PPT-style nosetip)."""
    out = np.zeros((h, w), dtype=np.uint8)
    import cv2

    cv2.ellipse(
        out,
        (int(round(x)), int(round(y))),
        (int(rx), int(ry)),
        0.0,
        0.0,
        360.0,
        1,
        thickness=-1,
    )
    return out


def merge_tip_circle_into_seg(
    seg: np.ndarray,
    x: float,
    y: float,
    tip_class: int = 3,
    radius: int = 16,
) -> np.ndarray:
    """Official-format 4-class mask: draw tip circle over semantic prediction."""
    out = seg.copy()
    h, w = out.shape
    tip = circle_mask(h, w, x, y, radius=radius).astype(bool)
    out[tip] = tip_class
    return out


def merge_tip_ellipse_into_seg(
    seg: np.ndarray,
    x: float,
    y: float,
    tip_class: int = 3,
    rx: int = 24,
    ry: int = 12,
) -> np.ndarray:
    """Official-format mask: draw fixed-axis horizontal tip ellipse over seg."""
    out = seg.copy()
    h, w = out.shape
    tip = ellipse_mask(h, w, x, y, rx=rx, ry=ry).astype(bool)
    out[tip] = tip_class
    return out


@torch.no_grad()
def tip_distance_metrics(
    pred_xy: torch.Tensor,
    gt_xy: torch.Tensor,
    valid: torch.Tensor,
    image_h: torch.Tensor,
    thresholds: tuple[int, ...] = (5, 10, 20),
) -> dict[str, float]:
    """
    pred_xy/gt_xy: (B,2), valid: (B,) bool, image_h: (B,)
    """
    if valid.ndim > 1:
        valid = valid.view(-1)
    if not valid.any():
        out = {"tip_mae_px": float("nan"), "tip_mae_norm_h": float("nan"), "tip_n": 0}
        for t in thresholds:
            out[f"tip_pck@{t}"] = float("nan")
        return out

    d = torch.linalg.norm(pred_xy[valid] - gt_xy[valid], dim=1)
    h = image_h[valid].clamp(min=1.0)
    out = {
        "tip_mae_px": float(d.mean().item()),
        "tip_mae_norm_h": float((d / h).mean().item()),
        "tip_n": int(valid.sum().item()),
    }
    for t in thresholds:
        out[f"tip_pck@{t}"] = float((d <= t).float().mean().item())
    return out


def soft_argmax_xy(heat: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Differentiable peak. heat: (B,1,H,W) -> (B,2) as (x,y)."""
    if heat.ndim == 3:
        heat = heat.unsqueeze(1)
    b, _, h, w = heat.shape
    # spatial softmax; smaller temperature -> sharper peak
    logits = heat.view(b, -1) / max(temperature, 1e-4)
    prob = torch.softmax(logits, dim=1)
    ys = torch.arange(h, device=heat.device, dtype=heat.dtype)
    xs = torch.arange(w, device=heat.device, dtype=heat.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=0)  # 2,HW
    return torch.matmul(prob, grid.t())  # B,2


class HeatmapMSELoss(torch.nn.Module):
    """
    Peak-aware heatmap loss on logits (BCE) + soft-argmax L1.
    Prefer logits+BCE over sigmoid+MSE: avoids mid-map collapse (~0.5 everywhere).
    """

    def __init__(
        self,
        weight: float = 1.0,
        peak_boost: float = 50.0,
        coord_weight: float = 1.0,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.weight = weight
        self.peak_boost = peak_boost
        self.coord_weight = coord_weight
        self.temperature = temperature

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor | None = None,
        tip_xy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # logits/target: B,1,H,W  (target in [0,1])
        if valid is not None:
            if not valid.any():
                return logits.sum() * 0.0
            logits = logits[valid]
            target = target[valid]
            if tip_xy is not None:
                tip_xy = tip_xy[valid]

        # emphasize peak; keep enough bg weight so map does not stay ~0.5
        spatial_w = 1.0 + self.peak_boost * target
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss = self.weight * (bce * spatial_w).sum() / spatial_w.sum().clamp(min=1.0)

        if tip_xy is not None and self.coord_weight > 0:
            # soft-argmax on probability map
            pred_xy = soft_argmax_xy(torch.sigmoid(logits), temperature=self.temperature)
            loss = loss + self.coord_weight * F.l1_loss(pred_xy, tip_xy)
        return loss
