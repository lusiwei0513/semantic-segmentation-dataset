"""Convert LabelMe JSON annotations to semantic segmentation masks."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

LABEL_TO_ID = {
    "background": 0,
    "body": 1,
    "windshield": 2,
    "nose_tip": 3,
}


def _resolve_image(json_path: Path, image_path_field: str | None) -> Path | None:
    candidates = []
    if image_path_field:
        candidates.append(json_path.parent / image_path_field)
        candidates.append(json_path.parent / Path(image_path_field).name)
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"):
        candidates.append(json_path.with_suffix(ext))
    for p in candidates:
        if p.exists():
            return p
    return None


def _fill_polygon(mask: np.ndarray, points, class_id: int) -> None:
    pts = np.array(points, dtype=np.int32)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return
    cv2.fillPoly(mask, [pts], int(class_id))


def _fill_rectangle(mask: np.ndarray, points, class_id: int) -> None:
    if len(points) < 2:
        return
    arr = np.asarray(points, dtype=np.float64)
    x_min = int(np.floor(arr[:, 0].min()))
    x_max = int(np.ceil(arr[:, 0].max()))
    y_min = int(np.floor(arr[:, 1].min()))
    y_max = int(np.ceil(arr[:, 1].max()))
    if x_max <= x_min or y_max <= y_min:
        return
    cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), int(class_id), thickness=-1)


def _body_width(shapes: list) -> float | None:
    """Estimate train front body width from body polygon/rectangle."""
    widths = []
    for shape in shapes:
        if shape.get("label") != "body":
            continue
        pts = shape.get("points") or []
        st = (shape.get("shape_type") or "polygon").lower()
        if st == "rectangle" and len(pts) >= 2:
            widths.append(abs(float(pts[0][0]) - float(pts[1][0])))
        elif len(pts) >= 2:
            xs = [float(p[0]) for p in pts]
            widths.append(max(xs) - min(xs))
    if not widths:
        return None
    return float(max(widths))


def _fill_nose_tip_ellipse(
    mask: np.ndarray,
    points,
    class_id: int,
    body_w: float | None,
    image_w: int,
    *,
    width_ratio: float = 0.08,
    height_ratio: float = 0.5,
    min_rx: int = 12,
    max_rx: int = 48,
) -> None:
    """Horizontal ellipse centered on the annotated tip point (PPT-style nosetip)."""
    if not points:
        return
    x, y = float(points[0][0]), float(points[0][1])
    ref_w = body_w if body_w and body_w > 1 else 0.45 * image_w
    rx = int(round(0.5 * width_ratio * ref_w))
    rx = max(min_rx, min(max_rx, rx))
    ry = max(6, int(round(rx * height_ratio)))
    cv2.ellipse(
        mask,
        (int(round(x)), int(round(y))),
        (rx, ry),
        0.0,
        0.0,
        360.0,
        int(class_id),
        thickness=-1,
    )


def _nose_tip_center(shapes: list) -> list[float] | None:
    """Prefer point annotation; otherwise use centroid of a compact nose polygon."""
    point_centers = []
    poly_centers = []
    for shape in shapes:
        if shape.get("label") != "nose_tip":
            continue
        pts = shape.get("points") or []
        if not pts:
            continue
        st = (shape.get("shape_type") or "polygon").lower()
        if st == "point" or len(pts) == 1:
            point_centers.append([float(pts[0][0]), float(pts[0][1])])
            continue
        arr = np.array(pts, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 2:
            continue
        # Ignore oversized mislabels (e.g. whole cabin marked as nose_tip).
        bw = float(arr[:, 0].max() - arr[:, 0].min())
        bh = float(arr[:, 1].max() - arr[:, 1].min())
        if bw * bh > 8000:  # too large for a tip marker
            continue
        poly_centers.append([float(arr[:, 0].mean()), float(arr[:, 1].mean())])
    if point_centers:
        return point_centers[0]
    if poly_centers:
        return poly_centers[0]
    return None


def json_to_mask(
    data: dict,
    nose_tip_radius: int = 16,
    nose_width_ratio: float = 0.08,
) -> np.ndarray:
    h = int(data["imageHeight"])
    w = int(data["imageWidth"])
    mask = np.zeros((h, w), dtype=np.uint8)
    shapes = data.get("shapes", []) or []
    body_w = _body_width(shapes)

    # Paint body / windshield first; nose_tip ellipse last so it carves into body.
    priority = {"body": 0, "windshield": 1, "nose_tip": 2}
    shapes_sorted = sorted(
        shapes,
        key=lambda s: priority.get(s.get("label", ""), 99),
    )

    for shape in shapes_sorted:
        label = shape.get("label")
        if label not in LABEL_TO_ID or label == "background":
            continue
        if label == "nose_tip":
            continue  # handled once below as PPT-style ellipse
        class_id = LABEL_TO_ID[label]
        st = shape.get("shape_type", "polygon")
        pts = shape.get("points") or []
        if st == "polygon":
            _fill_polygon(mask, pts, class_id)
        elif st == "rectangle":
            _fill_rectangle(mask, pts, class_id)
        else:
            if len(pts) >= 3:
                _fill_polygon(mask, pts, class_id)

    tip = _nose_tip_center(shapes)
    if tip is not None:
        _ = nose_tip_radius  # CLI compat
        _fill_nose_tip_ellipse(
            mask,
            [tip],
            LABEL_TO_ID["nose_tip"],
            body_w,
            w,
            width_ratio=nose_width_ratio,
        )
    return mask


def prepare(
    labelme_dir: Path,
    out_dir: Path,
    nose_tip_radius: int,
    copy_images: bool = True,
    nose_width_ratio: float = 0.08,
) -> None:
    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(labelme_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No LabelMe JSON in {labelme_dir}")

    rows = []
    for jf in tqdm(json_files, desc="prepare"):
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
        img_path = _resolve_image(jf, data.get("imagePath"))
        if img_path is None:
            print(f"[skip] image missing for {jf.name}")
            continue

        mask = json_to_mask(
            data,
            nose_tip_radius=nose_tip_radius,
            nose_width_ratio=nose_width_ratio,
        )
        stem = jf.stem
        out_img = images_dir / f"{stem}{img_path.suffix.lower()}"
        out_mask = masks_dir / f"{stem}.png"

        if copy_images:
            shutil.copy2(img_path, out_img)

        # PIL: OpenCV imwrite often fails on Windows paths with non-ASCII chars
        Image.fromarray(mask, mode="L").save(out_mask)

        uniq, counts = np.unique(mask, return_counts=True)
        rows.append(
            {
                "stem": stem,
                "image": out_img.name,
                "mask": out_mask.name,
                "pixels": {int(u): int(c) for u, c in zip(uniq, counts)},
            }
        )

    meta_path = out_dir / "prepare_summary.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_samples": len(rows),
                "label_to_id": LABEL_TO_ID,
                "nose_tip_radius": nose_tip_radius,
                "nose_tip_shape": "horizontal_ellipse",
                "nose_width_ratio": nose_width_ratio,
                "samples": rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Prepared {len(rows)} samples -> {out_dir}")
    print(f"Summary: {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LabelMe -> PNG masks")
    parser.add_argument("--labelme-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--nose-tip-radius", type=int, default=16)
    parser.add_argument(
        "--nose-width-ratio",
        type=float,
        default=0.08,
        help="Nose ellipse full width as fraction of body width (PPT-style).",
    )
    parser.add_argument(
        "--masks-only",
        action="store_true",
        help="Only regenerate masks; keep existing images/ folder.",
    )
    args = parser.parse_args()
    prepare(
        args.labelme_dir.resolve(),
        args.out_dir.resolve(),
        args.nose_tip_radius,
        copy_images=not args.masks_only,
        nose_width_ratio=args.nose_width_ratio,
    )


if __name__ == "__main__":
    main()
