"""Export fold test-set overlays for front UNet / DeepLab / KP."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from src.dataset import build_transforms, load_fold_pairs  # noqa: E402
from src.keypoint import (  # noqa: E402
    decode_heatmap_peak,
    merge_tip_ellipse_into_seg,
)
from src.letterbox import letterbox_meta, unletterbox_mask, unletterbox_xy  # noqa: E402
from src.metrics import confusion_matrix, iou_from_cm  # noqa: E402

FRONT_NAMES = ["background", "body", "windshield", "nose_tip"]
FRONT_PALETTE = np.array(
    [[0, 0, 0], [0, 120, 255], [0, 220, 120], [255, 60, 60]], dtype=np.uint8
)


def colorize(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, c in enumerate(FRONT_PALETTE):
        out[mask == i] = c
    return out


def overlay_rgb(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = colorize(mask).astype(np.float32)
    base = rgb.astype(np.float32)
    return np.clip(base * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8)


def resolve(base: Path, p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (base / path).resolve()


def build_seg_model(cfg: dict):
    import segmentation_models_pytorch as smp

    m = cfg["model"]
    arch = str(m.get("arch", "Unet")).lower()
    encoder = m["encoder"]
    classes = int(m["num_classes"])

    def _make(w):
        if arch in {"deeplabv3plus", "deeplabv3+", "deeplab"}:
            return smp.DeepLabV3Plus(
                encoder_name=encoder, encoder_weights=w, in_channels=3, classes=classes
            )
        return smp.Unet(
            encoder_name=encoder, encoder_weights=w, in_channels=3, classes=classes
        )

    return _make(None)


def draw_tip_marker(rgb: np.ndarray, xy: tuple[float, float]) -> np.ndarray:
    out = rgb.copy()
    x, y = int(round(xy[0])), int(round(xy[1]))
    cv2.drawMarker(out, (x, y), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2)
    cv2.circle(out, (x, y), 8, (0, 255, 255), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--mode", choices=["seg", "kp"], default="seg")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--max-images", type=int, default=0)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    run_cfg = ckpt.get("cfg", cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "seg":
        model = build_seg_model(run_cfg)
        model.load_state_dict(ckpt["model"])
    else:
        from scripts.train_front_kp import SegTipUnet

        mcfg = run_cfg["model"]
        model = SegTipUnet(mcfg["encoder"], None, int(mcfg.get("seg_classes", 3)))
        model.load_state_dict(ckpt["model"])

    model.to(device).eval()

    prepared = resolve(ROOT, cfg["data"]["prepared_dir"])
    fold_json = resolve(ROOT, cfg["data"]["fold_json"])
    pairs = load_fold_pairs(prepared, fold_json, split=args.split)
    if args.max_images > 0:
        pairs = pairs[: args.max_images]

    out_dir = args.out_dir or (args.weights.parent / f"{args.split}_vis")
    for sub in ("pred_overlay", "gt_overlay", "compare", "pred_mask"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    image_size = int(run_cfg["data"].get("image_size", cfg["data"]["image_size"]))
    tip_rx = int(cfg["data"].get("tip_ellipse_rx", 24))
    tip_ry = int(cfg["data"].get("tip_ellipse_ry", 12))
    tf = build_transforms(image_size, train=False)
    num_classes = 4
    cm = torch.zeros(num_classes, num_classes)
    per_image = []

    for img_path, mask_path in tqdm(pairs, desc=f"{args.split}_vis_front"):
        rgb = np.array(Image.open(img_path).convert("RGB"))
        gt = np.array(Image.open(mask_path).convert("L"))
        h0, w0 = rgb.shape[:2]
        meta = letterbox_meta(h0, w0, image_size)
        t = tf(image=rgb, mask=np.zeros((h0, w0), dtype=np.uint8))
        tip_xy = None
        pred3 = None

        with torch.no_grad():
            if args.mode == "seg":
                pred_sq = (
                    model(t["image"].unsqueeze(0).to(device))
                    .argmax(1)[0]
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )
            else:
                seg_logits, heat = model(t["image"].unsqueeze(0).to(device))
                pred3 = seg_logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
                px, py = decode_heatmap_peak(heat.sigmoid())
                x_c = float(px[0].item())
                y_c = float(py[0].item())
                # heatmap may be lower-res; map to letterbox canvas then original
                hh, ww = heat.shape[-2], heat.shape[-1]
                x_lb = x_c * (image_size / max(ww, 1))
                y_lb = y_c * (image_size / max(hh, 1))
                tip_xy = unletterbox_xy(x_lb, y_lb, meta)
                pred_sq = merge_tip_ellipse_into_seg(
                    pred3, x_lb, y_lb, tip_class=3, rx=tip_rx, ry=tip_ry
                )

        if tip_xy is not None and pred3 is not None:
            # redraw ellipse at original-resolution tip for cleaner PPT overlays
            pred = merge_tip_ellipse_into_seg(
                unletterbox_mask(pred3, meta),
                tip_xy[0],
                tip_xy[1],
                tip_class=3,
                rx=tip_rx,
                ry=tip_ry,
            )
        else:
            pred = unletterbox_mask(pred_sq, meta)

        cm += confusion_matrix(
            torch.from_numpy(pred), torch.from_numpy(gt.astype(np.int64)), num_classes
        )
        iou_i, miou_i = iou_from_cm(
            confusion_matrix(
                torch.from_numpy(pred),
                torch.from_numpy(gt.astype(np.int64)),
                num_classes,
            )
        )
        per_image.append(
            {
                "stem": img_path.stem,
                "miou": float(miou_i),
                "iou": {FRONT_NAMES[i]: float(iou_i[i]) for i in range(num_classes)},
            }
        )

        pred_ov = overlay_rgb(rgb, pred)
        gt_ov = overlay_rgb(rgb, gt)
        if tip_xy is not None:
            pred_ov = draw_tip_marker(pred_ov, tip_xy)
        gap = np.ones((rgb.shape[0], 8, 3), dtype=np.uint8) * 255
        compare = np.concatenate([rgb, gap, gt_ov, gap, pred_ov], axis=1)
        Image.fromarray(pred).save(out_dir / "pred_mask" / f"{img_path.stem}.png")
        Image.fromarray(pred_ov).save(
            out_dir / "pred_overlay" / f"{img_path.stem}.jpg", quality=90
        )
        Image.fromarray(gt_ov).save(
            out_dir / "gt_overlay" / f"{img_path.stem}.jpg", quality=90
        )
        Image.fromarray(compare).save(
            out_dir / "compare" / f"{img_path.stem}.jpg", quality=90
        )

    iou, miou = iou_from_cm(cm)
    ranked = sorted(per_image, key=lambda x: x["miou"], reverse=True)
    picks = [ranked[0], ranked[len(ranked) // 2], ranked[-1]] if ranked else []
    report = {
        "weights": str(args.weights),
        "config": str(args.config),
        "mode": args.mode,
        "split": args.split,
        "image_size": image_size,
        "num": len(pairs),
        "miou": float(miou),
        "iou": {FRONT_NAMES[i]: float(iou[i]) for i in range(num_classes)},
        "ppt_picks_best_mid_worst": [p["stem"] for p in picks],
    }
    (out_dir / "metrics_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"{args.split} mIoU={float(miou):.4f}  n={len(pairs)}  -> {out_dir}")
    for i, n in enumerate(FRONT_NAMES):
        print(f"  {n}: {float(iou[i]):.4f}")
    print("PPT picks:", [p["stem"] for p in picks])


if __name__ == "__main__":
    main()
