"""Export fold test-set overlays for side UNet / DeepLab (HxW letterbox)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from src.dataset import build_transforms, load_fold_pairs, parse_image_size  # noqa: E402
from src.metrics import confusion_matrix, iou_from_cm  # noqa: E402

SizeLike = Union[int, Sequence[int]]

SIDE_NAMES = ["background", "body", "windshield", "bogie", "door"]
SIDE_PALETTE = np.array(
    [
        [0, 0, 0],
        [0, 120, 255],
        [0, 220, 120],
        [255, 180, 0],
        [200, 80, 255],
    ],
    dtype=np.uint8,
)


def colorize(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, c in enumerate(SIDE_PALETTE):
        out[mask == i] = c
    return out


def overlay_rgb(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = colorize(mask).astype(np.float32)
    base = rgb.astype(np.float32)
    return np.clip(base * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8)


def resolve(base: Path, p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (base / path).resolve()


def letterbox_meta_hw(h0: int, w0: int, image_size: SizeLike) -> dict:
    th, tw = parse_image_size(image_size)
    scale = min(tw / float(w0), th / float(h0))
    nh = max(1, int(round(h0 * scale)))
    nw = max(1, int(round(w0 * scale)))
    pad_h = th - nh
    pad_w = tw - nw
    top = pad_h // 2
    left = pad_w // 2
    return {
        "h0": h0,
        "w0": w0,
        "th": th,
        "tw": tw,
        "scale": scale,
        "nh": nh,
        "nw": nw,
        "top": top,
        "left": left,
    }


def unletterbox_mask(mask_canvas: np.ndarray, meta: dict) -> np.ndarray:
    top, left = meta["top"], meta["left"]
    nh, nw = meta["nh"], meta["nw"]
    crop = mask_canvas[top : top + nh, left : left + nw]
    return cv2.resize(crop, (meta["w0"], meta["h0"]), interpolation=cv2.INTER_NEAREST)


def build_model(cfg: dict):
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

    try:
        return _make(None)
    except Exception:
        return _make(None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--max-images", type=int, default=0)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    run_cfg = ckpt.get("cfg", cfg)
    model = build_model(run_cfg)
    model.load_state_dict(ckpt["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    prepared = resolve(ROOT, cfg["data"]["prepared_dir"])
    fold_json = resolve(ROOT, cfg["data"]["fold_json"])
    pairs = load_fold_pairs(prepared, fold_json, split=args.split)
    if args.max_images > 0:
        pairs = pairs[: args.max_images]

    out_dir = args.out_dir or (args.weights.parent / f"{args.split}_vis")
    for sub in ("pred_overlay", "gt_overlay", "compare", "pred_mask"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    image_size = cfg["data"]["image_size"]
    if "image_size" in run_cfg.get("data", {}):
        image_size = run_cfg["data"]["image_size"]
    tf = build_transforms(image_size, train=False)
    num_classes = int(run_cfg["model"]["num_classes"])
    cm = torch.zeros(num_classes, num_classes)
    per_image = []

    for img_path, mask_path in tqdm(pairs, desc=f"{args.split}_vis_side"):
        rgb = np.array(Image.open(img_path).convert("RGB"))
        gt = np.array(Image.open(mask_path).convert("L"))
        h0, w0 = rgb.shape[:2]
        meta = letterbox_meta_hw(h0, w0, image_size)
        t = tf(image=rgb, mask=np.zeros((h0, w0), dtype=np.uint8))
        with torch.no_grad():
            pred_sq = (
                model(t["image"].unsqueeze(0).to(device))
                .argmax(1)[0]
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
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
                "iou": {SIDE_NAMES[i]: float(iou_i[i]) for i in range(num_classes)},
            }
        )

        pred_ov = overlay_rgb(rgb, pred)
        gt_ov = overlay_rgb(rgb, gt)
        # Side images are very wide; downscale viz for PPT (pred_mask stays full-res)
        vis_h = 720
        if rgb.shape[0] > vis_h:
            s = vis_h / float(rgb.shape[0])
            nw = max(1, int(round(rgb.shape[1] * s)))
            rgb_v = cv2.resize(rgb, (nw, vis_h), interpolation=cv2.INTER_AREA)
            gt_v = cv2.resize(gt_ov, (nw, vis_h), interpolation=cv2.INTER_AREA)
            pred_v = cv2.resize(pred_ov, (nw, vis_h), interpolation=cv2.INTER_AREA)
        else:
            rgb_v, gt_v, pred_v = rgb, gt_ov, pred_ov
        gap = np.ones((rgb_v.shape[0], 8, 3), dtype=np.uint8) * 255
        compare = np.concatenate([rgb_v, gap, gt_v, gap, pred_v], axis=1)
        Image.fromarray(pred).save(out_dir / "pred_mask" / f"{img_path.stem}.png")
        Image.fromarray(pred_v).save(
            out_dir / "pred_overlay" / f"{img_path.stem}.jpg", quality=90
        )
        Image.fromarray(gt_v).save(
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
        "split": args.split,
        "image_size": image_size,
        "num": len(pairs),
        "miou": float(miou),
        "iou": {SIDE_NAMES[i]: float(iou[i]) for i in range(num_classes)},
        "ppt_picks_best_mid_worst": [p["stem"] for p in picks],
    }
    (out_dir / "metrics_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"{args.split} mIoU={float(miou):.4f}  n={len(pairs)}  -> {out_dir}")
    for i, n in enumerate(SIDE_NAMES[:num_classes]):
        print(f"  {n}: {float(iou[i]):.4f}")
    print("PPT picks:", [p["stem"] for p in picks])


if __name__ == "__main__":
    main()
