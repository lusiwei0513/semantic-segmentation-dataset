"""Inference / simple deployment helper for trained U-Net."""

from __future__ import annotations

import argparse
import json
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

from src.dataset import IMAGENET_MEAN, IMAGENET_STD, build_transforms  # noqa: E402

CLASS_NAMES = ["background", "body", "windshield", "nose_tip"]
PALETTE = np.array(
    [
        [0, 0, 0],        # background
        [0, 120, 255],    # body — blue
        [0, 220, 120],    # windshield — green
        [255, 60, 60],    # nose_tip — red
    ],
    dtype=np.uint8,
)


def build_model(cfg: dict):
    import segmentation_models_pytorch as smp

    m = cfg["model"]
    return smp.Unet(
        encoder_name=m["encoder"],
        encoder_weights=None,
        in_channels=3,
        classes=m["num_classes"],
    )


def colorize(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(PALETTE):
        out[mask == i] = color
    return out


def predict_one(model, image_bgr: np.ndarray, image_size: int, device) -> np.ndarray:
    h0, w0 = image_bgr.shape[:2]
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tf = build_transforms(image_size, train=False)
    t = tf(image=image, mask=np.zeros((h0, w0), dtype=np.uint8))
    tensor = t["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)

    # Undo letterbox pad: map back roughly via resize to original
    # (PadIfNeeded pads bottom/right in albumentations by default)
    # Safer: resize prediction to original size directly for deployment simplicity.
    pred_resized = cv2.resize(pred, (w0, h0), interpolation=cv2.INTER_NEAREST)
    return pred_resized


def overlay(image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = colorize(mask)
    color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(image_bgr, 1 - alpha, color_bgr, alpha, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="U-Net inference")
    parser.add_argument("--weights", type=Path, required=True, help="best.pt path")
    parser.add_argument("--input", type=Path, required=True, help="image file or directory")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=None)
    args = parser.parse_args()

    ckpt = torch.load(args.weights, map_location="cpu")
    cfg = ckpt.get("cfg")
    if cfg is None:
        with open(ROOT / "config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    image_size = args.image_size or int(cfg["data"]["image_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    if args.input.is_dir():
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            paths.extend(args.input.glob(ext))
        paths = sorted(paths)
    else:
        paths = [args.input]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.out_dir / "masks"
    vis_dir = args.out_dir / "overlay"
    mask_dir.mkdir(exist_ok=True)
    vis_dir.mkdir(exist_ok=True)

    for p in tqdm(paths, desc="infer"):
        try:
            with Image.open(p) as im:
                rgb = np.array(im.convert("RGB"))
        except OSError:
            print(f"[skip] {p}")
            continue
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        pred = predict_one(model, img, image_size, device)
        Image.fromarray(pred, mode="L").save(mask_dir / f"{p.stem}.png")
        vis = overlay(img, pred)
        Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)).save(vis_dir / f"{p.stem}.jpg")

    meta = {
        "weights": str(args.weights),
        "num_images": len(paths),
        "class_names": CLASS_NAMES,
        "palette_rgb": PALETTE.tolist(),
    }
    with open(args.out_dir / "infer_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Wrote masks/overlays -> {args.out_dir}")


if __name__ == "__main__":
    main()
