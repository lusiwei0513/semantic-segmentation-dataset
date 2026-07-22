"""Export val-set overlays for PPT/report (front 4-class seg OR front KP)."""

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

from src.dataset import build_transforms, list_pairs, split_pairs  # noqa: E402
from src.metrics import confusion_matrix, iou_from_cm  # noqa: E402

FRONT_NAMES = ["background", "body", "windshield", "nose_tip"]
FRONT_PALETTE = np.array(
    [[0, 0, 0], [0, 120, 255], [0, 220, 120], [255, 60, 60]], dtype=np.uint8
)


def colorize(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, c in enumerate(palette):
        out[mask == i] = c
    return out


def overlay_rgb(rgb: np.ndarray, mask: np.ndarray, palette: np.ndarray, alpha=0.45):
    color = colorize(mask, palette).astype(np.float32)
    base = rgb.astype(np.float32)
    return np.clip(base * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8)


def resolve(base: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (base / path).resolve()


def predict_seg(model, rgb, image_size, device):
    h0, w0 = rgb.shape[:2]
    tf = build_transforms(image_size, train=False)
    t = tf(image=rgb, mask=np.zeros((h0, w0), dtype=np.uint8))
    with torch.no_grad():
        pred = model(t["image"].unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy().astype(np.uint8)
    return cv2.resize(pred, (w0, h0), interpolation=cv2.INTER_NEAREST)


def predict_kp(model, rgb, image_size, device, tip_radius: int):
    from src.keypoint import decode_heatmap_peak, merge_tip_circle_into_seg

    h0, w0 = rgb.shape[:2]
    tf = build_transforms(image_size, train=False)
    t = tf(image=rgb, mask=np.zeros((h0, w0), dtype=np.uint8))
    with torch.no_grad():
        seg_logits, heat = model(t["image"].unsqueeze(0).to(device))
        pred3 = seg_logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        px, py = decode_heatmap_peak(heat.sigmoid())
        x = float(px[0].item()) * w0 / image_size
        y = float(py[0].item()) * h0 / image_size
    pred3 = cv2.resize(pred3, (w0, h0), interpolation=cv2.INTER_NEAREST)
    # scale tip from letterboxed canvas is approximate; remap via same resize ratio
    # better: decode on padded canvas then map — for PPT this resize is acceptable
    pred = merge_tip_circle_into_seg(pred3, x, y, tip_class=3, radius=tip_radius)
    return pred, (x, y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--mode", choices=["seg", "kp"], default="seg")
    ap.add_argument("--max-images", type=int, default=0, help="0=all val")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "seg":
        import segmentation_models_pytorch as smp

        mcfg = ckpt.get("cfg", cfg)["model"]
        model = smp.Unet(
            encoder_name=mcfg["encoder"],
            encoder_weights=None,
            in_channels=3,
            classes=int(mcfg["num_classes"]),
        )
        model.load_state_dict(ckpt["model"])
        names = FRONT_NAMES
        tip_radius = 16
    else:
        from scripts.train_front_kp import SegTipUnet

        mcfg = ckpt.get("cfg", cfg)["model"]
        model = SegTipUnet(mcfg["encoder"], None, int(mcfg.get("seg_classes", 3)))
        model.load_state_dict(ckpt["model"])
        names = FRONT_NAMES
        tip_radius = int(cfg["data"].get("tip_radius_eval", 16))

    model.to(device).eval()
    prepared = resolve(ROOT, cfg["data"]["prepared_dir"])
    pairs = list_pairs(prepared)
    _, val_pairs = split_pairs(
        pairs, float(cfg["data"]["val_ratio"]), int(cfg["data"]["seed"])
    )
    if args.max_images > 0:
        val_pairs = val_pairs[: args.max_images]

    out_dir = args.out_dir or (args.weights.parent / "val_vis")
    for sub in ("pred_overlay", "gt_overlay", "compare", "pred_mask"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    image_size = int(cfg["data"]["image_size"])
    num_classes = 4
    cm = torch.zeros(num_classes, num_classes)
    per_image = []

    for img_path, mask_path in tqdm(val_pairs, desc="val_vis"):
        rgb = np.array(Image.open(img_path).convert("RGB"))
        gt = np.array(Image.open(mask_path).convert("L"))
        if args.mode == "seg":
            pred = predict_seg(model, rgb, image_size, device)
        else:
            pred, _ = predict_kp(model, rgb, image_size, device, tip_radius)

        cm += confusion_matrix(
            torch.from_numpy(pred), torch.from_numpy(gt.astype(np.int64)), num_classes
        )
        iou_i, miou_i = iou_from_cm(
            confusion_matrix(
                torch.from_numpy(pred), torch.from_numpy(gt.astype(np.int64)), num_classes
            )
        )
        per_image.append(
            {
                "stem": img_path.stem,
                "miou": float(miou_i),
                "iou": {names[i]: float(iou_i[i]) for i in range(num_classes)},
            }
        )

        pred_ov = overlay_rgb(rgb, pred, FRONT_PALETTE)
        gt_ov = overlay_rgb(rgb, gt, FRONT_PALETTE)
        gap = np.ones((rgb.shape[0], 8, 3), dtype=np.uint8) * 255
        # Original | GT | Pred
        compare = np.concatenate([rgb, gap, gt_ov, gap, pred_ov], axis=1)
        Image.fromarray(pred).save(out_dir / "pred_mask" / f"{img_path.stem}.png")
        Image.fromarray(pred_ov).save(out_dir / "pred_overlay" / f"{img_path.stem}.jpg", quality=92)
        Image.fromarray(gt_ov).save(out_dir / "gt_overlay" / f"{img_path.stem}.jpg", quality=92)
        Image.fromarray(compare).save(out_dir / "compare" / f"{img_path.stem}.jpg", quality=92)

    iou, miou = iou_from_cm(cm)
    ranked = sorted(per_image, key=lambda x: x["miou"], reverse=True)
    picks = [ranked[0], ranked[len(ranked) // 2], ranked[-1]] if ranked else []
    report = {
        "weights": str(args.weights),
        "mode": args.mode,
        "num_val": len(val_pairs),
        "val_miou": float(miou),
        "val_iou": {names[i]: float(iou[i]) for i in range(num_classes)},
        "ppt_picks_best_mid_worst": [p["stem"] for p in picks],
        "per_image": sorted(per_image, key=lambda x: x["miou"]),
    }
    (out_dir / "metrics_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"val mIoU={float(miou):.4f}")
    for i, n in enumerate(names):
        print(f"  {n}: {float(iou[i]):.4f}")
    print(f"overlays -> {out_dir}")
    print("PPT picks (best/mid/worst):", [p["stem"] for p in picks])


if __name__ == "__main__":
    main()
