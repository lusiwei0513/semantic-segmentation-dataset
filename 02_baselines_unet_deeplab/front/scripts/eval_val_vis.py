"""Evaluate best checkpoint on val split and export overlays for reporting."""

from __future__ import annotations

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

from src.dataset import SegDataset, list_pairs, split_pairs, build_transforms  # noqa: E402
from src.metrics import confusion_matrix, iou_from_cm  # noqa: E402

CLASS_NAMES = ["background", "body", "windshield", "nose_tip"]
PALETTE = np.array(
    [
        [0, 0, 0],
        [0, 120, 255],
        [0, 220, 120],
        [255, 60, 60],
    ],
    dtype=np.uint8,
)


def colorize(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, c in enumerate(PALETTE):
        out[mask == i] = c
    return out


def overlay_rgb(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = colorize(mask).astype(np.float32)
    base = rgb.astype(np.float32)
    return np.clip(base * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8)


def build_model(cfg: dict):
    import segmentation_models_pytorch as smp

    m = cfg["model"]
    return smp.Unet(
        encoder_name=m["encoder"],
        encoder_weights=None,
        in_channels=3,
        classes=m["num_classes"],
    )


def predict_full(model, rgb: np.ndarray, image_size: int, device) -> np.ndarray:
    h0, w0 = rgb.shape[:2]
    tf = build_transforms(image_size, train=False)
    t = tf(image=rgb, mask=np.zeros((h0, w0), dtype=np.uint8))
    tensor = t["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor).argmax(1)[0].cpu().numpy().astype(np.uint8)
    return cv2.resize(pred, (w0, h0), interpolation=cv2.INTER_NEAREST)


def main() -> None:
    run_dir = ROOT / "outputs" / "unet_resnet34_front_gpt189"
    weights = run_dir / "best.pt"
    cfg_path = ROOT / "config_front_gpt.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ckpt = torch.load(weights, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(ckpt.get("cfg", cfg))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    prepared = (ROOT / cfg["data"]["prepared_dir"]).resolve()
    pairs = list_pairs(prepared)
    train_pairs, val_pairs = split_pairs(
        pairs, val_ratio=float(cfg["data"]["val_ratio"]), seed=int(cfg["data"]["seed"])
    )
    image_size = int(cfg["data"]["image_size"])
    num_classes = int(cfg["model"]["num_classes"])

    out_dir = run_dir / "val_vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pred_overlay").mkdir(exist_ok=True)
    (out_dir / "gt_overlay").mkdir(exist_ok=True)
    (out_dir / "compare").mkdir(exist_ok=True)
    (out_dir / "pred_mask").mkdir(exist_ok=True)

    cm = torch.zeros(num_classes, num_classes, device=device)
    per_image = []

    for img_path, mask_path in tqdm(val_pairs, desc="val_vis"):
        with Image.open(img_path) as im:
            rgb = np.array(im.convert("RGB"))
        with Image.open(mask_path) as im:
            gt = np.array(im.convert("L"))
        pred = predict_full(model, rgb, image_size, device)

        # metrics at original resolution
        pred_t = torch.from_numpy(pred).to(device)
        gt_t = torch.from_numpy(gt.astype(np.int64)).to(device)
        cm += confusion_matrix(pred_t.unsqueeze(0), gt_t.unsqueeze(0), num_classes)
        iou_i, miou_i = iou_from_cm(
            confusion_matrix(pred_t.unsqueeze(0), gt_t.unsqueeze(0), num_classes)
        )
        per_image.append(
            {
                "stem": img_path.stem,
                "miou": float(miou_i),
                "iou": {CLASS_NAMES[i]: float(iou_i[i]) for i in range(num_classes)},
            }
        )

        pred_ov = overlay_rgb(rgb, pred)
        gt_ov = overlay_rgb(rgb, gt)
        # side-by-side compare: GT | Pred
        gap = np.ones((rgb.shape[0], 8, 3), dtype=np.uint8) * 255
        compare = np.concatenate([gt_ov, gap, pred_ov], axis=1)

        Image.fromarray(pred).save(out_dir / "pred_mask" / f"{img_path.stem}.png")
        Image.fromarray(pred_ov).save(out_dir / "pred_overlay" / f"{img_path.stem}.jpg")
        Image.fromarray(gt_ov).save(out_dir / "gt_overlay" / f"{img_path.stem}.jpg")
        Image.fromarray(compare).save(out_dir / "compare" / f"{img_path.stem}.jpg")

    iou, miou = iou_from_cm(cm)
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    best = max(history, key=lambda r: r["val_miou"])

    report = {
        "dataset": str(prepared),
        "num_train": len(train_pairs),
        "num_val": len(val_pairs),
        "weights": str(weights),
        "best_epoch_from_history": best["epoch"],
        "best_val_miou_from_history": best["val_miou"],
        "best_val_iou_from_history": best["val_iou"],
        "val_miou_fullres_recomputed": float(miou),
        "val_iou_fullres_recomputed": {
            CLASS_NAMES[i]: float(iou[i]) for i in range(num_classes)
        },
        "early_stop_epoch": history[-1]["epoch"],
        "per_image": sorted(per_image, key=lambda x: x["miou"]),
    }
    (out_dir / "metrics_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # training curve
    try:
        import matplotlib.pyplot as plt

        epochs = [r["epoch"] for r in history]
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(epochs, [r["train_miou"] for r in history], label="train")
        ax[0].plot(epochs, [r["val_miou"] for r in history], label="val")
        ax[0].axvline(best["epoch"], color="r", ls="--", label=f"best ep{best['epoch']}")
        ax[0].set_title("mIoU")
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)
        ax[1].plot(epochs, [r["train_loss"] for r in history], label="train")
        ax[1].plot(epochs, [r["val_loss"] for r in history], label="val")
        ax[1].set_title("Loss")
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "training_curves.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print("curve plot skipped:", e)

    # pick top/mid/low examples for a contact sheet
    ranked = sorted(per_image, key=lambda x: x["miou"], reverse=True)
    picks = []
    if ranked:
        picks = [ranked[0], ranked[len(ranked) // 2], ranked[-1]]
        # unique
        seen = set()
        uniq = []
        for p in picks:
            if p["stem"] not in seen:
                uniq.append(p)
                seen.add(p["stem"])
        picks = uniq

    print("=" * 60)
    print(f"best history mIoU={best['val_miou']:.4f} @ epoch {best['epoch']}")
    print(f"fullres val mIoU={float(miou):.4f}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {float(iou[i]):.4f}")
    print(f"overlays -> {out_dir}")
    print("example picks:", [p["stem"] for p in picks])


if __name__ == "__main__":
    main()
