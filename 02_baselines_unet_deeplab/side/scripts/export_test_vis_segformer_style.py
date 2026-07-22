"""Export fold test-set as SegFormer-style 1x2: input | pred overlay (matplotlib).

Side UNet-seg (argmax exclusive classes). Does NOT touch the existing 3-panel test_vis/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from src.dataset import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_transforms,
    load_fold_pairs,
)

SizeLike = Union[int, Sequence[int]]

# Match SegFormer visualization.py (body / windshield / bogie / door)
SEGFORMER_COLORS = np.array(
    [
        [0.2, 0.6, 1.0],  # body
        [1.0, 0.85, 0.2],  # windshield
        [0.2, 0.9, 0.4],  # bogie
        [1.0, 0.3, 0.3],  # door
    ],
    dtype=np.float32,
)


def denormalize(image_chw: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    x = image_chw.detach().cpu().float().numpy()
    mean_a = np.array(mean)[:, None, None]
    std_a = np.array(std)[:, None, None]
    x = x * std_a + mean_a
    return np.clip(x.transpose(1, 2, 0), 0, 1)


def save_input_pred(
    img: np.ndarray,
    pred_mask: np.ndarray,
    out_path: Path,
    title: str = "pred overlay",
) -> None:
    """img float HxWx3 in [0,1]; pred_mask uint8 with class ids (0=bg)."""
    overlay = img.copy()
    for cls_id, color in enumerate(SEGFORMER_COLORS, start=1):
        m = pred_mask == cls_id
        if m.any():
            overlay[m] = overlay[m] * 0.55 + color * 0.45

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img)
    axes[0].set_title("input")
    axes[0].axis("off")
    axes[1].imshow(np.clip(overlay, 0, 1))
    axes[1].set_title(title)
    axes[1].axis("off")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def resolve(base: Path, p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (base / path).resolve()


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

    out_dir = args.out_dir or (
        args.weights.parent / f"{args.split}_vis_segformer_style"
    )
    compare_dir = out_dir / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)

    image_size = cfg["data"]["image_size"]
    if "image_size" in run_cfg.get("data", {}):
        image_size = run_cfg["data"]["image_size"]
    tf = build_transforms(image_size, train=False)

    stems: list[str] = []
    for img_path, _mask_path in tqdm(pairs, desc=f"{args.split}_vis_sf_side"):
        rgb = np.array(Image.open(img_path).convert("RGB"))
        h0, w0 = rgb.shape[:2]
        t = tf(image=rgb, mask=np.zeros((h0, w0), dtype=np.uint8))
        image_t = t["image"]
        with torch.no_grad():
            pred = (
                model(image_t.unsqueeze(0).to(device))
                .argmax(1)[0]
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
        img = denormalize(image_t)
        out_path = compare_dir / f"{img_path.stem}.png"
        save_input_pred(img, pred, out_path, title="pred overlay")
        stems.append(img_path.stem)

    report = {
        "style": "segformer_1x2_input_pred",
        "view": "side",
        "model": "unet_seg_wide",
        "weights": str(args.weights),
        "config": str(args.config),
        "split": args.split,
        "fold_json": str(fold_json),
        "image_size": image_size,
        "n": len(stems),
        "out_dir": str(out_dir),
        "stems": stems,
    }
    (out_dir / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"OK n={len(stems)} -> {compare_dir}")
    print("examples:", stems[:3])


if __name__ == "__main__":
    main()
