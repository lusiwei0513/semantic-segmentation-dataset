"""Export fold test-set as SegFormer-style 1x2: input | pred overlay (matplotlib).

Front UNet-KP: argmax seg (body/windshield) + cyan tip marker on letterbox canvas.
Does NOT touch the existing 3-panel test_vis/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from src.dataset import IMAGENET_MEAN, IMAGENET_STD, build_transforms, load_fold_pairs  # noqa: E402
from src.keypoint import decode_heatmap_peak  # noqa: E402

# body / windshield (SegFormer palette); tip drawn as cyan marker
SEGFORMER_COLORS = np.array(
    [
        [0.2, 0.6, 1.0],  # body
        [1.0, 0.85, 0.2],  # windshield
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
    tip_xy: tuple[float, float] | None = None,
    title: str = "pred overlay",
) -> None:
    """img float HxWx3; pred_mask class ids 0=bg,1=body,2=windshield (+optional tip)."""
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
    if tip_xy is not None:
        axes[1].scatter([tip_xy[0]], [tip_xy[1]], c="cyan", marker="x", s=50)
    axes[1].axis("off")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def resolve(base: Path, p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (base / path).resolve()


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    out_dir = args.out_dir or (
        args.weights.parent / f"{args.split}_vis_segformer_style"
    )
    compare_dir = out_dir / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)

    image_size = int(run_cfg["data"].get("image_size", cfg["data"]["image_size"]))
    tf = build_transforms(image_size, train=False)

    stems: list[str] = []
    for img_path, _mask_path in tqdm(pairs, desc=f"{args.split}_vis_sf_front"):
        rgb = np.array(Image.open(img_path).convert("RGB"))
        h0, w0 = rgb.shape[:2]
        t = tf(image=rgb, mask=np.zeros((h0, w0), dtype=np.uint8))
        image_t = t["image"]
        with torch.no_grad():
            seg_logits, heat = model(image_t.unsqueeze(0).to(device))
            pred = seg_logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
            px, py = decode_heatmap_peak(heat.sigmoid())
            x_c = float(px[0].item())
            y_c = float(py[0].item())
            hh, ww = heat.shape[-2], heat.shape[-1]
            # map heatmap peak onto letterbox canvas (same as SegFormer viz)
            tip_xy = (
                x_c * (image_size / max(ww, 1)),
                y_c * (image_size / max(hh, 1)),
            )

        img = denormalize(image_t)
        out_path = compare_dir / f"{img_path.stem}.png"
        save_input_pred(img, pred, out_path, tip_xy=tip_xy, title="pred overlay")
        stems.append(img_path.stem)

    report = {
        "style": "segformer_1x2_input_pred",
        "view": "front",
        "model": "unet_kp",
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
