"""Auto-label all front-view images for the PPT task (separate from seg_train).

- Uses manual LabelMe JSON when available (gold).
- Otherwise uses U-Net checkpoint + geometric nose_tip refinement.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT / "seg_train"))

LABEL_TO_ID = {"background": 0, "body": 1, "windshield": 2, "nose_tip": 3}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
PALETTE = np.array([[0, 0, 0], [255, 80, 120], [0, 220, 120], [40, 40, 40]], dtype=np.uint8)


def imread_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(mask.astype(np.uint8), mode="L").save(path)


def overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = np.zeros_like(rgb)
    for i, c in enumerate(PALETTE):
        color[mask == i] = c
    return (rgb * (1 - alpha) + color * alpha).astype(np.uint8)


def json_to_mask(data: dict, nose_radius: int) -> np.ndarray:
    h, w = int(data["imageHeight"]), int(data["imageWidth"])
    mask = np.zeros((h, w), dtype=np.uint8)
    priority = {"body": 0, "windshield": 1, "nose_tip": 2}
    shapes = sorted(data.get("shapes", []), key=lambda s: priority.get(s.get("label", ""), 99))
    for shape in shapes:
        label = shape.get("label")
        if label not in LABEL_TO_ID or label == "background":
            continue
        cid = LABEL_TO_ID[label]
        st = shape.get("shape_type", "polygon")
        pts = shape.get("points") or []
        if st == "polygon" and len(pts) >= 3:
            cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], int(cid))
        elif st == "rectangle" and len(pts) >= 2:
            (x1, y1), (x2, y2) = pts[0], pts[1]
            cv2.rectangle(
                mask,
                (int(min(x1, x2)), int(min(y1, y2))),
                (int(max(x1, x2)), int(max(y1, y2))),
                int(cid),
                -1,
            )
        elif st == "point" and pts:
            x, y = pts[0]
            cv2.circle(mask, (int(round(x)), int(round(y))), nose_radius, int(cid), -1)
    return mask


def refine_nose_tip(mask: np.ndarray, radius: int) -> np.ndarray:
    """Place nose_tip disk at lowest center of body silhouette (PPT-style tip)."""
    out = mask.copy()
    out[out == 3] = 1  # clear old tip into body first if overlapping body region
    body = ((mask == 1) | (mask == 2) | (mask == 3)).astype(np.uint8)
    if body.sum() == 0:
        return mask
    ys, xs = np.where(body > 0)
    y_max = int(ys.max())
    band = body.copy()
    band[: max(0, y_max - max(20, (ys.max() - ys.min()) // 8)), :] = 0
    bys, bxs = np.where(band > 0)
    if len(bxs) == 0:
        cx, cy = int(xs.mean()), y_max
    else:
        cx, cy = int(np.median(bxs)), int(bys.max())
    # restore body/windshield from original (except tip disk)
    out = mask.copy()
    out[out == 3] = 0
    # tip should sit on body: paint tip last
    cv2.circle(out, (cx, cy), radius, 3, -1)
    return out


def mask_to_labelme(mask: np.ndarray, image_path: str, image_h: int, image_w: int) -> dict:
    shapes = []
    for cid, name in ID_TO_LABEL.items():
        if cid == 0:
            continue
        bin_m = (mask == cid).astype(np.uint8)
        if bin_m.sum() == 0:
            continue
        contours, _ = cv2.findContours(bin_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 8:
                continue
            if name == "nose_tip":
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
                    shapes.append(
                        {
                            "label": name,
                            "points": [[float(cx), float(cy)]],
                            "group_id": None,
                            "shape_type": "point",
                            "flags": {},
                        }
                    )
            else:
                pts = cnt.squeeze(1)
                if pts.ndim != 2 or len(pts) < 3:
                    continue
                # simplify
                eps = 0.002 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, eps, True).squeeze(1)
                if approx.ndim != 2 or len(approx) < 3:
                    approx = pts
                shapes.append(
                    {
                        "label": name,
                        "points": [[float(x), float(y)] for x, y in approx],
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {},
                    }
                )
    return {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path,
        "imageData": None,
        "imageHeight": image_h,
        "imageWidth": image_w,
    }


def build_model(cfg: dict):
    import os

    import segmentation_models_pytorch as smp

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    m = cfg["model"]
    try:
        return smp.Unet(
            encoder_name=m["encoder"],
            encoder_weights=None,
            in_channels=3,
            classes=m["num_classes"],
        )
    except Exception:
        return smp.Unet(
            encoder_name=m["encoder"],
            encoder_weights=None,
            in_channels=3,
            classes=m["num_classes"],
        )


def predict_mask(model, rgb: np.ndarray, image_size: int, device) -> np.ndarray:
    from src.dataset import build_transforms  # ppt_seg_task src after we create it

    h0, w0 = rgb.shape[:2]
    tf = build_transforms(image_size, train=False, mode="front")
    t = tf(image=rgb, mask=np.zeros((h0, w0), dtype=np.uint8))
    tensor = t["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor).argmax(1)[0].cpu().numpy().astype(np.uint8)
    return cv2.resize(pred, (w0, h0), interpolation=cv2.INTER_NEAREST)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--front-dir", type=Path, default=PROJECT / "front_images")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data_front")
    parser.add_argument("--weights", type=Path, default=PROJECT / "seg_train/outputs/unet_resnet34/best.pt")
    parser.add_argument("--nose-radius", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=512)
    args = parser.parse_args()

    out = args.out_dir
    for sub in ("images", "masks", "overlays", "labelme_json"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    imgs = sorted(
        [p for p in args.front_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    json_map = {p.stem: p for p in args.front_dir.glob("*.json")}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    cfg = None
    if args.weights.exists():
        ckpt = torch.load(args.weights, map_location="cpu")
        cfg = ckpt.get("cfg") or {
            "model": {"encoder": "resnet34", "num_classes": 4, "encoder_weights": None}
        }
        # import transforms from local src
        sys.path.insert(0, str(ROOT))
        model = build_model(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(f"Loaded weights: {args.weights} on {device}")
    else:
        print(f"[warn] weights missing: {args.weights}; unlabeled images will be skipped")

    stats = {"gold": 0, "pseudo": 0, "skip": 0}
    for img_path in tqdm(imgs, desc="auto_label_front"):
        stem = img_path.stem
        # unify stem matching: json stems equal image stems in this dataset
        rgb = imread_rgb(img_path)
        h, w = rgb.shape[:2]
        source = None
        mask = None

        jf = json_map.get(stem)
        if jf is not None:
            data = json.loads(jf.read_text(encoding="utf-8"))
            mask = json_to_mask(data, args.nose_radius)
            mask = refine_nose_tip(mask, args.nose_radius)
            source = "gold"
            stats["gold"] += 1
        elif model is not None:
            mask = predict_mask(model, rgb, args.image_size, device)
            mask = refine_nose_tip(mask, args.nose_radius)
            source = "pseudo"
            stats["pseudo"] += 1
        else:
            stats["skip"] += 1
            continue

        # save with stable names
        ext = img_path.suffix.lower()
        out_img = out / "images" / f"{stem}{ext}"
        if not out_img.exists():
            shutil.copy2(img_path, out_img)
        save_mask(out / "masks" / f"{stem}.png", mask)
        Image.fromarray(overlay(rgb, mask)).save(out / "overlays" / f"{stem}.jpg")
        lm = mask_to_labelme(mask, out_img.name, h, w)
        lm["flags"]["source"] = source
        (out / "labelme_json" / f"{stem}.json").write_text(
            json.dumps(lm, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    summary = {"num_images": len(imgs), **stats, "label_to_id": LABEL_TO_ID}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
