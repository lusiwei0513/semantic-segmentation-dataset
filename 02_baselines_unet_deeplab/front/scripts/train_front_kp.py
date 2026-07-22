"""Train UNet: semantic head + tip Gaussian heatmap; eval via peak->ellipse mIoU."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from src.dataset_kp import (  # noqa: E402
    SegTipDataset,
    expand_joint_pred,
    list_pairs_with_tip,
    load_fold_items,
    split_pairs,
)
from src.keypoint import (  # noqa: E402
    HeatmapMSELoss,
    decode_heatmap_peak,
    merge_tip_ellipse_into_seg,
    tip_distance_metrics,
)
from src.metrics import CEDiceLoss, confusion_matrix, iou_from_cm  # noqa: E402


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


def resolve_path(base: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (base / path).resolve()


class SegTipUnet(nn.Module):
    """UNet with C_seg logits + 1 heatmap channel."""

    def __init__(self, encoder: str, weights: str | None, seg_classes: int):
        super().__init__()
        import segmentation_models_pytorch as smp

        try:
            self.net = smp.Unet(
                encoder_name=encoder,
                encoder_weights=weights,
                in_channels=3,
                classes=seg_classes + 1,
            )
        except Exception as e:
            print(f"[warn] pretrained failed: {e}; random init")
            self.net = smp.Unet(
                encoder_name=encoder,
                encoder_weights=None,
                in_channels=3,
                classes=seg_classes + 1,
            )
        self.seg_classes = seg_classes
        # Heat channel bias <- negative so sigmoid starts near 0 (not ~0.5 flat).
        self._init_heat_bias(-4.0)

    def _init_heat_bias(self, value: float) -> None:
        head = getattr(self.net, "segmentation_head", None)
        if head is None:
            return
        for m in head.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels == self.seg_classes + 1:
                if m.bias is not None:
                    with torch.no_grad():
                        m.bias[self.seg_classes :].fill_(value)
                break

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        return out[:, : self.seg_classes], out[:, self.seg_classes : self.seg_classes + 1]


@torch.no_grad()
def eval_official_miou(
    model: SegTipUnet,
    loader: DataLoader,
    device: torch.device,
    tip_rx: int,
    tip_ry: int,
    mode: str,
    full_names: list[str],
) -> tuple[float, dict, dict]:
    """Peak -> fixed ellipse merge, then multiclass mIoU + tip MAE/PCK."""
    model.eval()
    n_cls = len(full_names)
    cm = torch.zeros(n_cls, n_cls, device=device)
    tip_stats = []
    for batch in tqdm(loader, leave=False, desc="val_official"):
        images = batch["image"].to(device)
        masks = batch["mask"].cpu().numpy()
        tip_xy = batch["tip_xy"].cpu().numpy()
        has_tip = batch["has_tip"].cpu().numpy()

        seg_logits, heat = model(images)
        pred_seg = seg_logits.argmax(1).cpu().numpy()
        px, py = decode_heatmap_peak(heat.sigmoid())
        px = px.cpu().numpy()
        py = py.cpu().numpy()

        for i in range(images.size(0)):
            if mode == "joint":
                gt = expand_joint_pred(masks[i].copy())
                pred = expand_joint_pred(pred_seg[i].copy())
            else:
                gt = masks[i].copy()
                pred = pred_seg[i].copy()

            if has_tip[i] > 0.5:
                gt = merge_tip_ellipse_into_seg(
                    gt, float(tip_xy[i, 0]), float(tip_xy[i, 1]), 3, tip_rx, tip_ry
                )
            pred = merge_tip_ellipse_into_seg(
                pred, float(px[i]), float(py[i]), 3, tip_rx, tip_ry
            )
            cm += confusion_matrix(
                torch.from_numpy(pred).to(device),
                torch.from_numpy(gt).to(device),
                n_cls,
            )
            if has_tip[i] > 0.5:
                tip_stats.append(
                    {
                        "pred": (float(px[i]), float(py[i])),
                        "gt": (float(tip_xy[i, 0]), float(tip_xy[i, 1])),
                        "h": float(batch["image_h"][i]),
                    }
                )

    iou, miou = iou_from_cm(cm)
    iou_dict = {full_names[i]: float(iou[i]) for i in range(n_cls)}

    if tip_stats:
        pred_xy = torch.tensor([t["pred"] for t in tip_stats], dtype=torch.float32)
        gt_xy = torch.tensor([t["gt"] for t in tip_stats], dtype=torch.float32)
        valid = torch.ones(len(tip_stats), dtype=torch.bool)
        hs = torch.tensor([t["h"] for t in tip_stats], dtype=torch.float32)
        tip_m = tip_distance_metrics(pred_xy, gt_xy, valid, hs)
    else:
        tip_m = tip_distance_metrics(
            torch.zeros(0, 2),
            torch.zeros(0, 2),
            torch.zeros(0, dtype=torch.bool),
            torch.zeros(0),
        )
    return miou, iou_dict, tip_m


def run_epoch(model, loader, seg_crit, heat_crit, optimizer, device, scaler, train: bool):
    model.train(train)
    total = 0.0
    n = 0
    tip_acc = []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, leave=False, desc="train" if train else "val"):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            heat = batch["heat"].to(device, non_blocking=True)
            has_tip = batch["has_tip"].to(device, non_blocking=True).bool()
            tip_xy = batch["tip_xy"].to(device, non_blocking=True)
            image_h = batch["image_h"].to(device, non_blocking=True)

            with autocast(enabled=scaler is not None and device.type == "cuda"):
                seg_logits, heat_logits = model(images)
                heat_pred = torch.sigmoid(heat_logits)
                loss_seg = seg_crit(seg_logits, masks)
                # BCE-with-logits on heat channel (not sigmoid+MSE)
                loss_heat = heat_crit(heat_logits.float(), heat.float(), has_tip, tip_xy)
                loss = loss_seg + loss_heat

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            bs = images.size(0)
            total += loss.item() * bs
            n += bs

            with torch.no_grad():
                px, py = decode_heatmap_peak(heat_pred)
                pred_xy = torch.stack([px, py], dim=1)
                tip_acc.append(tip_distance_metrics(pred_xy, tip_xy, has_tip, image_h))

    keys = [k for k in tip_acc[0].keys() if k != "tip_n"] if tip_acc else []
    tip_mean = {}
    for k in keys:
        vals = [d[k] for d in tip_acc if d.get("tip_n", 0) > 0 and np.isfinite(d[k])]
        tip_mean[k] = float(np.mean(vals)) if vals else float("nan")
    tip_mean["tip_n"] = int(sum(d.get("tip_n", 0) for d in tip_acc))
    return total / max(n, 1), tip_mean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "config_front_kp.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    prepared = resolve_path(ROOT, cfg["data"]["prepared_dir"])
    out_dir = resolve_path(ROOT, cfg["paths"]["output_dir"]) / cfg["paths"]["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = str(cfg["data"].get("mode", "front"))
    keypoints_dir = None
    if cfg["data"].get("keypoints_dir"):
        keypoints_dir = resolve_path(ROOT, cfg["data"]["keypoints_dir"])
    labelme_dir = None
    if cfg["data"].get("labelme_dir"):
        labelme_dir = resolve_path(ROOT, cfg["data"]["labelme_dir"])

    items = list_pairs_with_tip(prepared, keypoints_dir=keypoints_dir, labelme_dir=labelme_dir)
    fold_json = cfg["data"].get("fold_json")
    if fold_json:
        fold_path = resolve_path(ROOT, fold_json)
        train_items = load_fold_items(items, fold_path, "train")
        val_items = load_fold_items(items, fold_path, "val")
        test_items = load_fold_items(items, fold_path, "test")
    else:
        train_items, val_items = split_pairs(items, cfg["data"]["val_ratio"], cfg["data"]["seed"])
        test_items = []

    print(
        f"mode={mode} train={len(train_items)} val={len(val_items)} "
        f"test={len(test_items)} prepared={prepared}"
    )

    image_size = int(cfg["data"]["image_size"])
    sigma = float(cfg["data"]["tip_sigma"])
    tip_rx = int(cfg["data"].get("tip_ellipse_rx", 24))
    tip_ry = int(cfg["data"].get("tip_ellipse_ry", 12))
    strong_aug = bool(cfg["train"].get("strong_aug", True))
    full_names = list(cfg["data"]["class_names"])
    seg_classes = int(cfg["model"]["seg_classes"])

    train_loader = DataLoader(
        SegTipDataset(train_items, image_size, True, sigma, strong_aug, mode),
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        SegTipDataset(val_items, image_size, False, sigma, False, mode),
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegTipUnet(
        cfg["model"]["encoder"],
        cfg["model"].get("encoder_weights", "imagenet"),
        seg_classes,
    ).to(device)
    print(
        f"device={device} encoder={cfg['model']['encoder']} "
        f"seg_classes={seg_classes} sigma={sigma} tip_ellipse={tip_rx}x{tip_ry}"
    )

    seg_crit = CEDiceLoss(
        num_classes=seg_classes,
        class_weights=cfg["train"].get("class_weights"),
        ce_weight=float(cfg["train"].get("ce_weight", 1.0)),
        dice_weight=float(cfg["train"].get("dice_weight", 1.0)),
    )
    heat_crit = HeatmapMSELoss(
        weight=float(cfg["train"].get("heat_weight", 1.0)),
        peak_boost=float(cfg["train"].get("heat_peak_boost", 50.0)),
        coord_weight=float(cfg["train"].get("tip_coord_weight", 0.05)),
        temperature=float(cfg["train"].get("soft_argmax_temp", 0.1)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cfg["train"]["epochs"])
    )
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp) if use_amp else None
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    best_miou = -1.0
    best_tip_mae = 1e9
    patience = int(cfg["train"].get("early_stop_patience", 20))
    bad = 0
    history = []

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        tr_loss, tr_tip = run_epoch(
            model, train_loader, seg_crit, heat_crit, optimizer, device, scaler, True
        )
        va_loss, _ = run_epoch(
            model, val_loader, seg_crit, heat_crit, optimizer, device, None, False
        )
        off_miou, off_iou, off_tip = eval_official_miou(
            model, val_loader, device, tip_rx, tip_ry, mode, full_names
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "official_miou": off_miou,
            "official_iou": off_iou,
            "val_tip": off_tip,
            "train_tip": tr_tip,
        }
        history.append(row)
        fg_parts = {k: v for k, v in off_iou.items() if k != "background"}
        print(
            f"[{epoch:03d}] loss {tr_loss:.4f}/{va_loss:.4f} "
            f"fg_mIoU={off_miou:.4f} "
            + " ".join(f"{k}={v:.3f}" for k, v in fg_parts.items())
            + f" (bg={off_iou.get('background', float('nan')):.3f})"
            + f" | tip_mae={off_tip.get('tip_mae_px', float('nan')):.2f}px "
            f"PCK5={off_tip.get('tip_pck@5', float('nan')):.3f} "
            f"PCK10={off_tip.get('tip_pck@10', float('nan')):.3f} "
            f"PCK20={off_tip.get('tip_pck@20', float('nan')):.3f}"
        )

        writer.add_scalar("loss/train", tr_loss, epoch)
        writer.add_scalar("loss/val", va_loss, epoch)
        writer.add_scalar("miou/fg_val", off_miou, epoch)
        for k, v in off_iou.items():
            writer.add_scalar(f"iou_official/{k}", v, epoch)
        for k, v in off_tip.items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                writer.add_scalar(f"tip/{k}", v, epoch)

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "cfg": cfg,
            "official_miou": off_miou,
            "tip_metrics": off_tip,
            "class_names": full_names,
            "method": "seg_plus_gaussian_tip",
            "mode": mode,
        }
        torch.save(ckpt, out_dir / "last.pt")

        tip_mae = off_tip.get("tip_mae_px", 1e9)
        improved = off_miou > best_miou + 1e-6 or (
            abs(off_miou - best_miou) <= 1e-6 and tip_mae < best_tip_mae
        )
        if improved:
            best_miou = off_miou
            best_tip_mae = tip_mae if np.isfinite(tip_mae) else best_tip_mae
            bad = 0
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  ^ saved best.pt (fg_mIoU={best_miou:.4f}, tip_mae={tip_mae:.2f})")
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stop at epoch {epoch} (best fg_mIoU={best_miou:.4f})")
                break

    # optional test once with best
    if test_items and (out_dir / "best.pt").exists():
        best_ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model"])
        test_loader = DataLoader(
            SegTipDataset(test_items, image_size, False, sigma, False, mode),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        te_miou, te_iou, te_tip = eval_official_miou(
            model, test_loader, device, tip_rx, tip_ry, mode, full_names
        )
        test_report = {"test_official_miou": te_miou, "test_iou": te_iou, "test_tip": te_tip}
        (out_dir / "test_report.json").write_text(
            json.dumps(test_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"[TEST] fg_mIoU={te_miou:.4f} tip_mae={te_tip.get('tip_mae_px', float('nan')):.2f} "
            + " ".join(f"{k}={v:.3f}" for k, v in te_iou.items() if k != "background")
        )

    (out_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with open(out_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    writer.close()
    print(f"Done. best fg_mIoU={best_miou:.4f}  weights={out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
