"""Train U-Net for front-view semantic segmentation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import (  # noqa: E402
    SegDataset,
    list_pairs,
    load_fold_pairs,
    split_pairs,
    split_pairs_three_way,
)
from src.metrics import CEDiceLoss, confusion_matrix, iou_from_cm  # noqa: E402

DEFAULT_CLASS_NAMES = ["background", "body", "windshield", "nose_tip"]


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


def resolve_path(base: Path, p: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def build_model(cfg: dict):
    import os

    import segmentation_models_pytorch as smp

    m = cfg["model"]
    weights = m.get("encoder_weights", "imagenet")
    arch = str(m.get("arch", "Unet")).lower()
    encoder = m["encoder"]
    classes = int(m["num_classes"])
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    def _make(w):
        if arch in {"deeplabv3plus", "deeplabv3+", "deeplab"}:
            return smp.DeepLabV3Plus(
                encoder_name=encoder,
                encoder_weights=w,
                in_channels=3,
                classes=classes,
            )
        return smp.Unet(
            encoder_name=encoder,
            encoder_weights=w,
            in_channels=3,
            classes=classes,
        )

    try:
        return _make(weights)
    except Exception as e:
        print(f"[warn] failed loading encoder_weights={weights!r}: {e}")
        print("[warn] falling back to randomly initialized encoder")
        return _make(None)


def run_epoch(model, loader, criterion, optimizer, device, scaler, train: bool, num_classes: int):
    model.train(train)
    total_loss = 0.0
    cm_sum = torch.zeros(num_classes, num_classes, device=device)

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, leave=False, desc="train" if train else "val"):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            with autocast(enabled=scaler is not None and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, masks)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * images.size(0)
            pred = logits.argmax(1)
            cm_sum += confusion_matrix(pred, masks, num_classes)

    n = max(len(loader.dataset), 1)
    iou, miou = iou_from_cm(cm_sum)
    return total_loss / n, iou.detach().cpu(), miou


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--encoder", type=str, default=None, help="override encoder name")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="resume from checkpoint (e.g. outputs/.../last.pt)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.encoder:
        cfg["model"]["encoder"] = args.encoder
        cfg["paths"]["run_name"] = f"unet_{args.encoder.replace('-', '_')}"

    class_names = list(cfg["data"].get("class_names") or DEFAULT_CLASS_NAMES)
    prepared_dir = resolve_path(ROOT, cfg["data"]["prepared_dir"])
    out_dir = resolve_path(ROOT, cfg["paths"]["output_dir"]) / cfg["paths"]["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = list_pairs(prepared_dir)
    if not pairs:
        raise SystemExit(
            f"No image/mask pairs in {prepared_dir}. Run prepare_masks.py first."
        )

    fold_json = cfg["data"].get("fold_json")
    test_pairs: list = []
    if fold_json:
        fj = resolve_path(ROOT, fold_json)
        train_pairs = load_fold_pairs(prepared_dir, fj, "train")
        val_pairs = load_fold_pairs(prepared_dir, fj, "val")
        test_pairs = load_fold_pairs(prepared_dir, fj, "test")
        print(f"fold={fj.name} train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")
    elif float(cfg["data"].get("test_ratio", 0) or 0) > 0:
        train_pairs, val_pairs, test_pairs = split_pairs_three_way(
            pairs,
            val_ratio=float(cfg["data"]["val_ratio"]),
            test_ratio=float(cfg["data"]["test_ratio"]),
            seed=int(cfg["data"]["seed"]),
        )
        print(
            f"train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)} "
            f"prepared={prepared_dir}"
        )
    else:
        train_pairs, val_pairs = split_pairs(
            pairs, val_ratio=cfg["data"]["val_ratio"], seed=cfg["data"]["seed"]
        )
        print(f"train={len(train_pairs)} val={len(val_pairs)} prepared={prepared_dir}")
    print(f"classes={class_names}")
    print(
        "note: train/val(/test) are fixed for the whole run (seed/fold). "
        "Overfitting is reduced by stronger aug + weight_decay + early stop, not by reshuffling each epoch."
    )

    image_size = int(cfg["data"]["image_size"])
    strong_aug = bool(cfg["train"].get("strong_aug", True))
    train_ds = SegDataset(train_pairs, image_size=image_size, train=True, strong_aug=strong_aug)
    val_ds = SegDataset(val_pairs, image_size=image_size, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=True,
        drop_last=int(cfg["train"]["batch_size"]) > 1,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} encoder={cfg['model']['encoder']}")

    model = build_model(cfg).to(device)
    criterion = CEDiceLoss(
        num_classes=int(cfg["model"]["num_classes"]),
        class_weights=cfg["train"].get("class_weights"),
        ce_weight=float(cfg["train"].get("ce_weight", 1.0)),
        dice_weight=float(cfg["train"].get("dice_weight", 1.0)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    total_epochs = int(cfg["train"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs
    )
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp) if use_amp else None

    writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    best_miou = -1.0
    patience = int(cfg["train"].get("early_stop_patience", 20))
    bad_epochs = 0
    history = []
    start_epoch = 1

    resume_path = args.resume
    if resume_path is not None:
        resume_path = resume_path if resume_path.is_absolute() else (ROOT / resume_path).resolve()
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_path = out_dir / "best.pt"
        if best_path.exists():
            best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            best_miou = float(best_ckpt.get("val_miou", -1.0))
            best_epoch = int(best_ckpt.get("epoch", 0))
            bad_epochs = max(0, int(ckpt.get("epoch", 0)) - best_epoch)
        else:
            best_miou = float(ckpt.get("val_miou", -1.0))
        # Restore LR schedule position for epochs already done.
        for _ in range(start_epoch - 1):
            scheduler.step()
        hist_candidates = [
            out_dir / "history.json",
            out_dir / "history_partial.json",
        ]
        for hp in hist_candidates:
            if hp.exists():
                with open(hp, encoding="utf-8") as f:
                    history = json.load(f)
                history = [r for r in history if int(r["epoch"]) < start_epoch]
                break
        print(
            f"resume={resume_path} start_epoch={start_epoch} "
            f"best_mIoU={best_miou:.4f} bad_epochs={bad_epochs} history={len(history)}"
        )

    num_classes = int(cfg["model"]["num_classes"])
    for epoch in range(start_epoch, total_epochs + 1):
        tr_loss, tr_iou, tr_miou = run_epoch(
            model, train_loader, criterion, optimizer, device, scaler, True, num_classes
        )
        va_loss, va_iou, va_miou = run_epoch(
            model, val_loader, criterion, optimizer, device, None, False, num_classes
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "train_miou": tr_miou,
            "val_miou": va_miou,
            "val_iou": {class_names[i]: float(va_iou[i]) for i in range(num_classes)},
        }
        history.append(row)
        print(
            f"[{epoch:03d}] loss {tr_loss:.4f}/{va_loss:.4f} "
            f"mIoU {tr_miou:.4f}/{va_miou:.4f} "
            + " ".join(f"{class_names[i]}={va_iou[i]:.3f}" for i in range(num_classes))
        )

        writer.add_scalar("loss/train", tr_loss, epoch)
        writer.add_scalar("loss/val", va_loss, epoch)
        writer.add_scalar("miou/train", tr_miou, epoch)
        writer.add_scalar("miou/val", va_miou, epoch)
        for i, name in enumerate(class_names):
            writer.add_scalar(f"iou_val/{name}", float(va_iou[i]), epoch)

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "cfg": cfg,
            "val_miou": va_miou,
            "class_names": class_names,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if va_miou >= best_miou:
            best_miou = va_miou
            bad_epochs = 0
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  ^ saved best.pt (mIoU={best_miou:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stop at epoch {epoch} (best mIoU={best_miou:.4f})")
                break

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with open(out_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    if test_pairs:
        test_ds = SegDataset(test_pairs, image_size=image_size, train=False)
        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=int(cfg["train"]["num_workers"]),
            pin_memory=True,
        )
        # evaluate best weights on held-out test
        best_path = out_dir / "best.pt"
        if best_path.exists():
            best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(best_ckpt["model"])
        te_loss, te_iou, te_miou = run_epoch(
            model, test_loader, criterion, optimizer, device, None, False, num_classes
        )
        test_report = {
            "test_loss": te_loss,
            "test_miou": te_miou,
            "test_iou": {class_names[i]: float(te_iou[i]) for i in range(num_classes)},
            "n_test": len(test_pairs),
        }
        (out_dir / "test_report.json").write_text(
            json.dumps(test_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"TEST best.pt: loss={te_loss:.4f} mIoU={te_miou:.4f} "
            + " ".join(f"{class_names[i]}={te_iou[i]:.3f}" for i in range(num_classes))
        )

    writer.close()
    print(f"Done. best mIoU={best_miou:.4f}  weights={out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
