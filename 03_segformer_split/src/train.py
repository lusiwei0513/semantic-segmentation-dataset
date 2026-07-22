#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""正/侧视图分开训练入口（SegFormer-B0）。"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.rail_vehicle_dataset import RailVehicleDataset, collate_same_view
from src.losses.multitask_loss import MultiTaskLoss
from src.metrics.official_metrics import official_front_metrics, official_side_metrics
from src.models.segformer_multitask import SegFormerMultiTask
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.io import ensure_dir, load_yaml, resolve_path
from src.utils.logger import SimpleLogger
from src.utils.seed import seed_worker, set_seed
from src.utils.visualization import save_prediction_visualization


def build_optimizer(model: SegFormerMultiTask, cfg: dict):
    opt_cfg = cfg["optimizer"]
    param_groups = [
        {"params": [p for p in model.backbone_parameters() if p.requires_grad], "lr": float(opt_cfg["backbone_lr"])},
        {"params": [p for p in model.head_parameters() if p.requires_grad], "lr": float(opt_cfg["head_lr"])},
    ]
    param_groups = [g for g in param_groups if len(g["params"]) > 0]
    return torch.optim.AdamW(param_groups, weight_decay=float(opt_cfg.get("weight_decay", 0.01)))


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    total_epochs = int(cfg["training"]["epochs"])
    warmup_epochs = int(cfg.get("scheduler", {}).get("warmup_epochs", 0))
    total_steps = max(1, total_epochs * steps_per_epoch)
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_fold_ids(splits_dir: Path, fold: int, split: str) -> List[str]:
    payload = json.loads((splits_dir / f"fold_{fold}.json").read_text(encoding="utf-8"))
    return payload[split]


def make_loader(cfg: dict, sample_ids: List[str], view: str, train: bool) -> DataLoader:
    data_cfg = cfg["data"]
    ds = RailVehicleDataset(
        metadata_csv=resolve_path(data_cfg["metadata"]),
        processed_root=resolve_path(data_cfg["root"]),
        sample_ids=sample_ids,
        view=view,
        front_size=tuple(data_cfg["front_size"]),
        side_size=tuple(data_cfg["side_size"]),
        mean=data_cfg["mean"],
        std=data_cfg["std"],
        heatmap_sigma=float(cfg["loss"]["heatmap_sigma"]),
        train=train,
        seed=int(cfg["training"]["seed"]),
        keypoint_out_stride=int(cfg["model"].get("keypoint_out_stride", 4)),
    )
    g = torch.Generator()
    g.manual_seed(int(cfg["training"]["seed"]))
    nw = int(data_cfg["num_workers"])
    kwargs = dict(
        batch_size=int(data_cfg["batch_size"]),
        shuffle=train,
        num_workers=nw,
        collate_fn=collate_same_view,
        pin_memory=torch.cuda.is_available(),
    )
    if nw > 0:
        kwargs.update(worker_init_fn=seed_worker, persistent_workers=False, prefetch_factor=2)
    if train:
        kwargs["generator"] = g
    return DataLoader(ds, **kwargs)


def set_backbone_trainable(model: SegFormerMultiTask, trainable: bool) -> None:
    for p in model.backbone_parameters():
        p.requires_grad = trainable


def move_batch(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def train_one_epoch(model, criterion, optimizer, scheduler, loader, device, cfg, scaler, epoch, logger):
    model.train()
    amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    accum = max(1, int(cfg["training"].get("gradient_accumulation_steps", 1)))
    grad_clip = float(cfg["training"].get("grad_clip_norm", 0.0))
    max_batches = cfg["training"].get("max_train_batches")

    n_steps = len(loader)
    if max_batches is not None:
        n_steps = min(n_steps, int(max_batches))

    loss_sum = 0.0
    n_updates = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        if step >= n_steps:
            break
        batch = move_batch(batch, device)
        with torch.amp.autocast("cuda", enabled=amp):
            outputs = model(batch["image"])
            losses = criterion(outputs, batch)
            loss = losses["loss_total"] / accum
        if amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        loss_sum += float(losses["loss_total"].detach().cpu())
        n_updates += 1
        if n_updates % accum == 0:
            if amp:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        if (step + 1) % 10 == 0 or step == 0 or (step + 1) == n_steps:
            logger.log(
                f"epoch={epoch} step={step+1}/{n_steps} "
                f"loss={loss_sum / max(1, n_updates):.4f} "
                f"lr={optimizer.param_groups[-1]['lr']:.2e}"
            )
    return {"train_loss": loss_sum / max(1, n_updates)}


@torch.no_grad()
def validate(model, criterion, loader, device, cfg, view: str, max_batches: Optional[int] = None) -> Dict[str, float]:
    model.eval()
    amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    tip_rx = int(cfg.get("eval", {}).get("tip_ellipse_rx", 24))
    tip_ry = int(cfg.get("eval", {}).get("tip_ellipse_ry", 12))
    stride = int(cfg["model"].get("keypoint_out_stride", 4))

    metrics_acc: Dict[str, List[float]] = {}
    loss_sum = 0.0
    n = 0

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = move_batch(batch, device)
        with torch.amp.autocast("cuda", enabled=amp):
            outputs = model(batch["image"])
            losses = criterion(outputs, batch)
        loss_sum += float(losses["loss_total"].detach().cpu())
        n += 1

        if view == "front":
            m = official_front_metrics(
                outputs["segmentation_logits"],
                batch["segmentation"],
                batch["valid_seg_tasks"],
                outputs.get("nose_tip_heatmap_logits"),
                batch["nose_tip_heatmap"],
                batch["valid_nose_tip"],
                tip_out_stride=stride,
                tip_rx=tip_rx,
                tip_ry=tip_ry,
                image_hw=tuple(batch["image"].shape[-2:]),
            )
        else:
            m = official_side_metrics(
                outputs["segmentation_logits"],
                batch["segmentation"],
                batch["valid_seg_tasks"],
            )
        for k, v in m.items():
            if isinstance(v, float) and (v != v):  # nan
                continue
            metrics_acc.setdefault(k, []).append(float(v))

    out = {"val_loss": loss_sum / max(1, n)}
    for k, vals in metrics_acc.items():
        out[k] = float(sum(vals) / len(vals))
    # aliases for logging
    if "official_miou" in out:
        out["overall_macro_miou"] = out["official_miou"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--view", type=str, default=None, choices=["front", "side"])
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--init-checkpoint", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    fold = int(args.fold if args.fold is not None else cfg["training"].get("fold", 0))
    view = str(args.view or cfg["training"].get("view") or cfg["data"].get("view") or "front")
    assert view in ("front", "side"), view
    set_seed(int(cfg["training"]["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    out_root = resolve_path(args.output_dir) if args.output_dir else resolve_path("outputs")
    run_name = f"{view}_fold{fold}"
    run_dir = ensure_dir(out_root if args.output_dir else out_root / "train" / run_name)
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    vis_dir = ensure_dir(run_dir / "visualizations")
    logger = SimpleLogger(run_dir / "train.log")
    logger.log(f"device={device} view={view} fold={fold} config={args.config}")
    if device.type == "cuda":
        logger.log(
            f"gpu={torch.cuda.get_device_name(0)} "
            f"vram_gb={torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}"
        )

    splits_dir = resolve_path(cfg["data"]["splits_dir"])
    train_ids = load_fold_ids(splits_dir, fold, "train")
    val_ids = load_fold_ids(splits_dir, fold, "val")
    logger.log(f"train_ids={len(train_ids)} val_ids={len(val_ids)}")

    train_loader = make_loader(cfg, train_ids, view, train=True)
    val_loader = make_loader(cfg, val_ids, view, train=False)
    logger.log(f"train_batches={len(train_loader)} val_batches={len(val_loader)} n_train={len(train_loader.dataset)} n_val={len(val_loader.dataset)}")

    model_cfg = cfg["model"]
    pretrained = model_cfg.get("pretrained", True)
    model = SegFormerMultiTask(
        backbone_name=model_cfg.get("backbone", "nvidia/mit-b0"),
        pretrained=pretrained,
        num_segmentation_channels=4,
        decoder_channels=int(model_cfg.get("decoder_channels", 256)),
        keypoint_head=bool(model_cfg.get("keypoint_head", view == "front")),
        keypoint_out_stride=int(model_cfg.get("keypoint_out_stride", 4)),
    ).to(device)
    logger.log(f"pretrained={pretrained!r} keypoint_head={bool(model_cfg.get('keypoint_head', view=='front'))}")

    criterion = MultiTaskLoss(
        bce_weight=float(cfg["loss"]["bce_weight"]),
        dice_weight=float(cfg["loss"]["dice_weight"]),
        loss_weights=cfg["loss"]["loss_weights"],
        pos_weights=cfg["loss"].get("pos_weights"),
        heat_peak_boost=float(cfg["loss"].get("heat_peak_boost", 0.0)),
        tip_coord_weight=float(cfg["loss"].get("tip_coord_weight", 0.0)),
        soft_argmax_temp=float(cfg["loss"].get("soft_argmax_temp", 0.1)),
    )
    logger.log(
        f"tip_loss peak_boost={cfg['loss'].get('heat_peak_boost', 0)} "
        f"coord_w={cfg['loss'].get('tip_coord_weight', 0)} "
        f"sigma={cfg['loss'].get('heatmap_sigma')}"
    )

    freeze_epochs = int(cfg["training"].get("freeze_backbone_epochs", 0))
    if freeze_epochs > 0:
        set_backbone_trainable(model, False)
        logger.log(f"freeze backbone for first {freeze_epochs} epochs")

    optimizer = build_optimizer(model, cfg)
    accum = max(1, int(cfg["training"].get("gradient_accumulation_steps", 1)))
    steps_per_epoch = max(1, (len(train_loader) + accum - 1) // accum)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=steps_per_epoch)

    amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_epoch = 0
    best_score = -1.0
    patience = int(cfg["training"].get("early_stopping_patience", 15))
    bad_epochs = 0
    history = []

    if args.resume:
        payload = load_checkpoint(resolve_path(args.resume), model, optimizer, map_location=str(device))
        start_epoch = int(payload.get("epoch", 0)) + 1
        best_score = float(payload.get("meta", {}).get("best_score", -1.0))
        hist_path = run_dir / "history.json"
        if hist_path.exists():
            try:
                history = json.loads(hist_path.read_text(encoding="utf-8"))
                if not isinstance(history, list):
                    history = []
            except Exception:
                history = []
        logger.log(f"resume from {args.resume} epoch={start_epoch} best={best_score:.4f}")
    elif args.init_checkpoint:
        payload = load_checkpoint(
            resolve_path(args.init_checkpoint), model, optimizer=None, map_location=str(device)
        )
        logger.log(f"init weights from {args.init_checkpoint} (prev_epoch={payload.get('epoch')})")

    epochs = int(cfg["training"]["epochs"])
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        if epoch == freeze_epochs and freeze_epochs > 0:
            set_backbone_trainable(model, True)
            optimizer = build_optimizer(model, cfg)
            scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=steps_per_epoch)
            logger.log("unfreeze backbone")

        train_stats = train_one_epoch(
            model, criterion, optimizer, scheduler, train_loader, device, cfg, scaler, epoch, logger
        )
        logger.log(f"epoch={epoch} train done, validating...")
        val_stats = validate(
            model,
            criterion,
            val_loader,
            device,
            cfg,
            view=view,
            max_batches=cfg["training"].get("max_val_batches"),
        )

        score = float(val_stats.get("official_miou", val_stats.get("overall_macro_miou", -val_stats["val_loss"])))
        row = {"epoch": epoch, **train_stats, **val_stats, "seconds": round(time.time() - t0, 2)}
        history.append(row)
        tip_msg = ""
        if "tip_mae_px" in val_stats:
            tip_msg = f" tip_mae={val_stats['tip_mae_px']:.2f}"
        logger.log(
            f"epoch={epoch} train_loss={train_stats['train_loss']:.4f} "
            f"val_loss={val_stats['val_loss']:.4f} "
            f"official_mIoU={val_stats.get('official_miou', float('nan')):.4f}"
            f"{tip_msg} time={row['seconds']}s"
        )

        save_checkpoint(
            ckpt_dir / "last.pt",
            model,
            optimizer,
            epoch=epoch,
            meta={"best_score": best_score, "fold": fold, "view": view},
        )

        if score > best_score:
            best_score = score
            bad_epochs = 0
            save_checkpoint(
                ckpt_dir / "best.pt",
                model,
                optimizer,
                epoch=epoch,
                meta={"best_score": best_score, "fold": fold, "view": view, "val": val_stats},
            )
            logger.log(f"new best official_mIoU={best_score:.4f}")
            try:
                batch = move_batch(next(iter(val_loader)), device)
                with torch.no_grad():
                    out = model(batch["image"])
                tip = out.get("nose_tip_heatmap_logits")
                save_prediction_visualization(
                    batch["image"][0],
                    out["segmentation_logits"][0].cpu(),
                    tip[0].cpu() if tip is not None else None,
                    vis_dir / f"epoch_{epoch:03d}_{view}.png",
                    mean=cfg["data"]["mean"],
                    std=cfg["data"]["std"],
                    title=f"{view} fold{fold} ep{epoch}",
                    valid_seg_tasks=batch["valid_seg_tasks"][0].cpu(),
                )
            except Exception as exc:
                logger.log(f"vis failed: {exc!r}")
        else:
            bad_epochs += 1
            logger.log(f"no improve ({bad_epochs}/{patience})")

        (run_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if bad_epochs >= patience:
            logger.log("early stopping")
            break

    logger.log(f"done. best_official_mIoU={best_score:.4f} ckpt={ckpt_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
