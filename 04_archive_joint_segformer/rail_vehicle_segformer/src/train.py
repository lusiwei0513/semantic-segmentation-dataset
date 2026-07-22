#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""正式训练入口：共享 MiT-B0 + 正/侧视图多任务。"""
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
from src.metrics.keypoint_metrics import keypoint_metrics
from src.metrics.segmentation_metrics import segmentation_metrics
from src.models.segformer_multitask import SegFormerMultiTask
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.io import ensure_dir, load_yaml, resolve_path
from src.utils.logger import SimpleLogger
from src.utils.seed import seed_worker, set_seed
from src.utils.visualization import save_prediction_visualization


def build_optimizer(model: SegFormerMultiTask, cfg: dict):
    opt_cfg = cfg["optimizer"]
    param_groups = [
        {
            "params": [p for p in model.backbone_parameters() if p.requires_grad],
            "lr": float(opt_cfg["backbone_lr"]),
        },
        {
            "params": [p for p in model.head_parameters() if p.requires_grad],
            "lr": float(opt_cfg["head_lr"]),
        },
    ]
    # 过滤空参数组
    param_groups = [g for g in param_groups if len(g["params"]) > 0]
    return torch.optim.AdamW(
        param_groups, weight_decay=float(opt_cfg.get("weight_decay", 0.01))
    )


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


def make_loader(
    cfg: dict,
    sample_ids: List[str],
    view: str,
    train: bool,
) -> DataLoader:
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
        kwargs.update(
            worker_init_fn=seed_worker,
            persistent_workers=False,  # Windows 下避免 worker 常驻占满内存
            prefetch_factor=2,
        )
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


def train_one_epoch(
    model,
    criterion,
    optimizer,
    scheduler,
    front_loader,
    side_loader,
    device,
    cfg,
    scaler,
    epoch: int,
    logger: SimpleLogger,
) -> Dict[str, float]:
    model.train()
    amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    accum = max(1, int(cfg["training"].get("gradient_accumulation_steps", 1)))
    grad_clip = float(cfg["training"].get("grad_clip_norm", 0.0))
    max_batches = cfg["training"].get("max_train_batches")

    front_iter = iter(front_loader)
    side_iter = iter(side_loader)
    n_front = len(front_loader)
    n_side = len(side_loader)
    n_steps = max(n_front, n_side)
    if max_batches is not None:
        n_steps = min(n_steps, int(max_batches))

    loss_sum = 0.0
    n_updates = 0
    optimizer.zero_grad(set_to_none=True)

    for step in range(n_steps):
        for view, it, n_avail in [
            ("front", front_iter, n_front),
            ("side", side_iter, n_side),
        ]:
            if n_avail == 0:
                continue
            try:
                batch = next(it)
            except StopIteration:
                it = iter(front_loader if view == "front" else side_loader)
                if view == "front":
                    front_iter = it
                else:
                    side_iter = it
                batch = next(it)

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
def validate(
    model,
    criterion,
    front_loader,
    side_loader,
    device,
    cfg,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    metrics_acc: Dict[str, List[float]] = {}
    loss_sum = 0.0
    n = 0

    def run_loader(loader, tag: str):
        nonlocal loss_sum, n
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            batch = move_batch(batch, device)
            with torch.amp.autocast("cuda", enabled=amp):
                outputs = model(batch["image"])
                losses = criterion(outputs, batch)
            loss_sum += float(losses["loss_total"].detach().cpu())
            n += 1
            seg_m = segmentation_metrics(
                outputs["segmentation_logits"],
                batch["segmentation"],
                batch["valid_seg_tasks"],
            )
            tip_m = keypoint_metrics(
                outputs["nose_tip_heatmap_logits"],
                batch["nose_tip_heatmap"],
                batch["valid_nose_tip"],
                body_masks=batch["segmentation"][:, 0:1],
            )
            for k, v in {**seg_m, **tip_m}.items():
                metrics_acc.setdefault(f"{tag}_{k}", []).append(v)
                metrics_acc.setdefault(f"overall_{k}", []).append(v)

    run_loader(front_loader, "front")
    run_loader(side_loader, "side")

    out = {"val_loss": loss_sum / max(1, n)}
    for k, vals in metrics_acc.items():
        out[k] = float(sum(vals) / len(vals))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default="",
        help="只加载模型权重做微调（不恢复 epoch/optimizer）",
    )
    parser.add_argument("--output-dir", type=str, default="")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    fold = int(args.fold if args.fold is not None else cfg["training"].get("fold", 0))
    set_seed(int(cfg["training"]["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    out_root = resolve_path(args.output_dir) if args.output_dir else resolve_path("outputs")
    run_name = "fold_boost" if "boost" in str(args.config).lower() else f"fold_{fold}"
    if args.output_dir:
        run_dir = ensure_dir(out_root)
    else:
        run_dir = ensure_dir(out_root / "train" / run_name)
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    vis_dir = ensure_dir(run_dir / "visualizations")
    logger = SimpleLogger(run_dir / "train.log")
    logger.log(f"device={device} fold={fold} config={args.config}")

    splits_dir = resolve_path(cfg["data"]["splits_dir"])
    train_ids = load_fold_ids(splits_dir, fold, "train")
    val_ids = load_fold_ids(splits_dir, fold, "val")
    logger.log(f"train={len(train_ids)} val={len(val_ids)}")

    front_train = make_loader(cfg, train_ids, "front", train=True)
    side_train = make_loader(cfg, train_ids, "side", train=True)
    front_val = make_loader(cfg, val_ids, "front", train=False)
    side_val = make_loader(cfg, val_ids, "side", train=False)

    model_cfg = cfg["model"]
    model = SegFormerMultiTask(
        backbone_name=model_cfg.get("backbone", "nvidia/mit-b0"),
        pretrained=bool(model_cfg.get("pretrained", True)),
        num_segmentation_channels=4,
        decoder_channels=int(model_cfg.get("decoder_channels", 256)),
        keypoint_head=bool(model_cfg.get("keypoint_head", True)),
        keypoint_out_stride=int(model_cfg.get("keypoint_out_stride", 4)),
    ).to(device)
    logger.log(
        f"pretrained={bool(model_cfg.get('pretrained', True))} "
        f"backbone={model_cfg.get('backbone', 'nvidia/mit-b0')}"
    )

    criterion = MultiTaskLoss(
        bce_weight=float(cfg["loss"]["bce_weight"]),
        dice_weight=float(cfg["loss"]["dice_weight"]),
        loss_weights=cfg["loss"]["loss_weights"],
        pos_weights=cfg["loss"].get("pos_weights"),
    )

    freeze_epochs = int(cfg["training"].get("freeze_backbone_epochs", 0))
    if freeze_epochs > 0:
        set_backbone_trainable(model, False)
        logger.log(f"freeze backbone for first {freeze_epochs} epochs")

    optimizer = build_optimizer(model, cfg)
    accum = max(1, int(cfg["training"].get("gradient_accumulation_steps", 1)))
    steps_per_epoch = max(
        1,
        (max(len(front_train), len(side_train)) * 2 + accum - 1) // accum,
    )
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
        logger.log(
            f"init weights from {args.init_checkpoint} "
            f"(prev_epoch={payload.get('epoch')}, reset training schedule)"
        )

    epochs = int(cfg["training"]["epochs"])
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        if epoch == freeze_epochs and freeze_epochs > 0:
            set_backbone_trainable(model, True)
            optimizer = build_optimizer(model, cfg)
            scheduler = build_scheduler(
                optimizer,
                cfg,
                steps_per_epoch=max(
                    1,
                    (max(len(front_train), len(side_train)) * 2 + accum - 1) // accum,
                ),
            )
            logger.log("unfreeze backbone")

        train_stats = train_one_epoch(
            model,
            criterion,
            optimizer,
            scheduler,
            front_train,
            side_train,
            device,
            cfg,
            scaler,
            epoch,
            logger,
        )
        logger.log(f"epoch={epoch} train done, validating...")
        val_stats = validate(
            model,
            criterion,
            front_val,
            side_val,
            device,
            cfg,
            max_batches=cfg["training"].get("max_val_batches"),
        )

        # 主监控指标：overall macro mIoU（没有则用 -val_loss）
        score = float(val_stats.get("overall_macro_miou", -val_stats["val_loss"]))
        row = {
            "epoch": epoch,
            **train_stats,
            **{k: v for k, v in val_stats.items() if k in (
                "val_loss",
                "overall_macro_miou",
                "overall_macro_dice",
                "front_macro_miou",
                "side_macro_miou",
                "front_tip_mean_px_error",
            ) or k.endswith("_iou")},
            "seconds": round(time.time() - t0, 2),
        }
        history.append(row)
        logger.log(
            f"epoch={epoch} train_loss={train_stats['train_loss']:.4f} "
            f"val_loss={val_stats['val_loss']:.4f} "
            f"mIoU={val_stats.get('overall_macro_miou', float('nan')):.4f} "
            f"time={row['seconds']}s"
        )

        # 每轮保存最新
        save_checkpoint(
            ckpt_dir / "last.pt",
            model,
            optimizer,
            epoch=epoch,
            meta={"best_score": best_score, "fold": fold},
        )

        if score > best_score:
            best_score = score
            bad_epochs = 0
            save_checkpoint(
                ckpt_dir / "best.pt",
                model,
                optimizer,
                epoch=epoch,
                meta={"best_score": best_score, "fold": fold, "val": val_stats},
            )
            logger.log(f"new best score={best_score:.4f}")
            # 保存一张验证可视化
            model.eval()
            try:
                batch = move_batch(next(iter(front_val)), device)
                with torch.no_grad():
                    out = model(batch["image"])
                save_prediction_visualization(
                    batch["image"][0],
                    out["segmentation_logits"][0].cpu(),
                    out["nose_tip_heatmap_logits"][0].cpu(),
                    vis_dir / f"epoch_{epoch:03d}_front.png",
                    mean=cfg["data"]["mean"],
                    std=cfg["data"]["std"],
                    title=f"fold{fold} epoch{epoch} front",
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

    logger.log(f"done. best_score={best_score:.4f} ckpt={ckpt_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
