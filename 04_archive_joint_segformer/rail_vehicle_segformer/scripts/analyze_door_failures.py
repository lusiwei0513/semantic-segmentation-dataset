"""Analyze worst side-door predictions and save visualizations."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.rail_vehicle_dataset import RailVehicleDataset, collate_same_view
from src.models.segformer_multitask import SegFormerMultiTask
from src.utils.checkpoint import load_checkpoint
from src.utils.io import ensure_dir, load_yaml, resolve_path
from src.utils.visualization import denormalize

DOOR = 3  # channel index in multi-label: body0 windshield1 bogie2 door3
THR = 0.5


def iou_pr(gt: np.ndarray, pred: np.ndarray):
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    tp = inter
    fp = np.logical_and(~gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()
    iou = float(inter / union) if union else (1.0 if tp == 0 and fp == 0 else 0.0)
    prec = float(tp / (tp + fp)) if (tp + fp) else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) else 0.0
    return iou, prec, rec, int(tp), int(fp), int(fn), int(gt.sum()), int(pred.sum())


def main():
    cfg = load_yaml("configs/train_clean_cont.yaml")
    ckpt = resolve_path("../experiments/joint_fold0_clean_cont/checkpoints/best.pt")
    if not ckpt.exists():
        ckpt = resolve_path("../experiments/joint_fold0_clean_v2/checkpoints/best.pt")
    out_dir = ensure_dir(resolve_path("outputs/visualizations/door_fail_analysis"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = json.loads(
        (resolve_path(cfg["data"]["splits_dir"]) / "fold_0.json").read_text(encoding="utf-8")
    )
    # use val+test for more samples, or all side with door
    ids = sorted(set(splits["val"]) | set(splits["test"]))

    ds = RailVehicleDataset(
        processed_root=resolve_path(cfg["data"]["root"]),
        metadata_csv=resolve_path(cfg["data"]["metadata"]),
        sample_ids=ids,
        view="side",
        front_size=tuple(cfg["data"]["front_size"]),
        side_size=tuple(cfg["data"]["side_size"]),
        mean=cfg["data"]["mean"],
        std=cfg["data"]["std"],
        train=False,
        heatmap_sigma=float(cfg["loss"]["heatmap_sigma"]),
        keypoint_out_stride=int(cfg["model"].get("keypoint_out_stride", 4)),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_same_view)

    model = SegFormerMultiTask(
        backbone_name=cfg["model"].get("backbone", "nvidia/mit-b0"),
        pretrained=True,
        decoder_channels=int(cfg["model"].get("decoder_channels", 256)),
        keypoint_head=True,
        keypoint_out_stride=int(cfg["model"].get("keypoint_out_stride", 4)),
    ).to(device)
    load_checkpoint(ckpt, model, map_location=str(device))
    model.eval()

    rows = []
    with torch.no_grad():
        for batch in loader:
            sid = batch["sample_id"][0]
            valid = batch["valid_seg_tasks"][0].cpu().numpy()
            if float(valid[DOOR]) < 0.5:
                continue
            img = batch["image"].to(device)
            out = model(img)
            logits = out["segmentation_logits"][0].cpu()
            prob = torch.sigmoid(logits).numpy()
            gt = batch["segmentation"][0].cpu().numpy()
            door_gt = gt[DOOR] > 0.5
            door_pr = prob[DOOR] >= THR
            iou, prec, rec, tp, fp, fn, ngt, npr = iou_pr(door_gt, door_pr)

            # geometry of GT door
            ys, xs = np.where(door_gt)
            if len(ys):
                gh = int(ys.max() - ys.min() + 1)
                gw = int(xs.max() - xs.min() + 1)
            else:
                gh = gw = 0

            rows.append(
                {
                    "sample_id": sid,
                    "iou": iou,
                    "prec": prec,
                    "rec": rec,
                    "fp": fp,
                    "fn": fn,
                    "gt_px": ngt,
                    "pred_px": npr,
                    "gt_h": gh,
                    "gt_w": gw,
                    "fp_ratio": fp / max(ngt, 1),
                    "fn_ratio": fn / max(ngt, 1),
                    "image": batch["image"][0].cpu(),
                    "prob_door": prob[DOOR],
                    "gt_door": door_gt.astype(np.float32),
                    "prob_body": prob[0],
                    "gt_body": (gt[0] > 0.5).astype(np.float32),
                }
            )

    rows.sort(key=lambda r: r["iou"])
    print(f"side samples with door (val+test): {len(rows)}")
    print(f"ckpt: {ckpt}")
    if not rows:
        return

    ious = np.array([r["iou"] for r in rows])
    print(
        f"door IoU: mean={ious.mean():.3f} med={np.median(ious):.3f} "
        f"p25={np.percentile(ious,25):.3f} min={ious.min():.3f}"
    )
    # failure mode buckets
    under = [r for r in rows if r["rec"] < 0.7 and r["prec"] >= 0.7]  # miss door
    over = [r for r in rows if r["prec"] < 0.7 and r["rec"] >= 0.7]  # FP bleed
    both = [r for r in rows if r["prec"] < 0.7 and r["rec"] < 0.7]
    ok = [r for r in rows if r["iou"] >= 0.75]
    print(f"IoU>=0.75: {len(ok)}")
    print(f"mainly underseg (low recall): {len(under)}")
    print(f"mainly overseg/FP (low prec): {len(over)}")
    print(f"both bad: {len(both)}")

    # overlap of FP with body GT
    fp_on_body = []
    for r in rows:
        fp = (r["prob_door"] >= THR) & (r["gt_door"] < 0.5)
        body = r["gt_body"] > 0.5
        if fp.sum() == 0:
            fp_on_body.append(0.0)
        else:
            fp_on_body.append(float((fp & body).sum() / fp.sum()))
    print(f"among FP pixels, fraction on GT-body: mean={np.mean(fp_on_body):.3f}")

    mean_cfg, std_cfg = cfg["data"]["mean"], cfg["data"]["std"]

    # save worst 8
    worst = rows[:8]
    print("\nWORST 8:")
    for r in worst:
        print(
            f"  {r['sample_id']}: IoU={r['iou']:.3f} P={r['prec']:.3f} R={r['rec']:.3f} "
            f"gt_px={r['gt_px']} pred={r['pred_px']} gt_box={r['gt_w']}x{r['gt_h']}"
        )

    for rank, r in enumerate(worst):
        img = denormalize(r["image"], mean_cfg, std_cfg)
        gt = r["gt_door"]
        pr = (r["prob_door"] >= THR).astype(np.float32)
        # panels: input | GT door | Pred door | error (FP red FN cyan)
        err = np.zeros((*gt.shape, 3), dtype=np.float32)
        err[..., 0] = ((pr > 0.5) & (gt < 0.5)).astype(np.float32)  # FP red
        err[..., 1] = ((gt > 0.5) & (pr < 0.5)).astype(np.float32)  # FN green
        err[..., 2] = ((gt > 0.5) & (pr > 0.5)).astype(np.float32)  # TP blue

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        axes = axes.ravel()
        axes[0].imshow(img)
        axes[0].set_title(f"{r['sample_id']}")
        axes[1].imshow(img)
        axes[1].imshow(np.dstack([gt, gt * 0, gt * 0]), alpha=0.45)
        axes[1].set_title(f"GT door px={r['gt_px']}")
        axes[2].imshow(img)
        axes[2].imshow(np.dstack([pr, pr * 0.2, pr * 0.2]), alpha=0.45)
        axes[2].set_title(f"Pred door IoU={r['iou']:.3f} P={r['prec']:.3f} R={r['rec']:.3f}")
        axes[3].imshow(img)
        axes[3].imshow(err, alpha=0.55)
        axes[3].set_title("TP=blue  FP=red  FN=green")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(
            f"rank{rank} door fail | gt_box={r['gt_w']}x{r['gt_h']}",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"worst_{rank:02d}_{r['sample_id']}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    # also a summary strip of IoU histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ious, bins=20, color="#c44", edgecolor="white")
    ax.axvline(ious.mean(), color="k", ls="--", label=f"mean={ious.mean():.3f}")
    ax.set_xlabel("door IoU (val+test side)")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "door_iou_hist.png", dpi=120)
    plt.close(fig)

    # write markdown summary
    md = [
        "# Door 低分分析",
        "",
        f"- checkpoint: `{ckpt}`",
        f"- 样本数 (fold0 val+test 且有 door): **{len(rows)}**",
        f"- door IoU mean/med/min: **{ious.mean():.3f}** / {np.median(ious):.3f} / {ious.min():.3f}",
        f"- IoU≥0.75: {len(ok)}; 漏检为主: {len(under)}; 误检为主: {len(over)}; 双差: {len(both)}",
        f"- FP 落在 GT-body 上的比例均值: **{np.mean(fp_on_body):.3f}**",
        "",
        "## 为何“边界规整”仍可能 IoU 偏低",
        "",
        "1. **多扇门 / 大跨度**: 侧视常有多扇门，漏一扇会大幅拉低 recall。",
        "2. **与车身互斥 GT**: GT 门从 body 挖空；模型是多标签 Sigmoid，门边缘常扩到 body → FP（红）。",
        "3. **letterbox 压缩**: 侧视极宽，resize 后门变窄，边界几像素误差占比大。",
        "4. **阈值偏低**: door 像素面积小，BCE/Dice 对边界偏移敏感。",
        "",
        "## Worst samples",
        "",
    ]
    for rank, r in enumerate(worst):
        md.append(
            f"- `worst_{rank:02d}_{r['sample_id']}.png`: IoU={r['iou']:.3f} "
            f"P={r['prec']:.3f} R={r['rec']:.3f} gt={r['gt_w']}x{r['gt_h']}"
        )
    (out_dir / "README.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote -> {out_dir}")


if __name__ == "__main__":
    main()
