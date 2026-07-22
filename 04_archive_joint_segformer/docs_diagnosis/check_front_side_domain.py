#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
检查正视图与侧视图是否存在明显设备/成像域差异。

重要：
1. 正侧视图天然容易区分，因此 view probe 准确率高并不能证明需要两个完整模型。
2. 如果 device_id 与 view 完全一一绑定，则无法从现有数据中分离设备差异和视图差异。
3. 本脚本只做数据诊断，最终仍需比较：
   - 共享编码器 + 视图专用头
   - 两个完整模型
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torchvision.models import ResNet50_Weights, resnet50


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(data_root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else data_root / p


def safe_float(value: float) -> float:
    if value is None or not np.isfinite(value):
        return float("nan")
    return float(value)


def image_statistics(
    image_path: Path,
    dark_threshold: int = 15,
    bright_threshold: int = 240,
) -> Dict[str, float]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"无法读取图像: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    height, width = gray.shape
    rgb_mean = image_rgb.reshape(-1, 3).mean(axis=0)
    rgb_std = image_rgb.reshape(-1, 3).std(axis=0)

    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = gray.std()
    saturation = hsv[..., 1].mean()
    dark_ratio = (gray <= dark_threshold).mean()
    bright_ratio = (gray >= bright_threshold).mean()

    return {
        "width_actual": int(width),
        "height_actual": int(height),
        "aspect_ratio": safe_float(width / max(height, 1)),
        "r_mean": safe_float(rgb_mean[0]),
        "g_mean": safe_float(rgb_mean[1]),
        "b_mean": safe_float(rgb_mean[2]),
        "r_std": safe_float(rgb_std[0]),
        "g_std": safe_float(rgb_std[1]),
        "b_std": safe_float(rgb_std[2]),
        "gray_mean": safe_float(gray.mean()),
        "contrast": safe_float(contrast),
        "saturation": safe_float(saturation),
        "dark_ratio": safe_float(dark_ratio),
        "bright_ratio": safe_float(bright_ratio),
        "laplacian_variance": safe_float(lap_var),
        "file_size_kb": safe_float(image_path.stat().st_size / 1024.0),
    }


def detect_confounding(df: pd.DataFrame, view_col: str, device_col: str) -> Tuple[bool, List[str]]:
    warnings_out: List[str] = []

    if device_col not in df.columns or df[device_col].isna().all():
        warnings_out.append("缺少有效 device_id，无法判断设备差异。")
        return False, warnings_out

    table = pd.crosstab(df[device_col], df[view_col])
    devices_per_view = df.groupby(view_col)[device_col].nunique(dropna=True)
    views_per_device = df.groupby(device_col)[view_col].nunique(dropna=True)

    perfect = bool(
        len(devices_per_view) >= 2
        and (devices_per_view == 1).all()
        and (views_per_device == 1).all()
    )

    if perfect:
        warnings_out.append(
            "SEVERE_CONFOUNDING：每个视图只对应一个设备，且每个设备只对应一个视图。"
            "无法从现有数据中分离视图差异与设备差异。"
        )
    elif (views_per_device == 1).any():
        warnings_out.append(
            "PARTIAL_CONFOUNDING：至少部分设备只拍摄一种视图，设备与视图存在部分混淆。"
        )

    return perfect, warnings_out


class EmbeddingExtractor:
    def __init__(self, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        model.eval().to(self.device)

        self.model = model
        self.transform = weights.transforms()

    @torch.inference_mode()
    def extract(self, paths: List[Path], batch_size: int = 16) -> np.ndarray:
        feats: List[np.ndarray] = []
        for start in tqdm(range(0, len(paths), batch_size), desc="提取预训练特征"):
            batch_paths = paths[start : start + batch_size]
            tensors = []
            for path in batch_paths:
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    tensors.append(self.transform(image))
            batch = torch.stack(tensors).to(self.device)
            out = self.model(batch)
            feats.append(out.cpu().numpy())
        return np.concatenate(feats, axis=0)


def centroid_cosine_distance(x: np.ndarray, labels: np.ndarray) -> Optional[float]:
    classes = np.unique(labels)
    if len(classes) != 2:
        return None
    a = x[labels == classes[0]].mean(axis=0)
    b = x[labels == classes[1]].mean(axis=0)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return None
    return float(1.0 - np.dot(a, b) / denom)


def linear_mmd(x: np.ndarray, labels: np.ndarray) -> Optional[float]:
    classes = np.unique(labels)
    if len(classes) != 2:
        return None
    a = x[labels == classes[0]].mean(axis=0)
    b = x[labels == classes[1]].mean(axis=0)
    return float(np.mean((a - b) ** 2))


def probe_accuracy(
    x: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    random_state: int,
    min_samples_per_class: int,
) -> Optional[Dict[str, float]]:
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or counts.min() < min_samples_per_class:
        return None

    splits = min(n_splits, int(counts.min()))
    if splits < 2:
        return None

    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced"),
    )
    pred = cross_val_predict(clf, x, y, cv=cv)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "n_splits": int(splits),
    }


def pca_plot(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    if len(embeddings) < 3:
        return

    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 6))
    for label in np.unique(labels):
        mask = labels == label
        ax.scatter(reduced[mask, 0], reduced[mask, 1], label=str(label), alpha=0.75)

    ax.set_title(title)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def aggregate_stats(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    numeric_cols = [
        "width_actual",
        "height_actual",
        "aspect_ratio",
        "r_mean",
        "g_mean",
        "b_mean",
        "gray_mean",
        "contrast",
        "saturation",
        "dark_ratio",
        "bright_ratio",
        "laplacian_variance",
        "file_size_kb",
    ]
    available = [c for c in numeric_cols if c in df.columns]
    return df.groupby(group_cols, dropna=False)[available].agg(["count", "mean", "std", "median"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-embeddings", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_root = Path(cfg["data_root"]).resolve()
    metadata_path = resolve_path(data_root, cfg["metadata_csv"])
    output_dir = resolve_path(data_root, cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    columns = cfg["columns"]
    sample_col = columns["sample_id"]
    image_col = columns["image_path"]
    vehicle_col = columns["vehicle_id"]
    view_col = columns["view"]
    device_col = columns["device_id"]

    df = pd.read_csv(metadata_path)
    required = [sample_col, image_col, view_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"metadata 缺少字段: {missing}")

    valid_views = set(cfg.get("valid_views", ["front", "side"]))
    invalid_views = sorted(set(df[view_col].dropna().unique()) - valid_views)
    if invalid_views:
        raise ValueError(f"发现非法 view: {invalid_views}")

    if sample_col in df.columns and df[sample_col].duplicated().any():
        duplicates = df.loc[df[sample_col].duplicated(), sample_col].tolist()
        raise ValueError(f"sample_id 重复: {duplicates[:10]}")

    paths = [resolve_path(data_root, p) for p in df[image_col].astype(str)]
    missing_files = [str(p) for p in paths if not p.exists()]
    if missing_files:
        raise FileNotFoundError(f"缺少 {len(missing_files)} 个图像，例如: {missing_files[:5]}")

    image_cfg = cfg.get("image", {})
    stats = []
    errors = []
    for i, path in enumerate(tqdm(paths, desc="计算图像统计")):
        try:
            row = image_statistics(
                path,
                dark_threshold=int(image_cfg.get("dark_threshold", 15)),
                bright_threshold=int(image_cfg.get("bright_threshold", 240)),
            )
            row["_row_index"] = i
            stats.append(row)
        except Exception as exc:
            errors.append({"index": i, "path": str(path), "error": repr(exc)})

    if errors:
        (output_dir / "image_errors.json").write_text(
            json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    stats_df = pd.DataFrame(stats).set_index("_row_index")
    df = df.join(stats_df)
    df.to_csv(output_dir / "image_statistics.csv", index=False, encoding="utf-8-sig")

    basic_view = aggregate_stats(df, [view_col])
    basic_view.to_csv(output_dir / "basic_stats_by_view.csv", encoding="utf-8-sig")

    if device_col in df.columns and not df[device_col].isna().all():
        basic_device = aggregate_stats(df, [device_col])
        basic_device.to_csv(output_dir / "basic_stats_by_device.csv", encoding="utf-8-sig")
        crosstab = pd.crosstab(df[device_col], df[view_col])
        crosstab.to_csv(output_dir / "view_device_crosstab.csv", encoding="utf-8-sig")
    else:
        crosstab = pd.DataFrame()

    perfect_confounding, warnings_out = detect_confounding(df, view_col, device_col)

    summary: Dict[str, object] = {
        "n_samples": int(len(df)),
        "n_front": int((df[view_col] == "front").sum()),
        "n_side": int((df[view_col] == "side").sum()),
        "n_vehicles": (
            int(df[vehicle_col].nunique(dropna=True))
            if vehicle_col in df.columns
            else None
        ),
        "n_devices": (
            int(df[device_col].nunique(dropna=True))
            if device_col in df.columns
            else None
        ),
        "perfect_view_device_confounding": perfect_confounding,
        "warnings": warnings_out,
    }

    if not args.skip_embeddings:
        extractor = EmbeddingExtractor(device=cfg["embedding"].get("device", "auto"))
        embeddings = extractor.extract(
            paths,
            batch_size=int(cfg["embedding"].get("batch_size", 16)),
        )
        np.save(output_dir / "embeddings.npy", embeddings)

        views = df[view_col].astype(str).to_numpy()
        summary["view_centroid_cosine_distance"] = centroid_cosine_distance(
            embeddings, views
        )
        summary["view_linear_mmd"] = linear_mmd(embeddings, views)

        probe_cfg = cfg["probe"]
        view_probe = probe_accuracy(
            embeddings,
            views,
            n_splits=int(probe_cfg.get("n_splits", 5)),
            random_state=int(probe_cfg.get("random_state", 42)),
            min_samples_per_class=int(probe_cfg.get("min_samples_per_class", 5)),
        )
        summary["view_probe"] = view_probe
        pca_plot(
            embeddings,
            views,
            output_dir / "pca_embeddings_by_view.png",
            "Pretrained embeddings grouped by view",
        )

        if device_col in df.columns and df[device_col].nunique(dropna=True) >= 2:
            devices = df[device_col].astype(str).to_numpy()
            device_probe = probe_accuracy(
                embeddings,
                devices,
                n_splits=int(probe_cfg.get("n_splits", 5)),
                random_state=int(probe_cfg.get("random_state", 42)),
                min_samples_per_class=int(probe_cfg.get("min_samples_per_class", 5)),
            )
            summary["device_probe"] = device_probe
            pca_plot(
                embeddings,
                devices,
                output_dir / "pca_embeddings_by_device.png",
                "Pretrained embeddings grouped by device",
            )

            within_view = {}
            for view_value in sorted(df[view_col].dropna().unique()):
                mask = (df[view_col].astype(str).to_numpy() == str(view_value))
                if len(np.unique(devices[mask])) >= 2:
                    within_view[str(view_value)] = probe_accuracy(
                        embeddings[mask],
                        devices[mask],
                        n_splits=int(probe_cfg.get("n_splits", 5)),
                        random_state=int(probe_cfg.get("random_state", 42)),
                        min_samples_per_class=int(
                            probe_cfg.get("min_samples_per_class", 5)
                        ),
                    )
            summary["device_probe_within_view"] = within_view

            within_device = {}
            for device_value in sorted(df[device_col].dropna().unique()):
                mask = (df[device_col].astype(str).to_numpy() == str(device_value))
                if len(np.unique(views[mask])) >= 2:
                    within_device[str(device_value)] = probe_accuracy(
                        embeddings[mask],
                        views[mask],
                        n_splits=int(probe_cfg.get("n_splits", 5)),
                        random_state=int(probe_cfg.get("random_state", 42)),
                        min_samples_per_class=int(
                            probe_cfg.get("min_samples_per_class", 5)
                        ),
                    )
            summary["view_probe_within_device"] = within_device

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_lines = [
        "# 正侧视图数据域诊断报告",
        "",
        f"- 总样本数：{summary['n_samples']}",
        f"- 正视图：{summary['n_front']}",
        f"- 侧视图：{summary['n_side']}",
        f"- 独立车辆数：{summary['n_vehicles']}",
        f"- 设备数：{summary['n_devices']}",
        f"- 设备与视图完全混淆：{summary['perfect_view_device_confounding']}",
        "",
        "## 警告",
    ]
    if warnings_out:
        report_lines.extend([f"- {w}" for w in warnings_out])
    else:
        report_lines.append("- 未发现明显设备—视图完全混淆。")

    report_lines.extend(
        [
            "",
            "## 解释原则",
            "",
            "- 正侧视图分类准确率高是正常现象，不能单独证明需要两个完整模型。",
            "- 如果同一视图内部仍能高准确率预测设备，说明设备域差异较强。",
            "- 如果设备与视图完全绑定，需要补充交叉设备数据，或用模型对照实验判断。",
            "- 当前约 200 张数据时，默认仍推荐共享编码器 + 视图专用头。",
            "",
            "## 自动结果",
            "",
            "```json",
            json.dumps(summary, ensure_ascii=False, indent=2),
            "```",
        ]
    )

    (output_dir / "report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    (output_dir / "warnings.txt").write_text(
        "\n".join(warnings_out), encoding="utf-8"
    )

    print(f"诊断完成，结果位于: {output_dir}")
    if perfect_confounding:
        print(
            "警告：设备与视图完全混淆。不能仅根据域分类结果决定拆成两个模型。"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
