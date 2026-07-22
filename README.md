# 轨道车辆语义分割 — 定量对比工作区

整理日期：2026-07-21  
硬件：RTX 3050 Laptop 4GB（本机实机训练与评测）

**完整实验流程（推荐先读）→ [`EXPERIMENT_PIPELINE.md`](./EXPERIMENT_PIPELINE.md)**

## 1. 文件夹地图

| 文件夹 | 内容 |
|--------|------|
| `01_data/` | 共享数据入口（junction → baselines 的 `data` / `prepared_*`；含 `split_raw`） |
| `02_baselines_unet_deeplab/` | UNet-R34 / DeepLabV3+；子目录 `front/`、`side/`；输出按 `unet_kp` / `unet_seg` / `deeplab` 分组 |
| `03_segformer_split/` | SegFormer-B0 正/侧分开训练与评测 |
| `04_archive_joint_segformer/` | 历史联合训练存档（原 `bb`，已去掉 `.bb` 嵌套；**非本机结果**） |
| `05_archive_misc/` | 旧工程碎片、`bb.zip`、data backup、基线杂项归档 |
| `docs/` | 标注 PPT、旧说明、对比草稿、清理说明 |
| `FOLDER_RENAME_MAP.md` | 旧名 → 新名对照 |

CUDA 环境：`02_baselines_unet_deeplab/front/.venv`（torch 2.6.0+cu124）

---

## 2. 先读：数据共享 ≠「某模型管正视、某模型管侧视」

**所有 CNN / SegFormer 主对比共用同一套数据与划分**，不是「这个模型拥有正视、那个模型拥有侧视」。

| 视角 | 共享数据（全体模型共用） |
|------|--------------------------|
| 正视 | `01_data/data_unet/front` + `01_data/data/front/splits/fold_0.json` |
| 侧视 | `01_data/data_unet/side` + `01_data/data/side/splits/fold_0.json` |
| 正视 tip 关键点（仅 KP 头需要） | `01_data/data/front/processed/keypoints` |

**结果路径按「模型 × 视图」各自独立**（见下表）。不要把「正视表里的 UNet-KP」和「侧视表里的 UNet-seg」读成一对；那是同一视角下的不同模型，或同一模型族在不同视角的跑次。

### UNet-KP 说明（正视专用）

- **UNet-KP = 正视专用**：分割头 + **鼻尖 heatmap**；评测用 tip 椭圆 IoU / MAE / PCK。
- **侧视没有 KP 头**（侧视类别无 nose_tip）：侧视对比请用 **UNet-seg**（以及 DeepLab / SegFormer）。

---

## 3. 模型 × 视图 结果矩阵（主对比，fold_0 test）

图例：✅ 齐全（`best.pt` + `test_report.json` + 可视化）｜ N/A 设计上不存在｜ ⬜ 参考用方图

| 模型 | 正视 (512×512) | 侧视宽画布 (384×1536，主对比) | 侧视方图 (384，仅参考) |
|------|----------------|-------------------------------|------------------------|
| **UNet-KP** | ✅ | **N/A**（无侧视 KP） | N/A |
| **UNet-seg** | ✅ | ✅ | ⬜ |
| **DeepLabV3+-R34** | ✅ | ✅ | ⬜ |
| **SegFormer-B0** | ✅ | ✅ | —（未做方图） |

### 每个模型的正视结果路径

| 模型 | `best.pt` | `test_report.json` | 可视化 |
|------|-----------|--------------------|--------|
| UNet-KP | `02_baselines_unet_deeplab/front/outputs/unet_kp/front_unet_kp_tune/best.pt` | 同目录 `test_report.json` | 同目录 `test_vis/` |
| UNet-seg | `02_baselines_unet_deeplab/front/outputs/unet_seg/front_unet_seg_tune/best.pt` | 同目录 | 同目录 `test_vis/` |
| DeepLab | `02_baselines_unet_deeplab/front/outputs/deeplab/front_deeplab_v2/best.pt` | 同目录 | 同目录 `test_vis/` |
| SegFormer | `03_segformer_split/outputs/train/front_fold0/checkpoints/best.pt` | `03_segformer_split/outputs/eval/front_fold0_test/test_report.json` | `.../eval/front_fold0_test/compare/` |

### 每个模型的侧视结果路径

| 模型 | 变体 | `best.pt` | `test_report.json` | 可视化 |
|------|------|-----------|--------------------|--------|
| UNet-KP | — | **N/A** | N/A | N/A |
| UNet-seg | 宽 ★ | `02_baselines_unet_deeplab/side/outputs/unet_seg/side_unet_wide_384x1536/best.pt` | 同目录 | 同目录 `test_vis/` |
| UNet-seg | 方图 | `.../unet_seg/side_unet_square_384/best.pt` | 同目录 | 同目录 `test_vis/` |
| DeepLab | 宽 ★ | `02_baselines_unet_deeplab/side/outputs/deeplab/side_deeplab_wide_384x1536/best.pt` | 同目录 | 同目录 `test_vis/` |
| DeepLab | 方图 | `.../deeplab/side_deeplab_square_384/best.pt` | 同目录 | 同目录 `test_vis/` |
| SegFormer | 宽 ★ | `03_segformer_split/outputs/train/side_fold0/checkpoints/best.pt` | `03_segformer_split/outputs/eval/side_fold0_test/test_report.json` | `.../eval/side_fold0_test/compare/` |

★ = 主对比侧视协议。

**结论（2026-07-21 清点）：** UNet-seg / DeepLab / SegFormer 的正视+侧视主跑次均齐全；UNet-KP 仅正视（符合设计）。清理**没有**删掉任一模型「仅剩的正视」或「仅剩的侧视」主跑次。

---

## 4. 官方指标协议（fold_0 test，n=38）

| 视角 | 分辨率 | official mIoU |
|------|--------|----------------|
| **Front** | 512×512 | mean(body, windshield, **nose_tip 椭圆 IoU**)，rx=24, ry=12；另报 tip MAE / PCK |
| **Side** | **384×1536 letterbox**（主对比）；另有 384 方图旧结果 | mean(body, windshield, bogie, door) |

划分文件：`01_data/data/{front,side}/splits/fold_0.json`（同 `02_baselines_unet_deeplab/data/...`）。

---

## 5. 主结果表（fold_0 test）

### 正视图

| 模型 | official mIoU | body | windshield | nose_tip | tip MAE | PCK@20 | checkpoint |
|------|--------------:|-----:|-----------:|---------:|--------:|-------:|------------|
| **UNet-KP tune** | **0.780** | 0.922 | 0.846 | **0.572** | **6.3** | **0.95** | `.../unet_kp/front_unet_kp_tune/best.pt` |
| UNet-seg tune（像素 tip） | 0.757 | 0.949 | 0.879 | 0.442 | — | — | `.../unet_seg/front_unet_seg_tune/best.pt` |
| DeepLabV3+-R34 front | 0.742 | 0.945 | 0.871 | 0.410 | — | — | `.../deeplab/front_deeplab_v2/best.pt` |
| SegFormer-B0 | 0.739 | 0.942 | 0.862 | 0.414 | 17.3 | 0.76 | `03_segformer_split/.../front_fold0/checkpoints/best.pt` |

### 侧视图（宽画布 384×1536 — 主对比）

| 模型 | fg / official mIoU | body | windshield | bogie | door | checkpoint |
|------|-------------------:|-----:|-----------:|------:|-----:|------------|
| **UNet-seg side wide** | **0.810** | 0.969 | 0.558 | 0.873 | 0.840 | `.../unet_seg/side_unet_wide_384x1536/best.pt` |
| DeepLabV3+-R34 side wide | **0.800** | 0.967 | 0.513 | 0.873 | 0.848 | `.../deeplab/side_deeplab_wide_384x1536/best.pt` |
| **SegFormer-B0** | **0.801** | 0.962 | 0.529 | 0.850 | 0.839 | `03_segformer_split/.../side_fold0/checkpoints/best.pt` |

### 侧视图（方图 384×384 — 旧基线，仅参考）

| 模型 | mIoU | body | windshield | bogie | door | checkpoint |
|------|-----:|-----:|-----------:|------:|-----:|------------|
| UNet-seg side square | 0.739 | 0.944 | 0.453 | 0.834 | 0.726 | `.../unet_seg/side_unet_square_384/best.pt` |
| DeepLab side square | 0.695 | 0.935 | 0.442 | 0.770 | 0.632 | `.../deeplab/side_deeplab_square_384/best.pt` |

更细说明见 `03_segformer_split/COMPARISON_RESULTS.md`、`03_segformer_split/QUALITY_FAIRNESS.md`。

---

## 6. 清理与归档（删了什么 / 留了什么）

详见 `docs/CLEANUP_SUMMARY.md` 与 `05_archive_misc/cleanup_logs_20260721/`。

**删掉的（可恢复性低，但非主对比必需）：**

- SegFormer tip 消融 / peakboost 的额外 train+eval（约 600MB）
- 未进主表的旧 CNN 整树（早期无 tune、all189、旧 kp 非 v2、空的 SE-ResNet50 目录等）
- 各**主跑次**的 `last.pt`（**保留了 `best.pt`**）、`tb/`、`__pycache__`

**保留 / 归档的：**

- 上表全部主对比 `best.pt` + `test_report.json` + `test_vis`（或 SegFormer `compare/`）
- `04_archive_joint_segformer/`（历史联合训练，非本机主对比）
- `05_archive_misc/`（project13、bb.zip、data_backup、杂项日志）

**本轮可读化：** 输出改名为 `front_unet_kp_tune` 等；侧视 `outputs_side` → `outputs`；按模型族分组。

---

## 7. 复现评测（不重新训练）

```powershell
$py = ".\02_baselines_unet_deeplab\front\.venv\Scripts\python.exe"

# SegFormer test + 可视化
cd 03_segformer_split
& $py scripts\evaluate_test.py --config configs\train_front.yaml `
  --checkpoint outputs\train\front_fold0\checkpoints\best.pt --view front --split test --num-vis 999
& $py scripts\evaluate_test.py --config configs\train_side.yaml `
  --checkpoint outputs\train\side_fold0\checkpoints\best.pt --view side --split test --num-vis 999

# CNN test_vis（示例）
cd ..\02_baselines_unet_deeplab\front
& $py scripts\export_test_vis.py --config config_front_unet_kp_tune.yaml --run outputs\unet_kp\front_unet_kp_tune
cd ..\side
& $py ..\front\.venv\Scripts\python.exe scripts\export_test_vis.py --config config_side_unet_wide_384x1536.yaml --run outputs\unet_seg\side_unet_wide_384x1536
```
