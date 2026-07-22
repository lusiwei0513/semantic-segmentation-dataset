# 高速列车设计图语义分割实验全流程

> 仓库对应本地目录：`训练数据/`  
> GitHub：见仓库首页  
> 硬件参考：RTX 3050 Laptop 4GB · fold_0 正式对比

本文说明**从数据到交付 checkpoint** 的完整实验流程，以及有效结果所在位置。

---

## 1. 任务定义

| 视角 | 分割标签 | 额外任务 |
|------|----------|----------|
| **正视** | background, body, windshield, nose_tip | 鼻尖定位（UNet-KP：heatmap → 椭圆 IoU / MAE / PCK） |
| **侧视** | background, body, windshield, bogie, door | 无 |

**官方指标（test 只评一次）**

- 正视 official mIoU = mean(body, windshield, nose_tip_ellipse)，椭圆 rx=24, ry=12  
- 侧视 official mIoU = mean(body, windshield, bogie, door)（不含 background）

**划分（按 `vehicle_id`，防泄漏）**

| 集合 | fold_0 数量 | 作用 |
|------|------------|------|
| train | 133 | 唯一反传更新权重 |
| val | 18 | 不算梯度；early stop / 存 `best.pt` |
| test | 38 | 训练不可见；最终汇报 |

> 133/18/38 不是某篇论文硬性规定，而是 189 样本五折设计下 fold_0 的整数结果（约 7:1:2）。当前主结果为**单折 fold_0**，未跑满五折。

---

## 2. 仓库结构（有效部分）

```
01_data/                    # 数据入口（多为联接）+ 训练数据说明
02_baselines_unet_deeplab/  # UNet / DeepLab / UNet-KP
  front/                    # 正视训练代码、配置、outputs
  side/                     # 侧视训练代码、配置、outputs
  data_unet/{front,side}/   # 训练用 images + masks
  data/{front,side}/splits/ # fold_*.json
03_segformer_split/         # SegFormer-B0 正/侧分训（正式对比）
04_archive_joint_segformer/ # 联合训练历史存档（非正式 test 主表）
docs/                       # 对比与清理说明
EXPERIMENT_PIPELINE.md      # 本文件
README.md                   # 总览与结果矩阵
```

**模型输入（进网络）**：RGB 图 + 灰度 mask（由 LabelMe JSON 转换）；正视 KP 另加 tip 热图监督。  
**不是**训练时直接读 LabelMe `.json`。

---

## 3. 开源依赖

| 模型 | 实现 |
|------|------|
| UNet / DeepLab | [qubvel-org/segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)（`smp.Unet` / `DeepLabV3Plus` + ResNet34） |
| SegFormer-B0 | Hugging Face `transformers` · 骨干 [`nvidia/mit-b0`](https://huggingface.co/nvidia/mit-b0) |
| 共用 | PyTorch、Albumentations、LabelMe / X-AnyLabeling |

---

## 4. 实验流程（端到端）

```
标注 (LabelMe/X-AnyLabeling → JSON)
    → prepare_masks → data_unet images/masks
    → fold_0 划分
    → 冒烟 / 基线 (UNet、DeepLab)
    → 调参：类别权重 + early stop (patience 20–25)
    → 正视：UNet + KP 热图头
    → 侧视：letterbox [384,1536] 宽画布
    → SegFormer 正/侧分训对照
    → test 一次评测 + 可视化
    → 交付 best.pt
```

### 4.1 数据准备

1. 图 + JSON：`01_data/labelme_workspace/{front,side}`（或 `front_images` / `side_images`）  
2. 转 mask → `data_unet/{front,side}/`  
3. 划分 → `data/{front,side}/splits/fold_0.json`  
4. 正视 tip → `data/front/processed/keypoints/`

说明文档：`01_data/训练数据文件结构.md`

### 4.2 训练（示例命令）

环境：`02_baselines_unet_deeplab/front/.venv`

**正视 UNet-KP**

```powershell
cd 02_baselines_unet_deeplab/front
.\.venv\Scripts\python.exe -u scripts\train_front_kp.py --config config_front_unet_kp_tune.yaml
```

**侧视 UNet 宽幅**

```powershell
cd 02_baselines_unet_deeplab/side
..\front\.venv\Scripts\python.exe -u scripts\train.py --config config_side_unet_wide_384x1536.yaml
```

**SegFormer**

```powershell
cd 03_segformer_split
# 配置见 configs/train_front.yaml / train_side.yaml
python -u -m src.train --config configs/train_front.yaml
python -u -m src.train --config configs/train_side.yaml
```

训练时：**仅 train 反传**；每个 epoch 在 **val** 上算 mIoU 选 `best.pt`；**test 不参与训练**。

### 4.3 调参与结构改动（有数字对照）

| 改动 | 对照 | test 结果要点 |
|------|------|----------------|
| 正视 KP vs 像素 tip | UNet-KP vs UNet-seg | official **0.780** vs 0.757；tip IoU **0.572** vs 0.442；MAE **6.3** |
| 侧视宽幅 vs 方图 | UNet 384×1536 vs 384² | official **0.810** vs 0.739 |
| DeepLab 宽幅 vs 方图 | 同上 | **0.800** vs 0.695 |
| 类别权重 + early stop | 共用协议 | 无单独消融表；为上述实验底子 |

### 4.4 评测与可视化

- CNN：`*/outputs/**/test_report.json`，`test_vis/` 或 `test_vis_segformer_style/compare/`（左 input \| 右 pred）  
- SegFormer：`03_segformer_split/outputs/eval/{front,side}_fold0_test/compare/`（n=38，fold_0 **test**）  
- 正视 SegFormer 可视化需只画有效通道（body/windshield）；bogie/door 无效通道不应叠图。

---

## 5. 正式结果摘要（fold_0 test，n=38）

### 正视

| 模型 | official mIoU | body | windshield | nose_tip | tip MAE | PCK@20 |
|------|--------------:|-----:|-----------:|---------:|--------:|-------:|
| **UNet-R34 KP** | **0.780** | 0.922 | 0.846 | **0.572** | **6.3** | **0.95** |
| UNet-seg tune | 0.757 | 0.949 | 0.879 | 0.442 | — | — |
| DeepLabV3+ | 0.742 | 0.945 | 0.871 | 0.410 | — | — |
| SegFormer-B0 | 0.739 | 0.942 | 0.862 | 0.414 | 17.3 | 0.76 |

### 侧视（宽幅 384×1536，正式）

| 模型 | official mIoU | body | windshield | bogie | door |
|------|--------------:|-----:|-----------:|------:|-----:|
| **UNet-R34** | **0.810** | 0.969 | 0.558 | 0.873 | 0.840 |
| SegFormer-B0 | 0.801 | 0.962 | 0.529 | 0.850 | 0.839 |
| DeepLabV3+ | 0.800 | 0.967 | 0.513 | 0.873 | 0.848 |

---

## 6. 交付 checkpoint（预测新图用）

推荐交付（方法已在 fold_0 验证）：

| 用途 | 路径 |
|------|------|
| 正视主模型 | `02_baselines_unet_deeplab/front/outputs/unet_kp/front_unet_kp_tune/best.pt` |
| 侧视主模型 | `02_baselines_unet_deeplab/side/outputs/unet_seg/side_unet_wide_384x1536/best.pt` |
| 正视 SegFormer（可选） | `03_segformer_split/outputs/train/front_fold0/checkpoints/best.pt` |
| 侧视 SegFormer（可选） | `03_segformer_split/outputs/train/side_fold0/checkpoints/best.pt` |

权重文件经 **Git LFS** 跟踪（`*.pt`）。克隆后需：

```bash
git lfs install
git clone <本仓库 URL>
git lfs pull
```

若 LFS 配额不足，请从本机拷贝上表 `best.pt`，或使用 GitHub Release 附件。

**说明**：五折交叉可增强「评估可信度」，但不等于必须；交付单模型时，用上述已验证的 `best.pt` 即可对 189 张以外的新图推理。若要进一步吃更多标注车型，可在协议验证后用 train+val 重训一版 `final_deliver.pt`。

---

## 7. 联合训练存档（非主表）

`04_archive_joint_segformer/`：正侧交替联合训练历史；可视化曾默认只 dump front，侧视见各实验 `visualizations_side/`。  
**不要与 split + fold_0 official test 主表混比。**

---

## 8. 复现检查清单

- [ ] 安装依赖：`02_baselines_unet_deeplab/front/requirements.txt`、`03_segformer_split/requirements.txt`  
- [ ] `data_unet` 正/侧 images、masks 各 189；`fold_0.json` 为 133/18/38  
- [ ] 能加载上表 `best.pt` 并在 test 复现 `test_report.json` 量级  
- [ ] 新图推理：letterbox 至正视 512² / 侧视 384×1536，类别定义与 `classes.json` 一致  

---

## 9. 相关文档

| 文件 | 内容 |
|------|------|
| `README.md` | 目录总览与模型×视角矩阵 |
| `01_data/训练数据文件结构.md` | 训练数据树 |
| `01_data/labelme_workspace/README.md` | 标注软件打开方式 |
| `03_segformer_split/COMPARISON_RESULTS.md` | 定量对比 |
| `FOLDER_RENAME_MAP.md` | 目录重命名对照 |
| `docs/CLEANUP_SUMMARY.md` | 清理说明 |
