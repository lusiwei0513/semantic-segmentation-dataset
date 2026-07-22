# 轨道车辆语义分割实验

本仓库用于轨道车辆正视图和侧视图的语义分割实验，包含 UNet、DeepLabV3+ 和 SegFormer 三类模型。

实验在 RTX 3050 Laptop 4GB 上完成，主要环境为 PyTorch 2.6.0 + CUDA 12.4。仓库不包含本地虚拟环境，依赖文件位于：

```text
02_baselines_unet_deeplab/front/requirements.txt
03_segformer_split/requirements.txt
```

模型权重使用 Git LFS 管理。克隆仓库后需要执行：

```powershell
git lfs install
git lfs pull
```

## 目录说明

```text
00_checkpoints_交付/             整理后的模型权重和测试报告
01_data/                         数据、标注文件和数据划分
02_baselines_unet_deeplab/       UNet、UNet-KP、DeepLabV3+ 的训练与评测代码
03_segformer_split/              SegFormer 的正视、侧视训练与评测代码
04_joint_segformer/              正视与侧视联合训练实验，保留作实验记录
```

训练脚本读取转换后的图像和单通道 mask，不直接读取 LabelMe JSON。数据处理流程为：

```text
LabelMe 标注
→ 生成图像和单通道 mask
→ 按 fold_0 划分数据
→ 训练并在验证集上选择 best.pt
→ 在测试集上评测和导出可视化结果
```

## 任务和数据划分

| 视图 | 类别 | 输入尺寸 |
|---|---|---|
| 正视 | background、body、windshield、nose_tip | 512 × 512 |
| 侧视 | background、body、windshield、bogie、door | 384 × 1536，采用 Letterbox 预处理 |

本次对比统一使用 `fold_0`，按 `vehicle_id` 划分：

| 数据集 | 数量 |
|---|---:|
| train | 133 |
| val | 18 |
| test | 38 |

当前结果为单折结果，没有计算五折平均。训练阶段使用 train 更新参数，在 val 上选择 `best.pt`，test 只用于最终评测。

数据路径如下：

```text
正视图像和 mask：
01_data/data_unet/front

正视划分：
01_data/data/front/splits/fold_0.json

正视鼻尖关键点：
01_data/data/front/processed/keypoints
仅 UNet-KP 使用

侧视图像和 mask：
01_data/data_unet/side

侧视划分：
01_data/data/side/splits/fold_0.json
```

UNet-KP 只用于正视任务，其中包含鼻尖关键点热图分支；侧视任务使用普通 UNet 分割模型。

## 评价指标

- 正视 mIoU：`body`、`windshield` 和 `nose_tip` 三个前景类别 IoU 的平均值，不计 `background`。UNet-KP 根据预测的鼻尖关键点生成固定椭圆区域，再计算 `nose_tip` IoU。
- 侧视 mIoU：`body`、`windshield`、`bogie` 和 `door` 四个前景类别 IoU 的平均值，不计 `background`。
- UNet-KP 另外报告鼻尖定位 MAE 和 `PCK@20`。

## 测试结果

以下结果均来自 `fold_0` 测试集，共 38 张图像。

### 正视图

| 模型 | mIoU | body | windshield | nose_tip | tip MAE | PCK@20 |
|---|---:|---:|---:|---:|---:|---:|
| UNet-KP | 0.780 | 0.922 | 0.846 | 0.572 | 6.3 | 0.95 |
| UNet | 0.757 | 0.949 | 0.879 | 0.442 | — | — |
| DeepLabV3+ | 0.742 | 0.945 | 0.871 | 0.410 | — | — |
| SegFormer-B0 | 0.739 | 0.942 | 0.862 | 0.414 | 17.3 | 0.76 |

### 侧视图

| 模型 | mIoU | body | windshield | bogie | door |
|---|---:|---:|---:|---:|---:|
| UNet | 0.810 | 0.969 | 0.558 | 0.873 | 0.840 |
| SegFormer-B0 | 0.801 | 0.962 | 0.529 | 0.850 | 0.839 |
| DeepLabV3+ | 0.800 | 0.967 | 0.513 | 0.873 | 0.848 |

## 模型权重和结果目录

整理后的交付权重位于：

```text
00_checkpoints_交付/
```

原始训练输出位于：

```text
正视 UNet-KP：
02_baselines_unet_deeplab/front/outputs/unet_kp/front_unet_kp_tune

正视 UNet：
02_baselines_unet_deeplab/front/outputs/unet_seg/front_unet_seg_tune

正视 DeepLabV3+：
02_baselines_unet_deeplab/front/outputs/deeplab/front_deeplab_v2

正视 SegFormer：
03_segformer_split/outputs/train/front_fold0

侧视 UNet：
02_baselines_unet_deeplab/side/outputs/unet_seg/side_unet_wide_384x1536

侧视 DeepLabV3+：
02_baselines_unet_deeplab/side/outputs/deeplab/side_deeplab_wide_384x1536

侧视 SegFormer：
03_segformer_split/outputs/train/side_fold0
```

UNet 和 DeepLab 的结果目录中包含 `best.pt`、`test_report.json` 和 `test_vis/`。SegFormer 的权重位于 `checkpoints/best.pt`，测试报告和可视化结果位于对应的 `outputs/eval/` 目录。

## 训练命令

以下命令从仓库根目录开始执行。运行前需要先安装对应依赖。

### 正视 UNet-KP

```powershell
Push-Location "02_baselines_unet_deeplab/front"

python -u scripts/train_front_kp.py `
  --config config_front_unet_kp_tune.yaml

Pop-Location
```

### 侧视 UNet

```powershell
Push-Location "02_baselines_unet_deeplab/side"

python -u scripts/train.py `
  --config config_side_unet_wide_384x1536.yaml

Pop-Location
```

### SegFormer

```powershell
Push-Location "03_segformer_split"

python -u -m src.train `
  --config configs/train_front.yaml `
  --view front `
  --fold 0

python -u -m src.train `
  --config configs/train_side.yaml `
  --view side `
  --fold 0

Pop-Location
```

## 重新评测 SegFormer

以下命令加载已有的 `best.pt`，在测试集上重新计算指标并导出可视化结果，不会更新模型参数。

```powershell
Push-Location "03_segformer_split"

python scripts/evaluate_test.py `
  --config configs/train_front.yaml `
  --checkpoint outputs/train/front_fold0/checkpoints/best.pt `
  --view front `
  --fold 0 `
  --split test `
  --num-vis 999

python scripts/evaluate_test.py `
  --config configs/train_side.yaml `
  --checkpoint outputs/train/side_fold0/checkpoints/best.pt `
  --view side `
  --fold 0 `
  --split test `
  --num-vis 999

Pop-Location
```

评测结果默认写入：

```text
03_segformer_split/outputs/eval/front_fold0_test/
03_segformer_split/outputs/eval/side_fold0_test/
```