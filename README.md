# 轨道车辆语义分割实验

本仓库用于轨道车辆正视图和侧视图的语义分割实验，包含 UNet、DeepLabV3+ 和 SegFormer 三类模型。

实验在 RTX 3050 Laptop 4GB 上完成，环境为 torch 2.6.0+cu124。Python 环境位于：

```text
02_baselines_unet_deeplab/front/.venv
```

## 目录说明

```text
00_checkpoints_交付/             整理后的交付权重与 test_report
01_data/                         数据和数据划分
02_baselines_unet_deeplab/       UNet、UNet-KP、DeepLabV3+ 训练与评测
03_segformer_split/              SegFormer 训练与评测（说明见该目录 README）
04_joint_segformer/              正/侧分训之后的联合多任务实验（共享编码器；val 结果，非正式 test 主表）
```

训练直接读取转换后的图像和 mask，不直接读 LabelMe JSON。数据流：

```text
LabelMe 标注 → 图像 + 单通道 mask → fold_0 划分 → 训练 → test 评测 / 可视化
```

## 任务与划分

| 视图 | 类别 | 输入尺寸 |
|------|------|----------|
| 正视 | background, body, windshield, nose_tip | 512×512 |
| 侧视 | background, body, windshield, bogie, door | 主实验 384×1536 letterbox |

正式对比统一使用 `fold_0`（按 `vehicle_id` 划分）：train 133 / val 18 / test 38。当前为单折结果，未做五折平均。val 选 `best.pt`，test 只用于最终评测。

```text
正视：01_data/data_unet/front
划分：01_data/data/front/splits/fold_0.json
关键点：01_data/data/front/processed/keypoints   # 仅 UNet-KP

侧视：01_data/data_unet/side
划分：01_data/data/side/splits/fold_0.json
```

UNet-KP 只用于正视（鼻尖关键点热图）；侧视用普通 UNet 分割。

## 指标

- 正视 mIoU：`body`、`windshield`、`nose_tip` 三个前景类别 IoU 的平均值，不计 `background`。UNet-KP 的 `nose_tip` 由预测关键点生成固定椭圆后参与计算。
- 侧视 mIoU：`body`、`windshield`、`bogie`、`door` 四个前景类别 IoU 的平均值，不计 `background`。
- UNet-KP 另外报告鼻尖定位 MAE 和 `PCK@20`。

## 训练（复现）

UNet / DeepLab 基于 `segmentation_models.pytorch`（ResNet34）；SegFormer 用 `nvidia/mit-b0`。

```powershell
# 正视 UNet-KP
cd 02_baselines_unet_deeplab/front
.\.venv\Scripts\python.exe -u scripts\train_front_kp.py `
  --config config_front_unet_kp_tune.yaml

# 侧视 UNet（宽幅）
cd 02_baselines_unet_deeplab/side
..\front\.venv\Scripts\python.exe -u scripts\train.py `
  --config config_side_unet_wide_384x1536.yaml

# SegFormer（细节、tip 消融、公平约定见 03_segformer_split/README.md）
cd 03_segformer_split
python -u -m src.train --config configs/train_front.yaml
python -u -m src.train --config configs/train_side.yaml
```

## 测试结果（fold_0，38 张）

### 正视图

| 模型 | mIoU | body | windshield | nose_tip | tip MAE | PCK@20 |
|------|-----:|-----:|-----------:|---------:|--------:|-------:|
| UNet-KP | 0.780 | 0.922 | 0.846 | 0.572 | 6.3 | 0.95 |
| UNet | 0.757 | 0.949 | 0.879 | 0.442 | — | — |
| DeepLabV3+ | 0.742 | 0.945 | 0.871 | 0.410 | — | — |
| SegFormer-B0 | 0.739 | 0.942 | 0.862 | 0.414 | 17.3 | 0.76 |

### 侧视图

| 模型 | mIoU | body | windshield | bogie | door |
|------|-----:|-----:|-----------:|------:|-----:|
| UNet | 0.810 | 0.969 | 0.558 | 0.873 | 0.840 |
| SegFormer-B0 | 0.801 | 0.962 | 0.529 | 0.850 | 0.839 |
| DeepLabV3+ | 0.800 | 0.967 | 0.513 | 0.873 | 0.848 |


## 权重与运行目录

交付权重：`00_checkpoints_交付/`（`.pt` 用 Git LFS）。主模型训练输出：

```text
正视 KP：02_baselines_unet_deeplab/front/outputs/unet_kp/front_unet_kp_tune
正视 UNet：.../outputs/unet_seg/front_unet_seg_tune
正视 DeepLab：.../outputs/deeplab/front_deeplab_v2
正视 SegFormer：03_segformer_split/outputs/train/front_fold0
侧视 UNet：02_baselines_unet_deeplab/side/outputs/unet_seg/side_unet_wide_384x1536
侧视 DeepLab：.../outputs/deeplab/side_deeplab_wide_384x1536
侧视 SegFormer：03_segformer_split/outputs/train/side_fold0
```

CNN 目录含 `best.pt`、`test_report.json`、`test_vis/`；SegFormer 权重在 `checkpoints/best.pt`，报告与可视化在对应 `eval` 目录。

## 重新评测（不训练）

```powershell
$py = ".\02_baselines_unet_deeplab\front\.venv\Scripts\python.exe"

cd 03_segformer_split
& $py scripts\evaluate_test.py --config configs\train_front.yaml `
  --checkpoint outputs\train\front_fold0\checkpoints\best.pt `
  --view front --split test --num-vis 999

& $py scripts\evaluate_test.py --config configs\train_side.yaml `
  --checkpoint outputs\train\side_fold0\checkpoints\best.pt `
  --view side --split test --num-vis 999
```