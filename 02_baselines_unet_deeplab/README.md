# UNet / DeepLab 训练工程

本目录包含正视和侧视图的 UNet、DeepLabV3+ 训练代码、配置文件和实验结果。

## 目录结构

```text
02_baselines_unet_deeplab/
├── data/                  # 数据划分和关键点
├── data_unet/             # 训练使用的图像和 mask
├── prepared_front/        # 早期整理的正视数据
├── prepared_side/         # 早期整理的侧视数据
├── front/                 # 正视训练工程
└── side/                  # 侧视训练工程
```

正视和侧视工程共用下面的 Python 环境：

```text
front/.venv/Scripts/python.exe
```

## 数据路径

正视训练使用：

```text
../data_unet/front
../data/front/splits/fold_0.json
```

侧视训练使用：

```text
../data_unet/side
../data/side/splits/fold_0.json
```

UNet-KP 还会读取正视关键点：

```text
../data/front/processed/keypoints
```

`01_data/` 下的同名目录是目录联接，实际数据保存在本目录中。

## 正视实验

正视输入尺寸为 512×512。

| 模型 | 配置文件 | 输出目录 |
|------|----------|----------|
| UNet-KP | `front/config_front_unet_kp_tune.yaml` | `front/outputs/unet_kp/front_unet_kp_tune/` |
| UNet | `front/config_front_unet_seg_tune.yaml` | `front/outputs/unet_seg/front_unet_seg_tune/` |
| DeepLabV3+ | `front/config_front_deeplab_v2.yaml` | `front/outputs/deeplab/front_deeplab_v2/` |

UNet-KP 在分割之外增加了鼻尖关键点热图分支，仅用于正视图。

## 侧视实验

侧视主实验采用 384×1536 的宽幅 letterbox 输入（保留长宽比）。

| 模型 | 输入尺寸 | 配置文件 | 输出目录 |
|------|----------|----------|----------|
| UNet | 384×1536 | `side/config_side_unet_wide_384x1536.yaml` | `side/outputs/unet_seg/side_unet_wide_384x1536/` |
| DeepLabV3+ | 384×1536 | `side/config_side_deeplab_wide_384x1536.yaml` | `side/outputs/deeplab/side_deeplab_wide_384x1536/` |

每个输出目录中保留了 `best.pt`、`test_report.json`、`history.json` 和测试可视化。

## 评测示例

```powershell
$py = ".\02_baselines_unet_deeplab\front\.venv\Scripts\python.exe"

cd .\02_baselines_unet_deeplab\front
& $py scripts\export_test_vis.py `
  --config config_front_unet_kp_tune.yaml `
  --run outputs\unet_kp\front_unet_kp_tune

cd ..\side
& $py scripts\export_test_vis.py `
  --config config_side_unet_wide_384x1536.yaml `
  --run outputs\unet_seg\side_unet_wide_384x1536
```

重新训练时，输出位置由配置文件中的 `paths.output_dir` 和 `paths.run_name` 决定。