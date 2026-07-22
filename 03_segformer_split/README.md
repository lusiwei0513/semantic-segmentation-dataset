# SegFormer-B0 正/侧分开训练

本机环境：`../02_baselines_unet_deeplab/front/.venv`（torch 2.6.0+cu124，RTX 3050 4GB）。  
跨模型定量对比表见仓库根目录 [`README.md`](../README.md)。

## 目录与数据

```text
03_segformer_split/
  configs/          训练配置（train_front / train_side 为主）
  src/              训练与模型代码
  scripts/          评测、可视化、消融脚本
  pretrained/mit-b0 本地骨干权重
  outputs/train|eval
```

数据通过 junction 与 UNet/DeepLab **共用同一套** processed + fold：

- `data_front` → `02_baselines_unet_deeplab/data/front/processed`
- `data_side` → `02_baselines_unet_deeplab/data/side/processed`
- `splits_*` → 对应 `fold_*.json`（正式对比用 fold_0：133 / 18 / 38）

主跑次：`outputs/train/{front,side}_fold0/` + `outputs/eval/{front,side}_fold0_test/`。

## 标签与官方指标

| 视图 | 有效任务 | 输入尺寸 |
|------|----------|----------|
| front | body, windshield, nose_tip（bogie/door 永久无效） | 512×512 letterbox |
| side | body, windshield, bogie, door（nose_tip 永久无效） | 384×1536 letterbox |

- 分割头固定 4 通道 `[body, windshield, bogie, door]`，Sigmoid + BCE；缺失类用 `valid` mask，**不写全零真值进 loss**。
- Front `nose_tip`：圆盘质心 → 高斯热图训练；评测与 UNet-KP 对齐——peak 后椭圆 IoU（rx=24, ry=12）+ tip MAE / PCK@5/10/20。
- Front `official_miou` = mean(body, windshield, nose_tip_ellipse IoU)。
- Side `official_miou` = mean(body, windshield, bogie, door)（不计 background）。

## 训练

```powershell
$py = "..\02_baselines_unet_deeplab\front\.venv\Scripts\python.exe"
cd 03_segformer_split

& $py src\train.py --config configs\train_front.yaml --view front --fold 0
& $py src\train.py --config configs\train_side.yaml --view side --fold 0
```

可选 tip 强化：`configs/train_front_peakboost.yaml`（`heat_peak_boost` + 坐标 L1）。冒烟用 `configs/train_smoke.yaml` + `scripts/smoke_test.py`。

## 评测

```powershell
& $py scripts\evaluate_test.py --config configs\train_front.yaml `
  --checkpoint outputs\train\front_fold0\checkpoints\best.pt `
  --view front --split test --num-vis 999

& $py scripts\evaluate_test.py --config configs\train_side.yaml `
  --checkpoint outputs\train\side_fold0\checkpoints\best.pt `
  --view side --split test --num-vis 999
```

## Tip / σ 消融（front，fold_0 test n=38）

对照基线 `front_fold0`：`nose_tip` 权重=3，`heatmap_sigma`=16。

| Run | tip 权重 / σ | official mIoU | tip MAE↓ | PCK@20↑ |
|-----|-------------|--------------:|---------:|--------:|
| 基线 | 3 / 16 | **0.739** | **17.3** | **0.76** |
| A1 | 6 / 16 | 0.707 | 21.0 | 0.74 |
| A2 | 6 / 12 | 0.698 | 27.2 | 0.71 |
| A3 | 6 / 20 | 0.702 | 18.4 | 0.74 |

结论：加重 tip 到 6 **未优于**基线；三组里 A3（σ=20）tip 最好但仍略差于基线，A2（σ=12）最差。继续提 tip 优先 peak-boost / soft-argmax，而不是调整 w/σ。配置见 `configs/train_front_tip_a*.yaml`。