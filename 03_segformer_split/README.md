# SegFormer-B0 正/侧分开训练（定量对比）

## 硬件验证（本机 RTX 3050 4GB Laptop，2026-07-20）

使用 CUDA 环境：`../02_baselines_unet_deeplab/front/.venv`（torch 2.6.0+cu124）

| 配置 | peak VRAM | 结果 |
|------|----------:|------|
| front 512×512 bs=2 AMP + tip | 947 MB | OK |
| front 512×512 bs=1 AMP + tip | 540 MB | OK |
| side 384×768 bs=2 AMP | 1080 MB | OK |
| side 384×1024 bs=1 AMP | 825 MB | OK |
| side 384×1536 bs=1 AMP | 1352 MB | OK |
| side 448×2016 bs=1 AMP | 2393 MB | OK |

说明：系统自带 Python 3.11 为 **CPU-only torch**，不能用于 GPU 训练。

## 目录

```
训练数据/
  02_baselines_unet_deeplab/   # UNet/DeepLab + 修正后 data；front/ + side/
  03_segformer_split/         # 本工程：正/侧分开 SegFormer
  04_archive_joint_segformer/ # 历史联合训练存档（未在本机验证）
```

数据通过 junction 指向（**与 UNet/DeepLab 共用同一套数据**，不是「SegFormer 独占侧视」）：
- `data_front` → `02_baselines_unet_deeplab/data/front/processed`
- `data_side` → `02_baselines_unet_deeplab/data/side/processed`
- `splits_*` → 对应 fold json

本工程同时有正视与侧视主跑次：`outputs/train/{front,side}_fold0/` + `outputs/eval/{front,side}_fold0_test/`。  
完整「模型 × 视图」矩阵见根目录 `README.md` §3。

## 训练

```powershell
$py = "..\02_baselines_unet_deeplab\front\.venv\Scripts\python.exe"
cd 03_segformer_split

# 正视图（含 nose_tip）
& $py src\train.py --config configs\train_front.yaml --view front --fold 0

# 侧视图
& $py src\train.py --config configs\train_side.yaml --view side --fold 0
```

## 评测（与 UNet KP 对齐）

- Front `official_miou` = mean(body, windshield, nose_tip_ellipse IoU)，椭圆 rx=24,ry=12
- Side `official_miou` / `fg_miou` = mean(body, windshield, bogie, door)
- tip MAE / PCK@5/10/20（letterbox 像素）

```powershell
& $py scripts\evaluate_test.py --config configs\train_front.yaml --checkpoint outputs\train\front_fold0\checkpoints\best.pt --view front --split test --num-vis 999
& $py scripts\evaluate_test.py --config configs\train_side.yaml --checkpoint outputs\train\side_fold0\checkpoints\best.pt --view side --split test --num-vis 999
```
