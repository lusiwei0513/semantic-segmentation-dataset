# SegFormer-B0 vs UNet-R34 / DeepLab — 定量对比（fold_0，同 data）

生成时间：2026-07-21（路径整理后更新）  
硬件：本机 RTX 3050 Laptop 4GB（已实机冒烟+训练，非沿用 archive 历史结果）

## 质量承诺（未偷工减料）

| 项 | 做法 |
|----|------|
| 骨干 | 完整 MiT-B0 / SegFormer-B0 |
| 正视 tip | 保留 heatmap 训练 + 官方椭圆 IoU（rx=24, ry=12） |
| 训练方式 | 正/侧 **分开** |
| 划分 | `02_baselines_unet_deeplab/data/{front,side}/splits/fold_0.json` |
| 分辨率 | Front 512×512；Side **384×1536 letterbox**（主对比） |
| 显存适配 | 仅 bs / AMP / grad_accum / num_workers=0 |

详见 `QUALITY_FAIRNESS.md`。

## 目录（整理后）

- `02_baselines_unet_deeplab/` ← 原 `baselines_unet_deeplab` / `project13_split_labelme`
  - `front/` ← 原 `gpt标注训练正视图-seg_train`
  - `side/` ← 原 `cursor标注与训练第一版-ppt_seg_task`
- `03_segformer_split/` ← 原 `segformer_b0_split`
- `04_archive_joint_segformer/` ← 原 `bb`（联合训练；已去掉 `.bb` 嵌套；**非本机结果**）

## 读表前注意

- **数据共享**：正/侧 `fold_0` 与 `data_unet` 由全体模型共用，不是「某模型拥有正视、某模型拥有侧视」。
- **UNet-KP 仅正视**（鼻尖 heatmap）；侧视无 KP，侧视 UNet 行为是 **UNet-seg**。
- 下表按视角分行；完整「模型 × 视图」路径见根目录 `README.md` §3。

## Test 集对比（n=38，同一 fold_0 test）

### 正视图（含 nose_tip）

| 模型 | official mIoU | body | windshield | nose_tip IoU | tip MAE (px) | tip PCK@20 |
|------|--------------:|-----:|-----------:|-------------:|-------------:|-----------:|
| **SegFormer-B0（本机）** | **0.739** | 0.942 | 0.862 | 0.414 | 17.3 | 0.76 |
| UNet-KP tune | **0.780** | 0.922 | 0.846 | 0.572 | **6.3** | **0.95** |
| UNet-seg tune（像素 tip） | 0.757 | 0.949 | 0.879 | 0.442 | — | — |
| DeepLabV3+-R34 front | 0.742 | 0.945 | 0.871 | 0.410 (pixel tip) | — | — |

说明：SegFormer 车身/挡风很强；尖端定位仍弱于 UNet-KP（MAE 17 vs 6）。

### 侧视图 — 主对比（384×1536 letterbox）

| 模型 | fg / official mIoU | body | windshield | bogie | door |
|------|-------------------:|-----:|-----------:|------:|-----:|
| **UNet-seg side wide** | **0.810** | 0.969 | 0.558 | 0.873 | 0.840 |
| DeepLabV3+-R34 side wide | **0.800** | 0.967 | 0.513 | 0.873 | 0.848 |
| **SegFormer-B0（本机）** | **0.801** | 0.962 | 0.529 | 0.850 | **0.839** |

说明：宽画布下 UNet-seg / DeepLab / SegFormer 整体接近；挡风仍是共同短板。SegFormer 车门仍有竞争力。侧视无 UNet-KP 行（N/A）。

### 侧视图 — 方图参考（384×384，旧基线）

| 模型 | fg / official mIoU | body | windshield | bogie | door |
|------|-------------------:|-----:|-----------:|------:|-----:|
| UNet-seg side square | 0.739 | 0.944 | 0.453 | 0.834 | 0.726 |
| DeepLab side square | 0.695 | 0.935 | 0.442 | 0.770 | 0.632 |

## Val 最佳（训练日志）

- Front best val official_mIoU **0.799** @ep59（body 0.957 / ws 0.859 / tip_iou 0.581 / tip_mae≈25）
- Side best val official_mIoU **0.869** @ep54（body 0.973 / ws 0.678 / bogie 0.910 / door 0.884）
  - 侧视训练在后续 resume 中途曾被环境中断，**采用 best.pt（ep54）** 作为正式权重

## Checkpoint / 可视化（按模型 × 视图）

| 模型 | 正视 | 侧视 |
|------|------|------|
| UNet-KP | `02_.../front/outputs/unet_kp/front_unet_kp_tune/` | **N/A** |
| UNet-seg | `02_.../front/outputs/unet_seg/front_unet_seg_tune/` | `02_.../side/outputs/unet_seg/side_unet_wide_384x1536/` |
| DeepLab | `02_.../front/outputs/deeplab/front_deeplab_v2/` | `02_.../side/outputs/deeplab/side_deeplab_wide_384x1536/` |
| SegFormer | `03_segformer_split/outputs/train/front_fold0/checkpoints/best.pt` + `eval/front_fold0_test/` | `.../side_fold0/...` + `eval/side_fold0_test/` |

CNN 可视化在各跑次 `test_vis/compare/`；SegFormer 在 `outputs/eval/*_fold0_test/compare/`（n=38）。
