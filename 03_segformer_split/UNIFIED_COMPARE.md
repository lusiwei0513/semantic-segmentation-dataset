# 统一对比协议：UNet-KP vs SegFormer-B0（正视 tip）

## 1. `bb` 里的 SegFormer 是什么

路径：`04_archive_joint_segformer/rail_vehicle_segformer/`（原 `bb/`，已去掉中间 `.bb` 层；权重在 `experiments/`）

| 项 | 内容 |
|----|------|
| 训练方式 | **正+侧联合**一个共享模型（交替 front/side batch） |
| 指标 | val **overall macro mIoU**（body/ws/bogie/door 有效通道均值）；tip 为 **mean px error**（非官方椭圆 mIoU） |
| 最佳 val | fold0_side_wide：**0.837** overall；tip≈23 px（另一 run cont tip≈14） |
| Test | **未跑**；本机也无 `.pt` checkpoint |
| Tip 损失 | 均匀 MSE，`nose_tip=0.5`，`σ=4`，**无 peak-boost / 无坐标损失** |

**不能**直接和本机 UNet-KP / `03_segformer_split` 的 test official mIoU 横向比（协议、是否分开训练、分辨率、tip 定义都不同）。

## 2. 当前两边划分（本机正式对比用）

两边都应使用 **`02_baselines_unet_deeplab/data/{front,side}/splits/fold_0.json`**：

| 视图 | train | val | test |
|------|------:|----:|-----:|
| front | 133 | 18 | 38 |
| side | 133 | 18 | 38 |

- 按 **vehicle_id** 分组，避免同车泄漏
- **val**：早停 / 选 `best.pt`
- **test**：只评一次，写进报告

`bb` 联合划分是「正侧样本合计」train 266 / val 36 / test 76（车辆数同样 133/18/38），车辆集合与本机 fold 不一定逐 ID 一致，对比时以本机 `baselines` fold_0 为准。

## 3. 推荐统一设置（公平 + tip 更合理）

| 项 | UNet-KP（已有） | SegFormer（应对齐） |
|----|-----------------|---------------------|
| 训练 | 正 / 侧 **分开** | 正 / 侧 **分开**（`03_segformer_split`） |
| Front 分辨率 | 512 | 512 |
| Side 分辨率 | 384（letterbox 长边） | 384×1536 |
| Tip 评测 | peak→椭圆 rx24,ry12 + MAE/PCK | **同协议** |
| Tip 损失 | peak_boost=40 + coord L1=0.5 | **已实现**，配置见下 |
| 选模 | val official_mIoU | 同；辅看 tip_mae |

SegFormer tip 强化配置：`configs/train_front_peakboost.yaml`

```text
nose_tip: 2.0
heatmap_sigma: 16
heat_peak_boost: 40
tip_coord_weight: 0.5
```

旧基线 `train_front.yaml` 仍可复现（peak_boost=0）。

## 4. 训练命令

```powershell
$py = "..\02_baselines_unet_deeplab\front\.venv\Scripts\python.exe"
cd 03_segformer_split
& $py src\train.py --config configs\train_front_peakboost.yaml --view front --fold 0 `
  --output-dir outputs\train\front_fold0_peakboost
```

