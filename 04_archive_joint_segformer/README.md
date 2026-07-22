# 04_archive_joint_segformer — 联合训练存档

**性质：历史联合训练存档（原 `bb/` → 曾误嵌套为 `.bb/`），非正式 fold_0 test 对比结果。**

- 训练方式：正视图 + 侧视图 **共用一个 SegFormer-B0**（交替 batch）
- 指标：val overall macro mIoU；**未跑官方 test 协议**
- 正式对比请用：`03_segformer_split/`（正/侧分开）与 `02_baselines_unet_deeplab/`

整理日期：2026-07-21（去掉 `.bb` 嵌套，按任务分目录）

## 目录结构

```
04_archive_joint_segformer/
  README.md                 # 本说明
  rail_vehicle_segformer/   # 主工程代码（configs / src / scripts / data）
  experiments/              # 各次联合训练跑次（权重 + 日志）
  docs_diagnosis/           # 正侧视图域差异诊断包
  docs_experiment_plan/     # SegFormer-B0 实验方案与启动提示
  data_prepared/            # 当时用的 prepared_front / prepared_side
  third_party/SAV-main/     # 第三方 SAV 代码（未用于本对比）
  supplemental/补充+完善/   # 补充标注 JSON 等
  archive_bundles/          # rar/zip、标注工具安装包、辅助脚本
```

## 训练任务一览（`experiments/`）

| 目录 | 原名 | 说明 |
|------|------|------|
| `joint_fold0/` | `fold_0` | 基线联合训练 fold0 |
| `joint_fold0_boost/` | `fold_boost` | door 等加权 / boost 版 |
| `joint_fold0_clean/` | `fold0_clean` | clean 数据配置 |
| `joint_fold0_clean_v2/` | `fold0_clean_v2` | clean v2 |
| `joint_fold0_clean_cont/` | `fold0_clean_v2_cont` | clean v2 续训 |
| `joint_side_wide/` | `fold0_side_wide` | 侧视宽画布（历史最佳 val≈0.837） |
| `joint_side_wide_pause_ep2/` | `fold0_side_wide_pause_ep2` | 同上，ep2 暂停存档 |

每个跑次典型内容：`checkpoints/best.pt`、`checkpoints/last.pt`、`history.json`、`train.log`、`visualizations/`。

> **可视化只有 front：** `train.py` 刷新 best 时只 dump 一张 front val（`epoch_*_front.png`）。训练/验证仍交替跑 side；`history.json` 含 `front_macro_miou` / `side_macro_miou`。导出侧视对比图见 `rail_vehicle_segformer/README.md`（`scripts/visualize_predictions.py`）。

## 路径变更影响（去掉 `.bb`）

旧路径形如：

- `D:\.bb\rail_vehicle_segformer\...`
- `04_archive_joint_segformer\.bb\...`

现已全部升到本目录下。相对路径若曾假设 cwd 在 `.bb` 内或 `../.bb/...`，需改为：

| 用途 | 新相对路径（从 `rail_vehicle_segformer/`） |
|------|-------------------------------------------|
| 原始 front | `../data_prepared/prepared_front` |
| 原始 side | `../data_prepared/prepared_side` |
| 某次权重 | `../experiments/joint_side_wide/checkpoints/best.pt` |

训练入口仍可用相对工程根：

```powershell
cd 04_archive_joint_segformer\rail_vehicle_segformer
python src\train.py --config configs\train.yaml --fold 0
# 续训示例（指向 experiments）
python src\train.py --config configs\train.yaml --fold 0 `
  --resume ..\experiments\joint_fold0\checkpoints\last.pt
```

默认 `train.py` 仍会写到 `outputs/train/...`；本存档的历史权重已迁至 `experiments/`，不再放在代码树内。

## 与正式对比的关系

| | 本存档 | `03_segformer_split` |
|--|--------|----------------------|
| 训练 | 正+侧联合 | 正/侧分开 |
| Test | 未跑 | fold_0 test 官方 mIoU |
| 用途 | 参考 / 复现旧实验 | 主对比表 |
