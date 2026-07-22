# 04_joint_segformer

正/侧视图 **分开训练**（`02_baselines_unet_deeplab/`、`03_segformer_split/`）完成之后，进一步做的 **正侧联合多任务 / 共享编码器** 实验（SegFormer-B0，交替训练 front + side）。

## 目录结构

```text
04_joint_segformer/
├── README.md
├── rail_vehicle_segformer/   # 代码、配置、脚本；训练读 data/processed
├── experiments/             # 各次联合训练权重 / 日志 / 指标 / 可视化
└── data_prepared/           # 联合训练原始 prepared 数据（prepare 输入）
```

各目录内容如下：

- `rail_vehicle_segformer/`：联合训练代码、配置与脚本；训练实际读取 `rail_vehicle_segformer/data/processed/`（由 `data_prepared` 经 prepare 生成）。标签约定见 `rail_vehicle_segformer/LABEL_SPEC.md`。
- `experiments/`：各次训练的 `best.pt`、`history.json`、日志与可视化（PPT 用 val 指标来自 `history.json`）。
- `data_prepared/`：`prepared_front` / `prepared_side`，作为 prepare 流水线的只读输入（见 `configs/data.yaml` 的 `raw.*`）。

## `experiments` 目录

```text
experiments/
├── joint_fold0/                 # configs/train.yaml
├── joint_fold0_boost/           # configs/train_boost.yaml
├── joint_fold0_clean_v2/        # configs/train_clean.yaml
├── joint_fold0_clean_cont/      # configs/train_clean_cont.yaml
└── joint_side_wide/             # configs/train_side_wide.yaml
```

每个实验目录通常包含：

```text
checkpoints/best.pt
history.json
train.log
visualizations/          # 训练过程中按 epoch 落盘的 front 抽样
visualizations_side/     # best.pt 在侧视上的预测可视化
visualization_front/     # best.pt 在正视 test 上的预测可视化
```

```bash
# 正视 test（约 20 张）
python scripts/visualize_predictions.py \
  --config configs/train_side_wide.yaml \
  --checkpoint ../experiments/joint_side_wide/checkpoints/best.pt \
  --fold 0 --split test \
  --num-front 20 --num-side 0 \
  --out-dir ../experiments/joint_side_wide/visualization_front

# 侧视（与历史 visualizations_side 一致时可改 --split val）
python scripts/visualize_predictions.py \
  --config configs/train_side_wide.yaml \
  --checkpoint ../experiments/joint_side_wide/checkpoints/best.pt \
  --fold 0 --split val \
  --num-front 0 --num-side 20 \
  --out-dir ../experiments/joint_side_wide/visualizations_side
```

## 训练与数据
```bash
cd 04_joint_segformer/rail_vehicle_segformer
# 环境可用 02_baselines_unet_deeplab/front/.venv
python scripts/prepare_dataset.py --config configs/data.yaml
python scripts/create_group_folds.py --config configs/data.yaml --n-splits 5
python src/train.py --config configs/train.yaml --fold 0
```
