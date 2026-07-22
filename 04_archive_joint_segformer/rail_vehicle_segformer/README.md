# 轨道车辆正/侧视图多任务识别（SegFormer-B0）

本仓库按《轨道车辆正侧视图_SegFormer-B0_实验方案》完成**训练前工程准备**。  
当前阶段**不执行正式训练**；通过数据检查、单元测试与 smoke test 后停止。

## 环境

```bash
cd 04_archive_joint_segformer/rail_vehicle_segformer
pip install -r requirements.txt
# 若出现 NumPy 2.x 与 pandas 不兼容：pip install "numpy>=1.26,<2"
```

> 本目录属于**联合训练历史存档**（非正式 fold_0 test 对比）。总览见上级 `../README.md`。

## 推荐运行顺序

```bash
cd 04_archive_joint_segformer/rail_vehicle_segformer

python scripts/prepare_dataset.py --config configs/data.yaml

python scripts/validate_annotations.py --config configs/data.yaml

python scripts/compute_dataset_stats.py --config configs/data.yaml

python scripts/create_group_folds.py --config configs/data.yaml --n-splits 5

python scripts/visualize_samples.py --config configs/data.yaml --num-samples 20

pytest -q

python scripts/smoke_test.py --config configs/train_smoke.yaml
```

正式训练：

```bash
# 默认 fold 0（新跑次会写到 outputs/train/；历史权重在 ../experiments/）
python src/train.py --config configs/train.yaml --fold 0

# 显存不够时，可先改 configs/train.yaml：
#   front_size: [512, 512]
#   side_size: [384, 768]
#   amp: true

# 从历史存档断点继续
python src/train.py --config configs/train.yaml --fold 0 --resume ../experiments/joint_fold0/checkpoints/last.pt

# 五折依次训练
python src/train.py --config configs/train.yaml --fold 0
python src/train.py --config configs/train.yaml --fold 1
python src/train.py --config configs/train.yaml --fold 2
python src/train.py --config configs/train.yaml --fold 3
python src/train.py --config configs/train.yaml --fold 4
```

历史训练产物（已整理）：

```text
../experiments/joint_fold0/          # 原 outputs/train/fold_0
../experiments/joint_fold0_boost/
../experiments/joint_side_wide/      # 历史最佳 val
...
├── train.log
├── history.json
├── checkpoints/best.pt
├── checkpoints/last.pt
└── visualizations/epoch_*_front.png  # 仅 front（见下）
```

### 为何 `experiments/*/visualizations/` 只有 front？

联合训练本身**交替使用 front + side**（`train.py` 里 `front_loader`/`side_loader` 各一步；`validate()` 也会分别跑两边并写入 `front_macro_miou` / `side_macro_miou`）。  
但刷新 best 时只抽样 **一张 front val** 落盘（`next(iter(front_val))` → `epoch_XXX_front.png`），属于可视化便捷采样，**不等于没训侧视**。

需要正/侧对比图时，用已有脚本从 `best.pt` 导出（默认各 6 张）：

```bash
cd 04_archive_joint_segformer/rail_vehicle_segformer
python scripts/visualize_predictions.py \
  --config configs/train_side_wide.yaml \
  --checkpoint ../experiments/joint_side_wide/checkpoints/best.pt \
  --fold 0 --split val \
  --num-front 6 --num-side 6 \
  --out-dir outputs/visualizations/best_preds_side_wide
```

## 数据约定

- 原始数据只读：`../data_prepared/prepared_front` / `../data_prepared/prepared_side`
- 处理后数据：`data/processed/`
- 标签说明与未决问题：见 `LABEL_SPEC.md`
- 进度：见 `AGENT_PROGRESS.md`

## 模型

- 主干：HuggingFace `nvidia/mit-b0`（ImageNet 预训练）
- 分割：4 通道多标签（body / windshield / bogie / door）+ Sigmoid/BCEWithLogits
- 关键点：nose_tip 高斯热图（仅正视图有效）
- 无效任务：`valid_*` mask 屏蔽，不当作背景
