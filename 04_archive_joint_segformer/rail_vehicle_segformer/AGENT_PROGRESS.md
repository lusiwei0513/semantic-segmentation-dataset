# AGENT_PROGRESS

## 当前阶段

**阶段 8 完成 — 训练前工程准备已验收。按实验方案停止，不启动正式训练。**

## 各阶段完成项

### 阶段 1：检查数据
- [x] 确认 `prepared_front` / `prepared_side` 为互斥 ID 掩码
- [x] 撰写 `LABEL_SPEC.md`（含 Q1–Q4 未决问题）
- [x] 创建项目骨架 `04_archive_joint_segformer/rail_vehicle_segformer`

### 阶段 2：标准数据
- [x] `scripts/prepare_dataset.py`：复制图像、导出二值掩码、质心关键点 JSON、metadata.csv
- [x] 跳过侧视 body 缺失 5 张（不写入 metadata）
- [x] `validate_annotations.py`：**错误=0，警告=0**
- [x] 可视化 20 张 → `outputs/visualizations/data_check/`

### 阶段 3：划分
- [x] `create_group_folds.py`：StratifiedGroupKFold + vehicle_id，五折无泄漏
- [x] `compute_dataset_stats.py` → `outputs/dataset_stats.json`

### 阶段 4–7：工程
- [x] Dataset / letterbox（正 640×640、侧 512×1024；smoke 用更小尺寸）/ 增强 / valid mask / 高斯热图
- [x] MiT-B0（`nvidia/mit-b0`）+ 4 通道分割头 + nose_tip 热图头
- [x] BCE+Dice（按通道 valid mask）+ 热图 MSE
- [x] IoU/Dice/P/R + 关键点误差/PCK
- [x] 优化器分组 LR、AMP 开关、日志、checkpoint、seed、train 入口占位（禁止正式训练）

### 阶段 8：验收
- [x] `pytest -q`：**11 passed**
- [x] `smoke_test.py`：**PASS**（front+side 前向/损失/反向/优化器）
- [x] 预测图：`outputs/visualizations/smoke/front_pred.png`、`side_pred.png`
- [x] 日志：`outputs/logs/smoke_test.log`、`smoke_test_summary.json`

## Smoke 摘要

| 项 | 结果 |
|---|---|
| device | cpu（本机当时无 CUDA） |
| front loss_total | ≈ 3.17（bogie/door=0，已屏蔽） |
| side loss_total | ≈ 6.52（nose_tip=0，已屏蔽） |
| 显存 | CPU 运行，无 CUDA 峰值记录 |

## 数据规模

- metadata：373（front 189 + side 184）
- 跳过：5 张 side（body 缺失）
- nose_tip 有效：185；windshield 有效：360

## 问题 / 假设

1. body 与 door 等保持原始挖空互斥（LABEL_SPEC Q1）
2. nose_tip 用圆盘质心（Q2）
3. 缺失次要类用 valid mask，不删图（Q3/Q4）
4. 原始 `prepared_*` 未修改

## 下一步（需人工确认后）

1. 确认 LABEL_SPEC 未决问题
2. 有 GPU 时重跑 smoke 并记录显存
3. 再启动正式训练：`python src/train.py --config configs/base.yaml --fold 0`
