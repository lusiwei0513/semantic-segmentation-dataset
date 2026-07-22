# 轨道车辆正视图/侧视图部件识别实验方案
## 基于 SegFormer-B0 的训练前完整搭建计划

> 目标：让开发 Agent 按照本文档完成 **从项目初始化到正式训练开始前** 的全部准备工作。  
> 当前阶段只使用 **SegFormer-B0**，不执行正式训练，不追求最终指标。  
> 项目数据规模约 200 张，均为二维图像，包含正视图与侧视图。

---

# 1. 项目目标

建立一个面向轨道车辆二维图像的多任务视觉模型。

## 1.1 正视图任务

需要识别：

1. 车身主体掩码；
2. 前挡风玻璃掩码；
3. 车头最尖端位置。

其中：

- 车身主体和前挡风玻璃属于像素级分割任务；
- 车头最尖端属于关键点定位任务，不应作为普通语义分割类别。

## 1.2 侧视图任务

需要识别：

1. 车身主体掩码；
2. 转向架或车轮区域掩码；
3. 车门区域掩码；
4. 前挡风玻璃掩码。

## 1.3 当前实验目标

本阶段只完成：

- 数据格式确定；
- 数据检查；
- 数据划分；
- 数据加载器；
- 数据增强；
- SegFormer-B0 模型搭建；
- 多任务输出头搭建；
- 损失函数搭建；
- 评价指标搭建；
- 配置文件；
- 单元测试；
- 冒烟测试；
- 单 batch 前向与反向传播测试；
- 训练脚本能够正常启动。

本阶段 **不执行正式训练**。

---

# 2. 关键设计决定

## 2.1 使用一个共享模型，而不是正视图、侧视图分别训练两个模型

原因：

- 总数据量只有约 200 张；
- 正视图和侧视图分别建模会进一步减少样本；
- 两种视图仍共享大量底层视觉特征；
- 使用共享编码器可以提高小样本条件下的特征复用率。

模型结构：

```text
输入图像
   ↓
SegFormer-B0 / MiT-B0 共享编码器
   ↓
多尺度特征
   ├── 分割解码头
   │      ├── 车身主体
   │      ├── 前挡风玻璃
   │      ├── 转向架
   │      └── 车门
   │
   └── 关键点热图头
          └── 车头最尖端
```

## 2.2 分割采用多标签形式

推荐输出 4 个独立二值掩码：

```text
body
windshield
bogie
door
```

每个通道使用 Sigmoid，而不是所有类别共用 Softmax。

原因：

- “车身主体”可能与车门存在语义包含关系；
- 如果后续确定车身主体表示完整车体轮廓，车门像素可能同时属于 body 和 door；
- 多标签设计比互斥语义分割更灵活；
- 未标注类别可以通过 valid mask 屏蔽损失。

如果老师明确规定所有类别必须互斥，可以后续切换为 Softmax 方案；当前工程应优先实现多标签方案。

## 2.3 正视图和侧视图使用任务有效性掩码

不同视图只计算相应任务损失。

建议规则：

| 任务 | 正视图 | 侧视图 |
|---|---:|---:|
| body | 有效 | 有效 |
| windshield | 有效 | 有效 |
| bogie | 无效或可选 | 有效 |
| door | 无效或可选 | 有效 |
| nose_tip | 有效 | 无效 |

每个样本应包含：

```json
{
  "view": "front",
  "valid_tasks": {
    "body": true,
    "windshield": true,
    "bogie": false,
    "door": false,
    "nose_tip": true
  }
}
```

无效任务不得当作背景训练，而是完全不计算损失。

## 2.4 车头尖端使用热图预测

不直接回归 `(x, y)`。

处理方式：

1. 人工标注尖端像素坐标；
2. 根据坐标生成二维高斯热图；
3. 模型预测热图；
4. 取预测热图最大值作为尖端坐标。

推荐默认参数：

```yaml
heatmap_sigma: 4
heatmap_loss: mse
```

如果输入尺寸较大，可把 `sigma` 调整为 5～8。

---

# 3. 标签定义

在开始编码前，必须把标签标准写入项目文档，并保持所有图片一致。

## 3.1 body：车身主体

当前默认定义：

- 包含车体外部主体轮廓；
- 是否包含车门区域：允许包含；
- 是否包含挡风玻璃：默认不包含；
- 是否包含转向架：默认不包含；
- 是否包含车钩、雨刷、灯具、后视镜等附属部件：需要在项目 `LABEL_SPEC.md` 中逐项固定。

建议将完整定义写成：

```text
body 表示车辆主体外壳区域。
windshield、bogie、door 作为独立部件掩码。
body 是否覆盖 door 区域由当前标注数据决定；
工程按多标签输出设计，因此允许 body 与 door 重叠。
```

## 3.2 windshield：前挡风玻璃

需要明确：

- 只标玻璃本体；
- 不包含窗框；
- 不包含雨刷；
- 强反光区域仍属于玻璃；
- 被遮挡部分只标可见区域；
- 正视图和侧视图均使用同一标签名。

## 3.3 bogie：转向架或车轮区域

需要明确：

- 是标整个转向架总成，还是只标可见车轮；
- 侧裙遮挡后是否只标可见区域；
- 两个相邻车轮是否作为同一个连通区域；
- 是否包含悬挂、轴箱等部件。

默认建议：

```text
标注图像中可见的转向架及车轮整体区域，只标可见像素。
```

## 3.4 door：车门

需要明确：

- 标完整门板，还是只标门缝内区域；
- 车门玻璃是否属于 door；
- 推荐：门玻璃归入门区域还是不归入，需要统一。

默认建议：

```text
door 标注门板及明确属于车门的可见区域；
若车门上存在玻璃，而任务只关心前挡风玻璃，则车门玻璃可计入 door。
```

## 3.5 nose_tip：车头最尖端

标注格式：

```json
{
  "x": 812.0,
  "y": 436.0,
  "visible": true
}
```

定义必须明确：

- 尖端是车辆轮廓最前方的几何端点；
- 如果车钩比车体更突出，是否将车钩端点视为尖端；
- 如果图像透视导致多个候选点，采用统一判定规则；
- 如果尖端被遮挡，`visible=false`，该样本不计算关键点损失。

建议默认：

```text
nose_tip 表示车体主体轮廓最前方的端点，不包含车钩等可拆卸附属结构。
```

---

# 4. 数据目录规范

推荐项目目录：

```text
project_root/
├── configs/
│   ├── base.yaml
│   ├── data.yaml
│   ├── model_segformer_b0.yaml
│   └── train_smoke.yaml
├── data/
│   ├── raw/
│   │   ├── images/
│   │   └── annotations_original/
│   ├── processed/
│   │   ├── images/
│   │   ├── masks/
│   │   │   ├── body/
│   │   │   ├── windshield/
│   │   │   ├── bogie/
│   │   │   └── door/
│   │   ├── keypoints/
│   │   └── metadata.csv
│   └── splits/
│       ├── fold_0.json
│       ├── fold_1.json
│       ├── fold_2.json
│       ├── fold_3.json
│       └── fold_4.json
├── src/
│   ├── datasets/
│   │   ├── rail_vehicle_dataset.py
│   │   ├── transforms.py
│   │   └── heatmap.py
│   ├── models/
│   │   ├── segformer_multitask.py
│   │   ├── segmentation_head.py
│   │   └── keypoint_head.py
│   ├── losses/
│   │   ├── segmentation_losses.py
│   │   ├── keypoint_losses.py
│   │   └── multitask_loss.py
│   ├── metrics/
│   │   ├── segmentation_metrics.py
│   │   └── keypoint_metrics.py
│   ├── utils/
│   │   ├── seed.py
│   │   ├── visualization.py
│   │   ├── checkpoint.py
│   │   └── logger.py
│   ├── train.py
│   ├── validate.py
│   └── infer.py
├── scripts/
│   ├── prepare_dataset.py
│   ├── validate_annotations.py
│   ├── create_group_folds.py
│   ├── compute_dataset_stats.py
│   ├── visualize_samples.py
│   └── smoke_test.py
├── tests/
│   ├── test_dataset.py
│   ├── test_transforms.py
│   ├── test_model.py
│   ├── test_losses.py
│   └── test_metrics.py
├── outputs/
│   ├── logs/
│   ├── visualizations/
│   ├── checkpoints/
│   └── predictions/
├── requirements.txt
├── README.md
├── LABEL_SPEC.md
└── AGENT_PROGRESS.md
```

---

# 5. metadata.csv 规范

每张图片一行。

推荐字段：

```csv
sample_id,image_path,vehicle_id,view,width,height,body_mask,windshield_mask,bogie_mask,door_mask,keypoint_path
train_0001,images/train_0001.jpg,vehicle_001,front,1920,1080,masks/body/train_0001.png,masks/windshield/train_0001.png,,,keypoints/train_0001.json
train_0002,images/train_0002.jpg,vehicle_001,side,1920,1080,masks/body/train_0002.png,masks/windshield/train_0002.png,masks/bogie/train_0002.png,masks/door/train_0002.png,
```

要求：

- `sample_id` 唯一；
- `vehicle_id` 用于分组划分；
- `view` 只能取 `front` 或 `side`；
- 缺失且无效的任务路径留空；
- 掩码必须是单通道 PNG；
- 掩码像素只允许 0 和 255；
- 关键点保存为 JSON；
- 所有相对路径均以 `data/processed/` 为根目录。

---

# 6. 数据检查脚本

Agent 必须实现 `scripts/validate_annotations.py`，检查：

## 6.1 文件完整性

- 图像是否存在；
- 掩码是否存在；
- JSON 是否存在；
- 文件是否可读取；
- 是否存在重复 sample_id；
- 是否存在重复图像文件。

## 6.2 尺寸一致性

- 图像与所有有效掩码宽高完全一致；
- 关键点坐标必须落在图像范围内；
- `width`、`height` 与实际图像一致。

## 6.3 掩码合法性

- 掩码必须是单通道；
- 像素值只能是 `{0, 255}` 或 `{0, 1}`；
- 掩码非空；
- 面积占比异常样本需要报警；
- 连通区域数量异常需要记录，但不直接报错。

## 6.4 视图与任务一致性

- front 必须至少包含 body、windshield；
- front 若 `nose_tip.visible=true`，必须有合法坐标；
- side 必须至少包含 body、windshield、bogie、door；
- 无效任务不得被自动生成全零真值后参与损失。

## 6.5 输出报告

生成：

```text
outputs/data_validation_report.json
outputs/data_validation_report.md
```

报告应包括：

- 总样本数；
- 正视图数；
- 侧视图数；
- 独立车辆数；
- 每个任务有效样本数；
- 掩码面积占比统计；
- 错误列表；
- 警告列表。

只要存在严重错误，脚本以非零状态码退出。

---

# 7. 数据划分

## 7.1 使用五折分组交叉验证

使用 `vehicle_id` 作为 group。

必须满足：

- 同一车辆的所有图片只能出现在一个 fold；
- 尽量保持各 fold 的正视图/侧视图比例接近；
- 尽量保持不同车型或来源分布均衡；
- 不允许普通随机按图片切分。

建议使用：

```text
StratifiedGroupKFold
n_splits = 5
shuffle = true
random_state = 42
```

分层变量至少包含 `view`。

如果同一车辆只有一个视图，也仍按 vehicle_id 分组。

## 7.2 每折划分

每个 fold 文件包含：

```json
{
  "fold": 0,
  "train": ["sample_0001", "sample_0002"],
  "val": ["sample_0010"],
  "test": ["sample_0020"]
}
```

推荐流程：

- 外层 5 折产生 test；
- 在外层训练部分中，再按 vehicle_id 划分约 10%～15% 为 val；
- val 不得与 train 共享 vehicle_id。

## 7.3 固定随机种子

统一使用：

```yaml
seed: 42
```

脚本必须控制：

- Python random；
- NumPy；
- PyTorch CPU；
- PyTorch CUDA；
- DataLoader worker。

---

# 8. 输入尺寸与预处理

## 8.1 避免强制拉伸

正视图和侧视图长宽比差异较大，不得统一直接拉伸成正方形。

推荐：

### 正视图

```yaml
target_size: [640, 640]
resize_mode: letterbox
```

### 侧视图

```yaml
target_size: [512, 1024]
resize_mode: letterbox
```

如果显存不足，可暂时使用：

```yaml
front_size: [512, 512]
side_size: [384, 768]
```

## 8.2 batch 组织

推荐：

- 正视图和侧视图分别采样；
- 同一个 batch 内尺寸一致；
- 可以使用两个 DataLoader，训练时交替取 batch；
- 或使用自定义 batch sampler 按 view 分组。

训练前冒烟测试阶段使用：

```yaml
batch_size: 1
num_workers: 0
amp: true
```

正式训练时再调整。

## 8.3 归一化

使用 SegFormer ImageNet 预训练对应的归一化参数：

```yaml
mean: [0.485, 0.456, 0.406]
std: [0.229, 0.224, 0.225]
```

---

# 9. 数据增强

小数据集必须使用增强，但不能破坏轨道车辆结构。

## 9.1 允许的几何增强

- 水平翻转；
- 小角度旋转，建议 `[-5°, 5°]`；
- 小范围平移；
- 随机缩放；
- 轻度裁剪；
- letterbox 后的随机 padding。

注意：

- 水平翻转后关键点坐标必须同步变化；
- 所有掩码必须同步变换；
- 不使用上下翻转；
- 不使用大角度旋转；
- 不使用严重透视扭曲。

## 9.2 允许的光照与成像增强

- 亮度；
- 对比度；
- Gamma；
- 色温；
- 轻微饱和度；
- 高斯噪声；
- 轻微模糊；
- 轻微运动模糊；
- JPEG 压缩；
- 局部阴影；
- 局部高光。

## 9.3 挡风玻璃专项增强

可以加入轻度：

- 高光反射；
- 局部亮度梯度；
- 玻璃变暗；
- 蓝色或灰色轻微色偏。

增强必须只改变图像，不改变挡风玻璃掩码。

## 9.4 暂不使用

当前 baseline 阶段不使用：

- MixUp；
- CutMix；
- Mosaic；
- Copy-Paste；
- 强随机透视；
- 生成式数据增强。

这些内容后续作为独立消融实验。

---

# 10. 模型设计

## 10.1 编码器

使用：

```yaml
backbone: SegFormer MiT-B0
pretrained: ImageNet-1K
```

可以通过 Hugging Face Transformers、MMSegmentation 或 timm 实现，但整个项目只保留一种主实现，避免重复依赖。

推荐优先级：

1. Hugging Face `SegformerModel`；
2. MMSegmentation；
3. 自行复制官方实现。

若项目目标是快速、透明和便于 Agent 修改，推荐 Hugging Face 作为编码器，并自定义任务头。

## 10.2 分割头

输入 SegFormer 四级特征：

```text
C1, C2, C3, C4
```

处理：

1. 各层通过线性投影统一通道；
2. 上采样到最高分辨率特征尺度；
3. 拼接；
4. 通过卷积融合；
5. 输出 4 个分割 logits。

输出：

```text
[B, 4, H, W]
```

通道顺序固定：

```python
SEGMENTATION_CHANNELS = [
    "body",
    "windshield",
    "bogie",
    "door",
]
```

最终不在模型内部调用 Sigmoid，损失中使用 `BCEWithLogitsLoss`。

## 10.3 关键点热图头

输入优先使用高分辨率融合特征。

输出：

```text
[B, 1, Hk, Wk]
```

推荐输出分辨率为输入的 1/4。

标签热图同步生成相同尺寸。

预测坐标时：

1. 对热图取 Sigmoid；
2. 找最大值索引；
3. 映射回原始图像坐标；
4. 反向去除 letterbox padding 和缩放。

## 10.4 前向输出结构

模型必须返回字典：

```python
{
    "segmentation_logits": Tensor[B, 4, H, W],
    "nose_tip_heatmap_logits": Tensor[B, 1, Hk, Wk],
    "features": optional
}
```

---

# 11. 损失函数

## 11.1 分割损失

每个有效类别使用：

```text
BCEWithLogitsLoss + DiceLoss
```

默认：

```yaml
bce_weight: 1.0
dice_weight: 1.0
```

每个样本必须使用 task validity mask：

```python
valid_seg_tasks.shape == [B, 4]
```

如果某张正视图没有 bogie、door 标注，这两个通道不计算损失。

不得把缺失标注自动视为全背景。

## 11.2 类别不平衡

先实现可配置的 `pos_weight`，但默认不开启。

Agent 需要实现脚本统计各类正像素比例，输出建议权重。

如果挡风玻璃、转向架等类别面积过小，可在后续开启：

```yaml
use_pos_weight: true
```

## 11.3 关键点损失

默认：

```text
MSELoss(predicted_heatmap, gaussian_heatmap)
```

只对以下样本计算：

```text
view == front
and nose_tip.visible == true
```

## 11.4 总损失

```text
L_total
= λ_body × L_body
+ λ_windshield × L_windshield
+ λ_bogie × L_bogie
+ λ_door × L_door
+ λ_tip × L_tip
```

默认：

```yaml
loss_weights:
  body: 1.0
  windshield: 1.0
  bogie: 1.0
  door: 1.0
  nose_tip: 0.5
```

损失函数必须返回：

```python
{
    "loss_total": ...,
    "loss_body": ...,
    "loss_windshield": ...,
    "loss_bogie": ...,
    "loss_door": ...,
    "loss_nose_tip": ...
}
```

---

# 12. 评价指标

## 12.1 分割指标

分别统计：

- body IoU；
- windshield IoU；
- bogie IoU；
- door IoU；
- macro mIoU；
- Dice；
- Precision；
- Recall；
- Boundary F-score。

指标只在任务有效样本上计算。

必须分别输出：

```text
overall
front
side
```

## 12.2 关键点指标

至少包括：

- 平均像素误差；
- 中位像素误差；
- 归一化距离误差；
- PCK@0.02；
- PCK@0.05。

归一化建议使用车身包围框对角线：

```text
normalized_error
= EuclideanDistance(pred, gt)
  / body_bbox_diagonal
```

## 12.3 小数据统计方式

正式实验使用五折交叉验证，最终报告：

```text
mean ± std
```

当前训练前阶段只需确保所有指标函数通过单元测试。

---

# 13. 训练配置预设

虽然本阶段不执行正式训练，但必须准备可运行配置。

推荐初始配置：

```yaml
model:
  name: segformer_b0_multitask
  pretrained: true
  num_segmentation_channels: 4
  keypoint_head: true

data:
  front_size: [640, 640]
  side_size: [512, 1024]
  batch_size: 1
  num_workers: 4

optimizer:
  name: adamw
  backbone_lr: 1.0e-5
  head_lr: 1.0e-4
  weight_decay: 0.01

scheduler:
  name: cosine
  warmup_epochs: 5

training:
  epochs: 100
  amp: true
  gradient_accumulation_steps: 4
  early_stopping_patience: 15
  grad_clip_norm: 1.0
  seed: 42
```

## 13.1 分阶段微调策略

正式训练建议：

### 阶段 A

- 冻结编码器；
- 只训练分割头和关键点头；
- 约 10～20 epoch。

### 阶段 B

- 解冻 MiT-B0 后两级；
- 编码器使用较小学习率；
- 继续训练。

### 阶段 C

- 视验证集情况解冻全部编码器；
- 保持低学习率；
- 早停。

训练脚本应支持配置：

```yaml
freeze_backbone_epochs: 15
unfreeze_last_n_stages: 2
```

---

# 14. 显存策略

SegFormer-B0 优先使用 AMP。

训练前需要实现：

- 自动混合精度；
- 梯度累积；
- batch size 1；
- 可选梯度检查点；
- OOM 捕获和提示；
- 输出一次前向传播峰值显存。

默认测试尺寸：

```text
front: 512 × 512
side: 384 × 768
batch size: 1
```

如果显存允许，再改到正式尺寸。

---

# 15. 必须实现的脚本

## 15.1 prepare_dataset.py

功能：

- 根据原始标注导出标准二值 PNG；
- 导出关键点 JSON；
- 创建 metadata.csv；
- 不覆盖原始数据；
- 可重复执行；
- 记录转换日志。

## 15.2 validate_annotations.py

功能见第 6 节。

## 15.3 create_group_folds.py

功能：

- 按 vehicle_id 分组；
- 创建五折；
- 平衡 front/side；
- 检查数据泄漏；
- 输出 fold JSON 和统计报告。

## 15.4 compute_dataset_stats.py

输出：

- RGB 均值和标准差；
- 各类像素比例；
- 掩码面积分布；
- 图像尺寸分布；
- 长宽比分布；
- 正侧视图数量；
- 每个 vehicle_id 样本数。

## 15.5 visualize_samples.py

随机显示或保存：

- 原图；
- body 掩码叠加；
- windshield 掩码叠加；
- bogie 掩码叠加；
- door 掩码叠加；
- nose_tip 点；
- 增强前后对比；
- resize/letterbox 后结果。

至少保存 20 个样例到：

```text
outputs/visualizations/data_check/
```

## 15.6 smoke_test.py

必须完成：

1. 读取一张正视图；
2. 读取一张侧视图；
3. 构建一个 batch；
4. 实例化模型；
5. 执行前向传播；
6. 计算所有有效损失；
7. 执行反向传播；
8. 优化器执行一步；
9. 检查所有 loss 均为有限数；
10. 检查关键参数梯度不为 None；
11. 保存一次预测可视化；
12. 输出峰值显存。

---

# 16. 单元测试要求

## 16.1 数据集测试

- 正视图返回正确 task mask；
- 侧视图返回正确 task mask；
- 图像与掩码尺寸一致；
- 水平翻转后关键点坐标正确；
- letterbox 后关键点坐标正确；
- 空路径不会错误地生成负样本。

## 16.2 模型测试

- 前向输出 shape 正确；
- 不同输入尺寸均可运行；
- batch size 1 可运行；
- CPU 可运行；
- CUDA 可运行时能够运行；
- 模型参数量能被统计。

## 16.3 损失测试

- 无效任务不会贡献损失；
- 所有任务无效时有明确报错；
- 完美预测时 Dice loss 接近 0；
- 正负样本均可正常计算；
- loss 不产生 NaN。

## 16.4 指标测试

使用手工构造的小掩码验证：

- 完全一致时 IoU=1；
- 完全不相交时 IoU=0；
- ignore 样本不进入统计；
- 关键点误差计算正确。

---

# 17. 冒烟测试配置

创建：

```text
configs/train_smoke.yaml
```

内容建议：

```yaml
max_train_batches: 2
max_val_batches: 1
epochs: 1
batch_size: 1
num_workers: 0
front_size: [256, 256]
side_size: [256, 512]
amp: false
save_checkpoint: false
```

冒烟测试通过条件：

- 程序正常退出；
- 至少完成一次前向、反向和优化器更新；
- 无 NaN/Inf；
- 无尺寸错误；
- 无任务掩码错误；
- 输出日志；
- 输出一张预测叠加图。

---

# 18. 训练前验收标准

Agent 只有满足以下全部条件，才算完成本阶段。

## 18.1 数据

- [ ] metadata.csv 已生成；
- [ ] 所有图像可读取；
- [ ] 所有有效掩码可读取；
- [ ] 所有掩码为二值；
- [ ] 图像与掩码尺寸一致；
- [ ] 关键点坐标合法；
- [ ] 正侧视图标签规则正确；
- [ ] 数据检查报告无严重错误。

## 18.2 划分

- [ ] 已生成五折；
- [ ] train/val/test 不共享 vehicle_id；
- [ ] 每折正侧视图比例有统计；
- [ ] 有自动泄漏检查。

## 18.3 模型

- [ ] MiT-B0 ImageNet 预训练权重可加载；
- [ ] 4 通道分割头可运行；
- [ ] 关键点热图头可运行；
- [ ] 任意有效输入尺寸可前向；
- [ ] 输出 shape 有断言。

## 18.4 损失与指标

- [ ] 多标签 BCE + Dice 可运行；
- [ ] 任务有效性掩码可运行；
- [ ] 关键点热图损失可运行；
- [ ] IoU、Dice、Precision、Recall 可运行；
- [ ] 关键点误差和 PCK 可运行。

## 18.5 工程

- [ ] 所有单元测试通过；
- [ ] smoke_test.py 通过；
- [ ] 单 batch 反向传播成功；
- [ ] 日志正常；
- [ ] 可视化正常；
- [ ] README 提供运行命令；
- [ ] AGENT_PROGRESS.md 记录已完成、未完成和风险。

---

# 19. Agent 工作顺序

Agent 必须按以下顺序执行，不得一开始就写训练循环。

## 阶段 1：检查数据

1. 查看现有数据目录；
2. 确认图片和标注格式；
3. 生成 `LABEL_SPEC.md`；
4. 实现数据转换脚本；
5. 实现数据检查脚本；
6. 输出检查报告。

## 阶段 2：标准化数据

1. 创建 processed 目录；
2. 统一图片和掩码路径；
3. 生成 metadata.csv；
4. 生成关键点 JSON；
5. 可视化至少 20 张检查图。

## 阶段 3：数据划分

1. 读取 vehicle_id；
2. 创建五折；
3. 检查泄漏；
4. 输出各折统计。

## 阶段 4：DataLoader

1. 实现 Dataset；
2. 实现正侧视图尺寸策略；
3. 实现增强；
4. 实现任务有效性掩码；
5. 实现关键点热图；
6. 写数据单元测试。

## 阶段 5：模型

1. 加载 MiT-B0 预训练编码器；
2. 实现多尺度分割头；
3. 实现关键点头；
4. 实现前向输出字典；
5. 写模型单元测试。

## 阶段 6：损失和指标

1. BCE；
2. Dice；
3. task validity mask；
4. 关键点热图损失；
5. IoU、Dice、Precision、Recall；
6. 关键点误差和 PCK；
7. 写测试。

## 阶段 7：训练框架准备

1. 优化器；
2. 参数组学习率；
3. AMP；
4. 梯度累积；
5. 日志；
6. checkpoint；
7. 配置读取；
8. 固定随机种子。

## 阶段 8：验收

1. 运行全部测试；
2. 运行 smoke test；
3. 运行单 batch 反向；
4. 输出显存；
5. 输出预测可视化；
6. 更新 AGENT_PROGRESS.md；
7. 停止，不启动正式训练。

---

# 20. Agent 不得擅自做的事情

- 不得修改原始标注文件；
- 不得在未确认标签定义前自行猜测类别边界；
- 不得把缺失标注当作全背景；
- 不得随机按图片划分数据；
- 不得让同一 vehicle_id 同时出现在训练集和测试集；
- 不得直接把侧视图拉伸成正方形；
- 不得把车头尖端作为普通分割类别；
- 不得从随机权重训练编码器；
- 不得在 smoke test 未通过前启动正式训练；
- 不得在本阶段自动执行长时间训练；
- 不得以“代码能运行”代替数据可视化检查。

---

# 21. Agent 最终应交付的内容

```text
1. 完整项目代码
2. requirements.txt
3. README.md
4. LABEL_SPEC.md
5. AGENT_PROGRESS.md
6. 标准化 metadata.csv
7. 五折划分文件
8. 数据检查报告
9. 数据统计报告
10. 至少20张数据可视化
11. SegFormer-B0 多任务模型
12. 损失与指标实现
13. 单元测试
14. smoke test 日志
15. 单 batch 前向/反向日志
16. 峰值显存记录
17. 一张模型预测可视化
```

---

# 22. 推荐运行命令

Agent 应在 README 中保证以下命令可用：

```bash
python scripts/prepare_dataset.py --config configs/data.yaml

python scripts/validate_annotations.py --config configs/data.yaml

python scripts/compute_dataset_stats.py --config configs/data.yaml

python scripts/create_group_folds.py --config configs/data.yaml --n-splits 5

python scripts/visualize_samples.py --config configs/data.yaml --num-samples 20

pytest -q

python scripts/smoke_test.py --config configs/train_smoke.yaml
```

正式训练命令可以准备，但本阶段不执行：

```bash
python src/train.py --config configs/base.yaml --fold 0
```

---

# 23. 当前推荐 baseline 总结

```text
模型：
SegFormer-B0 / MiT-B0 ImageNet 预训练

输入：
正视图 letterbox 到 640×640
侧视图 letterbox 到 512×1024

输出：
4 个多标签分割通道
1 个车头尖端热图

分割任务：
body
windshield
bogie
door

关键点任务：
nose_tip

损失：
BCEWithLogits + Dice
关键点 MSE 热图损失
任务有效性掩码

评估：
五折 StratifiedGroupKFold
按 vehicle_id 分组
分别统计 front、side 和 overall

本阶段终点：
全部数据、模型、损失、指标、测试和训练框架准备完成；
smoke test 与单 batch 反向传播成功；
不执行正式训练。
```
