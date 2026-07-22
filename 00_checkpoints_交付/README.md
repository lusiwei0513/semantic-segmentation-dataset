# Checkpoint 说明

这里整理了本次实验使用的模型权重。  
如果只需要进行新图推理，使用 `01_主模型` 中的两个模型即可。

## 1. 主模型

目录：`01_主模型/`

### 正视模型

文件：

`front_UNetKP_R34_fold0_best.pt`

- 模型：UNet，ResNet34 编码器
- 任务：分割 + 鼻尖关键点定位
- 输入尺寸：512 × 512，采用 letterbox 预处理
- fold 0 测试结果：
  - mIoU：0.780
  - 鼻尖定位 MAE：约 6.3 像素

### 侧视模型

文件：

`side_UNet_R34_wide384x1536_fold0_best.pt`

- 模型：UNet，ResNet34 编码器
- 任务：侧视图分割
- 输入尺寸：384 × 1536，采用 letterbox 预处理
- fold 0 测试结果：
  - mIoU：0.810

两个模型旁边均保存了对应的 `test_report.json`，其中包含具体测试指标。

## 2. 正视模型对比

目录：`02_正式对比_正视_fold0/`

包含以下模型：

- `front_UNetKP_R34_best.pt`：UNet-R34，分割和关键点联合训练
- `front_UNetSeg_R34_best.pt`：UNet-R34，仅进行分割
- `front_DeepLabV3Plus_R34_best.pt`：DeepLabV3+
- `front_SegFormer_B0_best.pt`：SegFormer-B0

这些权重用于正视任务的模型对比。

## 3. 侧视模型对比

目录：`03_正式对比_侧视_fold0/`

包含以下模型：

- `side_UNet_R34_wide384x1536_best.pt`
- `side_DeepLabV3Plus_R34_wide384x1536_best.pt`
- `side_SegFormer_B0_best.pt`

以上模型均对应宽幅输入设置。

## 4. 联合训练（非正式主表）

目录：`04_联合训练_非正式主表/`

正/侧分训之后，进一步做的共享编码器、正侧交替联合训练权重，包括：

- `joint_fold0*.pt`
- `joint_side_wide_best.pt`

这部分实验只记录了验证集结果，没有在与正式对比完全相同的 fold 0 测试流程下重新评估，因此不放入最终测试结果进行横向比较。