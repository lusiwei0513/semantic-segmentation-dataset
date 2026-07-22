# 正侧视图数据域诊断报告

- 总样本数：378
- 正视图：189
- 侧视图：189
- 独立车辆数：189
- 设备数：1
- 设备与视图完全混淆：False

## 警告
- 未发现明显设备—视图完全混淆。

## 解释原则

- 正侧视图分类准确率高是正常现象，不能单独证明需要两个完整模型。
- 如果同一视图内部仍能高准确率预测设备，说明设备域差异较强。
- 如果设备与视图完全绑定，需要补充交叉设备数据，或用模型对照实验判断。
- 当前约 200 张数据时，默认仍推荐共享编码器 + 视图专用头。

## 自动结果

```json
{
  "n_samples": 378,
  "n_front": 189,
  "n_side": 189,
  "n_vehicles": 189,
  "n_devices": 1,
  "perfect_view_device_confounding": false,
  "warnings": [],
  "view_centroid_cosine_distance": 0.4957748055458069,
  "view_linear_mmd": 0.04426783695816994,
  "view_probe": {
    "accuracy": 1.0,
    "balanced_accuracy": 1.0,
    "n_splits": 5
  }
}
```