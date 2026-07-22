# 标签规范 LABEL_SPEC

> 本文档记录当前数据集标签定义、已知事实、假设，以及未决问题。  
> 未决问题不得自行猜测后写入训练真值逻辑；相关转换仅在「已确认/已记录假设」范围内进行。

---

## 1. 数据来源（只读，不修改）

| 视图 | 原始目录 | 图像数 | 掩码格式 |
|---|---|---:|---|
| front | `../data_prepared/prepared_front` | 189 | 单通道互斥 ID PNG |
| side | `../data_prepared/prepared_side` | 189 | 单通道互斥 ID PNG |

原始 `classes.json`：

**front**

| ID | 名称 |
|---:|---|
| 0 | background |
| 1 | body |
| 2 | windshield |
| 3 | nose_tip（半径约 16px 圆盘，非单点） |

绘制顺序：body → windshield → nose_tip（后画覆盖先画，车窗与尖点会挖空车身）。

**side**

| ID | 名称 |
|---:|---|
| 0 | background |
| 1 | body |
| 2 | windshield |
| 3 | bogie |
| 4 | door |

绘制顺序：body → windshield → door → bogie（后画覆盖先画，车身被挖空）。

---

## 2. 本项目目标任务

### 2.1 正视图（front）

| 任务 | 类型 | 是否有效 |
|---|---|---|
| body | 多标签分割通道 | 有效（若掩码中存在该类像素） |
| windshield | 多标签分割通道 | 有效（若掩码中存在该类像素） |
| bogie | — | **永久无效**（`valid=false`，不参与损失） |
| door | — | **永久无效** |
| nose_tip | 高斯热图关键点 | 有效当且仅当圆盘像素存在；否则 `visible=false` |

### 2.2 侧视图（side）

| 任务 | 类型 | 是否有效 |
|---|---|---|
| body | 多标签分割通道 | 有效（若存在像素） |
| windshield | 多标签分割通道 | 有效（若存在像素） |
| bogie | 多标签分割通道 | 有效（若存在像素） |
| door | 多标签分割通道 | 有效（若存在像素） |
| nose_tip | — | **永久无效** |

分割输出固定 4 通道顺序：

```text
[body, windshield, bogie, door]
```

使用 Sigmoid / BCEWithLogits，**不使用互斥 Softmax**。

---

## 3. 已确认的转换规则（可执行）

1. **不修改** `prepared_front` / `prepared_side` 原始文件。
2. 从互斥 ID 掩码导出各通道二值 PNG（0/255）：`channel = (mask == class_id)`。
3. 因原标注为「后画挖空」，当前各分割通道在像素级**互不重叠**；多标签框架仍保留，以便未来允许重叠。
4. `nose_tip`：对 ID=3 像素求质心 `(x, y)`（OpenCV 坐标：x=列，y=行），写入 JSON：
   ```json
   {"x": 812.0, "y": 436.0, "visible": true, "source": "disk_centroid", "n_pixels": 797}
   ```
   训练时由质心生成高斯热图，**不直接回归坐标**。
5. 某通道像素数为 0 → 该任务 `valid=false`，metadata 对应路径留空，**不得写入全零掩码参与损失**。
6. `vehicle_id`：取文件名中 UUID 后、视图后缀前的车型字符串（与 front/side 配对共享同一 UUID 前缀作为 `pair_id`）。

---

## 4. 当前数据中的缺失统计（检查结果）

| 视图 | 缺失类 | 样本数 |
|---|---|---:|
| front | windshield | 1 |
| front | nose_tip | 4 |
| side | body | 5 |
| side | windshield | 16 |
| side | bogie | 4 |
| side | door | 13 |

处理原则（与实验方案 2.3 / 6.4 对齐）：

- 缺失次要任务 → `valid_tasks[task]=false`，metadata 对应路径留空；
- **禁止**把缺失任务写成全零真值再算 loss；
- **body 缺失**的 5 张侧视样本：在 `prepare_dataset` 中 **SKIP**，不写入 metadata（最终 373 条）；
- 其余缺失类样本保留，依赖 valid mask。

---

## 5. 未决问题（暂停猜测）

以下问题在老师/标注方确认前，**不改变**当前「按原始互斥 ID 导出通道」的转换逻辑：

### Q1. body 是否应覆盖 door / windshield / bogie 区域？

- 现状：prepared 数据为挖空互斥，body 不包含这些区域。
- 实验方案允许多标签重叠，但未规定必须回填。
- **当前假设**：保持原始语义，不回填。

### Q2. nose_tip 原为半径 16px 圆盘，质心是否等于标注意图中的尖端点？

- **当前假设**：使用圆盘质心；JSON 中记录 `source=disk_centroid`。
- 若后续提供真实点标注，应替换 JSON 并重跑 prepare。

### Q3. 侧视「必须包含 body/windshield/bogie/door」与实际缺失样本如何取舍？

- **当前假设**：按样本级 valid mask 处理，不删除整图；缺失记警告；body 缺失记错误级警告。

### Q4. 正视 windshield 缺失的 1 张样本是否应剔除？

- **当前假设**：保留，`windshield` 无效。

---

## 6. 视图 × 任务有效性矩阵（默认）

| 任务 | front | side |
|---|---|---|
| body | 视像素存在 | 视像素存在 |
| windshield | 视像素存在 | 视像素存在 |
| bogie | 永远 false | 视像素存在 |
| door | 永远 false | 视像素存在 |
| nose_tip | 视圆盘存在 | 永远 false |
