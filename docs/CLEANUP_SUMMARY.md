# Cleanup summary (2026-07-21)

面向用户的「删了什么 / 留了什么」说明。详细机器日志：`05_archive_misc/cleanup_logs_20260721/`。

---

## 一句话结论

**没有删掉任一主对比模型「仅剩的正视」或「仅剩的侧视」主跑次。**  
UNet-seg / DeepLab / SegFormer 的正视+侧视 `best.pt` + `test_report.json` + 可视化仍在；UNet-KP 本来就只有正视。

---

## 删除了什么

### 1. SegFormer 额外 tip 实验（约 600MB）— 非主对比

| 类型 | 旧路径（清理前） |
|------|------------------|
| tip 消融训练 | `segformer_b0_split/outputs/train/front_fold0_tip_a{1,2,3}_*` |
| peakboost 训练 | `.../front_fold0_peakboost` |
| 对应 test eval | `outputs/eval/front_fold0_tip_a*_test` |

主对比仍保留：`front_fold0` / `side_fold0` 的 `best.pt` 与 `eval/*_fold0_test`。

### 2. 未进官方主表的旧 CNN 整树（过时实验）

| 旧跑次（已删整目录） | 为何删 |
|----------------------|--------|
| 侧视 `unet_resnet34_data_side`、`unet_resnet34_side_all189` | 早期 / 全量 189，非 fold_0 主表 |
| 正视 `unet_resnet34_data_front`、`unet_resnet34_front_all189` | 同上 |
| 正视旧 KP：`..._front_kp`、`..._front_kp_v2`、`..._kp_tune`（无 v2 后缀） | 已被主表 `kp_tune_v2`（现名 `front_unet_kp_tune`）取代 |
| 空目录 `deeplab_seresnet50_data_front_v2` | 无有效权重 |

这些**不是**当前主表里的 UNet-seg / DeepLab / KP 正式跑次。

### 3. 主跑次里只删「冗余文件」，权重保留

对**主对比跑次**（见下「保留」）：

- 删除各目录下的 `last.pt`（断点续训用）
- **保留** 同目录 `best.pt`
- 删除 TensorBoard `tb/`、`__pycache__`

---

## 保留了什么（主对比完整清单）

| 模型 | 正视 | 侧视宽 ★ | 侧视方图 |
|------|------|----------|----------|
| UNet-KP | ✅ `front_unet_kp_tune` | N/A（设计如此） | N/A |
| UNet-seg | ✅ `front_unet_seg_tune` | ✅ `side_unet_wide_384x1536` | ✅ square |
| DeepLab | ✅ `front_deeplab_v2` | ✅ `side_deeplab_wide_384x1536` | ✅ square |
| SegFormer | ✅ `front_fold0` + `eval/front_fold0_test` | ✅ `side_fold0` + `eval/side_fold0_test` | — |

路径前缀：`02_baselines_unet_deeplab/{front,side}/outputs/...` 或 `03_segformer_split/outputs/...`。  
更完整的路径表见根目录 `README.md` §3。

---

## 归档（未删除，只是搬走）

| 位置 | 内容 |
|------|------|
| `04_archive_joint_segformer/` | 原 `bb` 联合训练存档（第三轮已去掉 `.bb` 嵌套；**非本机主对比**） |
| `05_archive_misc/` | project13、bb.zip、data_backup、cleanup logs |
| `05_archive_misc/02_baselines_misc_20260721/` | 基线顶层日志 / rar / 标注脚本 |

第二轮仅**重命名**输出目录（如 `unet_resnet34_data_front_kp_tune_v2` → `outputs/unet_kp/front_unet_kp_tune`），见 `FOLDER_RENAME_MAP.md` §2。

---

## 是否需要从 archive 恢复？

**不需要。** 清点结果：主对比所需的正视+侧视权重与报告均在工作区活动路径中；archive 中也没有「被误删的唯一正/侧主跑次」需要捞回。  
若只要 tip 消融曲线，那些目录已永久删除（未进 archive），只能重训——本任务约定不重训。
