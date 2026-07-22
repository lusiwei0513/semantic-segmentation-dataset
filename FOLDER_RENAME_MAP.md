# Folder rename map

整理日期：2026-07-21（含第二轮基线可读化、第三轮联合存档去 `.bb`）

## 1. 顶层工程目录（第一轮）

| Old | New |
|-----|-----|
| `baselines_unet_deeplab/` | `02_baselines_unet_deeplab/` |
| `baselines_unet_deeplab/gpt标注训练正视图-seg_train/` | `02_baselines_unet_deeplab/front/` |
| `baselines_unet_deeplab/cursor标注与训练第一版-ppt_seg_task/` | `02_baselines_unet_deeplab/side/` |
| `segformer_b0_split/` | `03_segformer_split/` |
| `bb/` | `04_archive_joint_segformer/` |
| `split_raw/` | `01_data/split_raw/` |
| `project13/` | `05_archive_misc/project13/` |
| `project13_split_labelme/` | `05_archive_misc/project13_split_labelme/` |
| `bb.zip` | `05_archive_misc/bb.zip` |
| `baselines_unet_deeplab/data_backup_20260720_010059/` | `05_archive_misc/data_backup_20260720_010059/` |
| `README_COMPARE.md`, legacy `readme.md`, annotation PPT | `docs/` |

## 2. 训练跑次 / 输出目录（第二轮，可读名）

### 正视 `front/outputs/`

| Old | New |
|-----|-----|
| `outputs/unet_resnet34_data_front_kp_tune_v2/` | `outputs/unet_kp/front_unet_kp_tune/` |
| `outputs/unet_resnet34_data_front_tune/` | `outputs/unet_seg/front_unet_seg_tune/` |
| `outputs/deeplab_resnet34_data_front_v2/` | `outputs/deeplab/front_deeplab_v2/` |

### 侧视（`outputs_side/` → 统一为 `outputs/`）

| Old | New |
|-----|-----|
| `outputs_side/unet_resnet34_data_side_tune_384x1536/` | `outputs/unet_seg/side_unet_wide_384x1536/` |
| `outputs_side/deeplab_resnet34_data_side_v2_384x1536/` | `outputs/deeplab/side_deeplab_wide_384x1536/` |
| `outputs_side/unet_resnet34_data_side_tune/` | `outputs/unet_seg/side_unet_square_384/` |
| `outputs_side/deeplab_resnet34_data_side_v2/` | `outputs/deeplab/side_deeplab_square_384/` |

### 主配置文件重命名

| Old | New |
|-----|-----|
| `front/config_data_front_kp_tune.yaml` | `front/config_front_unet_kp_tune.yaml` |
| `front/config_data_front_unet_tune.yaml` | `front/config_front_unet_seg_tune.yaml` |
| `front/config_data_front_v2.yaml` | `front/config_front_deeplab_v2.yaml` |
| `side/config_data_side_unet_tune_384x1536.yaml` | `side/config_side_unet_wide_384x1536.yaml` |
| `side/config_data_side_deeplab_384x1536.yaml` | `side/config_side_deeplab_wide_384x1536.yaml` |
| `side/config_data_side_unet_tune.yaml` | `side/config_side_unet_square_384.yaml` |
| `side/config_data_side_v2.yaml` | `side/config_side_deeplab_square_384.yaml` |

旧配置移至各自 `configs_legacy/`。  
顶层杂项（日志、rar、标注脚本等）→ `05_archive_misc/02_baselines_misc_20260721/`。

## 3. 联合训练存档第三轮（去掉 `.bb` + 任务化整理）

路径均相对 `04_archive_joint_segformer/`。

| Old | New |
|-----|-----|
| `.bb/`（整层嵌套） | 内容直接升至本目录根下（`.bb` 已删除） |
| `.bb/rail_vehicle_segformer/` | `rail_vehicle_segformer/` |
| `.bb/prepared_front` / `prepared_side` | `data_prepared/prepared_*` |
| `.bb/正侧视图模型拆分诊断包/` | `docs_diagnosis/` |
| `.bb/SegFormer-B0_实验准备包/` + `_exp_plan*` | `docs_experiment_plan/` |
| `.bb/SAV-main/` | `third_party/SAV-main/` |
| `.bb/补充+完善/` | `supplemental/补充+完善/` |
| `.bb/*.rar` / `*.zip` / X-AnyLabeling exe | `archive_bundles/` |
| `rail_vehicle_segformer/outputs/train/fold_0/` | `experiments/joint_fold0/` |
| `.../fold_boost/` | `experiments/joint_fold0_boost/` |
| `.../fold0_clean/` | `experiments/joint_fold0_clean/` |
| `.../fold0_clean_v2/` | `experiments/joint_fold0_clean_v2/` |
| `.../fold0_clean_v2_cont/` | `experiments/joint_fold0_clean_cont/` |
| `.../fold0_side_wide/` | `experiments/joint_side_wide/` |
| `.../fold0_side_wide_pause_ep2/` | `experiments/joint_side_wide_pause_ep2/` |

详见 `04_archive_joint_segformer/README.md`。

## 4. Junctions（发现用入口）

Under `01_data/`：

- `prepared_front` → `02_baselines_unet_deeplab/prepared_front`
- `prepared_side` → `02_baselines_unet_deeplab/prepared_side`
- `data` → `02_baselines_unet_deeplab/data`
- `data_unet` → `02_baselines_unet_deeplab/data_unet`

Under `03_segformer_split/`：

- `data_front` → `02_baselines_unet_deeplab/data/front/processed`
- `data_side` → `02_baselines_unet_deeplab/data/side/processed`
- `splits_front` → `02_baselines_unet_deeplab/data/front/splits`
- `splits_side` → `02_baselines_unet_deeplab/data/side/splits`

基线 YAML 仍用相对路径 `../data/...`、`../data_unet/...`（从 `front/` / `side/` 出发）。
