# 02_baselines_unet_deeplab — UNet / DeepLab 基线

正视图、侧视图的 **UNet-ResNet34** 与 **DeepLabV3+-ResNet34** 训练与评测工程。  
整理后：**代码按视角分 `front/` / `side/`，权重按模型族分子目录，名称可读。**

---

## 顶层结构

```
02_baselines_unet_deeplab/
├── README.md                 ← 本说明
├── data/                     ← 处理后数据 + fold 划分 + 关键点（实体）
├── data_unet/                ← 主实验独占 mask（front / side / joint）
├── prepared_front/           ← LabelMe 转好的正视包（189）
├── prepared_side/            ← LabelMe 转好的侧视包（189）
├── front_images/             ← 正视 LabelMe 原图+json
├── side_images/              ← 侧视 LabelMe 原图+json
├── front/                    ← 正视训练工程
│   ├── .venv/                ← 共用 CUDA Python（torch 2.6+cu124）
│   ├── config_front_*.yaml   ← 主实验配置（仅 3 个）
│   ├── configs_legacy/       ← 旧配置（joint / smoke / all189 等）
│   ├── scripts/  src/
│   └── outputs/
│       ├── unet_kp/front_unet_kp_tune/     ★ 主结果
│       ├── unet_seg/front_unet_seg_tune/
│       └── deeplab/front_deeplab_v2/
└── side/                     ← 侧视训练工程
    ├── config_side_*.yaml    ← 主实验配置（宽画布 + 方图旧基线）
    ├── configs_legacy/
    ├── scripts/  src/
    └── outputs/              ← 已统一为 outputs/（不再用 outputs_side）
        ├── unet_seg/side_unet_wide_384x1536/   ★ 主结果
        ├── deeplab/side_deeplab_wide_384x1536/  ★ 主结果
        ├── unet_seg/side_unet_square_384/      （旧 384 方图，仅参考）
        └── deeplab/side_deeplab_square_384/
```

杂项日志 / rar / 标注流水线脚本已移至：  
`05_archive_misc/02_baselines_misc_20260721/`。

---

## Python 环境

**唯一推荐 venv：**

```
02_baselines_unet_deeplab/front/.venv/Scripts/python.exe
```

侧视训练也用同一解释器（不要再找 `seg_train\.venv`）。

```powershell
$py = ".\02_baselines_unet_deeplab\front\.venv\Scripts\python.exe"
```

---

## 数据共享说明

`data_unet/{front,side}` 与 `data/{front,side}/splits/fold_0.json` 由 **UNet-KP / UNet-seg / DeepLab 共用**（SegFormer 也读同一套，经 junction）。  
**不是**「正视数据归某个模型、侧视数据归另一个模型」。  
结果目录才按模型分开；完整矩阵见根 `README.md` §3。

---

## 主实验对照（fold_0 test）— 按模型列出正视与侧视

### 每个模型的正视结果（`front/`，512×512）

| 模型 | 配置 | 输出目录（含 best.pt / test_report / test_vis） |
|------|------|-----------------------------------------------|
| **UNet-KP** | `config_front_unet_kp_tune.yaml` | `front/outputs/unet_kp/front_unet_kp_tune/` |
| **UNet-seg** | `config_front_unet_seg_tune.yaml` | `front/outputs/unet_seg/front_unet_seg_tune/` |
| **DeepLab** | `config_front_deeplab_v2.yaml` | `front/outputs/deeplab/front_deeplab_v2/` |

正视数据（共用）：`../data_unet/front` + `../data/front/splits/fold_0.json`；  
KP 额外：`../data/front/processed/keypoints`。

### 每个模型的侧视结果（`side/`）

| 模型 | 变体 | 配置 | 输出目录 |
|------|------|------|----------|
| **UNet-KP** | — | — | **N/A**（侧视无鼻尖 KP 头；侧视请用 UNet-seg） |
| **UNet-seg** | 宽 ★ 主对比 | `config_side_unet_wide_384x1536.yaml` | `side/outputs/unet_seg/side_unet_wide_384x1536/` |
| **UNet-seg** | 方图参考 | `config_side_unet_square_384.yaml` | `side/outputs/unet_seg/side_unet_square_384/` |
| **DeepLab** | 宽 ★ | `config_side_deeplab_wide_384x1536.yaml` | `side/outputs/deeplab/side_deeplab_wide_384x1536/` |
| **DeepLab** | 方图参考 | `config_side_deeplab_square_384.yaml` | `side/outputs/deeplab/side_deeplab_square_384/` |

侧视数据（共用）：`../data_unet/side` + `../data/side/splits/fold_0.json`。

每个主跑次保留：`best.pt`、`test_report.json`、`history.json`、`test_vis/`。

---

## 三种模型怎么区分？

1. **UNet-KP（仅正视）**  
   分割头（bg/body/windshield）+ tip 高斯热力图；评测时 peak→椭圆并入 nose_tip，另报 tip MAE / PCK。  
   **侧视没有对应 KP 跑次**——侧视对比用 UNet-seg。

2. **UNet-seg（正视 + 侧视）**  
   多类像素分割；正视 tip 为椭圆 GT 像素类，无独立关键点头。侧视主对比为宽画布 384×1536。

3. **DeepLabV3+（正视 + 侧视）**  
   与 UNet-seg 同数据、同 fold，作架构对照；正+侧均有主跑次。

---

## 常用命令

```powershell
$py = ".\02_baselines_unet_deeplab\front\.venv\Scripts\python.exe"

# 正视 KP 评测可视化
cd .\02_baselines_unet_deeplab\front
& $py scripts\export_test_vis.py --config config_front_unet_kp_tune.yaml `
  --run outputs\unet_kp\front_unet_kp_tune

# 侧视宽画布 UNet
cd ..\side
& $py scripts\export_test_vis.py --config config_side_unet_wide_384x1536.yaml `
  --run outputs\unet_seg\side_unet_wide_384x1536
```

重新训练时，脚本会把权重写到配置里的 `paths.output_dir` / `paths.run_name` 下。

---

## 相对路径约定（勿随意挪 data）

从 `front/` 或 `side/` 出发：

| 配置字段 | 典型值 |
|----------|--------|
| `prepared_dir` | `../data_unet/front` 或 `../data_unet/side` |
| `fold_json` | `../data/front/splits/fold_0.json` 等 |
| `keypoints_dir` | `../data/front/processed/keypoints` |

`01_data/` 下同名目录是 junction，改实体请改本目录的 `data*` / `prepared_*`。
