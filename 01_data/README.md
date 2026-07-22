# 01_data — 共享训练数据入口

本目录是**数据总入口**：大部分子目录是指向 `02_baselines_unet_deeplab/` 的 **junction（目录联接）**，保证基线工程里的相对路径 `../data`、`../data_unet` 仍可用，同时在这里集中说明「每份数据做什么用」。

**训练可用结构总览（推荐先看）→ [`训练数据文件结构.md`](./训练数据文件结构.md)**

官方对比划分：**`data/{front,side}/splits/fold_0.json`**（test 各 38 张）。

---

## 目录一览

```
01_data/
├── README.md                 ← 本说明
├── split_raw/                ← 真实目录：正/侧原始裁切图
├── data/                     ← junction → 02.../data
├── data_unet/                ← junction → 02.../data_unet
├── prepared_front/           ← junction → 02.../prepared_front
└── prepared_side/            ← junction → 02.../prepared_side
```

物理文件实际存放在：`02_baselines_unet_deeplab/` 下同名目录（`split_raw` 除外，在本目录）。

---

## 各子目录用途

### 1. `split_raw/` — 正/侧原始拆分图（未打包）

从正侧合图拆出的**原始 JPG/JPEG**，尚未做成训练用 mask 包。

| 子路径 | 内容 |
|--------|------|
| `split_raw/front/` | 正视图原图 |
| `split_raw/side_images/` | 侧视图原图 |

用途：溯源、重新标注、核对文件名；**主训练一般不直接读这里**。

---

### 2. `prepared_front/` / `prepared_side/` — LabelMe 转好的「图+独占 mask」包

从 LabelMe（或等价标注）导出的 **189 张** 就绪数据：

| 子路径 | 内容 |
|--------|------|
| `images/` | RGB 图 |
| `masks/` | 单通道类别 mask（像素类 id） |
| `overlays_preview/` 等 | 可视化预览（检查标注用） |
| `classes.json` | 类别名与 id |

- **正视类别**：background / body / windshield / **nose_tip**
- **侧视类别**：background / body / windshield / bogie / door

用途：早期「全量 189、随机 80/20」实验；正式对比请改用下面的 `data_unet` + `fold_0`。

---

### 3. `data_unet/` — 主实验用的独占 mask 包（UNet / DeepLab / 对比协议）

与 `prepared_*` 同类，但是**按视角整理、给主流配置引用**的版本：

| 子路径 | 内容 |
|--------|------|
| `front/` | 正视：`images/` + `masks/` + `classes.json` |
| `side/` | 侧视：同上 |
| `joint/` | 正+侧合集（6 类：含 nose_tip / bogie / door） |

主配置里常见：`prepared_dir: ../data_unet/front` 或 `../data_unet/side`。

---

### 4. `data/` — 处理后的图像、关键点、五折划分

更完整的「工程数据树」，正/侧分开：

```
data/
├── front/
│   ├── raw/                 # 可选：原始副本
│   ├── processed/
│   │   ├── images/          # 处理后的图（约 189）
│   │   ├── masks/           # 按类分子目录（body/windshield/...）
│   │   ├── keypoints/       # 正视车头尖点 JSON（UNet-KP 用）
│   │   └── metadata.csv
│   ├── splits/
│   │   ├── fold_0.json … fold_4.json   # 五折；官方对比用 fold_0
│   │   └── …
│   └── dataset_summary.json
├── side/                    # 结构同 front（侧视无 tip 关键点）
├── processed/               # 历史合集副本（正视图+关键点等，兼容旧脚本）
└── splits/                  # 旧联合划分（若存在）
```

#### `splits/fold_*.json` 是什么？

每个 fold 给出 `train` / `val` / `test` 的样本 id 列表。  
**定量对比统一用 `fold_0`，test=38。**

#### `processed/keypoints/` 是什么？

正视车头尖点坐标（JSON）。  
**UNet-KP**（分割头 + tip 热力图头）训练时读取：`keypoints_dir: ../data/front/processed/keypoints`。

---

## 数据是共享的（请勿误读）

本目录只放**数据**，不「归属」某个模型。  
UNet-KP、UNet-seg、DeepLab、SegFormer **共用**下面同一套路径；差别只在「读不读 keypoints」以及「结果写到哪里」。

| 视角 | 全体模型共用的数据路径 |
|------|------------------------|
| **正视** | `data_unet/front` + `data/front/splits/fold_0.json` |
| **侧视** | `data_unet/side` + `data/side/splits/fold_0.json` |
| 正视 tip 关键点 | `data/front/processed/keypoints`（**仅 UNet-KP / 带 tip heatmap 的头需要**） |
| SegFormer | 通过 `03_segformer_split` 的 junction 指向同一套 `data/*/processed` 与 `splits` |

### 模型怎么用这些数据？（不是「正视归 A、侧视归 B」）

| 模型 | 正视数据 | 侧视数据 | 说明 |
|------|----------|----------|------|
| **UNet-KP** | ✅ `data_unet/front` + keypoints + fold_0 | **N/A** | 正视专用鼻尖 heatmap；侧视无 KP 头 |
| **UNet-seg** | ✅ 同上（无需 keypoints） | ✅ `data_unet/side` + fold_0 | 正+侧都有 |
| **DeepLab** | ✅ | ✅ | 正+侧都有 |
| **SegFormer** | ✅（经 junction） | ✅（经 junction） | 正+侧都有 |

**训练/评测权重与可视化不在 `01_data/`**，而在各工程 `outputs/`。完整「模型 × 视图」结果路径见根目录 `README.md` §3。

---

## 注意

1. **不要只改 `01_data` 里的 junction 目标而不改 `02` 实体目录**，否则训练相对路径会断。
2. `data/processed` 与 `data/front/processed` 存在部分镜像内容，以 **`data/{front,side}/`** 为准。
3. 标注原始 LabelMe 图在 `02_baselines_unet_deeplab/front_images`、`side_images`（图+json 成对），不在本目录。
4. 旧版「任务→数据路径」表容易被理解成「一个模型管正视、另一个管侧视」——那是误解；数据按**视角**共享，结果按**模型×视角**分目录。
