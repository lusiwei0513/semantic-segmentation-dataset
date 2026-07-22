# 01_data 数据说明

本目录保存训练数据及其说明。当前正视和侧视数据各 189 张，本次模型对比统一使用 `fold_0`：

- train：133
- val：18
- test：38

其中 `data`、`data_unet`、`prepared_front` 和 `prepared_side` 是目录联接，实体文件位于 `02_baselines_unet_deeplab/` 下的同名目录。`labelme_workspace` 和 `split_raw` 保存在本目录中。

## 目录结构

```text
01_data/
├── data/
├── data_unet/
├── labelme_workspace/
├── prepared_front/
├── prepared_side/
├── split_raw/
└── README.md
```

## 各目录用途

### `data_unet/`

训练时主要读取的图像和类别 mask。

```text
data_unet/
├── front/
│   ├── images/
│   ├── masks/
│   └── classes.json
└── side/
    ├── images/
    ├── masks/
    └── classes.json
```

正视类别为：

```text
background, body, windshield, nose_tip
```

侧视类别为：

```text
background, body, windshield, bogie, door
```

### `data/`

保存数据划分、关键点和部分处理后的数据。

```text
data/
├── front/
│   ├── processed/
│   │   └── keypoints/
│   └── splits/
│       ├── fold_0.json
│       ├── fold_1.json
│       └── ...
└── side/
    └── splits/
        ├── fold_0.json
        ├── fold_1.json
        └── ...
```

- `data/front/splits/`：正视数据划分。
- `data/side/splits/`：侧视数据划分。
- `data/front/processed/keypoints/`：正视鼻尖坐标，仅 UNet-KP 使用。

### `labelme_workspace/`

保存用于修改标注的图像和同名 LabelMe JSON 文件。

修改标注后，需要重新生成 mask，训练目录中的数据才会更新。

### `prepared_front/`、`prepared_side/`

早期由标注转换得到的数据包，保留作检查和备份。当前模型对比不从这两个目录读取。

### `split_raw/`

由正侧合图拆出的原始图像，主要用于溯源、核对文件名或重新标注，训练脚本一般不直接读取。

## 各任务使用的数据

| 任务 | 图像和 mask | 划分文件 | 额外输入 |
|---|---|---|---|
| 正视 UNet / DeepLab | `data_unet/front/` | `data/front/splits/fold_0.json` | 无 |
| 正视 UNet-KP | `data_unet/front/` | `data/front/splits/fold_0.json` | `data/front/processed/keypoints/` |
| 侧视 UNet / DeepLab | `data_unet/side/` | `data/side/splits/fold_0.json` | 无 |
| SegFormer | 通过工程内目录联接读取同一套数据 | 对应的 front / side 划分 | 无 |

UNet、DeepLab 和 SegFormer 共用同一套正视或侧视数据，没有分别复制训练集。

## 单张样本的对应关系

以正视样本 `front_<id>` 为例：

```text
data_unet/front/images/front_<id>.jpg
data_unet/front/masks/front_<id>.png
data/front/processed/keypoints/front_<id>.json
data/front/splits/fold_0.json
```

其中：

- `.jpg` 是模型输入图像；
- `.png` 是单通道类别 mask；
- `keypoints/*.json` 保存鼻尖坐标，仅 UNet-KP 使用；
- `fold_0.json` 决定样本属于 train、val 还是 test。

侧视样本的对应方式相同，但没有鼻尖关键点文件。

## 注意事项

1. 不要随意移动 `02_baselines_unet_deeplab/` 下的实体目录，否则目录联接和相对路径可能失效。
2. 当前模型对比统一使用 `data_unet/{front,side}` 和对应的 `fold_0.json`。
3. 修改 LabelMe 标注后，需要重新生成对应 mask。
4. 模型权重、测试结果和可视化文件保存在各模型工程的 `outputs/` 中，不在本目录。