# LabelMe / X-AnyLabeling 可直接打开的标注目录

本目录把「图片 + 同名 .json」整理为两个入口（目录联接，不占双倍磁盘）：

| 子文件夹 | 内容 | 数量 |
|----------|------|------|
| front/ | 正视图 + LabelMe JSON | 189 对 |
| side/ | 侧视图 + LabelMe JSON | 189 对 |

## 如何打开

### LabelMe
1. 启动 LabelMe
2. Open Dir，选择本目录下的 front 或 side
3. 图与 json 同目录同名，可直接查看/修改

### X-AnyLabeling
1. 打开文件夹，选择 front 或 side
2. 使用 LabelMe 兼容格式即可

## 说明
- 实体文件在 02_baselines_unet_deeplab/front_images 与 side_images
- prepared_front / prepared_side 是训练用 images+masks，不含标注 JSON
- 在此修改 json 会写回实体目录，请先备份
