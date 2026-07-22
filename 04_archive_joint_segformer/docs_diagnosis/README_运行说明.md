# 数据域诊断工具使用说明

## 1. 文件

```text
正侧视图是否需要两个完整模型_诊断方案.md
check_front_side_domain.py
build_metadata.py
diagnosis_config.yaml
metadata.csv / metadata_template.csv
requirements_diagnosis.txt
```

本包位于 `04_archive_joint_segformer/docs_diagnosis/`（原 `.bb/正侧视图模型拆分诊断包`）。

## 2. 准备 metadata.csv

至少包含：

```csv
sample_id,image_path,vehicle_id,view,device_id
```

`view` 只能是 `front` / `side`。路径可为绝对路径，或相对 `diagnosis_config.yaml` 中的 `data_root`（当前为 `.`）。

原始图：`../data_prepared/prepared_*`。可用 `python build_metadata.py` 重新生成。

## 3. 安装依赖

```bash
cd docs_diagnosis
pip install -r requirements_diagnosis.txt
```

## 4. 运行

```bash
python check_front_side_domain.py --config diagnosis_config.yaml
```

如果暂时无法下载 ImageNet 预训练权重：

```bash
python check_front_side_domain.py --config diagnosis_config.yaml --skip-embeddings
```

## 5. 输出

```text
outputs/domain_diagnosis/
├── image_statistics.csv
├── summary.json
├── report.md
├── view_device_crosstab.csv
├── pca_embeddings.png
├── basic_stats_by_view.csv
├── basic_stats_by_device.csv
└── warnings.txt
```

## 6. 注意

正侧视图本身容易被分类，因此 `view_probe_accuracy` 高并不能证明必须使用两个模型。

只有在同一视图内部，设备仍然容易被识别，并且两个完整模型在严格五折测试中稳定优于共享模型，才应考虑完全拆分。
