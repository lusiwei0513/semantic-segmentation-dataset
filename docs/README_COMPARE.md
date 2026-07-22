# 定量对比工作区说明

## 文件夹

| 名称 | 原名 | 用途 |
|------|------|------|
| `baselines_unet_deeplab/` | `project13_split_labelme` | UNet-R34 / DeepLab 基线 + **修正后 data** |
| `segformer_b0_split/` | 新建 | SegFormer-B0 **正/侧分开**训练与评测 |
| `04_archive_joint_segformer/` | 原 `bb/`（后曾误嵌 `.bb/`） | 历史联合训练存档；**非本机训练结果**；见该目录 README |

## 本机硬件结论（已实机验证，非依据旧 checkpoint）

- GPU: RTX 3050 Laptop **4GB**
- 系统 Python 3.11 = CPU-only torch，**不能**训 GPU
- 可用环境: `baselines_unet_deeplab/gpt标注训练正视图-seg_train/.venv`（torch 2.6.0+cu124）

### 真实数据冒烟（forward+backward+val，本机）

- Front 512×512 bs=2 AMP + nose_tip：**成功**（official_mIoU 上报、tip_mae 上报）
- Side 384×1536 bs=1 AMP：**成功**，无 OOM

## 公平配置（质量未牺牲）

见 `segformer_b0_split/QUALITY_FAIRNESS.md`

- 完整 MiT-B0；正视保留 tip；正侧分开；同 fold_0
- 仅用 batch / AMP / accum 适配显存

## 训练命令

```powershell
$py = "baselines_unet_deeplab\gpt标注训练正视图-seg_train\.venv\Scripts\python.exe"
cd segformer_b0_split
& $py src\train.py --config configs\train_front.yaml --view front --fold 0
& $py src\train.py --config configs\train_side.yaml --view side --fold 0
```
