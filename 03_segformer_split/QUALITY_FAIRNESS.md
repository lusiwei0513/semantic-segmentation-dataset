# 质量承诺 / 公平对比说明（不做偷工减料）
date: 2026-07-20
gpu: NVIDIA GeForce RTX 3050 4GB Laptop GPU
cuda_env: 02_baselines_unet_deeplab/front/.venv (torch 2.6.0+cu124)

## 不牺牲项
- 骨干：完整 SegFormer / MiT-B0（hidden_sizes 32/64/160/256，depths 2/2/2/2）
- 正视：保留 nose_tip（热图训练 + 官方椭圆 IoU rx=24,ry=12，与 UNet KP 一致）
- 训练方式：正/侧分开，不回退联合训练
- 划分：baselines fold_0（与 UNet/DeepLab 相同）
- 数据：02_baselines_unet_deeplab/data/{front,side}/processed

## 最终训练配置
### Front
- size: 512x512 letterbox
- batch_size: 2, AMP, grad_accum: 2 (effective batch 4)
- keypoint_head: true（heatmap + official ellipse IoU rx=24,ry=12）
- heatmap_sigma: 16（letterbox；热图分辨率下 ≈4，避免 σ过小导致 tip 学不动）
- pretrained: ./pretrained/mit-b0

### Side
- size: 384x1536 letterbox（短边对齐 UNet 384，宽边保留长宽比；非玩具尺寸）
- batch_size: 1, AMP, grad_accum: 4 (effective batch 4)
- keypoint_head: false（侧视无 tip）
- pretrained: ./pretrained/mit-b0

## 本机真实数据冒烟（非仅合成张量）
- Front 512 bs2 + tip：成功（上报 official_mIoU 与 tip_mae）
- Side 384x1536 bs1：成功，无 OOM

## 允许的显存适配（仅这些）
- batch 1–2、AMP、gradient accumulation、num_workers=0
- 禁止：更小骨干、砍 tip、正侧改联合、把输入缩到 256 等玩具尺寸

## 合成张量冒烟 peak VRAM（本机实测）
- front 512 bs2 AMP+tip: ~947 MB
- side 384x1536 bs1 AMP: ~1352 MB
- side 448x2016 bs1 AMP: ~2393 MB（可选更宽，非默认）
