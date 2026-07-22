# 推送到 GitHub（需本机已登录）

本地仓库已完成首次 commit（含代码、数据、test 结果、`best.pt` via LFS、`EXPERIMENT_PIPELINE.md`）。

因当前环境访问 `github.com` 超时 / 未登录，**推送需你在本机执行**（可开北航 VPN 后重试）。

## 一次性操作

```powershell
cd "F:\大三下学期\培养方案\保研\康国梁老师\语义分割\语义分割\训练数据"

# 1) 登录（浏览器授权）
$env:Path = "C:\Program Files\GitHub CLI;" + $env:Path
gh auth login -h github.com -p https -w

# 2) 创建远程仓库并推送
gh repo create rail-vehicle-semantic-segmentation --public --source=. --remote=origin --push --description "高铁设计图语义分割：UNet-KP / DeepLab / SegFormer，fold_0 实验与交付 checkpoint"

# 若仓库已存在：
# git remote add origin https://github.com/lusiwei0513/rail-vehicle-semantic-segmentation.git
# git push -u origin main
```

克隆他人机器时请先：

```bash
git lfs install
git clone https://github.com/lusiwei0513/rail-vehicle-semantic-segmentation.git
git lfs pull
```

## 实验流程文档

仓库根目录：**[`EXPERIMENT_PIPELINE.md`](./EXPERIMENT_PIPELINE.md)**

## 未上传（体积/安装包）

见 `.gitignore`：`.venv`、`05_archive_misc/`、zip/rar/exe、联合训练第三方与压缩包等。主实验 `best.pt`、代码、`data_unet`、划分与可视化已纳入。
