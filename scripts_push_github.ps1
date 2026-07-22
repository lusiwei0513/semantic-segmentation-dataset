# Push local main to GitHub under lusiwei0513
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $Root
$env:Path = "C:\Program Files\GitHub CLI;C:\Program Files\Git\cmd;" + $env:Path

git lfs install
gh auth status
if ($LASTEXITCODE -ne 0) {
  Write-Host "Please login first: gh auth login -h github.com -p https -w"
  exit 1
}

$repo = "rail-vehicle-semantic-segmentation"
$owner = "lusiwei0513"
$full = "$owner/$repo"

# Create if missing
gh repo view $full 2>$null
if ($LASTEXITCODE -ne 0) {
  gh repo create $full --public --source=. --remote=origin --description "高铁设计图语义分割 UNet-KP/DeepLab/SegFormer fold_0"
} else {
  $remotes = git remote
  if ($remotes -notcontains "origin") {
    git remote add origin "https://github.com/$full.git"
  }
}

git push -u origin main
Write-Host "Done: https://github.com/$full"
Write-Host "Pipeline doc: EXPERIMENT_PIPELINE.md"
