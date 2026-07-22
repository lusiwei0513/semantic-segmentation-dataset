# Sequential front tip/sigma ablation: A1 -> A2 -> A3, then test eval each best.pt
$ErrorActionPreference = "Stop"
$Root = "F:\大三下学期\培养方案\保研\康国梁老师\语义分割\语义分割\训练数据\03_segformer_split"
$Py = "F:\大三下学期\培养方案\保研\康国梁老师\语义分割\语义分割\训练数据\02_baselines_unet_deeplab\front\.venv\Scripts\python.exe"
Set-Location -LiteralPath $Root

if (-not (Test-Path -LiteralPath $Py)) {
    Write-Error "Python venv not found: $Py"
    exit 1
}

$Status = Join-Path $Root "outputs\train\TIP_ABLATION_STATUS.txt"
function Write-Status([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Add-Content -LiteralPath $Status -Value $line -Encoding UTF8
    Write-Host $line
}

New-Item -ItemType Directory -Force -Path (Join-Path $Root "outputs\train") | Out-Null
"Tip/sigma ablation queue started" | Set-Content -LiteralPath $Status -Encoding UTF8
Write-Status "python=$Py"

$jobs = @(
    @{ Name = "A1_w6_s16"; Config = "configs\train_front_tip_a1_w6_s16.yaml"; Out = "outputs\train\front_fold0_tip_a1_w6_s16" },
    @{ Name = "A2_w6_s12"; Config = "configs\train_front_tip_a2_w6_s12.yaml"; Out = "outputs\train\front_fold0_tip_a2_w6_s12" },
    @{ Name = "A3_w6_s20"; Config = "configs\train_front_tip_a3_w6_s20.yaml"; Out = "outputs\train\front_fold0_tip_a3_w6_s20" }
)

foreach ($j in $jobs) {
    $outDir = Join-Path $Root $j.Out
    $best = Join-Path $outDir "checkpoints\best.pt"
    if (Test-Path -LiteralPath $best) {
        Write-Status "$($j.Name): best.pt exists, skip train"
    } else {
        Write-Status "$($j.Name): TRAIN start -> $($j.Out)"
        & $Py -u src\train.py --config $j.Config --view front --fold 0 --output-dir $j.Out
        if ($LASTEXITCODE -ne 0) {
            Write-Status "$($j.Name): TRAIN FAILED exit=$LASTEXITCODE"
            exit $LASTEXITCODE
        }
        Write-Status "$($j.Name): TRAIN done"
    }

    $evalName = $j.Name.ToLower()
    $evalDir = Join-Path $Root ("outputs\eval\front_fold0_tip_{0}_test" -f $evalName)
    $report = Join-Path $evalDir "test_report.json"
    if (Test-Path -LiteralPath $report) {
        Write-Status "$($j.Name): test report exists, skip eval"
    } else {
        Write-Status "$($j.Name): EVAL test start"
        New-Item -ItemType Directory -Force -Path $evalDir | Out-Null
        & $Py -u scripts\evaluate_test.py --config $j.Config --checkpoint $best --view front --split test --output-dir $evalDir
        if ($LASTEXITCODE -ne 0) {
            Write-Status "$($j.Name): EVAL FAILED exit=$LASTEXITCODE"
            exit $LASTEXITCODE
        }
        Write-Status "$($j.Name): EVAL done"
    }
}

Write-Status "ALL DONE — compare tip_mae / PCK / official_mIoU in outputs/eval/front_fold0_tip_*_test/"
