@echo off
REM Front main runs: KP / UNet-seg / DeepLab (edit which config to use)
setlocal
cd /d "%~dp0"
set PYTHON=%~dp0.venv\Scripts\python.exe
set HF_HUB_OFFLINE=1

REM Default: UNet-KP official front protocol
"%PYTHON%" scripts\train_front_kp.py --config config_front_unet_kp_tune.yaml
if errorlevel 1 exit /b 1

echo.
echo Done. Best weights: outputs\unet_kp\front_unet_kp_tune\best.pt
echo Other configs: config_front_unet_seg_tune.yaml / config_front_deeplab_v2.yaml
endlocal
