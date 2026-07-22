@echo off
REM Side main runs: wide 384x1536 UNet + DeepLab (uses front/.venv)
setlocal
cd /d "%~dp0"
set PYTHON=%~dp0..\front\.venv\Scripts\python.exe
set HF_HUB_OFFLINE=1

echo ===== SIDE UNet wide 384x1536 =====
"%PYTHON%" scripts\train.py --config config_side_unet_wide_384x1536.yaml
if errorlevel 1 exit /b 1

echo ===== SIDE DeepLab wide 384x1536 =====
"%PYTHON%" scripts\train.py --config config_side_deeplab_wide_384x1536.yaml
if errorlevel 1 exit /b 1

echo Done. Checkpoints:
echo   outputs\unet_seg\side_unet_wide_384x1536\best.pt
echo   outputs\deeplab\side_deeplab_wide_384x1536\best.pt
endlocal
