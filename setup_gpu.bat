@echo off
echo ============================================
echo Setup MedGemma Environment with GPU Support
echo ============================================
echo.

REM Check if conda is available
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Conda not found. Please install Anaconda or Miniconda first.
    exit /b 1
)

echo Activating medgemma environment...
call conda activate medgemma

echo.
echo Checking current PyTorch installation...
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available())"

echo.
echo Uninstalling current PyTorch (if any)...
pip uninstall -y torch torchvision torchaudio

echo.
echo Installing PyTorch with CUDA 11.8 support...
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118

echo.
echo Verifying installation...
python -c "import torch; print('✓ PyTorch:', torch.__version__); print('✓ CUDA Available:', torch.cuda.is_available()); print('✓ GPUs:', torch.cuda.device_count())"

echo.
echo ============================================
echo Setup complete!
echo ============================================
pause
