"""
Check GPU availability and PyTorch CUDA installation.
"""
import torch
import sys

print("=" * 70)
print("GPU DIAGNOSTICS")
print("=" * 70)

# Check CUDA availability
print(f"\n1. PyTorch Version: {torch.__version__}")
print(f"2. CUDA Available: {torch.cuda.is_available()}")
print(f"3. CUDA Version (PyTorch): {torch.version.cuda}")
print(f"4. Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"5. GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("\n❌ NO GPU DETECTED - Model will use CPU (very slow)")
    print("\nTo fix this, reinstall PyTorch with CUDA support:")
    print("\nconda activate medgemma")
    print("pip uninstall torch torchvision")
    print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print("\nFor CUDA 11.8:")
    print("pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118")
    
print("\n" + "=" * 70)
