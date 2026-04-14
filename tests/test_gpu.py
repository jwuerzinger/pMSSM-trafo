"""
Quick test to verify GPU selection works correctly.
This script mimics the import order in train_pmssm.py.
Works on both NVIDIA (CUDA) and AMD (ROCm) GPUs.
"""
import os
# Set GPU before importing torch (CRITICAL!)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch

print("="*60)
print("GPU Configuration Test")
print("="*60)

print(f"\nGPU Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # Detect backend
    if hasattr(torch.version, 'hip') and torch.version.hip:
        print(f"Backend: ROCm (HIP {torch.version.hip})")
    elif torch.version.cuda:
        print(f"Backend: CUDA {torch.version.cuda}")

    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

    try:
        print(f"Device Capability: {torch.cuda.get_device_capability(0)}")
    except Exception:
        print("Device Capability: N/A (not supported on this backend)")

    # Test tensor on GPU
    x = torch.randn(10, 10).cuda()
    print(f"\nTensor device: {x.device}")
    print("GPU is working correctly!")
else:
    print("\nGPU not available - will use CPU")
    print("This is normal if:")
    print("  1. No GPU is available on this machine")
    print("  2. You're running on a CPU-only system")

print("\n" + "="*60)
