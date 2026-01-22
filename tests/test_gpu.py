"""
Quick test to verify GPU selection works correctly.
This script mimics the import order in train_pmssm.py
"""
import os
# Set GPU before importing torch (CRITICAL!)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch

print("="*60)
print("GPU Configuration Test")
print("="*60)

print(f"\nCUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    print(f"Current CUDA Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Device Capability: {torch.cuda.get_device_capability(0)}")

    # Test tensor on GPU
    x = torch.randn(10, 10).cuda()
    print(f"\nTensor device: {x.device}")
    print("✓ GPU is working correctly!")
else:
    print("\n⚠ CUDA not available - will use CPU")
    print("This is normal if:")
    print("  1. No GPU is available on this machine")
    print("  2. You're running on a CPU-only system")

print("\n" + "="*60)
