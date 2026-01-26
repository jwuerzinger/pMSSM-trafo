"""
Test script to verify sequential training mode.
"""
import os
# Set GPU before importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

import torch

print("="*60)
print("Sequential Mode Configuration Test")
print("="*60)

print(f"\nCUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs visible: {num_gpus}")

    print("\n" + "="*60)
    print("Testing Sequential Training Configuration")
    print("="*60)

    # Simulate sequential mode device selection
    if num_gpus >= 2:
        print("\n✓ Multiple GPUs available")
        print("  Default mode: Parallel training")
        print("  With --no-parallel flag: Sequential on cuda:0 (Physical GPU 1)")
        device_seq = "cuda:0"
    elif num_gpus == 1:
        print("\n✓ Single GPU available")
        print("  Mode: Sequential training (automatic)")
        device_seq = "cuda:0"
    else:
        print("\n✗ No GPUs available")
        print("  Mode: Sequential training on CPU")
        device_seq = "cpu"

    print(f"\nSequential training would use: {device_seq}")

    # Test MLP device selection
    print("\n" + "="*60)
    print("Testing MLP Device Configuration")
    print("="*60)

    if num_gpus >= 2:
        device_mlp = "cuda:1"
        print(f"\nMLP would use: {device_mlp} (Physical GPU 2)")
        print("  - Parallel mode: After transformers complete")
        print("  - Sequential mode: After transformers on cuda:0")
    elif num_gpus == 1:
        device_mlp = "cuda:0"
        print(f"\nMLP would use: {device_mlp} (Physical GPU 1)")
    else:
        device_mlp = "cpu"
        print(f"\nMLP would use: {device_mlp}")

    print("\n" + "="*60)
    print("Command-line Options:")
    print("="*60)
    print("  python train_pmssm.py")
    print("    → Parallel if 2+ GPUs, Sequential otherwise")
    print("\n  python train_pmssm.py --no-parallel")
    print("    → Sequential even with 2+ GPUs")
    print("\n  python train_pmssm.py --testing --epochs 100")
    print("    → Quick test with parallel/sequential based on GPUs")
    print("\n  python train_pmssm.py --no-parallel --testing --epochs 100")
    print("    → Quick test with sequential training")

else:
    print("\n⚠ CUDA not available - will use CPU for all training")

print("="*60)
