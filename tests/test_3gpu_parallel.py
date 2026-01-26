"""
Test script to verify 3-GPU parallel training setup.
"""
import os
# Set all GPUs visible
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import torch

print("="*60)
print("3-GPU Parallel Training Configuration Test")
print("="*60)

print(f"\nCUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs visible: {num_gpus}")

    for i in range(num_gpus):
        print(f"\nGPU {i} (Physical GPU {i}):")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Capability: {torch.cuda.get_device_capability(i)}")

        # Test tensor creation on each GPU
        try:
            torch.cuda.set_device(i)
            x = torch.randn(10, 10).cuda()
            print(f"  ✓ Can create tensors on cuda:{i}")
            print(f"  Memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
        except Exception as e:
            print(f"  ✗ Error on cuda:{i}: {e}")

    print("\n" + "="*60)
    print("Training Configuration:")
    print("="*60)

    if num_gpus >= 3:
        print("\n✓ Parallel training on 3 GPUs is possible!")
        print("\nGPU Assignment (Parallel Mode):")
        print("  - PMSSMTransformer → cuda:1 (Physical GPU 1)")
        print("  - PMSSMTransformerTabular → cuda:2 (Physical GPU 2)")
        print("  - MLP Baseline → cuda:3 (Physical GPU 3)")
        print("\nAll three models train simultaneously!")
        print(f"\nExpected speedup: ~3x faster than sequential on 1 GPU")
    else:
        print(f"\n⚠ Only {num_gpus} GPU(s) available")
        print("Need 3 GPUs for parallel training")
        print("\nSequential Mode:")
        print("  - All models will train on cuda:1 sequentially")
        print("  - Use --no-parallel flag to explicitly request sequential mode")

    print("\n" + "="*60)
    print("Usage:")
    print("="*60)
    print("  # Parallel training (default with 3+ GPUs)")
    print("  python train_pmssm.py")
    print("")
    print("  # Sequential training (even with 3+ GPUs)")
    print("  python train_pmssm.py --no-parallel")
    print("")
    print("  # Quick test")
    print("  python train_pmssm.py --testing --epochs 100")

else:
    print("\n⚠ CUDA not available - will use CPU for all training")

print("="*60)
