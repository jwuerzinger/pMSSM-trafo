"""
Test script to verify parallel GPU setup.
"""
import os
# Set GPU before importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

import torch

print("="*60)
print("Parallel GPU Configuration Test")
print("="*60)

print(f"\nCUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs visible: {num_gpus}")

    for i in range(num_gpus):
        print(f"\nGPU {i} (Physical GPU {i+1}):")
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
    print("GPU Configuration Summary:")
    print("  - PMSSMTransformer will use: cuda:0 (Physical GPU 1)")
    print("  - PMSSMTransformerTabular will use: cuda:1 (Physical GPU 2)")
    print("  - MLP will use: cuda:1 (Physical GPU 2)")
    print("="*60)

    if num_gpus >= 2:
        print("\n✓ Parallel training on 2 GPUs is possible!")
    else:
        print(f"\n⚠ Only {num_gpus} GPU(s) available - parallel training not possible")
else:
    print("\n⚠ CUDA not available - will use CPU")

print("="*60)
