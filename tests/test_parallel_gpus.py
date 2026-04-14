"""
Test script to verify parallel GPU setup.
Works on both NVIDIA (CUDA) and AMD (ROCm) GPUs.
"""
import os
# Set GPU before importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import torch

print("="*60)
print("Parallel GPU Configuration Test")
print("="*60)

print(f"\nGPU Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # Detect backend
    if hasattr(torch.version, 'hip') and torch.version.hip:
        print(f"Backend: ROCm (HIP {torch.version.hip})")
    elif torch.version.cuda:
        print(f"Backend: CUDA {torch.version.cuda}")

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs visible: {num_gpus}")

    for i in range(num_gpus):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")

        try:
            print(f"  Capability: {torch.cuda.get_device_capability(i)}")
        except Exception:
            print("  Capability: N/A (not supported on this backend)")

        # Test tensor creation on each GPU
        try:
            torch.cuda.set_device(i)
            x = torch.randn(10, 10).cuda()
            print(f"  Can create tensors on cuda:{i}")
            print(f"  Memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
        except Exception as e:
            print(f"  Error on cuda:{i}: {e}")

    print("\n" + "="*60)
    print("GPU Configuration Summary:")
    print("  - PMSSMTransformer will use: cuda:0")
    print("  - PMSSMTransformerTabular will use: cuda:1")
    print("  - MLP will use: cuda:1")
    print("="*60)

    if num_gpus >= 2:
        print("\nParallel training on 2 GPUs is possible!")
    else:
        print(f"\nOnly {num_gpus} GPU(s) available - parallel training not possible")
else:
    print("\nGPU not available - will use CPU")

print("="*60)
