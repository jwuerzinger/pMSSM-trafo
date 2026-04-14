#!/bin/bash
# Quick discovery job: detect GPU hardware, ROCm version, and driver info on compute nodes.
#SBATCH --job-name=discover_gpu
#SBATCH --partition=apudev
#SBATCH --account=mpp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

echo "=========================================="
echo " GPU Discovery Job"
echo " Node:    $(hostname)"
echo " Started: $(date)"
echo "=========================================="

echo ""
echo "=== Environment Variables ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<not set>}"
echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-<not set>}"
echo "GPU_DEVICE_ORDINAL=${GPU_DEVICE_ORDINAL:-<not set>}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-<not set>}"

echo ""
echo "=== ROCm Version ==="
cat /opt/rocm/.info/version 2>/dev/null || echo "/opt/rocm/.info/version not found"
ls -d /opt/rocm* 2>/dev/null || echo "No /opt/rocm directories"
rocm-smi --version 2>/dev/null || echo "rocm-smi --version not available"

echo ""
echo "=== GPU Hardware (rocm-smi) ==="
rocm-smi 2>/dev/null || echo "rocm-smi not available"
rocm-smi --showproductname 2>/dev/null || echo "rocm-smi --showproductname not available"

echo ""
echo "=== GPU Hardware (rocminfo) ==="
rocminfo 2>/dev/null | grep -E "Marketing Name|Name:|Device Type" | head -20 || echo "rocminfo not available"

echo ""
echo "=== AMDGPU Driver ==="
cat /proc/driver/amdgpu/version 2>/dev/null || echo "No amdgpu driver version file"
modinfo amdgpu 2>/dev/null | grep -E "^version|^description" || echo "modinfo amdgpu not available"

echo ""
echo "=== /dev/kfd and /dev/dri ==="
ls -la /dev/kfd 2>/dev/null || echo "No /dev/kfd"
ls -la /dev/dri/render* 2>/dev/null || echo "No /dev/dri/render*"

echo ""
echo "=== Kernel Module ==="
lsmod | grep -i amdgpu | head -5 || echo "No amdgpu kernel module loaded"

echo ""
echo "=== Available Python ==="
which python3 2>/dev/null && python3 --version 2>/dev/null || echo "No system python3"

echo ""
echo "=========================================="
echo " Done: $(date)"
echo "=========================================="
