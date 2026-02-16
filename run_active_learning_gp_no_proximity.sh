#!/bin/bash
# GP Active Learning for pMSSM — No Proximity Weighting
#
# Full production run with data generation
# Model: Deep Gaussian Process with ARD kernel
#
# Key difference from run_active_learning_gp.sh:
#   --proximity-sampling 0.0          # Disabled (default: 0.1)
#   --tolerance-sampling 0.0          # Disabled (default: 1.0)
#   Uses pure entropy-based selection without proximity/threshold filtering
#
# Features enabled:
#   --warm-starting               # Continue from previous iteration checkpoints
#   --early-stopping              # Stop when validation loss plateaus (patience=100)
#   --selection-strategy entropy_batch  # Sophisticated batch selection (default)
#
# Additional features available (add as needed):
#   --config-file <file.yaml>     # YAML config with parameter sweeps
#   --sweep-index <N>             # Sweep combination index
#   --eval-data-path <file.root>  # External validation dataset
#   --compute-full-metrics        # Comprehensive evaluation metrics
#   --track-lengthscales          # Save ARD lengthscales per iteration
#   --advanced-plots              # Advanced diagnostic plots

.pixi/envs/default/bin/python active_learning_gp.py \
    --model-type deep_gp \
    --epochs 10000 \
    --early-stopping \
    --patience 100 \
    --learning-rate 1e-3 \
    --generate-data \
    --n-samples 50000 \
    --n-iterations 40 \
    --n-select 200 \
    --n-candidates 50000 \
    --gen-workers 20 \
    --kernel RBF \
    --lengthscale 1.0 \
    --noise 1e-2 \
    --jitter 1e-3 \
    --use-ard \
    --proximity-sampling 0.0 \
    --tolerance-sampling 0.0 \
    --warm-starting
