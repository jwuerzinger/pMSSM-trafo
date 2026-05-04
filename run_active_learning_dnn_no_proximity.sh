#!/bin/bash
# DNN-based Active Learning for pMSSM — No Proximity Weighting
#
# Full production run with data generation
# Model: PMSSMFeedForward with MC Dropout uncertainty estimation
#
# Key difference from run_active_learning_dnn.sh:
#   --proximity-sampling 0.0          # Disabled (default: 0.1)
#   Uses pure variance-based selection without proximity weighting

.pixi/envs/cuda/bin/python active_learning_dnn.py \
    --y-transform log \
    --epochs 10_000 \
    --generate-data \
    --n-samples 50_000 \
    --n-iterations 40 \
    --n-select 20_000 \
    --n-candidates 50_000 \
    --entropy-pool-size 2000 \
    --gen-workers 20 \
    --proximity-sampling 0.0 \
    --warm-starting \
    --early-stopping \
    --output-dir active_learning_dnn_output_no_proximity
