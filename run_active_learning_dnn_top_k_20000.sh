#!/bin/bash
# DNN-based Active Learning for pMSSM — top_k selection, n_select=20000
#
# Full production run with data generation
# Model: PMSSMFeedForward with MC Dropout uncertainty estimation

.pixi/envs/cuda/bin/python active_learning_dnn.py \
    --y-transform log \
    --epochs 10_000 \
    --generate-data \
    --n-samples 2000 \
    --n-iterations 40 \
    --n-candidates 1_000_000 \
    --entropy-pool-size 5_000 \
    --gen-workers 20 \
    --mcmc-data-dir data/neutralino_v4 \
    --static-eval-size 100_000 \
    --warm-starting \
    --early-stopping \
    --selection-strategy top_k \
    --n-select 20_000 \
    --output-dir active_learning_dnn_output_top_k_n_select_20k \
    --gpu-ids 2,3
