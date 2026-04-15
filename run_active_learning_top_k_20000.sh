#!/bin/bash
# Transformer-based Active Learning for pMSSM
#
# Full production run with data generation
# Model: PMSSMTransformerTabular with MC Dropout uncertainty estimation
#
# Features enabled:
#   --warm-starting               # Continue from previous iteration checkpoints
#   --early-stopping              # Stop when validation loss plateaus (patience=200)
#
# Additional features available (add as needed):
#   --config-file <file.yaml>     # YAML config with parameter sweeps
#   --sweep-index <N>             # Sweep combination index
#   --eval-data-path <file.root>  # External validation dataset
#   --compute-full-metrics        # Comprehensive evaluation metrics

.pixi/envs/cuda/bin/python active_learning.py \
    --y-transform log \
    --epochs 10_000 \
    --generate-data \
    --n-samples 2000 \
    --n-iterations 40 \
    --n-candidates 1_000_000 \
    --entropy-pool-size 5_000 \
    --gen-workers 20 \
    --mcmc-data-dir data/19250082 \
    --static-eval-size 100_000 \
    --warm-starting \
    --early-stopping \
    --selection-strategy top_k \
    --n-select 20_000 \
    --output-dir active_learning_output_top_k_n_select_20k \
    --gpu-ids 2,3
    