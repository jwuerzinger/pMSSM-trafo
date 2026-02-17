#!/bin/bash
# Transformer-based Active Learning for pMSSM — No Proximity Weighting
#
# Full production run with data generation
# Model: PMSSMTransformerTabular with MC Dropout uncertainty estimation
#
# Key difference from run_active_learning.sh:
#   --proximity-sampling 0.0          # Disabled (default: 0.1)
#   Uses pure variance-based selection without proximity weighting
#
# Features enabled:
#   --warm-starting               # Continue from previous iteration checkpoints
#   --early-stopping              # Stop when validation loss plateaus (patience=200)
#   --y-transform log             # Log-space training for relic density
#
# Additional features available (add as needed):
#   --config-file <file.yaml>     # YAML config with parameter sweeps
#   --sweep-index <N>             # Sweep combination index
#   --eval-data-path <file.root>  # External validation dataset
#   --compute-full-metrics        # Comprehensive evaluation metrics

.pixi/envs/default/bin/python active_learning.py \
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
    --output-dir active_learning_output_no_proximity
