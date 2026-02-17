#!/bin/bash
# Transformer Active Learning - Medium Test
#
# Test run: 3 datasets, 2000 samples, 3 iterations
# Model: PMSSMTransformerTabular with MC Dropout
#
# New features available:
#   --config-file <file.yaml>     # YAML config with parameter sweeps
#   --eval-data-path <file.root>  # External validation dataset
#   --compute-full-metrics        # Comprehensive evaluation metrics

.pixi/envs/default/bin/python active_learning.py \
    --y-transform log \
    --n-datasets 3 \
    --generate-data \
    --n-samples 2000 \
    --n-iterations 3 \
    --epochs 500 \
    --dropout 0.1 \
    --mc-samples 30 \
    --n-candidates 20000 \
    --entropy-pool-size 1500 \
    --n-select 10 \
    --gen-workers 2 \
    --output-dir active_learning_output_medium_test
