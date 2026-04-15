#!/bin/bash
# Transformer-based Active Learning — ~1 hour run
#
# Timing estimate based on full run analysis:
#   - Full run: 40 iterations took ~34 hours (avg ~51 min/iteration)
#   - This run: 2 iterations with 50k initial samples
#   - Expected time: ~102 min (2 × 51 min), but early iterations are faster
#   - Actual estimate: ~45-60 min (iterations 1-2 average ~35-40 min each)
#
# Model: PMSSMTransformerTabular with MC Dropout
# Starting from same initial dataset size as full run (50k samples)
# Training in log space (matching GP pipeline)

.pixi/envs/cuda/bin/python active_learning.py \
    --y-transform log \
    --n-samples 50000 \
    --n-iterations 2 \
    --n-select 20000 \
    --n-candidates 50000 \
    --entropy-pool-size 2000 \
    --epochs 10000 \
    --dropout 0.1 \
    --mc-samples 30 \
    --generate-data \
    --gen-workers 20 \
    --warm-starting \
    --early-stopping \
    --patience 200 \
    --output-dir active_learning_output_1h
