#!/bin/bash
# GP Active Learning — ~1 hour run
#
# Timing estimate based on full run analysis:
#   - Full run: 40 iterations took ~10.5 hours (avg ~16 min/iteration)
#   - This run: 3 iterations with 50k initial samples
#   - Expected time: ~48 min (3 × 16 min) + overhead ≈ 1 hour
#
# Model: Deep Gaussian Process with ARD kernel
# Starting from same initial dataset size as full run (50k samples)

.pixi/envs/cuda/bin/python active_learning_gp.py \
    --model-type deep_gp \
    --n-samples 50000 \
    --n-iterations 3 \
    --n-select 200 \
    --n-candidates 50000 \
    --epochs 10000 \
    --early-stopping \
    --patience 100 \
    --learning-rate 1e-3 \
    --generate-data \
    --gen-workers 20 \
    --kernel RBF \
    --lengthscale 1.0 \
    --noise 1e-2 \
    --jitter 1e-3 \
    --use-ard \
    --warm-starting \
    --output-dir active_learning_gp_output_1h
