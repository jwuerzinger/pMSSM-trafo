#!/bin/bash
# GP Active Learning — ~2 hour run
#
# Timing estimate (4x H100, 96 cores):
#   - 10k initial samples, ExactGP with 2000 epochs + early stopping
#   - 10 AL iterations, selecting 500 points per iteration with data generation
#   - Per-iteration: ~8-12 min (training ~2-4 min, generation ~3-5 min, rest ~2 min)
#   - Total: ~80-120 min ≈ 2 hours
#
# To shorten: reduce --n-iterations or --epochs
# To lengthen: increase --n-iterations or --n-samples

.pixi/envs/cuda/bin/python active_learning_gp.py \
    --model-type exact_gp \
    --n-samples 10000 \
    --n-iterations 10 \
    --epochs 2000 \
    --early-stopping \
    --patience 200 \
    --learning-rate 1e-3 \
    --n-candidates 30000 \
    --n-select 500 \
    --generate-data \
    --gen-workers 10 \
    --kernel RBF \
    --lengthscale 1.0 \
    --noise 1e-2 \
    --jitter 1e-3 \
    --use-ard \
    --warm-starting \
    --output-dir active_learning_gp_output_2h
