#!/bin/bash
.pixi/envs/default/bin/python active_learning_gp.py \
    --model-type deep_gp \
    --epochs 10000 \
    --early-stopping \
    --patience 100 \
    --learning-rate 1e-3 \
    --generate-data \
    --n-samples 2000 \
    --n-iterations 40 \
    --n-select 250 \
    --n-candidates 5000 \
    --gen-workers 20 \
    --kernel RBF \
    --lengthscale 1.0 \
    --noise 1e-2 \
    --jitter 1e-3 \
    --use-ard \
    --warm-starting \
    --output-dir active_learning_gp_output_500
