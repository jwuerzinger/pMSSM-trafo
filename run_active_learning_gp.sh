#!/bin/bash
.pixi/envs/default/bin/python active_learning_gp.py \
    --model-type exact_gp \
    --training-iterations 10000 \
    --early-stopping \
    --patience 100 \
    --learning-rate 1e-3 \
    --generate-data \
    --n-samples 100000 \
    --n-iterations 40 \
    --n-select 20000 \
    --n-candidates 50000 \
    --gen-workers 20 \
    --kernel RBF \
    --lengthscale 1.0 \
    --noise 1e-2 \
    --jitter 1e-3 \
    --use-ard \
    --warm-starting
