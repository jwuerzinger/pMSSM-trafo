#!/bin/bash

.pixi/envs/cuda/bin/python active_learning_gp.py \
    --n-datasets 3 \
    --generate-data \
    --model-type exact_gp \
    --n-samples 2000 \
    --n-iterations 3 \
    --epochs 500 \
    --learning-rate 1e-3 \
    --n-candidates 20000 \
    --n-select 10 \
    --kernel RBF \
    --lengthscale 1.0 \
    --noise 1e-2 \
    --jitter 1e-3 \
    --use-ard \
    --warm-starting \
    --gen-workers 2 \
    --output-dir active_learning_gp_output_medium_test
