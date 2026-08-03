#!/bin/bash
# DNN Active Learning - Medium Test
#
# Test run: 3 datasets, 2000 samples, 3 iterations
# Model: PMSSMFeedForward with MC Dropout

.pixi/envs/cuda/bin/python active_learning_dnn.py \
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
    --mcmc-data-dir data/neutralino_v4 \
    --static-eval-size 1000 \
    --output-dir active_learning_dnn_output_medium_test
