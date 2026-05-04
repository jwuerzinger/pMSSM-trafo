#!/bin/bash
# DNN-based Active Learning — ~1 hour run
#
# Model: PMSSMFeedForward with MC Dropout
# Mirrors run_active_learning_1h.sh (transformer) for direct comparison.

.pixi/envs/cuda/bin/python active_learning_dnn.py \
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
    --output-dir active_learning_dnn_output_1h
