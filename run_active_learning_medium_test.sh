#!/bin/bash

python active_learning.py \
    --n-datasets 3 \
    --generate-data \
    --n-samples 2000 \
    --n-iterations 3 \
    --epochs 500 \
    --dropout 0.1 \
    --mc-samples 30 \
    --n-candidates 20000 \
    --n-select 10 \
    --gen-workers 2 \
    --output-dir active_learning_output_medium_test
