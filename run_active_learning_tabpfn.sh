#!/bin/bash
# TabPFN-based Active Learning for pMSSM
#
# Full production run with data generation
# Model: TabPFN with native predictive variance for uncertainty estimation
#
# TabPFN is a foundation model for tabular data — no training loop needed.
# Uncertainty is estimated via the predictive distribution (top_k) or
# ensemble diversity with multiple random_state runs (entropy_batch).

.pixi/envs/cuda/bin/python active_learning_tabpfn.py \
    --n-samples 2000 \
    --n-iterations 40 \
    --n-select 500 \
    --n-candidates 1_000_000 \
    --n-ensemble-samples 16 \
    --generate-data \
    --gen-workers 20 \
    --mcmc-data-dir data/19250082 \
    --static-eval-size 100_000 \
    --proximity-sampling 0.1 \
    --tolerance-sampling 1.0 \
    --output-dir active_learning_tabpfn_output \
    --gpu-id 0 \
    --selection-strategy top_k \
    # --selection-strategy entropy_batch \
    # --entropy-pool-size 5_000 \
