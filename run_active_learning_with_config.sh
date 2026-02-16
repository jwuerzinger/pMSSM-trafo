#!/bin/bash
# Example: Active Learning with YAML config file and parameter sweep
#
# This demonstrates using a config file to override CLI parameters
# and run systematic hyperparameter sweeps.
#
# Usage:
#   bash run_active_learning_with_config.sh
#
# To run all sweep combinations, modify sweep-index from 0 to 8
# Or use a loop: for i in {0..8}; do python active_learning.py --config-file sweep_config.yaml --sweep-index $i --testing; done

# First, create a config file with sweep parameters
cat > sweep_config.yaml <<EOF
# Hyperparameter sweep configuration for transformer model
epochs: [1000, 2000, 5000]  # 3 values
dropout: [0.1, 0.15, 0.2]   # 3 values
# Total: 3 × 3 = 9 combinations (sweep indices 0-8)

# Fixed parameters
y_transform: log
n_iterations: 5
n_candidates: 10000
n_select: 100
selection_strategy: entropy_batch
warm_starting: true
early_stopping: true
patience: 200
compute_full_metrics: true
candidate_generation: lhs
proximity_sampling: 0.1
EOF

echo "Created sweep_config.yaml with 9 parameter combinations"
echo ""

# Run a specific sweep combination (e.g., index 0: epochs=1000, dropout=0.1)
python active_learning.py \
    --config-file sweep_config.yaml \
    --sweep-index 0 \
    --testing

echo ""
echo "================================================"
echo "Sweep completed for combination 0"
echo "To run all 9 combinations, use sweep indices 0-8"
echo "================================================"
