#!/bin/bash
# Example: Active Learning with external evaluation dataset
#
# This demonstrates using an external dataset to validate model
# performance on held-out data throughout training.
#
# Usage:
#   1. Prepare an external eval dataset (ROOT file with same structure)
#   2. Update --eval-data-path with the correct path
#   3. Run: bash run_active_learning_with_eval.sh
#
# Features demonstrated:
#   --eval-data-path          External validation dataset
#   --compute-full-metrics    MSE, RMSE, R², accuracy in both normalized and physical space
#   --warm-starting           Continue from previous iteration checkpoints
#   --early-stopping          Stop when validation loss plateaus

# NOTE: Update this path to point to your evaluation dataset
EVAL_DATASET_PATH="/path/to/eval_dataset.root"

if [ ! -f "$EVAL_DATASET_PATH" ]; then
    echo "ERROR: Eval dataset not found at: $EVAL_DATASET_PATH"
    echo "Please update EVAL_DATASET_PATH in this script to point to a valid ROOT file"
    echo ""
    echo "For testing without an eval dataset, remove --eval-data-path option"
    exit 1
fi

echo "Running Active Learning with external eval dataset validation"
echo "Eval dataset: $EVAL_DATASET_PATH"
echo ""

python active_learning.py \
    --y-transform log \
    --testing \
    --n-iterations 5 \
    --epochs 1000 \
    --eval-data-path "$EVAL_DATASET_PATH" \
    --compute-full-metrics \
    --warm-starting \
    --early-stopping \
    --patience 100

echo ""
echo "================================================"
echo "Active Learning completed"
echo "Check output logs for 'External eval R²' metrics per iteration"
echo "Comprehensive metrics saved to: active_learning_output/iteration_*/metrics_al.csv"
echo "================================================"
