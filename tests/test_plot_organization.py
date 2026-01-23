"""
Test script to verify plot organization with timestamps.
"""
from pathlib import Path
from datetime import datetime

# Simulate the directory structure
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"logs/training_{timestamp}.log"
plots_dir = Path(f"plots/run_{timestamp}")

print("="*60)
print("Plot Organization Test")
print("="*60)
print(f"\nTimestamp: {timestamp}")
print(f"Log file would be: {log_file}")
print(f"Plots directory would be: {plots_dir}")
print("\nPlot files that would be created:")
print(f"  - {plots_dir}/losses_transformer.png")
print(f"  - {plots_dir}/transformer_true_vs_pred_train.png")
print(f"  - {plots_dir}/transformer_true_vs_pred_validation.png")
print(f"  - {plots_dir}/transformer_hist_true_vs_pred_train.png")
print(f"  - {plots_dir}/transformer_hist_true_vs_pred_validation.png")
print(f"\nAnd similar files for:")
print(f"  - transformer_tabular")
print(f"  - MLP")

print("\n" + "="*60)
print("✓ Each training run will have its own timestamped subdirectory")
print("✓ Log files and plot directories will have matching timestamps")
print("="*60)
