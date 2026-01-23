# Plot Organization

Plots are now organized into timestamped subdirectories that match the log files.

## Directory Structure

Each training run creates:
- **Log file**: `logs/training_YYYYMMDD_HHMMSS.log`
- **Plot directory**: `plots/run_YYYYMMDD_HHMMSS/`

Both use the same timestamp, making it easy to match plots to their corresponding training logs.

## Example

For a training run started on January 22, 2026 at 15:30:00:

```
logs/
  └── training_20260122_153000.log

plots/
  └── run_20260122_153000/
      ├── losses_transformer.png
      ├── transformer_true_vs_pred_train.png
      ├── transformer_true_vs_pred_validation.png
      ├── transformer_hist_true_vs_pred_train.png
      ├── transformer_hist_true_vs_pred_validation.png
      ├── losses_transformer_tabular.png
      ├── transformer_tabular_true_vs_pred_train.png
      ├── transformer_tabular_true_vs_pred_validation.png
      ├── transformer_tabular_hist_true_vs_pred_train.png
      ├── transformer_tabular_hist_true_vs_pred_validation.png
      ├── losses_MLP.png
      ├── MLP_true_vs_pred_train.png
      ├── MLP_true_vs_pred_validation.png
      ├── MLP_hist_true_vs_pred_train.png
      └── MLP_hist_true_vs_pred_validation.png
```

## Benefits

1. **No Overwriting**: Each run gets its own directory
2. **Easy Tracking**: Match plots to logs using the timestamp
3. **Historical Record**: Keep all past runs for comparison
4. **Clean Organization**: All plots for a run are in one place

## Files Modified

### [train_pmssm.py](../train_pmssm.py)
- Creates timestamped `plots_dir` matching the log file timestamp
- Passes `plot_dir` parameter to all plotting functions
- Logs the plot directory location at startup

### [pmssm.py](../pmssm.py)
Updated these functions to accept `plot_dir` parameter:
- `plot_losses()` - Training loss curves
- `scatter_true_vs_pred()` - Scatter plots
- `hist_true_vs_pred()` - 2D histograms with **log-scale colormap**

All default to `plot_dir="plots"` for backward compatibility.

**Note**: The 2D histogram now uses a logarithmic colormap scale (`LogNorm`) for better visualization of data distribution across different density ranges.

## Testing

Run the test to see the directory structure:

```bash
python tests/test_plot_organization.py
```

This shows the directory structure without actually creating files.
