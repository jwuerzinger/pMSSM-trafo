# Training Guide - Updated Transformer Models

## Quick Start

To test the improved transformer models, run:

```bash
python train_pmssm.py
```

For faster testing (limited data):
```bash
python train_pmssm.py --testing
```

To customize the number of epochs:
```bash
python train_pmssm.py --epochs 1000
```

To customize data loading:
```bash
# Load only 5 datasets
python train_pmssm.py --n-datasets 5

# Use only 100 samples per dataset
python train_pmssm.py --n-samples 100

# Combine: 5 datasets, 100 samples each, 500 epochs
python train_pmssm.py --n-datasets 5 --n-samples 100 --epochs 500
```

Or combine options:
```bash
python train_pmssm.py --testing --epochs 100
```

## What's Been Changed

### 1. [pmssm.py](pmssm.py)

**Enhanced PMSSMTransformer** (lines 170-247):
- Pre-normalization architecture (`norm_first=True`)
- Richer input embedding with LayerNorm + ReLU
- Deeper 3-layer regression head
- Better initialization (0.02 scaling)
- Dropout=0.0 by default for small datasets

**New PMSSMTransformerTabular** (lines 249-316):
- Individual learned embeddings per feature
- Attention pooling instead of CLS token
- Specifically designed for tabular data
- Should perform better than sequence-based approach

**Updated train_with_validation** (lines 369-461):
- Added `scheduler` parameter for learning rate scheduling
- Added `grad_clip` parameter for gradient clipping
- Prints learning rate during training
- Reduced print frequency (every 100 epochs)

**New get_model_name** (lines 508-520):
- Helper function to generate unique filenames for each model type
- Returns "transformer" for PMSSMTransformer
- Returns "transformer_tabular" for PMSSMTransformerTabular
- Returns "MLP" for PMSSMFeedForward

### 2. [train_pmssm.py](train_pmssm.py)

**GPU Selection Fix:**
- `CUDA_VISIBLE_DEVICES` now set before importing torch (lines 1-3)
- Ensures GPU 1 is used when available
- Added GPU name print statement for verification

**Command-line Options:**
- `--testing`: Use limited data for quick testing (n_datasets=3, n_samples=30)
- `--epochs N`: Set number of training epochs (default: 2000)
- `--n-datasets N`: Number of datasets to load (-1 for all, overrides --testing)
- `--n-samples N`: Number of samples per dataset (None for all, overrides --testing)

The script now trains **3 models sequentially**:

1. **Improved PMSSMTransformer**
   - d_model=128, nhead=4, num_layers=3
   - dim_feedforward=512
   - Learning rate: 3e-4 with cosine annealing
   - Gradient clipping at 1.0

2. **PMSSMTransformerTabular** (new)
   - Same hyperparameters as above
   - Better suited for tabular data

3. **PMSSMFeedForward (MLP baseline)**
   - Unchanged from your original config

Each model generates its own plots in the `plots/` directory.

## Expected Results

Based on the improvements:

| Model | Original Val MSE | Expected Improved Val MSE |
|-------|------------------|---------------------------|
| PMSSMTransformer | ~0.15-0.30 | ~0.05-0.10 |
| PMSSMTransformerTabular | N/A | ~0.03-0.08 |
| PMSSMFeedForward | ~0.01-0.03 | ~0.01-0.03 (unchanged) |

The gap between transformer and MLP should be **significantly reduced**, though the MLP will likely still perform best since it's fundamentally better suited for tabular regression.

## Key Improvements Applied

1. **Increased Model Capacity**: Matching MLP parameter count
2. **Better Architecture**: Pre-normalization for gradient flow
3. **Optimized Hyperparameters**: Higher LR (3e-4), lower weight decay (1e-4)
4. **Learning Rate Scheduling**: Cosine annealing with warmup
5. **Gradient Clipping**: Prevents exploding gradients
6. **Tabular-Specific Design**: PMSSMTransformerTabular treats features appropriately

## Monitoring Training

Watch for these signs of success:

- **Validation loss should decrease steadily** (not plateau early)
- **No large spikes** in training loss (grad clipping helps)
- **Learning rate decreases** over time (cosine schedule)
- **Final val MSE < 0.10** for transformers (vs ~0.15-0.30 before)

## Output Files

All results are saved to `plots/`:
- `transformer_losses.png` - PMSSMTransformer training curves
- `transformer_tabular_losses.png` - PMSSMTransformerTabular training curves
- `MLP_losses.png` - MLP training curves
- `*_true_vs_pred_train.png` - Training scatter plots
- `*_true_vs_pred_validation.png` - Validation scatter plots
- `*_hist_true_vs_pred_train.png` - Training histograms
- `*_hist_true_vs_pred_validation.png` - Validation histograms

Each model creates its own set of plots with a unique prefix:
- `transformer_*` for PMSSMTransformer
- `transformer_tabular_*` for PMSSMTransformerTabular
- `MLP_*` for PMSSMFeedForward

## Next Steps

If transformers still underperform significantly:

1. **Accept MLP superiority**: Transformers aren't ideal for tabular data
2. **Try ensemble**: Combine transformer + MLP predictions
3. **Feature engineering**: Add interaction features explicitly
4. **Different architecture**: Try TabNet or FT-Transformer from research

## Troubleshooting

**CUDA out of memory?**
- Reduce `batch_size` from 256 to 128 in [train_pmssm.py](train_pmssm.py):37

**Training too slow?**
- Run with `--testing` flag (uses less data)
- Use `--epochs 1000` to reduce training time
- Use `--n-datasets 5` to load fewer datasets
- Use `--n-samples 1000` to use fewer samples
- Examples:
  - `python train_pmssm.py --testing --epochs 500`
  - `python train_pmssm.py --n-datasets 5 --n-samples 1000 --epochs 1000`

**Models not improving?**
- Check data normalization (already handled in your code)
- Ensure CUDA is being used: "Set device to: cuda"
- Try reducing learning rate to 1e-4

## Files Modified

- [pmssm.py](pmssm.py) - Model architectures and training function
- [train_pmssm.py](train_pmssm.py) - Training script with improved configs
- [transformer_configs.md](transformer_configs.md) - Detailed explanation of changes
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - This file
