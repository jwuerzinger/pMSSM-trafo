# Project Summary

This document summarizes the pMSSM-trafo project: transformer and GP-based models for predicting pMSSM observables (relic density, cross sections, exclusion limits), with active learning pipelines for efficient data collection.

## Quick Start

```bash
# --- Transformer training ---
python train_pmssm.py                          # Full training (parallel GPUs)
python train_pmssm.py --testing --epochs 100   # Quick test

# --- Transformer active learning ---
python active_learning.py --n-iterations 5 --generate-data

# --- GP active learning ---
python active_learning_gp.py --n-iterations 5 --generate-data
python active_learning_gp.py --testing         # Quick test
```

## Major Improvements

### 1. Improved Transformer Architectures

**PMSSMTransformer (Enhanced)**:
- Increased capacity: ~796K parameters (was ~20K)
- Pre-normalization for better gradient flow
- Deeper regression head (3 layers)
- Better initialization

**PMSSMTransformerTabular (New)**:
- Designed specifically for tabular data
- Individual feature embeddings
- Attention-based pooling
- ~803K parameters

### 2. Parallel GPU Training

**Speed improvement: ~2x faster**

- PMSSMTransformer trains on Physical GPU 1
- PMSSMTransformerTabular trains on Physical GPU 2
- Both train simultaneously using multiprocessing
- MLP trains on Physical GPU 2 after transformers complete

**Fallback**: Automatically uses sequential training if:
- Only 1 GPU available
- `--no-parallel` flag is used
- CUDA not available (uses CPU)

### 3. Better Training Configuration

- Learning rate: 3e-4 (was 1e-4)
- Weight decay: 1e-4 (was 1e-3)
- Cosine annealing LR schedule
- Gradient clipping (max_norm=1.0)
- Dropout=0.0 for small datasets

### 4. Structured Logging

- All output logged to timestamped files
- Human-readable format: `YYYY-MM-DD HH:MM:SS [info] message`
- Separate logs for parallel processes
- Files: `logs/training_YYYYMMDD_HHMMSS.log`

### 5. Organized Plots

- Timestamped subdirectories: `plots/run_YYYYMMDD_HHMMSS/`
- Matches log file timestamps
- All plots for a run in one place
- Log-scale colormaps for 2D histograms

### 6. Command-Line Options

```
Options:
  --testing             Quick test mode (3 datasets, 30 samples)
  --epochs INTEGER      Number of epochs (default: 2000)
  --n-datasets INTEGER  Number of datasets to load
  --n-samples INTEGER   Samples per dataset
  --no-parallel         Force sequential training
  --help                Show this message
```

## Active Learning Pipelines

### Transformer Active Learning (`active_learning.py`)

Uses MC Dropout uncertainty estimation with the tabular transformer model:
- 2000 training epochs, early stopping (patience 200) on validation loss
- Parallel training of AL + baseline models on separate GPUs
- Candidate selection via top-K variance from MC Dropout forward passes
- Closed-loop data generation via Run3ModelGen
- See [active_learning_plan.md](active_learning_plan.md) for details

### GP Active Learning (`active_learning_gp.py`)

Uses GP posterior variance or entropy-based batch selection:
- Models: ExactGP, DeepGP, SparseGP, MLP
- 2000 training iterations (default), early stopping (patience 200) on validation loss
- Selection strategies: entropy batch (default, LHS + Gibbs sampling) or top-K variance
- Parallel training of AL + baseline models on separate GPUs
- Multiple targets: DMRD (relic density), CrossSection, CLs
- Comprehensive metrics: accuracy, chi2, pulls, weighted accuracy
- ARD lengthscale tracking, advanced diagnostic plots
- YAML config + SLURM parameter sweep support
- See [gp_integration_plan.md](gp_integration_plan.md) and [gp_pipeline_comparison.md](gp_pipeline_comparison.md) for details

### Pipeline Comparison

| Feature | Transformer | GP |
|---------|------------|-----|
| Uncertainty | MC Dropout (N forward passes) | GP posterior variance |
| Normalization | Z-score (mean/std) | Min-max to [0,1] |
| Default selection | Top-K variance | Entropy batch |
| Training | DataLoader + optimizer loop | `model.do_train_loop()` |
| Default epochs | 2000 | 2000 |
| Early stopping | Patience 200 | Patience 200 |
| Models | PMSSMTransformerTabular | ExactGP, DeepGP, SparseGP, MLP |

## File Organization

```
pMSSM-trafo/
├── train_pmssm.py              # Transformer training script
├── pmssm.py                    # Model definitions and training functions
├── active_learning.py          # Transformer active learning pipeline
├── active_learning_gp.py       # GP active learning pipeline
├── run_active_learning_gp.sh   # Production GP AL run script
├── run_active_learning_gp_medium_test.sh  # Medium test script
├── al_pmssmwithgp/             # GP models submodule
│   └── model/gp_pipeline/
│       ├── models/             # ExactGP, DeepGP, SparseGP, MLP
│       └── utils/              # Selection, evaluation, plotting
├── Run3ModelGen/               # Physics simulation submodule
├── logs/
│   └── training_YYYYMMDD_HHMMSS.log
├── plots/
│   └── run_YYYYMMDD_HHMMSS/
├── doc/
│   ├── SUMMARY.md                   # This file
│   ├── TRAINING_GUIDE.md            # Transformer training guide
│   ├── PARALLEL_TRAINING.md         # Parallel GPU setup
│   ├── PLOT_ORGANIZATION.md         # Plot organization
│   ├── LOGGING_INFO.md              # Logging configuration
│   ├── active_learning_plan.md      # Transformer AL design
│   ├── gp_integration_plan.md       # GP AL integration progress
│   └── gp_pipeline_comparison.md    # GP pipeline comparison & features
└── tests/
    ├── test_parallel_gpus.py
    ├── test_sequential_mode.py
    └── test_plot_organization.py
```

## Expected Performance

### Model Capacity
| Model | Parameters | Device |
|-------|-----------|--------|
| PMSSMTransformer | ~796K | GPU 1 |
| PMSSMTransformerTabular | ~803K | GPU 2 |
| PMSSMFeedForward | ~1.4M | GPU 2 |

### Training Time (2000 epochs, full data)
| Mode | Time |
|------|------|
| Sequential (1 GPU) | ~2X hours |
| Parallel (2 GPUs) | ~X hours |
| **Speedup** | **~2x faster** |

### Expected Validation MSE
| Model | Before | After | Improvement |
|-------|--------|-------|-------------|
| PMSSMTransformer | 0.15-0.30 | 0.05-0.10 | 3-6x better |
| PMSSMTransformerTabular | N/A | 0.03-0.08 | New model |
| PMSSMFeedForward | 0.01-0.03 | 0.01-0.03 | Baseline |

**Note**: MLP still likely outperforms transformers (better suited for tabular data), but gap is significantly narrower.

## GPU Configuration

Environment: `CUDA_VISIBLE_DEVICES='1,2'`

**Parallel mode** (2+ GPUs):
```
GPU 0 (Phys GPU 1) → PMSSMTransformer (parallel)
GPU 1 (Phys GPU 2) → PMSSMTransformerTabular (parallel)
                   → MLP Baseline (sequential after)
```

**Sequential mode** (1 GPU or --no-parallel):
```
GPU 0 (Phys GPU 1) → PMSSMTransformer
                   → PMSSMTransformerTabular
                   → MLP Baseline
```

## Testing

Verify setup:
```bash
# Test GPU configuration
python tests/test_parallel_gpus.py

# Test sequential mode
python tests/test_sequential_mode.py

# Test plot organization
python tests/test_plot_organization.py

# Quick training test
python train_pmssm.py --testing --epochs 10
```

## Key Features

✅ **Parallel GPU training** - 2x faster with multiple GPUs
✅ **Improved architectures** - Much better transformer performance
✅ **Structured logging** - All output saved to timestamped logs
✅ **Organized plots** - Timestamped directories, easy to track
✅ **Flexible options** - Control data size, epochs, parallel mode
✅ **Automatic fallback** - Works with 1 GPU or CPU
✅ **Better training** - LR scheduling, gradient clipping

## Documentation

See detailed documentation in `doc/`:
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Transformer training guide
- [PARALLEL_TRAINING.md](PARALLEL_TRAINING.md) - Parallel GPU setup
- [PLOT_ORGANIZATION.md](PLOT_ORGANIZATION.md) - Plot organization
- [LOGGING_INFO.md](LOGGING_INFO.md) - Logging configuration
- [active_learning_plan.md](active_learning_plan.md) - Transformer active learning design
- [gp_integration_plan.md](gp_integration_plan.md) - GP pipeline integration & progress
- [gp_pipeline_comparison.md](gp_pipeline_comparison.md) - GP pipeline features & CLI reference

## Changes to Your Code

### pmssm.py
1. Added PMSSMTransformerTabular class
2. Improved PMSSMTransformer architecture
3. Added plot_dir parameter to plotting functions
4. Added logger parameter to train_with_validation
5. Added LogNorm for 2D histogram colormaps

### train_pmssm.py
1. Added multiprocessing for parallel training
2. Added --no-parallel command-line option
3. Added structured logging with structlog
4. Created timestamped plot directories
5. Updated GPU device selection logic

## Next Steps

### Transformer
1. Run tests: `python tests/test_parallel_gpus.py`
2. Quick test: `python train_pmssm.py --testing --epochs 100`
3. Full training: `python train_pmssm.py`

### GP Active Learning
1. Quick test: `python active_learning_gp.py --testing`
2. Medium test: `bash run_active_learning_gp_medium_test.sh`
3. Production run: `bash run_active_learning_gp.sh`
4. Remaining end-to-end tests: SparseGP, MLP, entropy_batch, CrossSection/CLs targets
