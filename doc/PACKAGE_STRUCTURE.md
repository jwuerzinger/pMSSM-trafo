# pmssm/ Package Structure

Documentation for the unified `pmssm/` package architecture created in February 2026.

## Overview

The `pmssm/` package consolidates all shared functionality for pMSSM machine learning models:
- **13 modules**, ~5500 lines of code
- Replaces monolithic `pmssm.py` (~3000 lines with duplication)
- Shared by both transformer and GP active learning pipelines
- Eliminates ~1500 lines of duplicated code

## Package Organization

```
pmssm/
├── __init__.py              # Package exports (public API)
├── config.py                # Constants, parameter ranges, targets
├── data.py                  # Data loading, normalization
├── datasets.py              # PyTorch Dataset classes
├── models/                  # Neural network architectures
│   ├── __init__.py
│   ├── transformer.py       # PMSSMTransformer, PMSSMTransformerTabular
│   └── feedforward.py       # PMSSMFeedForward (MLP baseline)
├── selection.py             # Candidate generation (LHS), selection strategies
├── uncertainty.py           # MC Dropout, GP posterior uncertainty
├── training.py              # Training loops for transformer & GP
├── evaluation.py            # R², metrics, lengthscale tracking
├── visualization.py         # Plotting utilities
├── logging_utils.py         # Structured logging configuration
└── model_generation.py      # Run3ModelGen interface
```

## Module Reference

### config.py

**Purpose**: Central configuration and constants

**Key contents**:
- `PARAM_RANGES`: 19 pMSSM parameter ranges (min, max, log-space flag)
- `PARAM_NAMES`: Parameter names (meL, meR, mtauL, ..., mA, tanb)
- `TARGET_CONFIG`: Target observables (DMRD, CrossSection, CLs) with true values, thresholds, ROOT branch names
- `WANDB_PROJECT`: W&B project name
- `DEFAULT_DATA_DIR`: Default ROOT data location
- `RUN3_MODELGEN_PATH`: Path to Run3ModelGen submodule

**Usage**:
```python
from pmssm.config import PARAM_RANGES, TARGET_CONFIG

# Access parameter ranges
meL_min, meL_max = PARAM_RANGES['meL']

# Get target configuration
dmrd_config = TARGET_CONFIG['DMRD']
true_value = dmrd_config['true_value']    # 0.12
threshold = dmrd_config['threshold']       # 0.02
branch_name = dmrd_config['branch']        # 'DMRD'
```

**When to use**: Always import constants from here rather than hardcoding values.

### data.py

**Purpose**: Data loading and normalization

**Key functions**:
- `load_pmssm_data(data_dir, n_datasets, n_samples)`: Load pMSSM data from ROOT files
- `normalize_X(X, method)`: Normalize input features (z-score or minmax)
- `denormalize_X(X_norm, method)`: Reverse normalization
- `normalize_Y(Y, target)`: Normalize target observable
- `denormalize_Y(Y_norm, target)`: Reverse target normalization
- `get_normalization_params()`: Get normalization statistics

**Usage**:
```python
from pmssm.data import load_pmssm_data, normalize_X, normalize_Y

# Load data
X, Y = load_pmssm_data(data_dir="data/", n_datasets=10, n_samples=5000)

# Normalize for training
X_norm, norm_params = normalize_X(X, method='standard')  # z-score
Y_norm, _ = normalize_Y(Y, target='DMRD')

# Later: denormalize predictions for evaluation
Y_pred = denormalize_Y(Y_pred_norm, target='DMRD')
```

**Normalization methods**:
- `standard`: Z-score normalization (μ=0, σ=1) - used by transformers
- `minmax`: Min-max to [0,1] - used by GP models

**When to use**: Use these functions for all data loading and normalization to ensure consistency.

### datasets.py

**Purpose**: PyTorch Dataset classes

**Key classes**:
- `PMSSMDataset(X, Y)`: Simple dataset wrapper
- `PMSSMDatasetWithNorm(X, Y, X_mean, X_std)`: Dataset with embedded normalization params

**Usage**:
```python
from pmssm.datasets import PMSSMDataset
from torch.utils.data import DataLoader

dataset = PMSSMDataset(X_train, Y_train)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

for batch_X, batch_Y in dataloader:
    # Training loop
    pass
```

**When to use**: Use `PMSSMDataset` for simple cases, `PMSSMDatasetWithNorm` when normalization parameters need to travel with the data.

### models/

**Purpose**: Neural network architectures

#### models/transformer.py

**Classes**:
- `PMSSMTransformer`: Standard transformer with CLS token pooling (~796K parameters)
  - Pre-normalization architecture
  - Learnable positional encodings
  - 3-layer regression head

- `PMSSMTransformerTabular`: Transformer for tabular data (~803K parameters)
  - Individual learned embeddings per feature
  - Attention pooling instead of CLS token
  - Better suited for non-sequential pMSSM data

**Usage**:
```python
from pmssm.models.transformer import PMSSMTransformerTabular

model = PMSSMTransformerTabular(
    input_dim=19,
    d_model=128,
    nhead=4,
    num_layers=3,
    dim_feedforward=256,
    dropout=0.1
)

# Forward pass
predictions = model(X)

# With dropout for uncertainty
model.train()  # Enable dropout
predictions = model(X)
```

**When to use**: Default choice for transformer-based AL pipeline. `PMSSMTransformerTabular` typically performs better on pMSSM data.

#### models/feedforward.py

**Classes**:
- `PMSSMFeedForward`: Multi-layer perceptron baseline (~1.4M parameters)
  - Feature embedding layer
  - 4-layer fully connected network
  - ReLU activation
  - Often outperforms transformers on tabular data

**Usage**:
```python
from pmssm.models.feedforward import PMSSMFeedForward

model = PMSSMFeedForward(
    input_dim=19,
    hidden_dim=64,
    num_layers=4,
    dropout=0.0
)

predictions = model(X)
```

**When to use**: Use as baseline comparison for transformer models.

### selection.py

**Purpose**: Candidate generation and point selection strategies

**Key functions**:
- `generate_candidates_lhs(n_candidates)`: Latin Hypercube Sampling for candidate generation
- `generate_candidates_uniform(n_candidates)`: Uniform random sampling
- `select_top_k(uncertainties, n_select)`: Select top-K points by uncertainty
- `select_entropy_batch(...)`: Entropy-based batch selection with Gibbs sampling
- `select_proximity_weighted(...)`: Select points weighted by proximity to target value

**Usage**:
```python
from pmssm.selection import generate_candidates_lhs, select_top_k

# Generate candidate points
candidates = generate_candidates_lhs(n_candidates=10000)

# Compute uncertainties (e.g., MC Dropout variance)
uncertainties = compute_mc_dropout_variance(model, candidates)

# Select top-K most uncertain points
selected_indices = select_top_k(uncertainties, n_select=100)
selected_points = candidates[selected_indices]
```

**Selection strategies**:

**Top-K** (simple):
```python
indices = select_top_k(uncertainties, n_select=100)
```

**Entropy Batch** (diverse):
```python
indices = select_entropy_batch(
    model=model,
    candidates=candidates,
    n_select=100,
    entropy_blur=0.15,
    entropy_beta=50.0,
    tolerance_sampling=1.0,
    proximity_sampling=0.1
)
```

**Proximity Weighted** (focused):
```python
indices = select_proximity_weighted(
    predictions=predictions,
    uncertainties=uncertainties,
    n_select=100,
    target_value=0.12,
    proximity_width=0.1
)
```

**When to use**: Use `generate_candidates_lhs` for better space coverage. Choose selection strategy based on exploration vs. exploitation tradeoff.

### uncertainty.py

**Purpose**: Uncertainty estimation for active learning

**Key functions**:
- `mc_dropout_uncertainty(model, X, n_samples)`: MC Dropout for transformers
- `gp_posterior_variance(gp_model, X)`: GP posterior variance
- `gp_entropy(gp_model, X)`: GP entropy for batch selection

**Usage**:

**MC Dropout** (transformers):
```python
from pmssm.uncertainty import mc_dropout_uncertainty

model.train()  # Enable dropout
mean, variance = mc_dropout_uncertainty(
    model=model,
    X=candidates,
    n_samples=30
)

# Select points with high variance
```

**GP Posterior** (GP models):
```python
from pmssm.uncertainty import gp_posterior_variance

variance = gp_posterior_variance(gp_model, candidates)

# Select points with high posterior variance
```

**When to use**: Use `mc_dropout_uncertainty` for transformers, `gp_posterior_variance` for GP models.

### training.py

**Purpose**: Training loops for both transformer and GP models

**Key functions**:
- `train_transformer_worker(gpu_id, X, Y, idx_train, ...)`: Transformer training worker (multiprocessing)
- `train_gp_worker(gpu_id, X, Y, idx_train, ...)`: GP training worker (multiprocessing)
- `train_with_early_stopping(model, train_loader, val_loader, ...)`: Generic training with early stopping
- `get_best_checkpoint_path(output_dir, iteration)`: Find checkpoint from previous iteration for warm-starting

**Usage**:

**Transformer training** (in parallel):
```python
from pmssm.training import train_transformer_worker
import multiprocessing as mp

# Launch parallel training
al_process = mp.Process(
    target=train_transformer_worker,
    args=(gpu_id, X, Y, idx_train, idx_val, config)
)
al_process.start()
al_process.join()
```

**GP training** (standalone):
```python
from pmssm.training import train_gp_worker

# Train on specific GPU
train_gp_worker(
    gpu_id=0,
    X=X_train,
    Y=Y_train,
    idx_train=idx_train,
    idx_val=idx_val,
    config=config
)
```

**When to use**: Use these workers for active learning pipelines. They handle:
- GPU assignment
- Warm-starting from checkpoints
- Early stopping with patience
- Logging and checkpoint saving

### evaluation.py

**Purpose**: Model evaluation and metrics

**Key functions**:
- `compute_r2(y_true, y_pred)`: R² score
- `compute_mse(y_true, y_pred)`: Mean squared error
- `compute_rmse(y_true, y_pred)`: Root mean squared error
- `compute_accuracy(y_true, y_pred, threshold)`: Accuracy within threshold
- `compute_gof_metrics(y_true, y_pred, uncertainty)`: Goodness-of-fit (chi², pulls)
- `compute_weighted_accuracy(y_true, y_pred, alpha)`: Weighted accuracy
- `compute_comprehensive_metrics(model, X, Y, target)`: Full evaluation suite
- `extract_lengthscales(gp_model)`: Extract ARD lengthscales from GP models

**Usage**:

**Basic metrics**:
```python
from pmssm.evaluation import compute_r2, compute_mse

r2 = compute_r2(y_true, y_pred)
mse = compute_mse(y_true, y_pred)
```

**Comprehensive evaluation**:
```python
from pmssm.evaluation import compute_comprehensive_metrics

metrics = compute_comprehensive_metrics(
    model=model,
    X=X_eval,
    Y=Y_eval,
    target='DMRD'
)

# Returns dict with:
# - accuracy_1sig, accuracy_2sig, accuracy_3sig
# - weighted_accuracy_1, _2, _5, _10
# - mse, rmse, r2
# - chi2, reduced_chi2
# - pull_mean, pull_std
```

**GP lengthscales**:
```python
from pmssm.evaluation import extract_lengthscales

lengthscales = extract_lengthscales(gp_model)
# Returns dict: {'meL': 0.5, 'meR': 1.2, ..., 'tanb': 0.8}
```

**When to use**: Use `compute_comprehensive_metrics` for detailed evaluation. Use `extract_lengthscales` to track feature importance in GP models.

### visualization.py

**Purpose**: Plotting utilities

**Key functions**:
- `plot_losses(train_losses, val_losses, save_path)`: Loss curves
- `scatter_true_vs_pred(y_true, y_pred, save_path, mode)`: Scatter plot
- `hist_true_vs_pred(y_true, y_pred, save_path, mode)`: Histogram overlay
- `compare_random_predictions(y_true, y_pred, save_path, mode)`: Random sample bar chart
- `plot_iteration_metrics(results, save_path)`: AL vs baseline comparison
- `plot_gp_losses(...)`: GP-specific loss curves
- `scatter_true_vs_pred_gp(...)`: GP-specific scatter
- `hist_true_vs_pred_gp(...)`: GP-specific histogram
- `compare_random_predictions_gp(...)`: GP-specific bar chart

**Usage**:

**Training diagnostics**:
```python
from pmssm.visualization import plot_losses, scatter_true_vs_pred

# Loss curves
plot_losses(
    train_losses=train_losses,
    val_losses=val_losses,
    save_path="plots/losses.png",
    model_name="PMSSMTransformerTabular"
)

# Predictions vs truth
scatter_true_vs_pred(
    y_true=y_val,
    y_pred=y_pred,
    save_path="plots/scatter_val.png",
    mode="validation"
)
```

**AL progress tracking**:
```python
from pmssm.visualization import plot_iteration_metrics

# Compare AL vs baseline across iterations
plot_iteration_metrics(
    results=results,  # Dict with train_loss, val_loss, r2, dataset_size per iteration
    save_path="plots/iteration_metrics.png"
)
```

**When to use**: Use these functions to generate consistent plots across both pipelines.

### logging_utils.py

**Purpose**: Structured logging configuration

**Key functions**:
- `setup_logger(log_file, name)`: Configure structlog logger
- `get_logger(name)`: Get configured logger

**Usage**:
```python
from pmssm.logging_utils import setup_logger, get_logger

# Setup logger for script
logger = setup_logger(
    log_file="logs/training.log",
    name="train_pmssm"
)

# Log messages
logger.info("Starting training", epoch=1, loss=0.5)
logger.warning("High loss detected", loss=10.0)
logger.error("Training failed", error=str(e))
```

**Output format**:
```
2026-02-13 10:30:45 [info    ] Starting training              epoch=1 loss=0.5
2026-02-13 10:30:46 [warning ] High loss detected             loss=10.0
2026-02-13 10:30:47 [error   ] Training failed                error='...'
```

**When to use**: Use at the start of every script for consistent logging.

### model_generation.py

**Purpose**: Interface to Run3ModelGen for physics simulation

**Key functions**:
- `generate_models_from_csv(csv_path, output_dir, n_workers)`: Generate pMSSM models
- `create_modelgen_config(selected_points, output_dir)`: Create Run3ModelGen config

**Usage**:
```python
from pmssm.model_generation import generate_models_from_csv

# Save selected points
selected_points.to_csv("selected_points.csv", index=False)

# Generate models
generate_models_from_csv(
    csv_path="selected_points.csv",
    output_dir="scan/",
    n_workers=20
)
```

**When to use**: Use with `--generate-data` flag in active learning pipelines to automatically run SPheno + micromegas for selected points.

## Backward Compatibility

### pmssm.py

The old monolithic `pmssm.py` file is maintained as a backward compatibility wrapper:

```python
"""
DEPRECATED: This file is maintained for backward compatibility only.

All functionality has been moved to the pmssm/ package.
Please update imports:
    from pmssm import PMSSMTransformer  # OLD
    from pmssm.models.transformer import PMSSMTransformer  # NEW

This wrapper will be removed in a future version.
"""
from pmssm import *
```

**Migration guide**:

| Old Import | New Import |
|------------|------------|
| `from pmssm import PMSSMTransformer` | `from pmssm.models.transformer import PMSSMTransformer` |
| `from pmssm import PMSSMFeedForward` | `from pmssm.models.feedforward import PMSSMFeedForward` |
| `from pmssm import load_pmssm_data` | `from pmssm.data import load_pmssm_data` |
| `from pmssm import normalize_X` | `from pmssm.data import normalize_X` |
| `from pmssm import plot_losses` | `from pmssm.visualization import plot_losses` |

## Benefits of Modular Structure

### 1. Eliminates Duplication
Before restructuring:
- `active_learning.py`: 760 lines
- `active_learning_gp.py`: 990 lines
- Shared code duplicated in both files

After restructuring:
- ~1500 lines moved to shared package
- Single source of truth for common functionality

### 2. Improves Testability
Each module can be tested independently:
```python
# Test data loading
from pmssm.data import load_pmssm_data
X, Y = load_pmssm_data(...)

# Test normalization
from pmssm.data import normalize_X, denormalize_X
X_norm, params = normalize_X(X)
X_recovered = denormalize_X(X_norm, params)
assert np.allclose(X, X_recovered)
```

### 3. Easier Maintenance
Bug fixes and improvements in one place:
- Fix in `pmssm/data.py` → both pipelines benefit
- Add feature in `pmssm/selection.py` → available to both pipelines
- Update plotting in `pmssm/visualization.py` → consistent plots everywhere

### 4. Clear Responsibilities
Each module has a single, clear purpose:
- `data.py`: Loading and normalization
- `models/`: Neural network architectures
- `training.py`: Training loops
- `evaluation.py`: Metrics and evaluation
- `visualization.py`: Plotting

### 5. Scalability
Easy to add new features:
- New model? Add to `models/`
- New selection strategy? Add to `selection.py`
- New target observable? Update `config.py`
- New evaluation metric? Add to `evaluation.py`

## Usage Patterns

### Standard Imports

```python
# Configuration
from pmssm.config import PARAM_RANGES, TARGET_CONFIG

# Data handling
from pmssm.data import load_pmssm_data, normalize_X, normalize_Y

# Models
from pmssm.models.transformer import PMSSMTransformerTabular
from pmssm.models.feedforward import PMSSMFeedForward

# Active learning
from pmssm.selection import generate_candidates_lhs, select_top_k
from pmssm.uncertainty import mc_dropout_uncertainty

# Training
from pmssm.training import train_transformer_worker

# Evaluation
from pmssm.evaluation import compute_comprehensive_metrics

# Visualization
from pmssm.visualization import plot_losses, scatter_true_vs_pred

# Logging
from pmssm.logging_utils import setup_logger
```

### Typical Active Learning Script Structure

```python
# 1. Setup
from pmssm.logging_utils import setup_logger
logger = setup_logger("output/al.log", "active_learning")

# 2. Load and normalize data
from pmssm.data import load_pmssm_data, normalize_X, normalize_Y
X, Y = load_pmssm_data(...)
X_norm, norm_params = normalize_X(X, method='standard')
Y_norm, _ = normalize_Y(Y, target='DMRD')

# 3. Train model
from pmssm.models.transformer import PMSSMTransformerTabular
from pmssm.training import train_transformer_worker
model = PMSSMTransformerTabular(...)
train_transformer_worker(gpu_id=0, X=X_norm, Y=Y_norm, ...)

# 4. Generate candidates and select points
from pmssm.selection import generate_candidates_lhs, select_top_k
from pmssm.uncertainty import mc_dropout_uncertainty
candidates = generate_candidates_lhs(n_candidates=10000)
mean, variance = mc_dropout_uncertainty(model, candidates, n_samples=30)
selected_idx = select_top_k(variance, n_select=100)

# 5. Generate new data
from pmssm.model_generation import generate_models_from_csv
generate_models_from_csv(csv_path="selected.csv", output_dir="scan/", n_workers=20)

# 6. Evaluate
from pmssm.evaluation import compute_comprehensive_metrics
metrics = compute_comprehensive_metrics(model, X_eval, Y_eval, target='DMRD')

# 7. Visualize
from pmssm.visualization import plot_iteration_metrics
plot_iteration_metrics(results, save_path="plots/metrics.png")
```

## See Also

- [CLI_REFERENCE.md](CLI_REFERENCE.md) - Complete command-line interface reference
- [active_learning_plan.md](active_learning_plan.md) - Transformer AL design
- [gp_pipeline_comparison.md](gp_pipeline_comparison.md) - GP pipeline features
- [HARMONIZATION_SUMMARY.md](HARMONIZATION_SUMMARY.md) - Feb 2026 refactoring details
- [README.md](../README.md) - Quick start guide
