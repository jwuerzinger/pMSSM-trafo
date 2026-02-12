# GP Pipeline Comparison & Feature Integration

## Overview

This document compares the two GP-based pipelines in the repository and documents the features ported from the `al_pmssmwithgp` submodule into `active_learning_gp.py`.

- **Original pipeline**: `al_pmssmwithgp/model/gp_pipeline/` (entry: `main.py --config config3.yaml`)
- **Refactored pipeline**: `active_learning_gp.py` (main repo)

The original pipeline (`al_pmssmwithgp`) is a self-contained GP-based active learning system with its own data loading, model training, evaluation, and physics simulation. The refactored pipeline (`active_learning_gp.py`) integrates GP models into the main repo's infrastructure, sharing utilities with the transformer-based `active_learning.py` (data generation, candidate pooling, logging, plotting).

## Pipeline Comparison

### Execution Flow

| Step | al_pmssmwithgp (`pipeline.py`) | active_learning_gp.py |
|------|-------------------------------|----------------------|
| Data loading | Custom ROOT/CSV loading via `GPModelPipeline.__init__` | `pmssm.load_pmssm_data()` (shared with transformer pipeline) |
| Model init | `GPModelPipeline.initialize_model()` | `create_gp_model()` standalone function |
| Training | `gp_pipeline.train_model()` / `do_train_loop()` | `train_gp_worker()` with parallel GPU support |
| Warm-starting | Load checkpoint if iteration > 1 | Same, with `--warm-starting` flag |
| Evaluation | `evaluate_and_log()` on full true dataset | R2 on val split + optional `--compute-full-metrics` on val and/or `--eval-data-path` |
| Point selection | `EntropySelectionStrategy.select_new_points()` or random | `--selection-strategy entropy_batch` (default) or `top_k` |
| Data generation | `Run3PhysicsInterface.generate_targets()` | `generate_models_from_csv()` (shared with transformer pipeline) |
| Baseline comparison | None | Parallel baseline model trained on random data |
| Lengthscales | `save_lengthscale()` to CSV | `--track-lengthscales` flag, saves `lengthscales.csv` |
| Plotting | GP vs truth, 2D/4D slices, entropy histograms | Loss curves, diagnostics + optional `--advanced-plots` |

### Model Support

| Model | al_pmssmwithgp | active_learning_gp.py |
|-------|---------------|----------------------|
| ExactGP | Yes | Yes |
| DeepGP | Yes | Yes |
| SparseGP | Yes | Yes (ported) |
| MLP | Yes | Yes (ported, no uncertainty) |

### Target Functions

| Target | al_pmssmwithgp | active_learning_gp.py |
|--------|---------------|----------------------|
| DMRD (relic density) | Yes (default) | Yes (default) |
| CrossSection | Yes | Yes (ported via `--target`) |
| CLs | Yes | Yes (ported via `--target`) |
| Toy (synthetic) | Yes | Not ported (testing only) |

### Selection Strategies

| Strategy | al_pmssmwithgp | active_learning_gp.py |
|----------|---------------|----------------------|
| Random | Yes (`is_active=False`) | Via MLP fallback |
| Top-K variance | No | Yes (`--selection-strategy top_k`) |
| Entropy batch | Yes (LHS + threshold filter + Gibbs sampling) | Yes (default: `--selection-strategy entropy_batch`) |

### Training Configuration

| Setting | al_pmssmwithgp | active_learning_gp.py |
|---------|---------------|----------------------|
| Default iterations | 1000 | 2000 (`--training-iterations`) |
| Early stopping | Not supported | Patience 200 on val loss (`--early-stopping --patience 200`) |
| Best model tracking | ExactGP: train loss (bug); others: val loss | All models: val loss (fixed) |
| Learning rate | 0.005 (in `do_train_loop`) | 1e-3 (`--learning-rate`) |
| Warm starting | Checkpoint loading | Same (`--warm-starting`) |

### Diagnostic Plots

| Plot | al_pmssmwithgp | active_learning_gp.py |
|------|---------------|----------------------|
| Loss curves | Via pipeline | `plot_gp_losses` (train + val) |
| True vs predicted scatter | `plotGPTrueDifference` | `scatter_true_vs_pred_gp` (train + val) |
| Histogram overlay | Not present | `hist_true_vs_pred_gp` (train + val) |
| Random sample predictions | Not present | `compare_random_predictions_gp` (train + val) |
| Advanced diagnostics | 2D/4D slices, entropy histograms | `plot_advanced_diagnostics` (`--advanced-plots`) |

## Ported Features

### 1. Multiple Target Functions
**Source**: `models/base.py` lines 63-69

Added `TARGET_CONFIG` dict mapping target names to true values, thresholds, and ROOT branch names. The `transform_y()` function is now target-aware (CLs values are untransformed). CLI: `--target DMRD|CrossSection|CLs`.

### 2. Comprehensive Evaluation Metrics
**Source**: `utils/evaluation.py`

Imported directly from submodule: `compute_accuracy`, `compute_gof_metrics`, `compute_weighted_accuracy`, `misclassified`. Added `compute_comprehensive_metrics()` wrapper that evaluates a model and returns a consolidated dict with: accuracy, weighted accuracy (alpha=1,2,5,10), MSE, RMSE, R2, chi2, reduced chi2, pulls, and weighted variants. CLI: `--compute-full-metrics`. Saves per-iteration CSV files.

### 3. MLP Model Support
**Source**: `models/mlp.py`

3-layer MLP (input -> 64 -> 32 -> 1, ReLU) imported directly. Since MLP has no native uncertainty estimate, the pipeline warns and falls back to random selection when used with active learning. Useful as a comparison model. CLI: `--model-type mlp`.

### 4. Sparse GP Support
**Source**: `models/sparse_gp.py`

Variational GP with inducing points (KMeans or vanilla initialization), trained with VariationalELBO. CLI: `--model-type sparse_gp --inducing-strategy kmeans|vanilla`.

### 5. Entropy-Based Batch Active Learning
**Source**: `utils/selection.py` — `EntropySelectionStrategy`

The most complex feature. Adapted `select_new_points()` to work with explicit model + normalization arguments instead of a pipeline object. The algorithm:

1. Sample 1M points via Latin Hypercube Sampling
2. Filter candidates near the decision threshold (`--tolerance-sampling`)
3. Weight by proximity to threshold (`--proximity-sampling`)
4. Select top candidates by entropy score into focused pool
5. Compute full covariance matrix on focused pool
6. Iteratively select points using batch entropy (log-determinant scoring)
7. Use Gibbs sampling with temperature beta for stochastic selection

CLI: `--selection-strategy entropy_batch --entropy-blur 0.15 --entropy-beta 50 --tolerance-sampling 1.0 --proximity-sampling 0.1`

### 6. Lengthscale Tracking
**Source**: `models/base.py:save_lengthscale`

Extracts learned ARD lengthscales from ExactGP/SparseGP kernels after training. Saves one row per iteration to `lengthscales.csv`. Not applicable to MLP/DeepGP. CLI: `--track-lengthscales` (on by default).

### 7. Full True Dataset Evaluation
**Source**: `utils/evaluation.py:evaluate_and_log`

Loads a separate evaluation dataset (ROOT or CSV) and evaluates the model on it, independent of the training/validation split. CLI: `--eval-data-path /path/to/data.root`.

### 8. Advanced Plotting
**Source**: `utils/plotting.py`

Added `plot_advanced_diagnostics()` that generates: true vs predicted scatter, residual distribution, and residuals vs true values. CLI: `--advanced-plots`.

### 9. YAML Config + Parameter Sweep
**Source**: `main.py:apply_sweep_combination`

Added `load_config_with_sweep()` that loads a YAML config file and treats list-valued parameters as sweep dimensions. Enables SLURM array jobs. CLI: `--config-file sweep.yaml --sweep-index $SLURM_ARRAY_TASK_ID`.

### 10. Early Stopping on Validation Loss

Added early stopping with configurable patience to all four model types (`ExactGP`, `DeepGP`, `SparseGP`, `MLP`). Each model's `do_train_loop()` now accepts a `patience` parameter:

- **`patience=None`**: Disables early stopping (backward compatible)
- **`patience=N`**: Stops training after N consecutive iterations without validation loss improvement

Bugs fixed during integration:
- **ExactGP**: Best model was tracked on *training* loss instead of validation loss — fixed
- **SparseGP**: Best-model tracking code was incorrectly indented inside `with torch.no_grad()` — fixed

CLI: `--early-stopping` (default on), `--patience 200` (default). Disable with `--no-early-stopping`.

### 11. Diagnostic Plot Parity with active_learning.py

Added four diagnostic plot functions that match the transformer pipeline's per-iteration diagnostics:

- `plot_gp_losses(train_losses, val_losses, ...)` — training and validation loss curves
- `scatter_true_vs_pred_gp(y_true, y_pred, mode, ...)` — scatter plot for train and validation
- `hist_true_vs_pred_gp(y_true, y_pred, mode, ...)` — overlapping histograms of true and predicted
- `compare_random_predictions_gp(y_true, y_pred, mode, ...)` — bar chart of random sample predictions

All plots are generated per model (AL and baseline) per iteration, saved to `iteration_NNN/plots/{al,baseline}/`.

## Key Architectural Differences

The original pipeline uses an **object-oriented design** where `GPModelPipeline` holds all state (model, data, config). The refactored pipeline uses **standalone functions** with explicit arguments, which:

- Enables parallel training on multiple GPUs (workers can't share object state across processes)
- Allows importing individual functions for testing/notebooks
- Keeps the baseline comparison logic clean (two independent training runs)

## Output Directory Structure

```
active_learning_gp_output/
├── active_learning.log              # Main pipeline log
├── iteration_001/
│   ├── selected_points.csv          # Selected points with uncertainty/entropy
│   ├── model_checkpoint.pt          # AL model checkpoint
│   ├── al_training.log              # AL model training log
│   ├── baseline_training.log        # Baseline model training log
│   ├── gof_al.csv                   # [If --compute-full-metrics] GoF metrics
│   ├── gof_baseline.csv             # [If --compute-full-metrics] GoF metrics
│   ├── plots/
│   │   ├── al/                      # Active Learning model diagnostics
│   │   │   ├── losses_exact_gp.png
│   │   │   ├── exact_gp_true_vs_pred_train.png
│   │   │   ├── exact_gp_true_vs_pred_validation.png
│   │   │   ├── exact_gp_hist_true_vs_pred_train.png
│   │   │   ├── exact_gp_hist_true_vs_pred_validation.png
│   │   │   ├── exact_gp_random_predictions_train.png
│   │   │   └── exact_gp_random_predictions_validation.png
│   │   └── baseline/                # Baseline model diagnostics
│   │       └── ... (same plot set)
│   ├── modelgen_config.yaml         # [If --generate-data]
│   └── scan/                        # [If --generate-data]
├── iteration_002/
│   └── ...
├── plots/
│   └── iteration_metrics.png        # Comparison plot (Loss, R², Dataset Size)
├── lengthscales.csv                 # [If --track-lengthscales] ARD per iteration
└── summary.json                     # Configuration and all results
```

## Usage Examples

### Basic (uses entropy_batch selection, early stopping, 2000 iterations by default)
```bash
python active_learning_gp.py --model-type exact_gp --n-iterations 10
```

### With top-K selection (simpler, faster)
```bash
python active_learning_gp.py --selection-strategy top_k --n-iterations 10
```

### Disable early stopping (train for full iteration budget)
```bash
python active_learning_gp.py --no-early-stopping --training-iterations 500
```

### Custom early stopping patience
```bash
python active_learning_gp.py --patience 100 --training-iterations 3000
```

### Full metrics with true dataset evaluation
```bash
python active_learning_gp.py --compute-full-metrics \
    --eval-data-path /path/to/true_data.root
```

### Sparse GP with lengthscale tracking
```bash
python active_learning_gp.py --model-type sparse_gp \
    --inducing-strategy kmeans --track-lengthscales
```

### MLP comparison (no uncertainty — falls back to random selection)
```bash
python active_learning_gp.py --model-type mlp --n-iterations 5
```

### Production run with data generation
```bash
python active_learning_gp.py \
    --model-type exact_gp \
    --n-iterations 40 \
    --n-samples 100000 \
    --n-select 20000 \
    --n-candidates 50000 \
    --generate-data \
    --gen-workers 20
```

### Parameter sweep via SLURM
```bash
# sweep_config.yaml:
#   learning_rate: [0.001, 0.01, 0.1]
#   kernel: [RBF, Matern]
#SBATCH --array=0-5
python active_learning_gp.py --config-file sweep_config.yaml \
    --sweep-index $SLURM_ARRAY_TASK_ID
```

### CrossSection target
```bash
python active_learning_gp.py --target CrossSection --model-type exact_gp
```

## CLI Options

### Core Options

| Option | Default | Description |
|--------|---------|-------------|
| `--testing` | off | Testing mode (small data, few iterations) |
| `--n-iterations` | 1 | Number of active learning iterations |
| `--n-candidates` | 1000 | Candidate pool size |
| `--n-select` | 10 | Points to select per iteration |
| `--n-datasets` | None | Number of ROOT datasets to load |
| `--n-samples` | None | Number of samples to use |
| `--output-dir` | active_learning_gp_output | Output directory |
| `--generate-data` | off | Generate new models via Run3ModelGen |
| `--gen-workers` | 1 | Parallel generation workers |

### Model Options

| Option | Default | Description |
|--------|---------|-------------|
| `--target` | DMRD | Target function (DMRD, CrossSection, CLs) |
| `--model-type` | exact_gp | Model: exact_gp, deep_gp, sparse_gp, mlp |
| `--kernel` | RBF | Kernel type (RBF, Matern, RQK, RBF+Matern) |
| `--lengthscale` | 1.0 | Initial kernel lengthscale |
| `--noise` | 1e-2 | Initial noise level |
| `--jitter` | 1e-3 | Cholesky jitter |
| `--use-ard` | on | Automatic Relevance Determination |
| `--use-dkl` | off | Deep Kernel Learning (ExactGP only) |
| `--inducing-strategy` | kmeans | Inducing point init (sparse_gp only) |

### Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--learning-rate` | 1e-3 | Optimizer learning rate |
| `--training-iterations` | 2000 | Max training iterations per AL iteration |
| `--early-stopping` | on | Early stopping on validation loss |
| `--patience` | 200 | Iterations without improvement before stop |
| `--batch-size` | 256 | Batch size (DeepGP/SparseGP) |
| `--warm-starting` | on | Warm-start from previous iteration checkpoint |

### Selection Options

| Option | Default | Description |
|--------|---------|-------------|
| `--selection-strategy` | entropy_batch | Point selection (entropy_batch or top_k) |
| `--entropy-blur` | 0.15 | Entropy smoothing (entropy_batch only) |
| `--entropy-beta` | 50.0 | Gibbs sampling temperature (entropy_batch only) |
| `--tolerance-sampling` | 1.0 | Threshold filter width (entropy_batch only) |
| `--proximity-sampling` | 0.1 | Proximity weighting width (entropy_batch only) |

### Evaluation & Plotting Options

| Option | Default | Description |
|--------|---------|-------------|
| `--compute-full-metrics` | off | Comprehensive GoF metrics (accuracy, chi2, pulls) |
| `--eval-data-path` | None | Path to true evaluation dataset (ROOT/CSV) |
| `--track-lengthscales` | on | Save ARD lengthscales per iteration |
| `--advanced-plots` | off | Advanced diagnostic plots (heatmaps, residuals) |

### Config & Sweep Options

| Option | Default | Description |
|--------|---------|-------------|
| `--config-file` | None | YAML config override |
| `--sweep-index` | None | Sweep combination index (for SLURM arrays) |
