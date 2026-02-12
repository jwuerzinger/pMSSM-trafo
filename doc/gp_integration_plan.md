# GP/DeepGP Active Learning Integration

## Overview

Integrate Gaussian Process (ExactGP) and Deep GP models from the `al_pmssmwithgp` repo into the existing active learning pipeline. Creates `active_learning_gp.py` — an analogous script to `active_learning.py` that swaps the tabular transformer for GP models while keeping the same infrastructure (data loading, parallel generation, baseline comparison, metrics tracking).

## Key Design Decisions

| Aspect | Transformer (`active_learning.py`) | GP (`active_learning_gp.py`) |
|--------|-------------------------------------|------------------------------|
| Uncertainty | MC Dropout (N forward passes) | Native GP posterior variance |
| Normalization | Z-score (mean/std) | Min-max to [0,1] |
| Target transform | Z-score | log(y / 0.12) for DMRD |
| Training API | DataLoader + optimizer loop | `model.do_train_loop()` |
| Model creation | Separate from data | Takes x_train, y_train in constructor |
| Default epochs | 2000 | 2000 |
| Early stopping | Patience 200 on val loss | Patience 200 on val loss (all model types) |
| Selection strategy | Top-K (MC Dropout variance) | Entropy batch (default) or Top-K |

## Files Created / Modified

- `active_learning_gp.py` — main GP active learning script
- `run_active_learning_gp.sh` — production run script (40 iterations, 100k samples, 20k selections)
- `run_active_learning_gp_medium_test.sh` — medium test (3 datasets, 2000 samples, 500 iterations)
- `doc/gp_integration_plan.md` — this file
- `doc/gp_pipeline_comparison.md` — detailed comparison of original vs refactored pipelines
- `al_pmssmwithgp/model/gp_pipeline/models/exact_gp.py` — added early stopping, fixed val-loss tracking
- `al_pmssmwithgp/model/gp_pipeline/models/deep_gp.py` — added early stopping
- `al_pmssmwithgp/model/gp_pipeline/models/sparse_gp.py` — added early stopping, fixed indentation
- `al_pmssmwithgp/model/gp_pipeline/models/mlp.py` — added early stopping

## Progress Tracker

- [x] Save plan to `doc/gp_integration_plan.md`
- [x] Add dependencies to pixi.toml: gpytorch, corner, scikit-learn, scipy
- [x] Create `active_learning_gp.py`
  - [x] Imports and constants (GP_RANGE_DICT, DMRD_TRUE_VALUE)
  - [x] Normalization functions (build_norm_tensors, normalize_x, transform_y)
  - [x] GP model creation and training functions
  - [x] GP uncertainty computation (posterior variance, batched for large candidate pools)
  - [x] R² computation on validation set
  - [x] train_gp_worker for sequential or parallel training
  - [x] CLI options (shared + GP-specific, defaults from config3.yaml)
  - [x] Main loop (same structure as active_learning.py)
  - [x] Diagnostic plots (loss curves, scatter, histogram, random predictions — matching active_learning.py)
  - [x] Model checkpoint saving/loading for uncertainty reuse
- [x] Create `run_active_learning_gp.sh`
- [x] Verify imports work (gp_pipeline.models.exact_gp, deep_gp)
- [x] Smoke test with `--testing` flag → passes end-to-end (exit 0)
- [x] Test with ExactGP model type → R² computed, uncertainty computed, points selected
- [x] Implement parallel training (AL on cuda:0, Baseline on cuda:3 via `mp.Process` + `torch.cuda.set_device`)
- [x] Fix device mismatch: `torch.cuda.set_device(gpu_id)` in worker before model construction
- [x] Implement warm-starting: `warm_start_path` in `train_gp_worker` loads prev checkpoint before training
- [x] Dynamic GPU IDs in log messages (`AL_GPU_ID`, `BASELINE_GPU_ID` constants)
- [x] Test with DeepGP model type → passes end-to-end, uncertainty computed, points selected
- [x] Fix CG non-convergence warning in ExactGP: raise `max_cholesky_size` to `max(n_train+1, 5000)` in `do_train_loop` to force Cholesky for typical dataset sizes; also raised `max_cg_iterations` to 500 as fallback for larger datasets
- [x] Early stopping with configurable patience (default 200) on validation loss for all model types
  - [x] ExactGP: fixed best model tracking from train loss to val loss, added patience
  - [x] DeepGP: added patience parameter and early break
  - [x] SparseGP: fixed indentation of best-model tracking, added patience
  - [x] MLP: uncommented and parameterised existing early stopping logic
- [x] Changed default training iterations from 1000 to 2000 (matching active_learning.py)
- [x] Changed default selection strategy from `top_k` to `entropy_batch`
- [x] Full diagnostic plot parity with `active_learning.py`:
  - `plot_gp_losses` (training/validation loss curves)
  - `scatter_true_vs_pred_gp` (true vs predicted scatter, train + val)
  - `hist_true_vs_pred_gp` (histogram overlay, train + val)
  - `compare_random_predictions_gp` (random sample predictions, train + val)
- [ ] Full multi-iteration test with data generation

## Feature Integration (from al_pmssmwithgp)

Ported all major features from the `al_pmssmwithgp` submodule's pipeline. See `doc/gp_pipeline_comparison.md` for full details.

- [x] Multiple target functions (DMRD, CrossSection, CLs) via `--target`
- [x] Comprehensive evaluation metrics (accuracy, chi2, pulls, weighted) via `--compute-full-metrics`
- [x] Lengthscale tracking via `--track-lengthscales`
- [x] Full true dataset evaluation via `--eval-data-path`
- [x] Sparse GP model support via `--model-type sparse_gp`
- [x] MLP model support via `--model-type mlp`
- [x] Entropy-based batch AL selection via `--selection-strategy entropy_batch` (now default)
- [x] Advanced diagnostic plots via `--advanced-plots`
- [x] YAML config + parameter sweep support via `--config-file` / `--sweep-index`
- [x] Early stopping on validation loss via `--early-stopping` / `--patience` (default: on, patience=200)
- [ ] End-to-end test with entropy_batch selection
- [ ] End-to-end test with SparseGP model
- [ ] End-to-end test with MLP model
- [ ] End-to-end test with CrossSection/CLs target
