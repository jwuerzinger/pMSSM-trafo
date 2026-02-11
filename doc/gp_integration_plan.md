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

## Files Created

- `active_learning_gp.py` — main GP active learning script
- `run_active_learning_gp.sh` — run script with default config3.yaml parameters
- `doc/gp_integration_plan.md` — this file

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
  - [x] Diagnostic plots (loss curves, true vs predicted scatter)
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
- [ ] Full multi-iteration test with data generation
