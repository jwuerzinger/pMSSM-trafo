# Cross-Run Analysis Metrics

Reference for all metrics computed by `analyse_runs.py` and orchestrated via `run_analysis.sh`.

## Overview

The analysis loads `state.pt` checkpoints from completed active learning runs and computes three categories of metrics: **quality** (does the model learn?), **physical properties** (what does the training data look like?), and **diversity** (does the run explore broadly?). All diversity and variance metrics use only the **9 free pMSSM parameters** (those whose scan ranges are not degenerate): `meL`, `meR`, `M_1`, `M_2`, `mu`, `At`, `Ab`, `Atau`, `tanb`.

## Quality Metrics

| Metric | Description | Source |
|--------|-------------|--------|
| **Hit rate** | Fraction of training points within tolerance of the target relic density (0.12). Reported at 10%, 20%, and 50% relative tolerance: `\|Y - 0.12\| / 0.12 < tol`. | `state.pt -> Y` |
| **Validation R²** | AL model R² on the held-out validation split, per iteration. | `state.pt -> al_r2` |
| **Static random eval R²** | R² on a fixed held-out random sample (100k points). Unbiased generalization metric. | `state.pt -> al_on_static_random_r2` |
| **MCMC eval R²** | R² on MCMC posterior samples. Measures how well the model generalizes to the physically motivated region. | `state.pt -> al_on_mcmc_r2` |
| **Convergence iteration** | Earliest iteration where R² (static random) exceeds 0.5, 0.7, and 0.9. `NaN` if the threshold is never reached. | Derived from R² trajectory |

## Physical Property Metrics

| Metric | Description | Source |
|--------|-------------|--------|
| **Mean relic density** | Mean of Omega across all accumulated training points. Should converge toward 0.12 for effective runs. | `state.pt -> Y` |
| **Std relic density** | Standard deviation of Omega. Narrowing over iterations indicates the run is focusing on the target region. | `state.pt -> Y` |
| **Per-parameter variance** | Variance of each free parameter in normalized [0, 1] space. High variance means broadly sampled. | `state.pt -> X` |
| **Log-generalised variance** | `log det(Cov(X_free))` where `X_free` is the free parameters in [0, 1] space. A scalar summary of the volume of parameter space covered. Log scale because the raw determinant is very small in 9 dimensions. | `state.pt -> X` |

## Diversity Metrics

| Metric | Description | Source |
|--------|-------------|--------|
| **Per-parameter Shannon entropy** | Shannon entropy of a 20-bin histogram over each free parameter in [0, 1] space. Maximum is `log2(20) ~ 4.32` bits (perfectly uniform). Mean across all 9 parameters is also reported. | `state.pt -> X` |
| **k-NN mean distance** | For each training point, the mean Euclidean distance to its k=5 nearest neighbours in normalized free-parameter space, averaged over all points. Higher values indicate sparser, more spread-out coverage. | `state.pt -> X` |
| **MMD vs MCMC** | Maximum Mean Discrepancy between the AL training data and MCMC posterior samples using a Gaussian RBF kernel (bandwidth = median heuristic). Lower values indicate the AL distribution is closer to the physically motivated MCMC distribution. | `state.pt -> X` vs MCMC data |
| **MCMC cluster coverage** | K-means (k=20) is fitted once on the MCMC data. Coverage is the fraction of clusters that contain at least one AL training point within a distance threshold. Higher is better. Uncertainty via Wilson score interval. | `state.pt -> X` vs MCMC data |

## Uncertainty Estimation

- **Bootstrap** (default n=500): All scalar metrics (hit rate, mean/std Omega, per-parameter entropy, k-NN distance, per-parameter variance, generalised variance) are resampled with replacement. Reported as mean +/- 1 standard deviation.
- **Permutation test** (default n=200): For MMD, labels between the AL and MCMC pools are shuffled to build a null distribution. The reported CI comes from this null.
- **Wilson score interval**: For cluster coverage (a binomial proportion), the Wilson interval provides the confidence bound.

## Outputs

### Plots

| File | Description |
|------|-------------|
| `r2_trajectories.png` | R² (validation, static random, MCMC) vs iteration for all runs |
| `hit_rate_trajectories.png` | Hit rate vs iteration for each run at 10/20/50% tolerance |
| `quality_summary.png` | Bar chart of final hit rates and R² values across runs |
| `relic_density_summary.png` | Mean +/- std of Omega per run with horizontal line at 0.12 |
| `diversity_summary.png` | Bar chart of mean entropy, k-NN distance, MMD, cluster coverage |
| `param_entropy_heatmap.png` | Heatmap of per-parameter Shannon entropy for each run |
| `param_variance_heatmap.png` | Heatmap of per-parameter variance for each run |
| `pairwise_scatter_per_run/pairwise_scatter_<label>.png` | 2-D projections of training data overlaid with MCMC samples, one figure per run |

### CSV

`summary.csv` contains one row per run with all scalar metrics and their bootstrap uncertainties as separate `_se` columns.

## CLI Reference

```
python analyse_runs.py \
    --run-dirs DIR [DIR ...]          # Run output directories (must contain state.pt)
    --labels NAME [NAME ...]          # Human-readable labels (auto-derived if omitted)
    --mcmc-data-dir DIR               # MCMC ROOT files for MMD + cluster coverage
    --output-dir DIR                  # Output directory (default: analysis_output)
    --target {DMRD,CrossSection,CLs}  # Target variable (default: DMRD)
    --n-bootstrap N                   # Bootstrap resamples (default: 500, 0 to skip)
    --n-permutations N                # MMD permutation resamples (default: 200)
    --n-clusters N                    # K-means clusters for coverage (default: 20)
    --knn-k N                         # Nearest neighbours (default: 5)
    --include-baseline                # Also plot baseline model curves
    --seed N                          # RNG seed (default: 0)
```

## Wrapper Script

`run_analysis.sh` orchestrates multiple comparisons:

1. **Transformer strategy** -- entropy vs top-k selection (same model, 2 GPU)
2. **Model type** -- transformer vs ExactGP vs DeepGP vs TabPFN (single GPU, top-k)
3. **Model type (entropy)** -- same comparison using entropy selection for GP models (2 GPU)
4. **GP strategy** -- entropy vs top-k within each GP model type
5. **Parallelism** -- 1-GPU vs 2-GPU for transformer, ExactGP, DeepGP
6. **All runs** -- full overview of every completed run
