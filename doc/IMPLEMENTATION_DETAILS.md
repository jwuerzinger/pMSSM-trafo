# Implementation Details & Advanced Topics

This document covers important implementation details, performance analysis, and advanced features of the pMSSM active learning pipelines.

## Table of Contents
1. [Uncertainty Computation](#uncertainty-computation)
2. [Warm Starting Performance Impact](#warm-starting-performance-impact)
3. [Proximity Weighting Explained](#proximity-weighting-explained)
4. [Key Pipeline Differences](#key-pipeline-differences)
5. [Numerical Stability (GP Jitter)](#numerical-stability-gp-jitter)

---

## Uncertainty Computation

### Critical Finding: Uncertainty is in Log-Space

Both pipelines compute uncertainty in **log-space** (transformed space), not physical relic density space.

#### Implementation

**Transformer ([pmssm/uncertainty.py:54-55](../pmssm/uncertainty.py#L54-L55)):**
```python
predictions = torch.stack(predictions, dim=0)  # (T, N, 1) MC Dropout samples
pred_mean = predictions.mean(dim=0)  # Mean in log-space
pred_var = predictions.var(dim=0)    # Variance in log-space
```

**GP ([pmssm/uncertainty.py:128-132](../pmssm/uncertainty.py#L128-L132)):**
```python
preds = model.likelihood(model(x_batch))  # Model trained in log-space
pred_mean = preds.mean  # Mean in log-space
pred_var = preds.variance  # GP posterior variance in log-space
```

#### Why This Makes Sense

1. **Models trained in log-space**: Both pipelines use `log(Y / 0.12)` for training
2. **Uncertainty is model-native**: The model's uncertainty is in the space it understands
3. **Better for large dynamic ranges**: Log-space variance captures relative uncertainty better than absolute physical-space variance

#### What Gets Transformed Back

**Only R² evaluation** uses physical space ([pmssm/evaluation.py:147-148](../pmssm/evaluation.py#L147-L148)):
```python
y_true_physical = inverse_transform_y(y_true_transformed, target=target)
y_pred_physical = inverse_transform_y(y_pred_transformed, target=target)
# Then compute R² in physical space for consistent evaluation
```

#### Summary Table

| Quantity | Space |
|----------|-------|
| **Model predictions** | Log-space: `log(Y / 0.12)` |
| **Uncertainty (variance)** | Log-space |
| **Point selection** | Based on log-space uncertainty |
| **R² evaluation** | Computed in physical space |

**Interpretation**: An uncertainty of 0.5 in log-space means the model is uncertain by ±0.5 in `log(Ωh²/0.12)` units, **not** ±0.5 in physical Ωh² units.

---

## Warm Starting Performance Impact

Warm starting loads the previous iteration's checkpoint so training starts from a good initialization instead of random weights.

### Performance Analysis

Based on production runs (40 iterations, 50k samples):

#### GP Pipeline (DeepGP)
- **Actual time**: 10.4 hours
- **Without warm starting**: ~66 hours
- **Savings**: 55 hours (6.3x speedup)
- **Epoch reduction**: 86% fewer epochs needed (872 → 120 avg per iteration)

#### Transformer Pipeline
- **Actual time**: 33.9 hours
- **Without warm starting**: ~53 hours
- **Savings**: 19 hours (1.6x speedup)
- **Epoch reduction**: 37% fewer epochs needed (339 → 215 avg per iteration)

### Why GP Benefits More

**DeepGP characteristics:**
- Internal representations transfer exceptionally well between iterations
- First iteration: 872 epochs to convergence
- Later iterations: Only 100-120 epochs (7-8x faster!)

**Transformer characteristics:**
- Dataset grows (adding 20k points each iteration)
- Needs more epochs to fit larger datasets
- Still benefits, but less dramatically

### Combined Impact (Early Stopping + Warm Starting)

Without these optimizations:

| Pipeline | Actual | Without Both | Total Speedup | Time Saved |
|----------|--------|--------------|---------------|------------|
| **GP** | 10.4 h | 753 h (31 days) | **72x** | 743 hours |
| **Transformer** | 33.9 h | 1553 h (65 days) | **46x** | 1519 hours |

**Conclusion**: Warm starting + early stopping make iterative active learning **practically feasible**. Without them, 40-iteration runs would be completely impractical.

### Configuration

Warm starting is **enabled by default** in all production scripts:

```bash
python active_learning.py --warm-starting  # Default: enabled
python active_learning_gp.py --warm-starting  # Default: enabled

# Disable for ablation studies only:
python active_learning.py --no-warm-starting
```

---

## Proximity Weighting Explained

Proximity weighting focuses point selection near the target relic density (Ωh² = 0.12).

### Mathematical Formula

```python
proximity = exp(-((pred_mean - threshold)² / proximity_sampling))
weighted_uncertainty = proximity × uncertainty
```

This is a **Gaussian weight** centered at the target value.

### Parameter Values

**`proximity_sampling`** = Gaussian width (σ) in log-space

| Value | Physical Range | Behavior |
|-------|----------------|----------|
| **0.0** (disabled) | Entire space | Pure uncertainty, no preference |
| **0.05** (tight) | Ωh² ≈ 0.11-0.13 | Very focused near target |
| **0.1** (typical) | Ωh² ≈ 0.10-0.14 | Moderately focused |
| **0.2** (loose) | Ωh² ≈ 0.08-0.18 | Broadly focused |

### Physical Interpretation

For `proximity_sampling = 0.1`:

```
Log-space distances from threshold (0.0):
  Δ = 0.0  → weight = 1.00  (full weight at target)
  Δ = ±0.3 → weight = 0.11  (11% weight)
  Δ = ±0.5 → weight = 0.007 (0.7% weight)

Physical relic density:
  Center: Ωh² = 0.12 (target)
  1σ range: Ωh² ≈ 0.10-0.14 (~90% of weight)
  2σ range: Ωh² ≈ 0.086-0.17 (~99% of weight)
```

### Effect on Selection

**Example:**
```
Predicted Ωh²:  0.01   0.05   0.10   0.12   0.14   0.20   0.50
Raw uncertainty: 0.8    0.9    0.7    0.5    0.6    0.95   0.99
                 ↓      ↓      ↓      ↓      ↓      ↓      ↓
Proximity:       0.00   0.01   0.82   1.00   0.82   0.04   0.00
Weighted:        0.00   0.01   0.57   0.50   0.49   0.04   0.00
                                      ^^^^^^^^
                Points near target selected despite lower raw uncertainty!
```

### When to Use

**Use proximity weighting when:**
- You care primarily about the cosmologically relevant region (Ωh² ≈ 0.12)
- You want to focus computational resources efficiently
- You're doing precision mapping near the target

**Don't use when:**
- You want to map the entire parameter space uniformly
- You're exploring new physics far from the target
- You want purely uncertainty-driven exploration

### Configuration

**Transformer:**
```bash
# Default: disabled
python active_learning.py --proximity-sampling 0.0  # No weighting
python active_learning.py --proximity-sampling 0.1  # Enable weighting
```

**GP:**
```bash
# Default: enabled with entropy batch selection
python active_learning_gp.py --proximity-sampling 0.1  # Moderate focus
python active_learning_gp.py --proximity-sampling 0.0  # Disable
python active_learning_gp.py --tolerance-sampling 0.0  # Also disable pre-filtering
```

**Shell scripts without proximity weighting:**
- [run_active_learning_no_proximity.sh](../run_active_learning_no_proximity.sh)
- [run_active_learning_gp_no_proximity.sh](../run_active_learning_gp_no_proximity.sh)

### Origin

Proximity weighting was originally implemented in the `al_pmssmwithgp` submodule by Irina:
- Implementation: [al_pmssmwithgp/model/gp_pipeline/utils/selection.py:145](../al_pmssmwithgp/model/gp_pipeline/utils/selection.py#L145)
- Config: [al_pmssmwithgp/model/gp_pipeline/config/config3.yaml:34](../al_pmssmwithgp/model/gp_pipeline/config/config3.yaml#L34)

---

## Key Pipeline Differences

### Data Normalization (CRITICAL)

| Pipeline | Input (X) | Target (Y) |
|----------|-----------|-----------|
| **Transformer** | Z-score: `(X - mean) / std` | Log-space: `log(Y / 0.12)` |
| **GP** | Min-max: `(X - min) / (max - min)` | Log-space: `log(Y / 0.12)` |

**Implication**: Input normalization differs! This affects model behavior and uncertainty calibration.

### Default Selection Strategy

| Pipeline | Default | Alternative |
|----------|---------|-------------|
| **Transformer** | `top_k` (variance ranking) | `entropy_batch` |
| **GP** | `entropy_batch` (sophisticated batch) | `top_k` |

### Learning Rate & Optimizer

| Pipeline | Optimizer | Learning Rate | Configurable? |
|----------|-----------|---------------|---------------|
| **Transformer** | AdamW | 3e-4 | No (hardcoded) |
| **GP** | Adam | 1e-3 | Yes (`--learning-rate`) |

### Multi-Target Support

| Pipeline | Targets | CLI |
|----------|---------|-----|
| **Transformer** | DMRD only | None |
| **GP** | DMRD, CrossSection, CLs | `--target` |

### Model Variants

| Pipeline | Models Available |
|----------|------------------|
| **Transformer** | PMSSMTransformerTabular (1 variant) |
| **GP** | ExactGP, DeepGP, SparseGP, MLP (4 variants) |

### Comprehensive Comparison

See [gp_pipeline_comparison.md](gp_pipeline_comparison.md) for detailed pipeline comparison.

---

## Numerical Stability (GP Jitter)

### What is Jitter?

Jitter is a small positive value added to the diagonal of the covariance matrix to ensure numerical stability during Cholesky decomposition:

```python
K_stable = K + jitter × I  # Add jitter to diagonal
L = cholesky(K_stable)     # Now numerically stable
```

### Why Needed?

The covariance matrix should theoretically be positive semi-definite (PSD), but can become non-PSD due to:
1. Floating-point precision errors
2. Nearly identical training points → singular matrix
3. Extreme kernel parameters (very small/large lengthscales)
4. Ill-conditioned matrices

### Common Error

```
linear_operator.utils.errors.NotPSDError:
Matrix not positive definite after repeatedly adding jitter up to 1.0e-06.
```

**Causes:**
- DeepGP with large datasets (50k+) is numerically fragile
- ARD lengthscales going to extreme values
- Duplicate or near-duplicate training points

### Solutions

**1. Increase jitter:**
```bash
--jitter 1e-2  # or even 1e-1
```

**2. Use more stable model:**
```bash
--model-type exact_gp  # More stable than DeepGP
```

**3. Use SparseGP:**
```bash
--model-type sparse_gp --inducing-strategy kmeans
```

**4. Check for duplicates** in data quality

### Typical Values

```bash
--jitter 1e-6  # GPyTorch default (very stable data)
--jitter 1e-4  # Mild stability issues
--jitter 1e-3  # Moderate stability issues (typical)
--jitter 1e-2  # Significant stability issues
--jitter 1e-1  # Very unstable (last resort)
```

### Trade-offs

- ✅ **Too little**: Numerical instability, Cholesky fails
- ✅ **Right amount**: Stable computation, minimal impact
- ⚠️ **Too much**: Overly smooth predictions, acts like added noise

**Think of jitter as**: A small "safety cushion" to prevent numerical explosions during matrix operations.

---

## See Also

- [HARMONIZATION_SUMMARY.md](HARMONIZATION_SUMMARY.md) - Log transformation harmonization details
- [gp_pipeline_comparison.md](gp_pipeline_comparison.md) - Comprehensive pipeline comparison
- [CLI_REFERENCE.md](CLI_REFERENCE.md) - Complete command-line reference
- [active_learning_plan.md](active_learning_plan.md) - Transformer AL design
- [README.md](../README.md) - Quick start guide
