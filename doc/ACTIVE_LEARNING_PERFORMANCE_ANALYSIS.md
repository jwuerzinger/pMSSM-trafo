# Active Learning Performance Analysis

## Context

This document summarizes the results of all active learning (AL) runs performed to date for pMSSM relic density prediction. The central finding is that the **random sampling baseline consistently matches or outperforms active learning** in R2 score across most configurations and model types. This analysis investigates why, and proposes a plan to improve AL performance.

All runs use the same setup: an AL model that selects new training points via uncertainty estimation, compared against a baseline model. Both models always have **identical dataset sizes** at every iteration. The baseline receives the same number of new points as survive the AL generation process, but drawn randomly from a pre-existing pool of 1.34M validated physics simulations rather than freshly generated.

---

## 1. Cross-Run Summary

### Completed Full Runs (40 iterations)

| Run | Model | Strategy | n_select | Prox. | Warm | y-xform | Final Size | AL R2 | BL R2 | Delta | Winner |
|-----|-------|----------|----------|-------|------|---------|------------|-------|-------|-------|--------|
| `13-02-2026` | Transformer | top_k | 20,000 | 0.1 | Yes | None | 600k | 0.938 | **0.981** | -0.042 | BL |
| `entropy_batch_17-02` | Transformer | entropy_batch | 200 | 0.1 | Yes | log | 55k | 0.970 | **0.971** | -0.001 | Tied |
| `gp_13-02-2026` | Deep GP | entropy_batch | 200 | 0.1 | Yes | log | 54k | 0.946 | **0.964** | -0.018 | BL |
| `gp_deepGP` | Deep GP | entropy_batch | 200 | 0.1 | Yes | log | 54k | 0.625 | 0.614 | +0.011 | AL* |

*Failure mode: both models stuck at low R2 (~0.6), likely bad local minimum.

### Incomplete Runs

| Run | Model | Strategy | n_select | Iters Done | Last AL R2 | Last BL R2 | Delta | Winner |
|-----|-------|----------|----------|------------|-----------|-----------|-------|--------|
| `09-02-2026` | Transformer | top_k | 10,000 | 30/30 | 0.643 | **0.723** | -0.080 | BL |
| `11-02-2026` | Transformer | top_k | 10,000 | 8/30 | 0.802 | **0.909** | -0.107 | BL |
| `11-02-2026_2` | Transformer | top_k | 20,000 | 6/40 | 0.832 | **0.927** | -0.095 | BL |
| `13-02-2026_2` | Transformer | top_k | 20,000 | 14/40 | **0.948** | 0.891 | +0.057 | AL |
| `1h` | Transformer | top_k | 20,000 | 2/2 | 0.822 | **0.843** | -0.020 | BL |
| `no_warm_no_prox` | Transformer | entropy_batch | 200 | 10/40* | 0.814 | **0.848** | -0.034 | BL |
| `gp_2h` | Exact GP | entropy_batch | 500 | 9/10 | **0.844** | 0.815 | +0.029 | AL |
| `gp_13-02-2026_2` | Deep GP | entropy_batch | 200 | 2/40 | **0.955** | 0.892 | +0.063 | AL |

---

## 2. R2 Trajectories for Key Runs

### Transformer + entropy_batch (best run: `entropy_batch_17-02-2026`)

| Iter | AL R2 | BL R2 | AL - BL |
|------|-------|-------|---------|
| 1 | 0.793 | 0.761 | +0.032 |
| 2 | 0.866 | 0.893 | -0.027 |
| 3 | 0.909 | 0.934 | -0.025 |
| 5 | 0.913 | 0.949 | -0.036 |
| 8 | 0.949 | 0.960 | -0.010 |
| 10 | 0.941 | 0.970 | -0.030 |
| 13 | 0.960 | 0.979 | -0.019 |
| 15 | 0.945 | 0.977 | -0.032 |
| 17 | 0.954 | 0.976 | -0.022 |
| 20 | 0.962 | 0.974 | -0.012 |
| 23 | 0.971 | 0.978 | -0.007 |
| 25 | 0.971 | 0.977 | -0.006 |
| 28 | 0.967 | 0.981 | -0.015 |
| 30 | 0.968 | 0.972 | -0.003 |
| 33 | 0.969 | 0.976 | -0.007 |
| 35 | 0.976 | 0.977 | -0.001 |
| 37 | 0.973 | 0.976 | -0.003 |
| **38** | **0.977** | 0.973 | **+0.005** |
| 39 | 0.972 | 0.962 | +0.010 |
| 40 | 0.970 | 0.971 | -0.001 |

Peak AL: **0.977** at iteration 38. AL briefly overtook baseline.

### Transformer + top_k (production run: `13-02-2026`)

| Iter | AL R2 | BL R2 | AL - BL |
|------|-------|-------|---------|
| 1 | 0.732 | 0.720 | +0.012 |
| 5 | 0.820 | 0.857 | -0.037 |
| 9 | 0.924 | 0.873 | +0.051 |
| 10 | 0.921 | 0.885 | +0.036 |
| 15 | 0.925 | 0.925 | +0.001 |
| 20 | 0.926 | 0.935 | -0.010 |
| 25 | 0.941 | 0.942 | -0.001 |
| 30 | 0.939 | 0.963 | -0.024 |
| 35 | 0.934 | 0.976 | -0.042 |
| 40 | 0.938 | 0.981 | **-0.042** |

AL led through iterations 7-15, then baseline overtook and kept improving. AL plateaued at ~0.94.

### Deep GP (production run: `gp_13-02-2026`)

| Iter | AL R2 | BL R2 | AL - BL |
|------|-------|-------|---------|
| 1 | 0.947 | 0.940 | +0.006 |
| 5 | 0.952 | 0.945 | +0.008 |
| 10 | 0.946 | 0.950 | -0.004 |
| 15 | 0.942 | 0.950 | -0.008 |
| 20 | 0.952 | 0.954 | -0.002 |
| 25 | 0.948 | 0.952 | -0.004 |
| 30 | 0.949 | 0.958 | -0.010 |
| 35 | 0.951 | 0.963 | -0.013 |
| 40 | 0.946 | **0.964** | -0.018 |

Deep GP achieves high R2 from iteration 1 but plateaus. Baseline slowly diverges upward.

### No warm starting ablation (`no_warmstarting_no_proximitysampling`, in progress)

| Iter | AL R2 | BL R2 | AL - BL |
|------|-------|-------|---------|
| 1 | 0.779 | 0.772 | +0.006 |
| 2 | 0.832 | 0.826 | +0.005 |
| 3 | 0.788 | 0.775 | +0.013 |
| 4 | 0.830 | 0.822 | +0.008 |
| 5 | 0.797 | 0.826 | -0.029 |
| 6 | 0.800 | 0.837 | -0.038 |
| 7 | 0.724 | 0.791 | -0.068 |
| 8 | 0.794 | 0.811 | -0.017 |
| 9 | 0.762 | 0.810 | -0.049 |
| 10 | 0.814 | 0.848 | **-0.034** |

Dramatically worse and more erratic than the warm-started entropy_batch run (which was at R2=0.941 at iteration 10). The AL model without warm starting shows large iteration-to-iteration variance (e.g., 0.830 -> 0.724 -> 0.794), indicating unstable retraining from scratch.

---

## 3. Key Findings

### 3.1 entropy_batch >> top_k

The entropy_batch selection strategy is dramatically more data-efficient than top_k:
- **entropy_batch**: R2 = 0.970 with only **55k total samples** (200 new/iter)
- **top_k**: R2 = 0.938 with **600k total samples** (20,000 new/iter)

Entropy_batch achieves higher R2 with **11x less data**. The top_k strategy floods the dataset with biased samples from uncertain regions, degrading overall model quality.

### 3.2 Warm starting is critical

Comparing the two entropy_batch transformer runs:
- **With warm starting** (iter 10): R2 = 0.941
- **Without warm starting** (iter 10): R2 = 0.814

Difference of **0.127 in R2** at the same iteration count. Without warm starting, the model must relearn from scratch every iteration, losing previously acquired knowledge.

### 3.3 Log y-transform improves performance

Earlier runs without log transform (09-02, 11-02 series) consistently achieved lower R2 than later runs with `--y-transform log`. The log transform `Y = log(Y/0.12)` better handles the large dynamic range of relic density values.

### 3.4 Transformer outperforms Deep GP

With comparable settings (50k initial, entropy_batch, 200/iter, warm starting):
- **Transformer**: R2 = 0.970 (peak 0.977)
- **Deep GP**: R2 = 0.946

The transformer is more flexible and benefits more from iterative data augmentation.

### 3.5 AL shows early-iteration advantage that erodes

In nearly every run, AL outperforms or matches the baseline in the first few iterations. The advantage erodes as more data accumulates:
- `13-02-2026` (top_k): AL led iterations 1-15, then baseline overtook
- `entropy_batch_17-02`: AL led iteration 1 only, then baseline led until convergence
- `13-02-2026_2` (top_k, 14 iters): AL still leading at +0.057 when run stopped

This pattern suggests AL is most valuable in the small-data regime.

### 3.6 Generation yield

Physics simulation yield (valid models / attempts) varies significantly across runs. Note: the baseline always receives the same number of samples as survive AL generation, so yield does not cause a data quantity gap — but it indicates how far into unphysical parameter space AL is selecting.

| Run | n_select | Yield (early) | Yield (late) | Attempts needed |
|-----|----------|---------------|--------------|-----------------|
| `09-02` (top_k) | 10,000 | ~6% (631/10k) | ~6% (556/10k) | 1 (no retry) |
| `13-02` (top_k) | 20,000 | ~37% (14.7k/40k) | ~31% (12.3k/40k) | 2 |
| `entropy_17-02` (trafo) | 200 | 61% (122, 3 attempts) | 64% (135, 9 attempts) | 3-10 |
| `gp_13-02` (deep GP) | 200 | 62% (123, 6 attempts) | **19% (37-68, 10 attempts)** | 6-10 |
| `no_warm` (trafo) | 200 | 67% (134, 3 attempts) | 72% (144, 4 attempts) | 2-4 |

Critical observations:
- **GP yield collapses catastrophically**: From 123 valid (iter 1) to as low as 37 valid (iter 23). The GP selects increasingly unphysical parameter points, hitting the max 10 retry attempts from iteration 4 onward.
- **Transformer entropy_batch yield is stable**: Consistently 119-139 valid across all 40 iterations, though the number of retry attempts increases from 3 to ~9.
- **No-warm-starting run has better yield**: Only 2-4 attempts needed vs 3-10 for the warm-started run, suggesting less-specialised models select more "generateable" points.
- The early run (09-02) had no retry mechanism and only achieved 6% yield per attempt.

---

## 4. Hypotheses for Baseline Dominance

### H1: Asymmetric data sources (primary hypothesis)

The comparison gives both models **identical dataset sizes** at every iteration — the baseline adds exactly as many random samples from the pre-existing pool as survive AL's generation process. However, the data sources differ fundamentally:
- **Baseline** draws from a pool of **1.34M pre-validated, pre-computed samples** that follow a representative (uniform/LHS) distribution across the parameter space.
- **AL** adds **freshly generated physics simulations** at uncertainty-selected points. These samples are biased toward the uncertain frontier of the model.

The asymmetry is not in data quantity but in **data quality and distribution**: the baseline's random samples provide representative coverage of the parameter space, while AL's samples concentrate in uncertain regions, potentially degrading overall model performance when evaluated on a uniformly-distributed validation set.

### H2: Distribution bias in AL-selected data

AL selects points where the model is most uncertain. These tend to be:
- Near decision boundaries (the Omega_h2 threshold)
- In extreme/sparse regions of parameter space
- Where physics constraints are most likely to reject the point

Adding these biased samples shifts the training distribution away from representative coverage. Since R2 is evaluated on a uniformly-distributed validation set, models trained on biased data are at a disadvantage.

### H3: Cold-start problem without warm starting

Without warm starting, the model retrains from random initialisation every iteration. This:
- Wastes computation relearning what was already known
- Introduces variance from random initialisation
- Prevents gradual refinement of the learned function

The ablation confirms this: R2 drops by 0.127 at iteration 10 without warm starting.

### H4: Diminishing returns of targeted selection at scale

As the dataset grows, each additional targeted sample has diminishing marginal value compared to uniform coverage. The initial AL advantage (iterations 1-15 in the top_k run) disappears once the baseline accumulates enough representative data to cover the space well.

### H5: GP yield collapse indicates over-specialisation

The Deep GP run shows a catastrophic decline in generation yield: from 123 valid models at iteration 1 to as low as 37 at iteration 23 (hitting the 10-attempt retry limit from iteration 4 onward). This means the GP is selecting parameter space points that are overwhelmingly unphysical. The GP's uncertainty estimates may be poorly calibrated, leading it to explore regions where the physics simulator cannot produce valid output. The transformer does not exhibit this problem (stable yield across all 40 iterations), suggesting it has better-calibrated uncertainty or selects more conservative candidates.

### H6: Evaluation metric mismatch

R2 is computed on a validation set drawn from the original data pool, which follows the initial sampling distribution (uniform/LHS). This inherently favors models trained on data that matches this distribution (i.e., the baseline). If evaluation weighted performance near the physical threshold (Omega_h2 ~ 0.12), AL might show more benefit.

---

## 5. Plan to Improve Active Learning Performance

### Priority 1: Fix the data source asymmetry

**Problem**: Both models receive identical dataset sizes, but the baseline draws pre-validated samples from a representative distribution (1.34M pool) while AL adds freshly generated samples from biased uncertain regions. The baseline benefits from the original sampling distribution matching the validation set.

**Actions**:
- **Option A (fair generation)**: Have the baseline also generate new data at random parameter points (rather than drawing from the pre-existing pool). This isolates the value of AL selection by making both data sources go through the same physics simulation pipeline.
- **Option B (pool-based AL)**: Give AL access to the pre-existing 1.34M pool. AL selects from this pool based on uncertainty, rather than generating from scratch. This isolates the value of targeted selection without the generation noise confound.
- **Recommended**: Implement both options as separate experiments to disentangle the effect of generation noise from selection strategy.

### Priority 2: Hybrid acquisition (uncertainty + diversity)

**Problem**: Pure uncertainty selection concentrates data in a narrow region, degrading coverage.

**Actions**:
- Mix AL-selected and random samples each iteration (e.g., 70% uncertainty-selected, 30% random)
- Implement DPP (Determinantal Point Process) or k-medoids diversity term in the acquisition function
- Try BADGE (Batch Active learning by Diverse Gradient Embeddings) which naturally balances uncertainty and diversity

### Priority 3: Reduce batch size further

**Problem**: The move from n_select=20,000 to n_select=200 dramatically improved performance (0.938 -> 0.970). Smaller batches allow the model to retrain on new data before making the next selection.

**Actions**:
- Try n_select=50 and n_select=100 to see if smaller batches continue to improve
- Consider n_select=1 (pure sequential AL) as an upper bound experiment if compute allows

### Priority 4: Evaluate on task-relevant metric

**Problem**: Global R2 may not capture AL's advantage in the physically interesting region near Omega_h2 ~ 0.12.

**Actions**:
- Compute R2 specifically for points near the physical threshold (e.g., 0.10 < Omega_h2 < 0.14)
- Plot prediction error as a function of distance from threshold
- Use the existing `--eval-data-path` flag to evaluate on a separate test set focused on the threshold region
- Use `--compute-full-metrics` for more comprehensive evaluation

### Priority 5: Improve generation yield

**Problem**: In later iterations, yield drops to 19-37% as AL selects extreme parameter points.

**Actions**:
- Constrain candidate generation to physically plausible regions (pre-filter using known constraints)
- Use a validity classifier to predict which parameter points will produce valid physics output before running expensive simulations
- Retry failed generations with nearby parameter perturbations

### Priority 6: Investigate model capacity and training

**Problem**: AL model may be under-fitting due to distribution shift from biased data.

**Actions**:
- Increase model capacity (larger transformer, more layers) to handle the more complex AL data distribution
- Experiment with curriculum learning: weight recent AL samples lower to prevent catastrophic forgetting
- Try sample-weighted loss where AL-generated samples have lower weight than original data

### Priority 7: Alternative uncertainty estimation

**Problem**: MC Dropout uncertainty may not be well-calibrated, leading to poor sample selection.

**Actions**:
- Try deep ensembles (3-5 models) for better-calibrated uncertainty
- Evaluate uncertainty calibration plots to verify MC Dropout quality
- Compare query-by-committee (ensemble disagreement) with MC Dropout variance

---

## 6. Recommended Next Experiments

In order of priority:

1. **Fair generation baseline** (Priority 1, Option A): Run with baseline also generating new data at random parameter points instead of drawing from pool. This isolates whether AL *selection* helps when both sides go through the same generation pipeline.

2. **Pool-based AL** (Priority 1, Option B): Run AL selecting from the existing 1.34M pool (no generation). Cheap experiment that isolates whether targeted selection from a representative pool beats random selection.

3. **Hybrid 70/30** (Priority 2): Run entropy_batch with 70% uncertainty-selected + 30% random samples per iteration.

4. **Threshold-focused evaluation** (Priority 4): Re-evaluate existing runs with R2 computed near Omega_h2 ~ 0.12.

5. **Smaller batch** (Priority 3): Run entropy_batch with n_select=50.
