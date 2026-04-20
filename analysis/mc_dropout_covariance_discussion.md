# MC Dropout covariance quality and its effect on entropy-batch selection

**Date:** 2026-04-20
**Context:** observed underperformance of the `entropy_batch` acquisition strategy relative to `top_k` for the MC-Dropout transformer surrogate, particularly in the cold-start regime. See [results_summary_20260417.md](results_summary_20260417.md) and the warm-start / strategy comparison tables for the empirical baseline.

## Question

MC Dropout is widely considered good enough for *pointwise* predictive variance. Is it good enough for the *full predictive covariance matrix* that `select_entropy_batch_mc` builds from the same samples — or are we asking MC Dropout to deliver a quantity it cannot reliably estimate?

## Short answer

Probably not, in the current configuration. The sample covariance constructed in [pmssm/selection.py:263-267](../pmssm/selection.py#L263-L267) with T = 20 MC samples over a focused pool of n_pool = 5,000 is:

- rank-deficient by construction (rank ≤ T−1 = 19, so ≥ 4,980 of the 5,000 eigenvalues are zero before regularisation),
- propped up by a `1e-4·I` diagonal term that drowns the null space in isotropic noise, and
- extremely noisy on the genuine off-diagonal entries: sample correlation from T = 20 uncorrelated variables has SE ≈ 1/√(T−2) ≈ 0.24, so spurious correlations of ±0.25 between independent pool points are routine.

Top-k only consumes the diagonal (for an ordinal ranking); it is insensitive to all of this. Entropy-batch's diversity signal is a log-det of a batch submatrix — explicitly a function of the off-diagonal entries — so it inherits the full estimation noise.

## Why this mechanism fits what we see

Empirically (see [results_summary_20260417.md](results_summary_20260417.md) and the warm-start × strategy table in the conversation log):

| Model | Covariance source | Strategy ranking | Comment |
|---|---|---|---|
| Transformer (MC Dropout) | rank-deficient, T = 20 | top-k ≳ entropy, strongly in cold regime | covariance unreliable |
| DeepGP | variational, full-rank | entropy > top-k in warm regime; inverted in cold | calibration depends on fit |
| ExactGP | analytic, full-rank | entropy ≥ top-k in warm regime; modest Δ | covariance is the right object |

ExactGP is the cleanest comparison: when the acquisition uses an analytic, full-rank, well-calibrated covariance, entropy's diversity mechanism wins (warm ΔR² = +0.04, Δloss = −0.07). The transformer is the opposite extreme — a covariance stitched together from 20 dropout masks — and entropy loses. DeepGP sits in between.

## Mechanism in detail

`select_entropy_batch_mc` ([pmssm/selection.py:160](../pmssm/selection.py#L160)):

1. Compute `(T, n_pool)` matrix of MC predictions.
2. Center across T and form `Σ̂ = Xᵀ X / (T − 1) + 1e-4·I`.
3. Iteratively build a batch: at each step, score each remaining pool point by the incremental log-det contribution it makes when appended to the current batch covariance, and Gibbs-sample (β = 50) among the scores. Diversity in the batch comes entirely from the off-diagonal structure of `Σ̂`.

Three independent failure modes of this estimator at T = 20, n_pool = 5000:

**(a) Rank deficiency.** The empirical covariance from T samples has rank at most T. The remaining 4,980 eigenvalue directions are determined purely by the regulariser. For n_pool ≫ T (our regime) the covariance is essentially `UΛUᵀ + εI` with U rank-20 and ε = 1e-4. The log-det of any submatrix of size larger than 20 is dominated by the `εI` contribution — i.e. by a constant unrelated to the acquisition problem.

**(b) Noise floor on off-diagonals.** Even in the rank-20 subspace, individual off-diagonal entries of a 20-sample covariance have large relative noise. Vershynin's bounds give ‖Σ̂ − Σ‖ = O(n_pool / T) in operator norm for sub-Gaussian predictors — we are two orders of magnitude away from the concentration regime.

**(c) Diagonal miscalibration compounds.** If the per-point variance map is itself off (cold-started weights haven't converged), the covariance is both rank-deficient *and* pointed at the wrong subspace. Entropy-batch's diversity picks are then spread across the axes of a noisy, miscalibrated spectrum — i.e. the diversity is in the noise, not in the predictions.

**Why top-k is robust.** Top-k only uses `diag(Σ̂)`, and only ordinally. A monotone transform preserves the ranking. Estimation noise in off-diagonals is simply ignored.

## Things to test

Cheapest first. Each of these should take less than a day to implement and can be run on existing MC Dropout checkpoints.

### 1. Stability of the batch under re-sampling (smoking-gun test)

On a single fixed surrogate checkpoint, run `select_entropy_batch_mc` twice with different MC Dropout seeds (different dropout masks, same weights, same pool).

- If the two selected batches **overlap ≳ 80 %**, the covariance is stable enough to define a consistent diversity notion.
- If they overlap **≪ 50 %**, the diversity signal is dominated by sampling noise. The current MC Dropout ensemble size is below what the acquisition needs.

Logging: Jaccard index of the two selected batches; histogram over 10 seed pairs.

### 2. Eigenvalue spectrum of `Σ̂`

One-line diagnostic in the existing pipeline: log `torch.linalg.eigvalsh(sample_cov).sort()` at each acquisition step.

Expected pathological signature: 20 eigenvalues of order `σ²_pred` followed by ~4,980 eigenvalues clamped at 1e-4. If this is what we see, the 5,000-point pool is operationally a 20-dimensional pool for entropy-batch purposes.

Mitigation if confirmed: see test 4.

### 3. Impact of MC sample count T

Re-run one paired acquisition (same pool, same candidates) with T ∈ {20, 50, 100, 200} and measure:

- batch overlap vs the T = 200 reference (does it plateau?),
- AL downstream: pick one model × strategy (transformer entropy, warm) and run 10 AL iterations at each T, compare R² trajectory.

If R² at T = 200 materially exceeds T = 20, the entropy-batch underperformance is mostly an undersampling artefact and can be fixed by paying more inference compute. If it doesn't, the problem is deeper (see test 5).

### 4. Match n_pool to T (no-code-change mitigation)

Set `entropy_pool_size = T × k` for k ∈ {5, 10, 25}. With T = 20 and k = 10 that's n_pool = 200, within the regime where the sample covariance is much closer to full rank and Vershynin-scale concentration kicks in.

Hypothesis: the tolerance + proximity pre-filter already does most of the "which region" work. Entropy-batch's job is only to de-correlate within that region. Shrinking the pool trades a small amount of candidate exploration for a materially better covariance estimate.

### 5. Swap raw sample covariance for a shrinkage estimator

Replace the current `Xᵀ X / (T − 1) + 1e-4·I` with Ledoit–Wolf shrinkage (a convex combination with a structured target, with analytic optimal mixing coefficient). This is a one-function change: `sklearn.covariance.LedoitWolf().fit(preds_2d).covariance_`.

Shrinkage does not recover the missing rank, but it replaces the 1e-4 regulariser with a statistically justified scalar. If the acquisition metric changes meaningfully, the `1e-4` prior was wrong.

### 6. Switch to a deep ensemble

Train K = 5 transformer surrogates with different seeds (keep total parameter budget similar or pay the 5× cost). Use ensemble predictions in place of MC samples.

Known from the literature: ensemble predictive covariances are consistently better calibrated than MC Dropout, especially in cold-start / small-data regimes. Downside: 5× training compute.

This is the test most likely to *confirm or rule out* the hypothesis — it changes the quantity entropy-batch consumes while leaving everything else fixed. If entropy-batch + ensemble beats top-k + ensemble, the story is cleanly "MC Dropout covariance is the bottleneck".

### 7. Per-iteration validity rate per strategy

Already hinted at by the final-N column in the strategy table, but worth making explicit. Log `n_generated / n_selected` per iteration per strategy on the transformer runs:

- If entropy's validity rate is consistently lower than top-k's in the cold regime, the batch it picks is genuinely out-of-manifold (spurious-correlation-driven spreads into the invalid regions of pMSSM space) — direct evidence for mechanism (c).

## Prioritisation

For a talk-ready story in ~1 week: **tests 1, 2, 4**. Those are all diagnostic, no retraining, answer the question definitively.

For a paper: **test 6**. The ensemble comparison turns the hypothesis into an ablation.

For the production AL pipeline: **test 5** if test 2 shows a pathological spectrum. It is a two-line change with no new hyperparameters.

## What this does not explain

- ExactGP top-k cold-start R² (0.26) being worse than ExactGP top-k warm (0.42). That is plausibly a dataset-starvation effect (top-k cold starves on 2,383 points vs warm's 3,689) and the LOVE-disable regression ([resubmission_20260420.md](resubmission_20260420.md)), not an entropy/covariance story.
- DeepGP top-k warm (N = 7,513) underperforming DeepGP top-k cold (N = 12,488). The warm-start run selects redundant points the stale weights cannot improve on; N collapses; R² collapses. That is a warm-start × acquisition-redundancy interaction, independent of covariance quality.

## References

- [pmssm/selection.py](../pmssm/selection.py) — both acquisition implementations.
- [al_pmssmwithgp/model/gp_pipeline/utils/selection.py](../al_pmssmwithgp/model/gp_pipeline/utils/selection.py) — `EntropySelectionStrategy` used as a black box by `select_entropy_batch_mc`.
- [results_summary_20260417.md](results_summary_20260417.md) — prior AL-run analysis with the strategy comparisons.
