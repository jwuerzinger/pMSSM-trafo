# Resubmission log — 2026-04-20

Two exact_gp jobs resubmitted 2026-04-20 to validate the LOVE re-enable patch.

## Why these were re-submitted

Analysis on 2026-04-20 traced a severe sampling-efficiency regression to commit `3706d93` ("disabling LOVE"). That commit set `gpytorch.settings.fast_pred_var(False)` in five exact_gp / sparse_gp code paths — `compute_uncertainty_gp`, `compute_gp_r2`, `compute_comprehensive_metrics`, `gp_predict`, and `plot_advanced_diagnostics`.

### Observed effect (pre-revert)

- `active_learning_exact_gp_output_20260415_113302` (pre-3706d93 entropy_batch run): 40 iters, yield ~60% sustained, final training set 10,600.
- `active_learning_exact_gp_output_20260416_171932` (post-3706d93, same config): yield collapsed from 3000 to 25 seeds/iter after iter ~10, final training set 3,201.
- `active_learning_exact_gp_top_k_output_20260417_164349` (post-3706d93 + selection fix): same lockup pattern, final training set 2,764.

### Mechanism

LOVE (Lanczos Variance Estimates) uses randomized probe vectors — its variance estimates have per-call stochasticity. With LOVE off, `compute_uncertainty_gp` returns exact deterministic variance. Once ARD lengthscales converge (around iter 10), the candidate-variance landscape stops changing, `top_k` / retry selection picks identical candidates each iteration, dedupe eliminates nearly all, and the physics scan starves. See [results_summary_20260417.md §9](results_summary_20260417.md) for the full diagnosis.

The flat exact-GP uncertainty profile that motivated `3706d93` did **not** improve with exact variance computation (still a dip at M_2=0, per §3 of the summary), so the change produced a severe regression without delivering its intended benefit.

### Fix

Reverted the five `fast_pred_var(False)` calls introduced by `3706d93` back to `fast_pred_var()` (default True) in:
- [pmssm/uncertainty.py:132](../pmssm/uncertainty.py#L132) — exact_gp branch of `compute_uncertainty_gp` (the acquisition-critical path)
- [pmssm/evaluation.py:138, 225](../pmssm/evaluation.py)
- [pmssm/visualization.py:75, 746](../pmssm/visualization.py)

Deep_gp branches left alone (LOVE doesn't apply to variational models). `select_entropy_batch` in [active_learning_gp.py](../active_learning_gp.py) also left alone — its `fast_pred_var(False)` with `batch_size=5_000` comes from commit `aba3c86` and avoids OOM on the main acquisition path; the lockup was in the retry path, not there.

## Resubmitted jobs

| Pipeline | Previous run (post-3706d93, broken) | New job | Expected output dir pattern |
|---|---|---|---|
| exact_gp `top_k` | `active_learning_exact_gp_top_k_output_20260417_164349` | **8102247** | `/ptmp/jwuerzin/output/active_learning_exact_gp_top_k_output_<TIMESTAMP>` |
| exact_gp `entropy_batch` | `active_learning_exact_gp_output_20260416_171932` | **8102248** | `/ptmp/jwuerzin/output/active_learning_exact_gp_output_<TIMESTAMP>` |

`<TIMESTAMP>` is set by the Slurm script at job start (format `YYYYMMDD_HHMMSS`).

Both jobs submitted ~2026-04-20, state PENDING at submit time (cluster maintenance reservation).

## Runs NOT re-submitted (not affected by the LOVE change)

- All transformer runs — no GP code paths.
- All deep_gp runs — LOVE branch was already `fast_pred_var(False)` independently of 3706d93 (not applicable to variational models).
- All tabpfn runs — TabPFN has its own uncertainty path in [active_learning_tabpfn.py](../active_learning_tabpfn.py), not GPyTorch.

## What to watch when the new runs finish

1. **exact_gp top_k yield trajectory**: seeds submitted/iter should stay ≳ 1500 through iter 40 (post-3706d93 run dropped to ~25/iter by iter 15). If it instead partially recovers then decays, the tolerance-cut filter is materially tightening exploration even with LOVE restored — would argue for loosening `--tolerance-sampling` for exact_gp.
2. **exact_gp entropy_batch**: should match the 20260415_113302 baseline (final N ≈ 10,600, late-iter AL R² ≈ 0.5).
3. **Uncertainty variance stats in the log**: "GP uncertainty stats: mean_var=..." should fluctuate across iterations rather than locking at ~1.36 from iter 15 onward (the signature of the exact-variance freeze).
4. **R² sanity check**: LOVE-restored `compute_gp_r2` will give slightly different R² numbers than the post-fix broken run. The difference should be small (LOVE is an approximation with bounded error in the mean).

## Related docs

- [results_summary_20260417.md §9](results_summary_20260417.md) — full diagnosis, including the LOVE investigation.
- [resubmission_20260417.md](resubmission_20260417.md) — the previous resubmission burst (selection-filter fix) which inadvertently inherited the LOVE-off state.
