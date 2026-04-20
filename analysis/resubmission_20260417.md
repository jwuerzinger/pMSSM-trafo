# Resubmission log — 2026-04-17

All re-runs share the timestamp **`20260417_164349`** (single `sbatch` burst at 16:43 UTC).

## Why these were re-submitted

Two behavior-changing fixes were made to the AL code on 2026-04-17:

### Fix 1 — tolerance cut added to the `top_k` selection path

Previously, `top_k` selection called `select_top_uncertain` directly, applying only the soft proximity weight. The hard tolerance cut (`--tolerance-sampling=1.0`, keep only candidates with predicted `Y_t ∈ [threshold ± tol]`) was silently ignored — it was wired up only in the `entropy_batch` path.

This was the root cause of the **TabPFN bootstrap trap** we diagnosed on 2026-04-17:
- TabPFN's uncertainty dip at M_2≈0 pushed `top_k` seeds to large |M_2| (median |M_2| ≈ 1447).
- At large |M_2|, ~47% of scans fail `SP_m_h = −1` (EWSB numerical failure) and ~22% fail `Ωh² ≥ 1` (bino overclosure).
- Physics filter yield crashed from 60% → 4% across iterations.
- Meanwhile the model never saw the overclosing labels (they were filtered out of the training set), so its predicted-mean at high |M_2| stayed near 0.12 and proximity weighting couldn't catch the problem — it only soft-weights, it doesn't gate.

After the fix, both `top_k` and `entropy_batch` now run through `select_top_uncertain_filtered` which applies the three-stage pre-filter (tolerance → proximity → rank), making the two selection strategies share an identical pipeline up through the scoring step. Only the final batch-construction differs (greedy argsort vs. iterative entropy-diverse selection).

Files: [pmssm/selection.py](../pmssm/selection.py), [active_learning.py](../active_learning.py), [active_learning_tabpfn.py](../active_learning_tabpfn.py), [active_learning_gp.py](../active_learning_gp.py).

### Fix 2 — `submit_al_gp_deep.sh` candidate-pool size aligned

The entropy-batch variant of the deep-GP script was passing `--n-candidates 500000` while its `_top_k` counterpart passed `1000000`. The two are now both 1M so the top_k vs entropy comparison is on an equal footing.

File: [slurm/submit_al_gp_deep.sh](../slurm/submit_al_gp_deep.sh).

## Before → after output-dir map

| Pipeline | Before (latest completed pre-fix) | After (submitted 2026-04-17 16:43) | New job |
|---|---|---|---|
| transformer `top_k` | `active_learning_transformer_top_k_no_warm_20260416_153902` | `active_learning_output_top_k_slurm_20260417_164349` | 8064510 |
| transformer `top_k` n_select=20k | `active_learning_output_top_k_n_select_20k_20260414_152657` | `active_learning_output_top_k_n_select_20k_20260417_164349` | 8064511 |
| exact_gp `top_k` | `active_learning_exact_gp_top_k_no_warm_20260416_171841` | `active_learning_exact_gp_top_k_output_20260417_164349` | 8064512 |
| deep_gp `top_k` | `active_learning_deep_gp_top_k_no_warm_20260416_153902` | `active_learning_deep_gp_top_k_output_20260417_164349` | 8064513 |
| deep_gp `entropy_batch` | `active_learning_deep_gp_no_warm_20260416_153606` (500K cand.) | `active_learning_deep_gp_output_20260417_164349` (1M cand.) | 8064514 |
| tabpfn `top_k` (default) | `active_learning_tabpfn_output_slurm_20260414_152657` | `active_learning_tabpfn_output_slurm_20260417_164349` | 8064515 |

Output dirs live in `/ptmp/jwuerzin/output/`.

## Runs NOT re-submitted (pre-filter stack unchanged)

- transformer entropy_batch: `active_learning_transformer_no_warm_20260416_153606`
- tabpfn entropy_batch: `active_learning_tabpfn_entropy_output_slurm_20260414_152657` (only reached iter 7; TabPFN entropy_batch is prohibitively expensive — see README)
- exact_gp entropy_batch: `active_learning_exact_gp_output_20260416_171932`

## What to watch for when these finish

1. **TabPFN top_k yield**: should jump back up from ~17% overall to something closer to 50–70%, matching the transformer top_k baseline. Median |M_2| of AL-selected seeds should shift *toward* 0 (the tolerance cut drops candidates predicted to overclose at large |M_2|), not away.
2. **Training distribution**: fraction of training points with |M_2|<200 should *decrease* for TabPFN (previously 44.6% — inflated by filter-survivor bias), converge to what the transformer top_k sees (~22%).
3. **Candidate-uncertainty plot for TabPFN top_k**: the previously-observed pattern (high uncertainty everywhere with dip at M_2=0) should shrink / shift because the model will now receive training data that corrects its overclosing-region extrapolation.
4. **Deep-GP entropy vs top_k comparison**: now valid, previously n-candidates were mismatched so any difference in selection quality was confounded with pool size.

## Aligned CLI defaults (after this change)

| Flag | transformer | tabpfn | gp |
|---|---|---|---|
| `--selection-strategy` | entropy_batch | **top_k** (intentional — entropy_batch is too slow for TabPFN) | entropy_batch |
| `--proximity-sampling` | 0.1 | 0.1 | 0.1 |
| `--tolerance-sampling` | 1.0 | 1.0 | 1.0 |
| `--entropy-pool-size` | 5000 | 5000 | 5000 (was 10000) |
| `--entropy-blur` | 0.15 | 0.15 | 0.15 |
| `--entropy-beta` | 50.0 | 50.0 | 50.0 |
| `--candidate-generation` | lhs | lhs | lhs (via generate_candidate_pool default) |
