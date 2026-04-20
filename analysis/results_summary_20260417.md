# pMSSM active-learning results — summary for slides

*Compiled 2026-04-17, rolled forward 2026-04-20 (added §§9–12). Covers runs completed 2026-04-14 through 2026-04-18, the 2026-04-17 re-submissions, the warm-start × strategy × model analysis, and the MC Dropout covariance investigation.*

---

## 1. What we studied

Four **surrogate-model architectures** combined with two **acquisition strategies**, predicting dark-matter relic density `Ωh²` from the 19-D pMSSM parameter vector:

| Model | Uncertainty mechanism | Notes |
|---|---|---|
| Transformer (tabular) | MC-Dropout variance over T=30 stochastic forward passes | baseline NN |
| ExactGP (GPyTorch, RBF + ARD) | Analytical posterior variance + GaussianLikelihood σ²ₙ | gold-standard Bayesian |
| DeepGP | Multi-layer GP, likelihood-sample average | GP + NN hybrid |
| TabPFN | Native Bayesian predictive variance from in-context ensemble (16 members) | pretrained prior |

Two selection strategies:
- **top_k**: rank candidates by (proximity-weighted) variance, take the N highest.
- **entropy_batch**: focused pool → sample-covariance → iterative diverse batch (`EntropySelectionStrategy` with Gibbs sampling, β=50).

All strategies share a pre-filter stack: LHS candidate pool → tolerance cut on predicted `Y_t` (default ±1 in log-space = `Y ∈ [0.044, 0.326]`) → Gaussian proximity weighting around the observed value 0.12.

Where feasible each (model × strategy) combination was also run both with **1 GPU** (sequential AL + baseline) and **2 GPUs** (AL + baseline trained in parallel on separate devices — no DDP, same per-model training recipe). The TabPFN × entropy_batch combination was not completed — the covariance step is prohibitively expensive on in-context ensembles, no run got past iter 7.

Each run targets **40 AL iterations**, 500 seeds/iter submitted to SPheno + micromegas scans with `--gen-workers 20`.

---

## 2. Runtimes (completed 40-iteration runs)

From `sacct` on viper/apu partition. "2GPU" = AL on gpu0, baseline on gpu1 in parallel; "1GPU" = sequential on same device.

| Model / strategy | GPU | Wall time | Job ID | Notes |
|---|---|---|---|---|
| Transformer entropy_batch | 2 | 5 h 59 m | 8047645 | |
| Transformer top_k         | 2 | 4 h 11 m | 8047657 | |
| ExactGP entropy_batch     | 2 | 13 h 14 m | 8048439 | longest GP run |
| ExactGP entropy_batch (output) | 2 | 6 h 36 m | 8048447 | |
| ExactGP top_k             | 2 | 10 h 50 m | 8048440 | |
| ExactGP top_k (output)    | 2 | 5 h 33 m | 8048448 | |
| DeepGP entropy_batch      | 2 | 13 h 21 m | 8047643 | |
| DeepGP top_k              | 2 | 16 h 49 m | 8047655 | most expensive |
| TabPFN top_k              | 2 | ~ same order as transformer | 8033834 etc. | |
| TabPFN entropy_batch      | — | **did not complete** (iter ≤ 7 in 24 h) | — | entropy-batch covariance is infeasible |
| Transformer entropy       | — | **TIMEOUT at 24 h** | 8037029 | ran out of walltime |

Rough ordering of cost: DeepGP > ExactGP ≫ Transformer. GP runs fill most of the 24 h slurm walltime; transformer finishes in < 6 h.

---

## 3. Uncertainty profiles (the qualitative finding)

Plotting *candidate std vs. each input* at iteration 40 reveals a striking asymmetry at `M_2 ≈ 0`:

| Model | Shape of σ(candidate) vs M_2 | Mechanism |
|---|---|---|
| ExactGP / DeepGP | **Dip** at M_2=0 | Posterior variance = `k(x,x) − kᵀ K⁻¹ k + σ²ₙ`. The data-dependent term shrinks where training data is dense; homoscedastic σ²ₙ adds a *global* constant and cannot localize. |
| TabPFN | **Dip** at M_2=0 (with small local spikes) | Bayesian predictive variance is coverage-driven, same qualitative behavior as the GP. |
| Transformer (MC-Dropout) | **Peak** at M_2=0 | Variance = `Var_T[ f_dropout(x) ]` across dropout masks — pure disagreement, no aleatoric/epistemic separation. |

**Why M_2≈0 is special (physics):**
The relic-density target is log-transformed (`Y_t = log(Y / 0.12)`). Near M_2 ≈ 0 the lightest neutralino becomes wino-like, annihilates efficiently into W⁺W⁻, and `Ωh² → 0`. In raw-Y space near-zero targets are compressed; in log-space they become very negative outliers → a *sharp cusp in Y_t at M_2 = 0*. MC-Dropout's mask disagreement surfaces this cusp as predictive variance; the GP absorbs the same signal into its single global noise scalar.

**LSP composition near M_2 ≈ 0** (|M_2| < 300, from the filtered training set of 300k models):

| LSP type | n | median Ωh² | models with 0.08 < Ω < 0.16 |
|---|---|---|---|
| wino | 87 454 | 0.0005 | 39 |
| higgsino | 6 254 | 0.0035 | **443** |
| mixed / well-tempered | 5 552 | 0.0022 | 174 |
| bino | 835 | 0.3530 | 66 |

The dip in the raw data is populated almost entirely by **wino LSPs**; the ones *at* the observed relic density are **higgsinos** and **well-tempered mixtures**.

Supporting plots (in [analysis/](.)):
- `m2_multimodality.png` — raw Ωh² vs M_2
- `m2_logspace.png` — log-space cusp and within-bin std inversion
- `m2_lsp_type.png` — LSP classification near M_2=0

---

## 4. Why high-|M_2| scans fail

Breakdown over the full 300k-model dataset, by seed |M_2| bin:

| \|M_2\| bin | pass all filters | Ωh² ≥ 1 | Ωh² ≤ 0 | SPheno m_h = −1 |
|---|---|---|---|---|
| [0, 100)    | **92 %** | 2 % | 5 % | 6 % |
| [100, 300)  | 79 %     | 7 % | 5 % | 14 % |
| [300, 700)  | 53 %     | 15 % | 4 % | 32 % |
| [700, 1200) | 36 %     | 21 % | 4 % | 44 % |
| [1200, 2000)| **31 %** | 22 % | 3 % | **47 %** |

Two independent physics effects compound as |M_2| grows:
1. **Bino overclosure** — at large M_2 the LSP is typically bino (|M_1| smallest); pure binos have poor annihilation and `Ωh²` runs away to ≫ 0.12. 54 % of all bino points in the dataset overclose the universe.
2. **Radiative-EWSB / SPheno failure (m_h = −1)** — with fixed heavy scalars (mqL3 = mtR = 4000) and a very heavy M_2, SPheno's EWSB solver fails to converge. This is the *dominant* failure mode at high |M_2| (47 %).

Wino LSPs pass 96 %, higgsinos 97 %, binos only 43 % — the physics filter is essentially a "neutralino composition" filter that strongly favors the small-|M_2| region.

---

## 5. The selection-filter bug (diagnosed and fixed 2026-04-17)

### Symptoms

TabPFN top_k training data was anomalously concentrated at `|M_2| < 200`: **44.6 %** of the final 4,995 training points lay there, vs **22.2 %** for the transformer top_k run (14,008 training points).

### Root cause — a bootstrap trap

- The **AL-selected seed distribution** was the *opposite*: TabPFN seeds had median `|M_2| = 1447` and only 8.3 % were in the central region (transformer top_k: median 715, 16.4 % central).
- TabPFN's uncertainty dips at `M_2=0` → top_k pushes selections outward toward large |M_2|.
- But large |M_2| seeds mostly fail the physics filter (see §4). The scan yield collapsed from 60 % in early iterations to ~4 % by iteration 30+. Over the full run: **3,395 valid points out of 20,000 seeds (17 %) for TabPFN** vs. **12,408 / 20,000 (62 %) for the transformer**.
- The few scans that *do* survive near-|M_2|=2000 submissions tend to be the edge cases that happened to land near M_2≈0 — so the training set ended up biased back toward the central region anyway, despite the AL's intent to avoid it.
- Meanwhile the model never saw the overclosing labels (they were filtered out before training), so its extrapolation over high-|M_2| regions stayed near the observed value and kept looking attractive to the acquisition function.

### Why the existing proximity filter didn't catch this

`proximity_sampling = 0.1` *was* active every iteration — but proximity is a *soft* Gaussian weight on `pred_var`, it does not drop candidates. A candidate with miscalibrated predicted mean near 0.12 still ranks high.

The `entropy_batch` path also applied a **hard** tolerance cut (`tolerance_sampling = 1.0`: drop all candidates with predicted `Y_t` outside `[-1, +1]`) *before* ranking. The `top_k` path silently ignored this flag.

### Fix

Added `select_top_uncertain_filtered` that mirrors the three stages of `select_entropy_batch_mc` (tolerance cut → proximity weight → rank). Wired into all four top_k call sites (main + retry) across the transformer, TabPFN, and GP pipelines. Both strategies now share the identical pre-filter stack up to the batch-construction step, which is the only place they differ.

Files: [pmssm/selection.py](../pmssm/selection.py), [active_learning.py](../active_learning.py), [active_learning_tabpfn.py](../active_learning_tabpfn.py), [active_learning_gp.py](../active_learning_gp.py), [README.md](../README.md).

Also aligned CLI defaults across the three pipelines:
- TabPFN `--selection-strategy`: now `top_k` (intentional — entropy_batch infeasible for TabPFN).
- GP `--entropy-pool-size`: 10 000 → 5 000 (matches what every script actually passed).
- `submit_al_gp_deep.sh` `--n-candidates`: 500 k → 1 M (matches the `_top_k` counterpart).

### Re-submitted runs

Six jobs resubmitted 2026-04-17 at 16:43 with timestamp `20260417_164349` (jobs 8064510–8064515): all five top_k variants + the deep_gp entropy variant (for n-candidates parity). See [resubmission_20260417.md](resubmission_20260417.md) for the before→after output-dir map.

**Expected effect** (predictions made 2026-04-17 — **not borne out**, see §9):
- TabPFN top_k yield should climb back toward 50–70 %. *(Actual: 21.1 → 24.8%)*
- TabPFN top_k training-data distribution should *decouple* from the filter-survivor bias; median |M_2| should move outward from 152 toward something closer to the transformer's 541. *(Actual: the 152 figure was the training-set median; the **selected-seed** median was already 1447 pre-fix and moved only to 1422.)*
- The candidate uncertainty plot at large |M_2| should shrink as TabPFN finally gets to see overclosing labels and learns to down-weight those regions. *(Not re-measured; unlikely given selections barely shifted.)*

---

## 6. Quantitative performance comparison (pre-fix runs)

From [all_runs/summary.csv](/ptmp/jwuerzin/analysis/all_runs/summary.csv). Hit rate = fraction of training points with `|Ω − 0.12| / 0.12 < tol`.

| Run | Hit rate ±50 % | Hit rate ±20 % | AL val R² | MMD² vs MCMC |
|---|---|---|---|---|
| transformer_entropy_2gpu | **0.290** | **0.116** | 0.23 | 0.016 |
| transformer_entropy_1gpu | 0.252 | 0.097 | 0.22 | 0.018 |
| transformer_top_k_2gpu | 0.224 | 0.089 | 0.26 | 0.024 |
| transformer_top_k_1gpu | 0.239 | 0.095 | 0.26 | 0.018 |
| exact_gp_entropy_2gpu | 0.040 | 0.020 | — | — |
| exact_gp_top_k_2gpu | 0.182 | 0.040 | 0.42 | 0.014 |
| exact_gp_top_k_1gpu | **0.229** | **0.078** | **0.46** | 0.016 |
| deep_gp_entropy_2gpu | 0.206 | 0.084 | 0.30 | 0.037 |
| deep_gp_top_k_2gpu | 0.151 | 0.055 | 0.30 | 0.037 |
| deep_gp_top_k_1gpu | 0.135 | 0.046 | 0.45 | 0.047 |
| tabpfn_top_k_2gpu | 0.166 | 0.063 | 0.35 | 0.026 |
| tabpfn_top_k_1gpu | 0.186 | 0.073 | 0.29 | 0.027 |

**High-level take** (on pre-fix data — TabPFN numbers will shift after the re-run):
- **Transformer entropy_batch on 2 GPUs** has the best hit rate at every tolerance (~29 % at ±50 %).
- **ExactGP top_k** is the best GP config (22 % hit rate, 0.46 val R² on 1-GPU variant).
- **ExactGP entropy_batch** surprisingly underperforms its top_k sibling on hit rate (4 % vs 22 %) — worth investigating; may be a symptom of the same tolerance-cut interaction.
- **TabPFN top_k** is middle-of-the-pack (19 %), but suffers from the filter bias documented above — the fix should improve these numbers.
- **DeepGP** is the most expensive and not the most accurate; hard to recommend unless the flexibility matters for some downstream task.

### 1-GPU vs 2-GPU

The codebase's "2-GPU" mode does *not* train one model across two devices — it simply runs the AL model and the baseline in parallel on separate GPUs. Per-model training is identical across GPU counts, so differences in the table above (e.g. transformer_entropy ±3.8 pp, exact_gp_top_k ±4.7 pp) are almost certainly **seed/training stochasticity**, not a systematic GPU-count effect. The permutation-test SEs reported in `summary.csv` capture *within-run* variance, not between-seed variance, so they overstate significance.

---

## 7. Distributional comparison (MMD²)

MMD² vs MCMC with Gaussian kernel (median-heuristic bandwidth, shared across all runs for comparability):

- All runs have MMD² in the range 0.01 – 0.05 — meaningfully above 0 (permutation test p-values are small), so AL training sets are distinguishable from the MCMC posterior, as expected.
- Transformer configurations have the *lowest* MMD² (~0.017), i.e., their training data most resembles the MCMC posterior.
- DeepGP top_k has the *highest* MMD² (~0.04–0.05), i.e., it covers the parameter space least like the posterior.
- Relative differences of ~0.001 look "highly significant" by z-score because permutation-test SEs are tiny (≲ 10⁻⁴), but those z-scores don't translate to physical/practical significance — treat MMD² as a qualitative ranking metric, not a precision measurement.

---

## 8. Open questions / things to check when the re-runs finish

1. **Does the filter fix close the TabPFN gap?** *Answered 2026-04-20 — **No** (see §9).* Yield climbed only 21.1 → 24.8%, well short of the 50% target; AL R² and selected-seed distribution essentially unchanged. The fix was necessary (top_k was silently ignoring the tolerance flag) but insufficient to break the bootstrap trap, because the cut fires on the model's *predicted* mean — which sits near 0.12 at high |M_2| precisely because overclosing labels never reach training.
2. **Why does exact_gp entropy_batch score so poorly (4 % hit rate, 0.13 mean Ω)?** Still open — the entropy variant was not re-submitted with the selection fix. What we do know from the post-fix `exact_gp top_k` run: AL R² went flat (0.48 → 0.46) and training-set size *shrank* 4578 → 2764 (yield 18.6 → 7.3%), so the tolerance cut appears to over-prune GP pipelines rather than rescue them. If so, exact_gp entropy_batch likely needs a *different* fix (e.g. looser tolerance or heteroscedastic noise) rather than the same one.
3. **Heteroscedastic likelihood for the GP?** If we want GP uncertainty to reflect the log-cusp at M_2 = 0 the same way MC-Dropout does, we need `HeteroskedasticNoise` instead of `GaussianLikelihood`. Not an easy swap but worth exploring — arguably more promising now given the GP regressions in §9.
4. **TabPFN entropy_batch feasibility.** Could we build a smaller-pool or amortized variant that reduces the covariance cost? Currently it's a hard blocker.
5. **An exact_gp entropy_batch 1-GPU run** is missing from the 1v2 grid — would need ≳ 26 h of walltime with a resume-logic workflow.
6. **Breaking the TabPFN bootstrap trap** (new, raised by §9). Options: (a) stop filtering overclosing labels out of the training set; (b) add an explicit confidence/OOD gate on top of the predicted-mean tolerance; (c) seed with synthetic high-|M_2| labels from a cheaper physics proxy.
7. **MC Dropout covariance quality for entropy_batch** (new, raised by §10 + §11). T = 20 samples over n_pool = 5000 gives a rank-deficient covariance that may explain why entropy_batch underperforms top_k on the transformer. Diagnostic plan in [mc_dropout_covariance_discussion.md](mc_dropout_covariance_discussion.md).
8. **Transformer warm-start divergence** (new, §10). Warm R² drops 0.2–0.36 below cold across both strategies; loss trajectories suggest divergence from ≈iter 3–7 onwards. Needs a reseed before claiming it in the talk.

---

## 9. Post-fix re-run results (analysis 2026-04-20)

The six resubmitted jobs (`20260417_164349`) completed between 2026-04-17 and 2026-04-18. Five finished cleanly at 40 iterations; transformer `top_k` `n_select=20k` (job 8064511) hit the 24 h Slurm wall at iter 18 and is excluded.

### 9.1 Methodological note — warm-start must be held constant

A first pass compared each new run against the `_no_warm_20260416_*` baselines listed in [resubmission_20260417.md](resubmission_20260417.md) and produced a misleading "exact_gp `top_k` improved 0.26 → 0.45". That gap was entirely warm-start (enabled in the new runs, disabled in those baselines). Re-comparing against warm-start-matched pre-fix runs (timestamp `20260414_152657` where available, else `20260415_*` for deep_gp which never completed on 04-14) removes the artifact. All numbers below use warm-start-matched baselines.

### 9.2 Model-quality & yield comparison

Averaged over the last 10 iterations. Yield = physics-survivors / AL-selected seeds over the full 40 iterations.

| Pipeline | AL R² last10 B→A | AL val-loss B→A | Yield B→A | Final N B→A |
|---|---|---|---|---|
| transformer `top_k` | 0.30 → 0.28 ≈ | 1.68 → 2.01 | 60.9 → **70.3%** ↑ | 11356 → 12872 |
| exact_gp `top_k` | 0.48 → 0.46 ≈ | 0.79 → 1.11 | 18.6 → **7.3%** ↓↓ | 4578 → 2764 |
| deep_gp `top_k` | 0.37 → **0.24** ↓ | 1.11 → 1.09 | 36.8 → 63.3% ↑ | 7513 → 11741 |
| deep_gp `entropy_batch` | 0.44 → **0.23** ↓ | 0.81 → 0.92 | 63.5 → 66.3% ≈ | 11778 → 12228 |
| tabpfn `top_k` | 0.36 → 0.34 ≈ | 1.24 → 1.28 | 21.1 → 24.8% ≈ | 4995 → 5582 |

### 9.3 Where the tolerance cut actually redirected selections

| Pipeline | Median \|M_2\| B→A | Frac \|M_2\| < 200 B→A |
|---|---|---|
| transformer `top_k` | 715 → 761 | 16.4 → 17.6% |
| exact_gp `top_k` | 1799 → 1507 | 0.0 → 0.0% |
| deep_gp `top_k` | 809 → 950 | **24.0 → 6.5%** ↓↓ |
| deep_gp `entropy_batch` | 1382 → 1272 | 0.6 → 1.3% |
| tabpfn `top_k` | 1447 → 1422 | 8.3 → 8.5% |

Only two runs' selections moved meaningfully: deep_gp `top_k` was pushed *away* from M_2 ≈ 0 (opposite of the TabPFN hypothesis in §5), and exact_gp `top_k` was pulled inward. TabPFN's selections are essentially unchanged, which directly explains why its yield and R² are unchanged.

### 9.4 Why the TabPFN prediction in §5 didn't pan out

§5 assumed the tolerance cut would drop high-|M_2| candidates predicted to overclose, breaking the yield collapse. But the cut gates on the *predicted* Y_t, and TabPFN's predicted mean at high |M_2| sits near 0.12 — because the overclosing labels it would need to learn from are filtered out of its training set by the pre-training physics filter (the closed loop §5 itself diagnosed). The gate therefore never triggers at high |M_2|, selections don't shift (median |M_2| 1447 → 1422), yield doesn't recover (21.1 → 24.8%). Escaping the trap requires attacking the label-filter side of the loop, not the selection side alone. See §8 Q6.

### 9.5 Unexpected regressions

- **Deep GP `top_k` AL R² 0.37 → 0.24.** Selections shifted *away* from the central M_2 ≈ 0 cusp (frac <200: 24.0 → 6.5%) — the structure §3 identified as most informative for learning the log-Y_t cusp. Plausibly the tolerance cut over-prunes central wino-LSP candidates whose predicted Ω the model (correctly) places near zero, even though those candidates would survive physics.
- **Deep GP `entropy_batch` AL R² 0.44 → 0.23.** This run was resubmitted only for n-candidates parity (500k → 1M), not the selection fix, so its regression hints at either seed noise or a nontrivial effect of pool size on the entropy-batch iterative selection.
- **ExactGP `top_k` training-set shrank 4578 → 2764.** Fewer seeds survived physics despite the fix supposedly improving hit rate. With R² flat this points to the tolerance cut over-pruning GP pipelines rather than helping them.

### 9.6 Caveat — single-seed variance

All five comparisons are n = 1 per cell. R² swings of 0.10–0.15 are plausible from seed and training stochasticity alone; the Deep GP drops of ~0.20 exceed that band but still warrant at least one confirmation seed before treating them as established. §6's 1-GPU-vs-2-GPU note made the same point from the pre-fix data.

### 9.7 Recommendations

1. Run a 2–3 seed sweep for Deep GP `top_k` and `entropy_batch` to separate noise from a real regression. If real, try `--tolerance-sampling` at 2.0 or 3.0 to recover central-region sampling.
2. Attack the TabPFN label-filter side of the bootstrap loop (§8 Q6) — the selection-side fix alone cannot break the trap.
3. Any future before/after write-up must hold warm-start fixed — §9.1 cost us one misleading conclusion already.

---

## 10. Warm-start × acquisition-strategy interaction (analysis 2026-04-20)

Extracted per-iteration metrics from paired warm-start and cold-start runs (warm = default, cold = `--no-warm-starting`) for all three models × both strategies. All cells are 40/40 iterations. **ΔR² = warm − cold** sign convention (positive = warm better).

### 10.1 Warm vs cold, per (model × strategy)

| Model | Strategy | Warm R² | Cold R² | ΔR² | Warm val-loss | Cold val-loss | Δloss | Epoch speedup |
|---|---|---|---|---|---|---|---|---|
| Transformer | entropy | 0.230 | 0.438 | **−0.208** | 1.52 | 1.02 | +0.51 | 1.5× |
| Transformer | top_k | 0.258 | **0.620** | **−0.363** | 1.86 | 0.88 | +0.98 | 1.55× |
| ExactGP † | entropy | **0.466** | 0.196 | **+0.270** | 0.94 | 2.30 | −1.37 | 6.7× |
| ExactGP † | top_k | 0.423 | 0.264 | +0.159 | 0.87 | 2.49 | −1.62 | ≈ |
| DeepGP | entropy | 0.445 | 0.446 | −0.001 | 0.81 | 0.92 | −0.12 | 5.6× |
| DeepGP | top_k | 0.301 | **0.537** | **−0.236** | 1.21 | 0.92 | +0.29 | ≈ |

† ExactGP row uses the `20260416_171932/171841` pair (N ≈ 3.2k), matched to the cold baseline. Not the production `20260415_113302` (N ≈ 10.6k, R² = 0.558) — they are different configs. See also [resubmission_20260420.md](resubmission_20260420.md) for the concurrent LOVE-disable investigation.

### 10.2 What this changes about the story

- **"Entropy always wins" does not survive dropping warm-start.** With warm-start on, entropy beats top-k for DeepGP (+0.14) and ExactGP (+0.04); with warm-start off, top-k wins for Transformer (+0.18) and DeepGP (+0.09). Only ExactGP keeps the entropy advantage in both regimes.
- **Warm-start helps GPs, hurts the transformer.** For ExactGP/DeepGP warm-start is neutral-to-positive on R² *and* 5.6–6.7× faster. For the transformer warm-start is only 1.5× faster *and* costs 0.2–0.36 R². The transformer warm run visibly diverges from iteration 3–7 onward (val-loss trajectory in both strategies climbs after early convergence).
- **Strategy × warm-start is not separable.** Warm-start damages top-k more than entropy for the transformer (ΔR²_gain 0.36 vs 0.21) and for DeepGP (ΔR²_gain 0.24 vs ≈0). Plausible mechanism: warm-start locks the model in an earlier basin; top-k's concentrated queries keep re-selecting neighbourhoods the stuck model can't improve on; entropy's forced diversity partially rescues it. ExactGP is the odd one out because its "warm start" is kernel-hyperparameter reuse, not a network-weight basin.

### 10.3 Confounders

- **Training-set size differs within rows.** Final N for DeepGP warm: top-k = 7,513 vs entropy = 11,778. Top-k + warm concentrates queries in high-variance regions that fail physics validity more often → starvation. The entropy > top-k advantage for DeepGP warm is partly a data-volume effect, not purely an acquisition-rule effect. Top-k cold yields *more* points than entropy cold for Transformer (14k vs 12.7k) and DeepGP (12.5k vs 12k), so this confound runs in both directions and does not alone reconstruct the R² pattern.
- **n = 1 per cell.** These are single-seed comparisons. R² swings of 0.10–0.15 are plausible from seed noise alone; the > 0.20 swings in §10.1 exceed that band but should still be confirmed before committing to the talk.

### 10.4 Follow-up tests

1. **Reseed the transformer warm-start runs** (both strategies). If R² ≈ 0.23–0.26 reproduces, warm-start divergence is real; if it regresses toward the cold baseline, it was seed noise.
2. **Per-iteration R² trajectory plot** for Transformer warm vs cold. Expected from the hypothesis: warm tracks cold until ≈ iter 7, then diverges. If the divergence is gradual from iter 1, the story is different.
3. **Per-iteration validity-rate trajectory** for top-k warm vs cold across models. Expected: top-k warm rate crashes faster than top-k cold because stale weights keep re-selecting uncertain-but-invalid regions.
4. **Same-N comparison.** Truncate each run to a fixed N (say 5,000 selected points) and recompute R². Would separate "better acquisition" from "more data".

---

## 11. MC Dropout covariance quality and entropy-batch (analysis 2026-04-20)

Transformer entropy_batch underperforms top_k in the cold-start regime (ΔR² = −0.18, see §10). Hypothesis developed in [mc_dropout_covariance_discussion.md](mc_dropout_covariance_discussion.md): the sample covariance the entropy-batch strategy consumes is estimated from T = 20 MC Dropout forward passes over a pool of n_pool = 5,000 candidates, making it rank-deficient by construction (rank ≤ T − 1 = 19) with a `1e-4·I` regulariser drowning the 4,980 null-space directions in isotropic noise.

Top-k uses only the diagonal (an ordinal ranking of marginal variances) and is insensitive to this. Entropy-batch's diversity signal is a log-det of a batch submatrix — explicitly a function of the off-diagonals — and inherits the full estimation noise.

Prediction: entropy-batch should degrade gracefully with covariance quality. ExactGP (analytic, full-rank, calibrated) shows entropy ≥ top-k (§10.1 entropy ΔR² +0.04). DeepGP (variational, better than MC Dropout) shows entropy > top-k in warm mode. Transformer (rank-20 MC covariance) shows top-k > entropy in the cold regime where calibration is worst. The ordering is consistent with the hypothesis but n = 1 per cell.

See [mc_dropout_covariance_discussion.md](mc_dropout_covariance_discussion.md) for the full mechanism and 7 prioritised diagnostic tests. Cheapest three (no retraining, no code changes beyond logging): batch stability under re-sampling, eigenvalue spectrum of the sample covariance, and sweeping T ∈ {20, 50, 100, 200} on a fixed checkpoint.

---

## 12. Pending jobs as of 2026-04-20

Two exact_gp re-submissions to validate the LOVE re-enable patch ([resubmission_20260420.md](resubmission_20260420.md)):

| JobID | Name | State | Reason |
|---|---|---|---|
| 8102247 | al_gp_exact_top_k | PENDING | ReqNodeNotAvail, Reserved for maintenance |
| 8102248 | al_gp_exact | PENDING | ReqNodeNotAvail, Reserved for maintenance |

Blocked by `test-image` reservation on apu (ends 2026-04-24 11:00). Site-wide `maint` reservation runs 2026-04-21 07:00 → 2026-04-26 07:00, so realistic start time is after 2026-04-26.

---

## 13. Files and artifacts

- **Analysis plots**: [/ptmp/jwuerzin/analysis/all_runs/](/ptmp/jwuerzin/analysis/all_runs/)
- **Run output dirs**: `/ptmp/jwuerzin/output/active_learning_*`
- **Resubmission logs**: [resubmission_20260417.md](resubmission_20260417.md) · [resubmission_20260420.md](resubmission_20260420.md)
- **MC Dropout covariance discussion**: [mc_dropout_covariance_discussion.md](mc_dropout_covariance_discussion.md)
- **M_2 investigation scripts & plots**: [m2_multimodality.py](m2_multimodality.py) · [m2_logspace.py](m2_logspace.py) · [m2_lsp_type.py](m2_lsp_type.py) (+ `.png` companions in this directory)
- **Code documentation**: [README.md](../README.md) §"Batch Acquisition Strategy" covers the pre-filter stack and strategy tradeoffs.
