# pMSSM-trafo

Machine learning models for predicting pMSSM (phenomenological Minimal Supersymmetric Standard Model) observables, with active learning pipelines for efficient data collection.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Models](#models)
- [Active Learning Pipeline](#active-learning-pipeline)
- [GP Active Learning Pipeline](#gp-active-learning-pipeline)
- [TabPFN Active Learning Pipeline](#tabpfn-active-learning-pipeline)
- [Batch Acquisition Strategy](#batch-acquisition-strategy)
- [Cross-Run Analysis](#cross-run-analysis)
- [Slurm Submission](#slurm-submission)
- [Multi-Seed Strategy Sweep](#multi-seed-strategy-sweep)
- [Project Structure](#project-structure)
- [Output](#output)
- [Multi-GPU Training](#multi-gpu-training)
- [Documentation](#documentation)
- [License](#license)

## Overview

This project trains and compares neural network and Gaussian Process architectures to predict dark matter relic density (Ωh²) and other observables from 19 pMSSM input parameters. Two active learning pipelines intelligently select the most informative parameter points for expensive physics simulations (SPheno + micromegas).

### Input Parameters (19 features)
- Slepton masses: `meL`, `meR`, `mtauL`, `mtauR`
- Squark masses: `mqL1`, `muR`, `mdR`, `mqL3`, `mtR`, `mbR`
- Gaugino masses: `M_1`, `M_2`, `M_3`
- Higgsino mass: `mu`
- Trilinear couplings: `At`, `Ab`, `Atau`
- Higgs sector: `mA`, `tanb`

### Output
- Dark matter relic density: Ωh²

## Installation

This project uses [pixi](https://pixi.sh/) for dependency management.

```bash
# Clone the repository
git clone <repository-url>
cd pMSSM-trafo

# Install dependencies
pixi install
```

### Requirements
- CUDA 12.6+ or ROCm 6.x (AMD MI300A tested)
- Python 3.12+
- PyTorch (GPU)

## Quick Start

### Basic Training

```bash
# Full training (all data, 2000 epochs)
pixi run python train_pmssm.py

# Quick test run (limited data)
pixi run python train_pmssm.py --testing

# Custom epochs
pixi run python train_pmssm.py --epochs 1000
```

### Advanced Options

```bash
# Control data loading
pixi run python train_pmssm.py --n-datasets 5 --n-samples 100

# Enable early stopping
pixi run python train_pmssm.py --early-stopping --patience 500

# Force sequential training (disable multi-GPU parallel training)
pixi run python train_pmssm.py --no-parallel
```

### Interactive Shell

Alternatively, activate the pixi environment for an interactive session:

```bash
pixi shell
python train_pmssm.py --testing
```

## Models

Three neural network architectures are trained and compared:

### 1. PMSSMTransformer
Standard transformer with CLS token pooling.
- Pre-normalization architecture for better gradient flow
- Learnable positional encodings
- 3-layer regression head

### 2. PMSSMTransformerTabular
Transformer designed specifically for tabular data.
- Individual learned embeddings per feature
- Attention pooling instead of CLS token
- Better suited for non-sequential data

### 3. PMSSMFeedForward (MLP Baseline)
Multi-layer perceptron for comparison.
- Feature embedding layer
- 4-layer fully connected network
- Typically performs best on tabular regression tasks

## Active Learning Pipeline

In addition to standard training, this project includes an **active learning pipeline** that intelligently selects the most informative pMSSM points for expensive physics simulations, significantly improving model performance with fewer samples.

### Why Active Learning?

- **Simulation Cost**: Each pMSSM point requires running SPheno + micromegas (computationally expensive)
- **High-Dimensional Space**: The 19-dimensional parameter space cannot be densely sampled
- **Efficient Learning**: Select points where the model is most uncertain to maximize information gain

### Quick Start

```bash
# Testing mode (quick verification, 3 min)
bash run_active_learning_medium_test.sh

# 1-hour test with 50k samples, log transformation
bash run_active_learning_1h.sh

# Full active learning with automatic data generation
pixi run python active_learning.py \
    --y-transform log \
    --n-iterations 10 \
    --n-candidates 10000 \
    --n-select 1000 \
    --generate-data \
    --epochs 2000

# Without data generation (just point selection)
pixi run python active_learning.py \
    --y-transform log \
    --n-iterations 5 \
    --n-candidates 5000 \
    --n-select 100
```

### Key Features

1. **Log Transformation** (NEW): Trains in log space matching GP pipeline for fair comparison
2. **Baseline Comparison**: Trains both active learning and random baseline models in parallel
3. **MC Dropout Uncertainty**: Uses Monte Carlo Dropout for uncertainty estimation
4. **Data Quality Checks**: Automatic detection of duplicates, leakage, and contamination
5. **Contamination-Free**: Strict separation between AL and baseline datasets
6. **Comprehensive Tracking**: Logs losses, R² scores, and dataset sizes across iterations
7. **Consistent R² Calculation**: Always computed in physical space regardless of training space

### Pipeline Stages

```
for each iteration:
    1. Train AL model on current dataset
    2. Train baseline model on random samples (for comparison)
    3. Generate candidate pool via Latin Hypercube Sampling
    4. Score candidates using uncertainty + proximity weighting
    5. Select batch via iterative entropy-based selection (see below)
    6. [Optional] Generate new models via Run3ModelGen
    7. Add generated data to AL training set
```

See [Batch Acquisition Strategy](#batch-acquisition-strategy) for details on steps 3-5.

### Output

```
active_learning_output/
├── active_learning.log          # Main pipeline log
├── iteration_001/
│   ├── selected_points.csv      # Selected points with uncertainty
│   ├── al_training.log          # AL model log
│   ├── baseline_training.log    # Baseline model log
│   ├── plots/
│   │   ├── al/                  # AL diagnostic plots
│   │   └── baseline/            # Baseline diagnostic plots
│   └── scan/                    # [If --generate-data] Generated models
├── iteration_002/...
├── plots/
│   └── iteration_metrics.png    # AL vs Baseline comparison
└── summary.json                 # All results and configuration
```

### Comparison Plots

The pipeline automatically generates comparison plots showing:
- **Loss curves**: Train/validation loss for both AL and baseline
- **R² scores**: Model performance across iterations
- **Dataset sizes**: Growth of training/validation sets

### Integration with Run3ModelGen

Use `--generate-data` to automatically generate new pMSSM models:

```bash
# Requires Run3ModelGen submodule
git submodule update --init
cd Run3ModelGen && pixi run build && cd ..

# Run with automatic data generation
pixi run python active_learning.py --generate-data --n-iterations 10
```

### Documentation

See [doc/active_learning_plan.md](doc/active_learning_plan.md) for detailed algorithm description and CLI reference.

## GP Active Learning Pipeline

An alternative active learning pipeline using Gaussian Process models for native uncertainty estimation (no MC Dropout needed).

⚠️ **Important**: GP and Transformer pipelines use **different input normalization**:
- **Transformer**: Z-score normalization `(X - mean) / std` → centered at 0, unit variance
- **GP**: Min-max normalization `(X - min) / (max - min)` → scaled to [0, 1]

This means models see the **same physical points in different normalized spaces**, which affects kernel/embedding behavior and uncertainty estimates. Both use the same log-space target transformation. See [Implementation Details](doc/IMPLEMENTATION_DETAILS.md#key-pipeline-differences) for full analysis.

### Models

| Model | Uncertainty | Use Case |
|-------|-----------|----------|
| ExactGP | GP posterior variance | Default, best for <10k samples |
| DeepGP | Deep GP posterior | Larger datasets |
| SparseGP | Variational GP with inducing points | Scalable to large datasets |
| MLP | None (random fallback) | Comparison baseline |

### Quick Start

```bash
# Quick test (3 min)
bash run_active_learning_gp_medium_test.sh

# 1-hour test with 50k samples
bash run_active_learning_gp_1h.sh

# Full run with entropy-based batch selection (default)
python active_learning_gp.py \
    --n-iterations 10 \
    --n-samples 100000 \
    --generate-data

# Simpler top-K variance selection
python active_learning_gp.py \
    --selection-strategy top_k \
    --n-iterations 10

# Production run
bash run_active_learning_gp.sh
```

### Key Features

1. **Native GP Uncertainty**: No MC Dropout needed - uses posterior variance directly
2. **Log Transformation**: Trains in log space: `log(Y / 0.12)` for DMRD target
3. **Entropy-Based Batch Selection** (default): LHS candidate sampling, threshold filtering, Gibbs sampling for diverse batches
4. **Early Stopping**: Patience 200 on validation loss (configurable via `--patience`, disable with `--no-early-stopping`)
5. **Multiple Targets**: DMRD (relic density), CrossSection, CLs via `--target`
6. **Comprehensive Metrics**: Accuracy, chi2, pulls, weighted accuracy via `--compute-full-metrics`
7. **ARD Lengthscale Tracking**: Per-iteration lengthscale CSV via `--track-lengthscales`
8. **Parallel Training**: AL and baseline models on separate GPUs
9. **YAML Config + SLURM Sweeps**: `--config-file sweep.yaml --sweep-index $SLURM_ARRAY_TASK_ID`

### ⚠️ Pipeline Comparison: Pre-filtering Strategies

GP and Transformer pipelines use **different pre-filtering strategies** due to computational constraints:

| Pipeline | Strategy | Reason |
|----------|----------|--------|
| **GP** | `--tolerance-sampling` (value-based) | Can evaluate 1M candidates cheaply (single forward pass per point) |
| **Transformer** | `--entropy-pool-size` (uncertainty-based) | MC Dropout requires 30× forward passes - limits candidate pool to ~50k |

**Key insight**: GP gets mean + variance from a single forward pass (analytical), while Transformer needs 30 MC Dropout samples to estimate uncertainty. This makes evaluating 1M candidates 30× more expensive for Transformer.

Both approaches achieve similar goals but are optimized for their computational profiles. See [IMPLEMENTATION_DETAILS.md](doc/IMPLEMENTATION_DETAILS.md#pre-filtering-strategies) for detailed explanation.

### Training Configuration

| Setting | Default | CLI Option |
|---------|---------|------------|
| Max training epochs | 2000 | `--epochs` |
| Early stopping | On | `--early-stopping/--no-early-stopping` |
| Patience | 200 | `--patience` |
| Learning rate | 1e-3 | `--learning-rate` |
| Selection | entropy_batch | `--selection-strategy` |
| Kernel | RBF with ARD | `--kernel`, `--use-ard` |

### Advanced Features

Both pipelines support several advanced capabilities for production workflows:

#### Config Files & Parameter Sweeps
Use YAML configuration files for systematic hyperparameter exploration:

```bash
# Create a sweep configuration
cat > sweep_config.yaml <<EOF
epochs: [1000, 2000, 5000]
dropout: [0.1, 0.15, 0.2]
# Total: 3 × 3 = 9 combinations
EOF

# Run specific combination
python active_learning.py --config-file sweep_config.yaml --sweep-index 0

# Or with SLURM array jobs
#SBATCH --array=0-8
python active_learning.py --config-file sweep_config.yaml --sweep-index $SLURM_ARRAY_TASK_ID
```

See [run_active_learning_with_config.sh](run_active_learning_with_config.sh) for a complete example.

#### Warm Starting (Default: Enabled)
Resume training from previous iteration checkpoints for faster convergence:

```bash
# Enabled by default
python active_learning.py --warm-starting

# Train from scratch each iteration
python active_learning.py --no-warm-starting
```

#### Early Stopping (Default: Enabled)
Automatically stop training when validation loss plateaus:

```bash
# Default: enabled with patience=200
python active_learning.py --early-stopping --patience 200

# Disable to see full training curves
python active_learning.py --no-early-stopping
```

#### External Evaluation Dataset
Evaluate models on a separate held-out dataset:

```bash
python active_learning.py \
    --eval-data-path /path/to/eval_data.root \
    --compute-full-metrics
```

#### Comprehensive Metrics
Compute full evaluation suite (accuracy, MSE, RMSE, chi², pulls):

```bash
python active_learning.py --compute-full-metrics
```

### Output

```
active_learning_gp_output/
├── active_learning.log
├── iteration_001/
│   ├── selected_points.csv
│   ├── model_checkpoint.pt
│   ├── plots/{al,baseline}/     # Losses, scatter, histogram, random predictions
│   └── scan/                    # [If --generate-data]
├── plots/iteration_metrics.png  # AL vs Baseline comparison
├── lengthscales.csv             # [If --track-lengthscales]
└── summary.json
```

### Documentation

See [doc/gp_integration_plan.md](doc/gp_integration_plan.md) for progress tracking and [doc/gp_pipeline_comparison.md](doc/gp_pipeline_comparison.md) for full feature reference and CLI options.

## TabPFN Active Learning Pipeline

A third pipeline using [TabPFN](https://github.com/PriorLabs/TabPFN) (Tabular Prior-Fitted Network), a pre-trained foundation model for tabular data. Unlike the transformer and GP pipelines, TabPFN requires no per-iteration training -- it performs in-context learning at inference time.

### Key Differences from GP/Transformer

| Aspect | TabPFN | Transformer / GP |
|--------|--------|-----------------|
| Training | None (pre-trained) | Per-iteration training with early stopping |
| Uncertainty | Ensemble of 16 forward passes | MC Dropout / GP posterior |
| GPU usage | Single GPU | 1-2 GPUs (parallel AL + baseline) |
| Warm starting | N/A | Enabled by default |

### Quick Start

```bash
# Production run (requires TABPFN_TOKEN)
export TABPFN_TOKEN="<your-api-key>"
python active_learning_tabpfn.py \
    --n-iterations 40 \
    --n-candidates 1000000 \
    --n-select 500 \
    --generate-data
```

### Documentation

See [doc/CLI_REFERENCE.md](doc/CLI_REFERENCE.md) for TabPFN-specific options.

## Batch Acquisition Strategy

Both pipelines choose informative points for simulation via a shared pre-filter stack, then pick via one of two selection strategies: `top_k` (greedy) or `entropy_batch` (diverse). The default is `entropy_batch` for the Transformer and GP pipelines, and `top_k` for the TabPFN pipeline — see the note under Stage 3 for why.

### Stage 1: Uncertainty Estimation

| Pipeline | Method | How It Works |
|----------|--------|-------------|
| **Transformer** | MC Dropout | Run T stochastic forward passes (default T=30) with dropout active; uncertainty = variance across passes |
| **TabPFN** | Native predictive variance | Variance of TabPFN's Bayesian in-context predictive distribution |
| **GP** | Posterior variance | Single forward pass through the GP likelihood gives analytical mean and variance |

### Stage 2: Candidate Pre-filtering (applied by **both** selection strategies)

1. **Candidate generation**: A pool of `n_candidates` points (default 20,000) is generated via Latin Hypercube Sampling across the 19D parameter space.

2. **Tolerance cut** (`--tolerance-sampling`, default 1.0): A **hard cut** that keeps only candidates whose predicted target lies within `[threshold − tol, threshold + tol]` in transformed space. With the default `log(Y/0.12)` transform and `tol=1.0`, this keeps candidates predicted to give `Y ∈ [0.044, 0.326]` — i.e. within a factor ~3 of the observed relic density. Candidates the model extrapolates as overclosing (Y ≫ 0.12) or far-sub-dominant are dropped outright. Set to 0 to disable.

   > **Note:** applied by *both* `top_k` and `entropy_batch` as of the current version. Previously only `entropy_batch` honored it.

3. **Proximity weighting** (`--proximity-sampling`, default 0.1): A **soft** Gaussian weight on the surviving candidates' variances:

   ```
   proximity = exp(-((pred_mean - threshold)^2) / sigma)
   weighted_variance = proximity * variance
   ```

   Candidates further from the target get down-weighted but not eliminated. Set to 0 to disable.

4. **Focused pool** (entropy_batch only, `--entropy-pool-size`): The top `n_pool` candidates by weighted variance are retained to reduce the downstream covariance computation:

   | Pipeline | Default pool size | Reason |
   |----------|-------------------|--------|
   | **GP** | 1M candidates evaluated, filtered near threshold | Single forward pass per point (cheap) |
   | **Transformer / TabPFN** | 1,500–5,000 | MC Dropout / ensemble requires multiple forward passes (expensive) |

### Stage 3: Selection Strategy

Two strategies operate on the pre-filtered pool from Stage 2. Both honor the tolerance cut and proximity weighting; they differ only in the batch-construction step.

#### `top_k` — greedy variance ranking

After tolerance + proximity, sort survivors by weighted variance and take the top `n_select`. No diversity: if several high-uncertainty candidates cluster in one region, they all get picked. Cheap and simple; appropriate when the filters have already narrowed the pool well.

Implemented in `select_top_uncertain_filtered` ([pmssm/selection.py](pmssm/selection.py)).

#### `entropy_batch` — iterative diverse batch (default for Transformer and GP)

Selects `n_select` points that are jointly informative — not just individually uncertain, but diverse relative to each other.

> **Note on TabPFN**: The TabPFN pipeline defaults to `top_k`, not `entropy_batch`. `entropy_batch` needs a full `T × n_pool` MC-dropout / ensemble prediction tensor to build the sample covariance, which is prohibitively expensive with TabPFN's in-context ensembles — in practice only a handful of iterations complete within a reasonable wall-clock budget. Use `--selection-strategy entropy_batch` explicitly if you want to pay the cost; otherwise the `top_k` path (which still applies the shared tolerance cut and proximity weighting) is recommended.

**Building the covariance matrix** over the focused pool:
- *Transformer*: Sample covariance from T MC Dropout predictions: `C = (X - X̄)^T (X - X̄) / (T - 1)`
- *GP*: Full posterior covariance from the GP likelihood

**Iterative selection** (implemented in `EntropySelectionStrategy`):

1. Compute the smoothed entropy for each candidate individually. Select the candidate with the highest entropy as the first point.

2. For each remaining slot in the batch:
   - For every unselected candidate, compute the **conditional batch entropy** — the information gain from adding that candidate *given the already-selected set*. This uses block matrix conditioning on the joint covariance:
     ```
     Cov = [[C_selected,  C_cross ],
            [C_cross^T,   C_new   ]]
     ```
   - The batch entropy formula is: `log|Sigma + I| - log|Sigma + 2I| + n * log(2)`
   - Select the next point via **Gibbs sampling** with temperature beta (default 50, quasi-deterministic).

3. Repeat until `n_select` points are chosen.

**Why this ensures diversity**: After selecting a point, its contribution to the joint uncertainty is "used up" — nearby candidates with correlated predictions see their conditional entropy reduced. The algorithm naturally spreads selections across the uncertainty landscape rather than clustering them in a single high-variance region.

### Acquisition Hyperparameters

| Parameter | Default | CLI Flag | Description |
|-----------|---------|----------|-------------|
| Selection strategy | `entropy_batch` (Transformer, GP); `top_k` (TabPFN, for cost reasons) | `--selection-strategy` | `entropy_batch` (diverse) or `top_k` (greedy) |
| MC samples | 30 | `--mc-samples` | Forward passes for MC Dropout uncertainty |
| Candidates | 20,000 | `--n-candidates` | LHS candidate pool size |
| Points per iteration | 10 | `--n-select` | Batch size to select |
| Entropy pool size | 1,500 | `--entropy-pool-size` | Pre-filtered pool for covariance (Transformer) |
| Proximity sampling | 0.1 | `--proximity-sampling` | Gaussian width for threshold focus (0 = disabled) |
| Entropy blur | 0.15 | `--entropy-blur` | Smoothing for numerical stability |
| Entropy beta | 50.0 | `--entropy-beta` | Gibbs temperature (higher = more deterministic) |
| Tolerance sampling | 1.0 | `--tolerance-sampling` | Value-based hard pre-filter width (both `top_k` and `entropy_batch`; 0 = disabled) |

See [Implementation Details](doc/IMPLEMENTATION_DETAILS.md) for proximity weighting analysis and pre-filtering strategy comparison.

## Cross-Run Analysis

`analyse_runs.py` loads completed run checkpoints (`state.pt`) and computes quality, physical property, and diversity metrics across multiple runs. The wrapper script `run_analysis.sh` orchestrates predefined comparisons (strategy, model type, parallelism).

```bash
# Run all predefined comparisons
bash run_analysis.sh

# Or run a custom comparison directly
python analyse_runs.py \
    --run-dirs /path/to/run_a /path/to/run_b \
    --labels "Run A" "Run B" \
    --mcmc-data-dir /path/to/mcmc_data \
    --output-dir my_analysis
```

**Outputs**: 8 PNG plots (R² trajectories, hit rate, diversity summary, parameter heatmaps, pairwise scatter) and a `summary.csv` with all scalar metrics and bootstrap uncertainties.

See [doc/ANALYSIS_METRICS.md](doc/ANALYSIS_METRICS.md) for a full description of all metrics, their interpretation, and CLI options.

## Slurm Submission

Slurm job scripts live in `slurm/` and read cluster-specific settings (partition, GPU gres) from `slurm/cluster.conf`:

```bash
# One-time setup
cp slurm/cluster.conf.template slurm/cluster.conf
# Edit cluster.conf for your cluster

# Submit a 2-GPU GP job
source slurm/cluster.conf
sbatch --partition="${CLUSTER_PARTITION}" --gres="${CLUSTER_GPU_GRES_2}" slurm/submit_al_gp_exact.sh

# Resume a timed-out run (+24 more iterations)
bash resume_slurm.sh slurm/submit_al_transformer_top_k_20k.sh /ptmp/output/previous_run 24
```

Available job scripts: `submit_al_transformer.sh`, `submit_al_transformer_top_k.sh`, `submit_al_transformer_top_k_20k.sh`, `submit_al_gp_exact.sh`, `submit_al_gp_exact_top_k.sh`, `submit_al_gp_deep.sh`, `submit_al_gp_deep_top_k.sh`, `submit_al_tabpfn.sh`, `submit_al_tabpfn_entropy.sh`. All parameters can be overridden via environment variables (see script headers for details).

## Multi-Seed Strategy Sweep

For multi-seed comparisons of `(model × selection_strategy × warm/cold)` configurations, use `slurm/submit_strategy_sweep.sh`. This meta-launcher submits one `sbatch` per `(config, seed)`, appends a row to a manifest CSV at submit time, and produces collision-free output directory names that encode the config.

### Selection strategies

A third "short-circuit" strategy, `top_k_tol_only`, complements the existing `top_k` and `entropy_batch`:

| Strategy | Tolerance cut | Proximity weighting | DPP-style covariance pick |
|---|---|---|---|
| `top_k_tol_only` | ✓ | ✗ | ✗ |
| `top_k` | ✓ | ✓ | ✗ |
| `entropy_batch` | ✓ | ✓ | ✓ |

All three apply the same tolerance pre-filter; they differ only in how far through the selection pipeline they continue. Every driver (`active_learning.py`, `active_learning_gp.py`, `active_learning_tabpfn.py`) accepts `--selection-strategy top_k_tol_only` and a new `--seed INT` flag for reproducibility.

### Running the sweep

```bash
# Full grid (100 jobs): 3 models × 3 strategies × 2 warm/cold × 5 seeds + tabpfn (2 × 1 × 5)
bash slurm/submit_strategy_sweep.sh

# Preview without submitting
DRY_RUN=1 bash slurm/submit_strategy_sweep.sh

# Partial sweeps (env-var filters)
MODELS=transformer STRATEGIES=top_k_tol_only WARM_MODES=warm SEEDS=1,2,3 \
    bash slurm/submit_strategy_sweep.sh

# Include the (expensive) tabpfn + entropy_batch combination
TABPFN_ALLOW_ENTROPY=1 bash slurm/submit_strategy_sweep.sh

# BUNDLED MODE: one multi-node sbatch per (model, strategy, warm) cell runs
# all requested seeds in parallel via srun, reducing the job count from 100
# to 20. Useful when AssocMaxJobsLimit caps you at 8 concurrent jobs.
BUNDLE_SEEDS=1 bash slurm/submit_strategy_sweep.sh
```

**Defaults** (override via env vars): `SEEDS=1,2,3,4,5`, `MODELS=transformer,deep_gp,exact_gp,tabpfn`, `STRATEGIES=top_k,top_k_tol_only,entropy_batch`, `WARM_MODES=warm,cold`. TabPFN auto-skips the warm/cold axis (no training, no warm-start to apply) and skips `entropy_batch` unless `TABPFN_ALLOW_ENTROPY=1`.

### Bundled mode (`BUNDLE_SEEDS=1`)

In default mode the launcher submits one sbatch per `(model, strategy, warm, seed)` — 100 independent jobs that compete for the `AssocMaxJobsLimit=8` concurrent-job cap. When this cap is the bottleneck, bundling lets you run ~5× more seeds in parallel without raising the cap:

- One sbatch per `(model, strategy, warm)` cell (20 bundles total).
- Each bundle allocates `n_seeds × 2-GPU nodes` and spawns one `srun` per seed in parallel via [slurm/submit_al_bundled.sh](slurm/submit_al_bundled.sh).
- All `n_seeds` seeds of the cell train concurrently on their own node; bundle wall-clock = max(seed₁, ..., seed_N).
- Manifest still gets one row per seed; the `job_id` column is shared across the seeds of a bundle, and `expected_run_dir` is unique per seed.

Trade-offs:
- Same per-seed compute — **total compute is unchanged**; only the scheduling shape differs.
- Multi-node allocations may queue longer than single-node ones when the cluster is busy.
- Multi-node `--gres=gpu:1` is rejected on apu, so TabPFN bundles also request `gpu:2` per node and let the second GPU idle.
- With 8 concurrent bundles × 5 nodes each, the sweep uses up to 40 nodes simultaneously.

### Output directory convention

Each job writes to `/ptmp/jwuerzin/output/active_learning_{model}_{strategy}_{warm}_seed{N}_{YYYYMMDD_HHMMSS}/`, where `warm` is `warm`, `cold`, or `tabpfn` (sentinel). The driver only appends this suffix if the caller's `--output-dir` does not already end with a timestamp — manual runs work as before.

### Manifest CSV

The launcher writes `/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv` (header on first run, appended thereafter). One row per submitted job:

| column | content |
|---|---|
| `sweep_id` | timestamp of the whole grid — filters "this sweep" from older ones |
| `submit_time` | per-job `sbatch` time |
| `model`, `strategy`, `warm_start`, `seed` | the config key |
| `job_id` | returned by `sbatch --parsable` |
| `expected_run_dir` | absolute path the driver will write to |
| `status` | `submitted` → `pending`/`running`/`completed`/`missing`/`failed`/`timeout`/... |
| `slurm_log` | stdout path |

### Refreshing status

```bash
# Update `status` in place by consulting sacct + checking for state.pt on disk.
# Defaults to refreshing only the latest sweep (lexicographically max sweep_id).
python scripts/update_sweep_manifest.py

# Refresh one specific sweep
python scripts/update_sweep_manifest.py --sweep-id 20260422_154733

# Refresh every row across all sweeps
python scripts/update_sweep_manifest.py --all
```

Idempotent — safe to re-run or cron. A job is marked `completed` only when `summary.json` appears on disk — the driver writes it once, at the very end of the AL loop, so this is the true end-of-run marker. (`state.pt` is overwritten after every iteration and is *not* sufficient evidence of completion; its presence flips the row to `running`.) If sacct reports `COMPLETED` but `summary.json` is missing, the row becomes `missing` (finalise step crashed or output dir mis-pointed).

### Multi-seed hit-rate plots

```bash
# Read manifest, group by (model, strategy, warm_start), render three views
python scripts/plot_hit_rate_trajectories_multiseed.py

# Filter to one sweep, require >=3 completed seeds, use ±1 SD bands instead
python scripts/plot_hit_rate_trajectories_multiseed.py \
    --sweep-id 20260422_154733 \
    --min-seeds 3 \
    --uncertainty sd
```

Default output dir: `/ptmp/jwuerzin/analysis/all_runs/`. The script writes three plot families (each with one panel per relative tolerance — 10 %, 20 %, 50 %):

  1. `hit_rate_settings_<model>.png` — one figure per model, overlaying every (strategy, warm) combo for that model. Colour = strategy, linestyle = warm.
  2. `hit_rate_models_<strategy>_<warm>.png` — one figure per setting, overlaying every model that ran with that setting. Colour = model.
  3. `hit_rate_best_per_model.png` — single figure with one curve per model, picking the (strategy, warm) setting that maximises mean final-iteration hit rate at the strictest tolerance. Falls back to looser tolerances if no config has data at the strictest one. The chosen settings are echoed to stdout.

Groups with fewer than `--min-seeds` completed seeds are silently skipped, so a partially-finished sweep still produces readable figures.

### Multi-seed MCMC R² plots

```bash
python scripts/plot_r2_mcmc_trajectories_multiseed.py

# Clip y-axis to focus on best-performing region
python scripts/plot_r2_mcmc_trajectories_multiseed.py --y-min -10 --y-max 1
```

Sister script to the hit-rate plotter: same manifest filtering, seed grouping, and visual encoding, but plots `al_on_mcmc_r2` (AL surrogate's R² on the held-out MCMC eval set). Outputs go to the same default dir under `r2_mcmc_strategy_<strategy>.png` (one per strategy) and `r2_mcmc_best_per_model.png`. Each figure has a single panel since R² is scalar per iteration. R² on MCMC is typically deeply negative because the MCMC chain distribution differs sharply from the iid training pool — the `--y-min` / `--y-max` flags are useful for comparing high-performing configs without the long tail of bad runs flattening the axis.

## Project Structure

```
pMSSM-trafo/
├── pmssm/                      # Unified package (13 modules)
│   ├── __init__.py             # Package exports
│   ├── config.py               # Constants, parameter ranges
│   ├── data.py                 # Data loading & normalization
│   ├── datasets.py             # PyTorch Dataset classes
│   ├── models/                 # Neural network architectures
│   │   ├── transformer.py      # PMSSMTransformer variants
│   │   └── feedforward.py      # PMSSMFeedForward (MLP)
│   ├── selection.py            # Candidate generation (LHS), selection strategies
│   ├── uncertainty.py          # MC Dropout & GP uncertainty
│   ├── training.py             # Training loops for transformer & GP
│   ├── evaluation.py           # R², metrics, lengthscales
│   ├── visualization.py        # Plotting utilities
│   ├── logging_utils.py        # Structured logging
│   ├── resume.py               # Checkpoint loading for state.pt
│   └── model_generation.py     # Run3ModelGen interface
├── train_pmssm.py              # Transformer training script
├── active_learning.py          # Transformer AL pipeline (MC Dropout)
├── active_learning_gp.py       # GP AL pipeline (ExactGP, DeepGP, SparseGP)
├── active_learning_tabpfn.py   # TabPFN AL pipeline (pre-trained foundation model)
├── analyse_runs.py             # Cross-run quality & diversity analysis
├── run_analysis.sh             # Wrapper for analyse_runs.py comparisons
├── resume_slurm.sh             # Resume timed-out Slurm jobs from checkpoint
├── slurm/                      # Slurm job submission scripts
│   ├── cluster.conf.template   # Cluster config template (partition, GPU gres)
│   ├── resume_args.sh          # Resume flag builder for slurm scripts
│   ├── submit_al_transformer.sh        # Transformer entropy 2-GPU
│   ├── submit_al_transformer_top_k.sh  # Transformer top-k 2-GPU
│   ├── submit_al_gp_exact.sh           # ExactGP entropy 2-GPU
│   ├── submit_al_gp_exact_top_k.sh     # ExactGP top-k 2-GPU
│   ├── submit_al_gp_deep.sh            # DeepGP entropy 2-GPU
│   ├── submit_al_gp_deep_top_k.sh      # DeepGP top-k 2-GPU
│   ├── submit_al_tabpfn.sh             # TabPFN top-k 1-GPU
│   ├── submit_al_tabpfn_entropy.sh     # TabPFN entropy 1-GPU
│   ├── submit_strategy_sweep.sh        # Multi-seed grid launcher (5 seeds × grid)
│   └── submit_al_bundled.sh            # Multi-node bundled-seed worker (srun-fork one seed per node)
├── scripts/
│   ├── update_sweep_manifest.py        # Refresh sweep manifest status from sacct + state.pt
│   └── plot_hit_rate_trajectories_multiseed.py  # Mean ± SEM hit-rate plot across seeds
├── pixi.toml                   # Dependency configuration
├── data/                       # ROOT files with pMSSM data
├── logs/                       # Slurm job logs (timestamped)
├── al_pmssmwithgp/             # GP models submodule (ExactGP, DeepGP, SparseGP)
├── Run3ModelGen/               # Submodule for pMSSM model generation
├── tests/                      # Unit tests
└── doc/                        # Documentation
    ├── ANALYSIS_METRICS.md     # Cross-run analysis metrics reference
    ├── CLI_REFERENCE.md        # Complete CLI options
    ├── IMPLEMENTATION_DETAILS.md  # Uncertainty, proximity weighting, normalization
    ├── PARALLEL_TRAINING.md    # Multi-GPU setup
    ├── SUMMARY.md              # Project overview
    └── ...                     # Additional guides
```

## Output

Training produces:
- **Log files**: `logs/training_YYYYMMDD_HHMMSS.log`
- **Plots** in `plots/run_YYYYMMDD_HHMMSS/`:
  - `losses_<model>.png` - Training/validation loss curves
  - `<model>_true_vs_pred_*.png` - Scatter plots
  - `<model>_hist_true_vs_pred_*.png` - 2D histograms

## Multi-GPU Training

With 3+ GPUs, models train in parallel:
- GPU 0: PMSSMTransformer
- GPU 1: PMSSMTransformerTabular
- GPU 2: MLP Baseline

See [doc/PARALLEL_TRAINING.md](doc/PARALLEL_TRAINING.md) for details.

## Documentation

### Getting Started
- **[README.md](README.md)** - This file (quick start, overview)
- [Project Summary](doc/SUMMARY.md) - Overview of all pipelines and status

### Active Learning Pipelines
- [Transformer AL Plan](doc/active_learning_plan.md) - Detailed design and algorithms
- [GP Pipeline Reference](doc/gp_pipeline_comparison.md) - GP features, CLI options, models
- [Implementation Details](doc/IMPLEMENTATION_DETAILS.md) - Uncertainty computation, proximity weighting, normalization differences

### Analysis
- [Analysis Metrics](doc/ANALYSIS_METRICS.md) - Cross-run quality & diversity metrics reference

### Reference
- [CLI Reference](doc/CLI_REFERENCE.md) - Complete command-line options for all pipelines
- [Parallel Training](doc/PARALLEL_TRAINING.md) - Multi-GPU setup and configuration
- [Logging Info](doc/LOGGING_INFO.md) - Structured logging with structlog
- [Plot Organization](doc/PLOT_ORGANIZATION.md) - Output plot structure

## License

MIT License - see [LICENSE](LICENSE) for details.
