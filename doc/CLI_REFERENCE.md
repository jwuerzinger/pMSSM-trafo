# CLI Reference

Complete command-line interface reference for both active learning pipelines.

## Quick Reference

```bash
# Transformer AL - Quick test (log-space training)
python active_learning.py --y-transform log --testing

# Transformer AL - Production (log-space training)
python active_learning.py --y-transform log --n-iterations 40 --generate-data --epochs 10000

# GP AL - Quick test
python active_learning_gp.py --testing

# GP AL - Production
python active_learning_gp.py --n-iterations 40 --generate-data --epochs 2000
```

## Common Options

Options available in both pipelines:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--testing` | flag | off | Testing mode (3 datasets, small sample size, few iterations) |
| `--n-iterations` | int | 1 | Number of active learning iterations |
| `--n-candidates` | int | varies | Candidate pool size for selection (Transformer: 10000, GP: 1000) |
| `--n-select` | int | varies | Points to select per iteration (Transformer: 100, GP: 10) |
| `--n-datasets` | int | None | Number of ROOT datasets to load |
| `--n-samples` | int | None | Number of samples to use from datasets |
| `--output-dir` | str | varies | Output directory (Transformer: `active_learning_output`, GP: `active_learning_gp_output`) |
| `--generate-data` | flag | off | Generate new models via Run3ModelGen after selection |
| `--gen-workers` | int | 1 | Number of parallel workers for model generation |
| `--epochs` | int | varies | Max training epochs per AL iteration (Transformer: 2000, GP: 2000) |
| `--early-stopping` | flag | on | Enable early stopping on validation loss |
| `--no-early-stopping` | flag | - | Disable early stopping (train for full epoch budget) |
| `--patience` | int | 200 | Patience for early stopping (iterations without improvement) |
| `--learning-rate` | float | varies | Optimizer learning rate (Transformer: 3e-4, GP: 1e-3) |
| `--warm-starting` | flag | on | Resume training from previous iteration checkpoint |
| `--no-warm-starting` | flag | - | Train from scratch each iteration |
| `--config-file` | str | None | YAML config file for parameter overrides |
| `--sweep-index` | int | None | Sweep combination index (for SLURM array jobs) |
| `--eval-data-path` | str | None | Path to external evaluation dataset (ROOT file) |
| `--compute-full-metrics` | flag | off | Compute comprehensive evaluation metrics (accuracy, MSE, RMSE, R², chi², pulls) |

## Transformer-Specific Options

Options only available in `active_learning.py`:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--y-transform` | str | log | Target transformation: `log` (log-space training) or `zscore` (z-score normalization) |
| `--dropout` | float | 0.1 | Dropout rate for PMSSMTransformerTabular |
| `--mc-samples` | int | 30 | Number of Monte Carlo forward passes for uncertainty estimation |
| `--selection-strategy` | str | top_k | Point selection strategy: `top_k`, `entropy_batch`, `proximity_weighted` |
| `--entropy-pool-size` | int | 5000 | Size of high-uncertainty pool for entropy-based selection |
| `--candidate-generation` | str | lhs | Candidate generation method: `lhs` (Latin Hypercube), `uniform` |
| `--proximity-sampling` | float | 0.0 | Proximity weighting width (0 = disabled, 0.1 = focus near θ=0.12) |

### Selection Strategies (Transformer)

**top_k** (default):
- Select top-K points by MC Dropout variance
- Simple, fast, no diversity guarantee

**entropy_batch**:
- Pre-filter candidates with high entropy
- Use Latin Hypercube Sampling for candidates
- More diverse selection

**proximity_weighted**:
- Weight uncertainty by proximity to target value (θ = 0.12 for DMRD)
- Focuses exploration near scientifically interesting region
- Formula: `u_weighted(x) = u(x) × exp(-((μ(x) - θ)² / (2σ²)))`

## GP-Specific Options

Options only available in `active_learning_gp.py`:

### Model Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--target` | str | DMRD | Target function: `DMRD` (relic density), `CrossSection`, `CLs` |
| `--model-type` | str | exact_gp | GP model: `exact_gp`, `deep_gp`, `sparse_gp`, `mlp` |
| `--kernel` | str | RBF | Kernel type: `RBF`, `Matern`, `RQK`, `RBF+Matern` |
| `--lengthscale` | float | 1.0 | Initial kernel lengthscale |
| `--noise` | float | 1e-2 | Initial noise level |
| `--jitter` | float | 1e-3 | Cholesky decomposition jitter |
| `--use-ard` | flag | on | Enable Automatic Relevance Determination (ARD) |
| `--no-ard` | flag | - | Disable ARD (single lengthscale for all dimensions) |
| `--use-dkl` | flag | off | Enable Deep Kernel Learning (ExactGP only) |
| `--inducing-strategy` | str | kmeans | Inducing point initialization: `kmeans`, `vanilla` (SparseGP only) |
| `--batch-size` | int | 256 | Batch size for DeepGP and SparseGP training |

### Selection Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--selection-strategy` | str | entropy_batch | Point selection: `entropy_batch` (default), `top_k` |
| `--entropy-blur` | float | 0.15 | Entropy smoothing parameter (entropy_batch only) |
| `--entropy-beta` | float | 50.0 | Gibbs sampling temperature (entropy_batch only) |
| `--tolerance-sampling` | float | 1.0 | Threshold filter width (entropy_batch only) |
| `--proximity-sampling` | float | 0.1 | Proximity weighting width (entropy_batch only) |

### Evaluation & Tracking Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--track-lengthscales` | flag | on | Save ARD lengthscales per iteration to `lengthscales.csv` |
| `--no-track-lengthscales` | flag | - | Disable lengthscale tracking |
| `--advanced-plots` | flag | off | Generate advanced diagnostic plots (heatmaps, residuals) |

## Usage Examples

### Quick Testing

Test both pipelines with minimal compute:

```bash
# Transformer - 3 datasets, 2 iterations, 100 epochs, log-space training
python active_learning.py --y-transform log --testing

# GP - 3 datasets, 2 iterations, 100 epochs
python active_learning_gp.py --testing
```

### Production Runs

Full active learning with data generation:

```bash
# Transformer - 40 iterations, 50k samples, 10k epochs, log-space training
python active_learning.py \
    --y-transform log \
    --n-iterations 40 \
    --n-samples 50000 \
    --n-select 20000 \
    --n-candidates 50000 \
    --epochs 10000 \
    --generate-data \
    --gen-workers 20 \
    --warm-starting \
    --early-stopping

# GP - 40 iterations, 100k samples, 2k epochs
python active_learning_gp.py \
    --model-type exact_gp \
    --n-iterations 40 \
    --n-samples 100000 \
    --n-select 20000 \
    --n-candidates 50000 \
    --epochs 2000 \
    --generate-data \
    --gen-workers 20 \
    --warm-starting \
    --early-stopping
```

### Advanced Selection Strategies

**Transformer with entropy-based selection**:
```bash
python active_learning.py \
    --y-transform log \
    --selection-strategy entropy_batch \
    --entropy-pool-size 5000 \
    --candidate-generation lhs \
    --n-iterations 10
```

**Transformer with proximity weighting** (focus near DMRD target):
```bash
python active_learning.py \
    --y-transform log \
    --selection-strategy proximity_weighted \
    --proximity-sampling 0.1 \
    --candidate-generation lhs \
    --n-iterations 10
```

**GP with top-K selection** (simpler, faster):
```bash
python active_learning_gp.py \
    --selection-strategy top_k \
    --n-iterations 10
```

**GP with entropy batch** (default, more diverse):
```bash
python active_learning_gp.py \
    --selection-strategy entropy_batch \
    --entropy-blur 0.15 \
    --entropy-beta 50.0 \
    --n-iterations 10
```

### Parameter Sweeps with YAML Config

Create a sweep configuration:

```bash
# Create sweep_config.yaml
cat > sweep_config.yaml <<EOF
epochs: [1000, 2000, 5000]  # 3 values
dropout: [0.1, 0.15, 0.2]   # 3 values (Transformer only)
# Total: 3 × 3 = 9 combinations

# Fixed parameters
y_transform: log
n_iterations: 5
n_candidates: 10000
n_select: 100
warm_starting: true
early_stopping: true
patience: 200
EOF

# Run specific combination (e.g., combo 0: epochs=1000, dropout=0.1)
python active_learning.py --config-file sweep_config.yaml --sweep-index 0

# Or use SLURM array job for all combinations
#SBATCH --array=0-8
python active_learning.py \
    --config-file sweep_config.yaml \
    --sweep-index $SLURM_ARRAY_TASK_ID
```

GP sweep example:

```bash
# gp_sweep_config.yaml
cat > gp_sweep_config.yaml <<EOF
epochs: [1000, 2000, 5000]
learning_rate: [0.001, 0.01]
kernel: [RBF, Matern]
# Total: 3 × 2 × 2 = 12 combinations

n_iterations: 5
model_type: exact_gp
use_ard: true
early_stopping: true
EOF

#SBATCH --array=0-11
python active_learning_gp.py \
    --config-file gp_sweep_config.yaml \
    --sweep-index $SLURM_ARRAY_TASK_ID
```

### External Evaluation Dataset

Evaluate on separate held-out dataset:

```bash
# Transformer
python active_learning.py \
    --y-transform log \
    --eval-data-path /path/to/eval_data.root \
    --compute-full-metrics \
    --n-iterations 10

# GP
python active_learning_gp.py \
    --eval-data-path /path/to/eval_data.root \
    --compute-full-metrics \
    --n-iterations 10
```

### Multi-Target GP Models

Train GP models for different target observables:

```bash
# Dark matter relic density (default)
python active_learning_gp.py --target DMRD --model-type exact_gp

# Direct detection cross section
python active_learning_gp.py --target CrossSection --model-type exact_gp

# LHC exclusion limits
python active_learning_gp.py --target CLs --model-type exact_gp
```

### Model Comparisons

Compare different model architectures:

```bash
# GP models
python active_learning_gp.py --model-type exact_gp --n-iterations 10
python active_learning_gp.py --model-type deep_gp --n-iterations 10
python active_learning_gp.py --model-type sparse_gp --inducing-strategy kmeans --n-iterations 10

# MLP baseline (no uncertainty - uses random selection)
python active_learning_gp.py --model-type mlp --n-iterations 10
```

### Custom Training Configuration

**Disable early stopping** (train for full epoch budget):
```bash
python active_learning.py --y-transform log --no-early-stopping --epochs 5000
```

**Custom early stopping patience**:
```bash
python active_learning.py --y-transform log --early-stopping --patience 500 --epochs 10000
```

**Train from scratch each iteration** (no warm-starting):
```bash
python active_learning.py --y-transform log --no-warm-starting --n-iterations 10
```

**Custom learning rate**:
```bash
# Transformer
python active_learning.py --y-transform log --learning-rate 1e-4 --epochs 5000

# GP
python active_learning_gp.py --learning-rate 1e-2 --epochs 3000
```

### Debugging and Analysis

**Full training curves** (no early stopping):
```bash
python active_learning.py --y-transform log --no-early-stopping --epochs 2000
```

**Longer patience** for better convergence:
```bash
python active_learning.py --y-transform log --patience 500 --epochs 10000
```

**Advanced GP diagnostics**:
```bash
python active_learning_gp.py \
    --advanced-plots \
    --track-lengthscales \
    --compute-full-metrics
```

**Comprehensive evaluation metrics**:
```bash
python active_learning.py --y-transform log --compute-full-metrics
```
Metrics include:
- Accuracy (within 1σ, 2σ, 3σ)
- Weighted accuracy (α = 1, 2, 5, 10)
- MSE, RMSE, R²
- Chi², reduced chi²
- Pulls (mean, std)
- All in both normalized and physical space

## Shell Scripts

Pre-configured shell scripts for common workflows:

### Transformer Active Learning

| Script | Purpose | Configuration |
|--------|---------|---------------|
| [run_active_learning.sh](../run_active_learning.sh) | Production run | 40 iterations, 50k samples, 10k epochs, log-space training, data generation |
| [run_active_learning_1h.sh](../run_active_learning_1h.sh) | 1-hour test | 2 iterations, 50k samples, log-space training |
| [run_active_learning_medium_test.sh](../run_active_learning_medium_test.sh) | Medium test | 3 datasets, 2000 samples, 3 iterations, 500 epochs, log-space training |
| [run_active_learning_with_config.sh](../run_active_learning_with_config.sh) | Config file example | Demonstrates YAML config and parameter sweeps |
| [run_active_learning_with_eval.sh](../run_active_learning_with_eval.sh) | External eval example | Shows external dataset evaluation |

### GP Active Learning

| Script | Purpose | Configuration |
|--------|---------|---------------|
| [run_active_learning_gp.sh](../run_active_learning_gp.sh) | Production run | DeepGP, 40 iterations, 100k samples, 2k epochs |
| [run_active_learning_gp_1h.sh](../run_active_learning_gp_1h.sh) | 1-hour test | DeepGP, 3 iterations, 50k samples, log-space training |
| [run_active_learning_gp_2h.sh](../run_active_learning_gp_2h.sh) | 2-hour test | ExactGP, shorter run for quick validation |
| [run_active_learning_gp_medium_test.sh](../run_active_learning_gp_medium_test.sh) | Medium test | 3 datasets, 2000 samples, 3 iterations |

## Config File Format

YAML configuration files can override any CLI parameter:

```yaml
# Single values - override defaults
y_transform: log
n_iterations: 10
epochs: 5000
learning_rate: 1e-3
early_stopping: true
patience: 200
warm_starting: true

# List values - create parameter sweep (Cartesian product)
dropout: [0.1, 0.15, 0.2]     # 3 values
mc_samples: [20, 30, 50]       # 3 values
# Total combinations: 3 × 3 = 9

# Use --sweep-index to select combination:
# 0 = dropout=0.1, mc_samples=20
# 1 = dropout=0.1, mc_samples=30
# ...
# 8 = dropout=0.2, mc_samples=50
```

**Key features**:
- Underscores in YAML keys (e.g., `n_iterations`, `early_stopping`)
- Hyphens in CLI (e.g., `--n-iterations`, `--early-stopping`)
- Automatic conversion between formats
- List values create Cartesian product for sweeps
- `--sweep-index` selects which combination to run
- Perfect for SLURM array jobs

## See Also

- [active_learning_plan.md](active_learning_plan.md) - Transformer AL design and algorithms
- [gp_pipeline_comparison.md](gp_pipeline_comparison.md) - GP pipeline features and comparison
- [PACKAGE_STRUCTURE.md](PACKAGE_STRUCTURE.md) - Code organization and architecture
- [README.md](../README.md) - Quick start guide
