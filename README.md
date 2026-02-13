# pMSSM-trafo

Machine learning models for predicting pMSSM (phenomenological Minimal Supersymmetric Standard Model) observables, with active learning pipelines for efficient data collection.

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
- CUDA 12.6+
- Python 3.13+
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
# Testing mode (quick verification)
pixi run python active_learning.py --testing

# Full active learning with automatic data generation
pixi run python active_learning.py \
    --n-iterations 10 \
    --n-candidates 10000 \
    --n-select 1000 \
    --generate-data \
    --epochs 2000

# Without data generation (just point selection)
pixi run python active_learning.py \
    --n-iterations 5 \
    --n-candidates 5000 \
    --n-select 100
```

### Key Features

1. **Baseline Comparison**: Trains both active learning and random baseline models in parallel
2. **MC Dropout Uncertainty**: Uses Monte Carlo Dropout for uncertainty estimation
3. **Data Quality Checks**: Automatic detection of duplicates, leakage, and contamination
4. **Contamination-Free**: Strict separation between AL and baseline datasets
5. **Comprehensive Tracking**: Logs losses, R² scores, and dataset sizes across iterations

### Pipeline Stages

```
for each iteration:
    1. Train AL model on current dataset
    2. Train baseline model on random samples (for comparison)
    3. Generate candidate points in pMSSM parameter space
    4. Compute uncertainty using MC Dropout
    5. Select top-K most uncertain points
    6. [Optional] Generate new models via Run3ModelGen
    7. Add generated data to AL training set
```

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

### Models

| Model | Uncertainty | Use Case |
|-------|-----------|----------|
| ExactGP | GP posterior variance | Default, best for <10k samples |
| DeepGP | Deep GP posterior | Larger datasets |
| SparseGP | Variational GP with inducing points | Scalable to large datasets |
| MLP | None (random fallback) | Comparison baseline |

### Quick Start

```bash
# Quick test
python active_learning_gp.py --testing

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
2. **Entropy-Based Batch Selection** (default): LHS candidate sampling, threshold filtering, Gibbs sampling for diverse batches
3. **Early Stopping**: Patience 200 on validation loss (configurable via `--patience`, disable with `--no-early-stopping`)
4. **Multiple Targets**: DMRD (relic density), CrossSection, CLs via `--target`
5. **Comprehensive Metrics**: Accuracy, chi2, pulls, weighted accuracy via `--compute-full-metrics`
6. **ARD Lengthscale Tracking**: Per-iteration lengthscale CSV via `--track-lengthscales`
7. **Parallel Training**: AL and baseline models on separate GPUs
8. **YAML Config + SLURM Sweeps**: `--config-file sweep.yaml --sweep-index $SLURM_ARRAY_TASK_ID`

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

## Project Structure

```
pMSSM-trafo/
├── pmssm/                      # 🆕 Unified package (13 modules, ~5500 lines)
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
│   └── model_generation.py     # Run3ModelGen interface
├── pmssm.py                    # Backward compatibility wrapper (re-exports from pmssm/)
├── train_pmssm.py              # Transformer training script
├── active_learning.py          # Transformer AL pipeline (~760 lines)
├── active_learning_gp.py       # GP AL pipeline (~990 lines)
├── plot_progress.py            # Visualization utility for AL progress
├── Shell Scripts:
│   ├── run_active_learning.sh                    # Transformer AL production run
│   ├── run_active_learning_medium_test.sh        # Transformer AL test run
│   ├── run_active_learning_with_config.sh        # 🆕 Config file & sweep example
│   ├── run_active_learning_with_eval.sh          # 🆕 External evaluation example
│   ├── run_active_learning_gp.sh                 # GP AL production run (DeepGP)
│   ├── run_active_learning_gp_2h.sh              # 🆕 GP AL 2-hour test run
│   └── run_active_learning_gp_medium_test.sh     # GP AL medium test
├── pixi.toml                   # Dependency configuration
├── data/                       # ROOT files with pMSSM data
├── logs/                       # Training logs (timestamped)
├── plots/                      # Output plots (organized by run)
├── al_pmssmwithgp/             # GP models submodule (ExactGP, DeepGP, SparseGP, MLP)
├── Run3ModelGen/               # Submodule for pMSSM model generation
├── tests/                      # Unit tests
└── doc/                        # Documentation
    ├── SUMMARY.md              # Project overview
    ├── TRAINING_GUIDE.md       # Transformer training guide
    ├── PARALLEL_TRAINING.md    # Multi-GPU setup
    ├── active_learning_plan.md # Transformer AL design
    ├── gp_integration_plan.md  # GP AL integration & progress
    └── gp_pipeline_comparison.md  # GP pipeline features & CLI
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

### Architecture & Code Organization
- [HARMONIZATION_SUMMARY.md](doc/HARMONIZATION_SUMMARY.md) - Feb 2026 refactoring details ⭐
- **pmssm/ package** - Unified codebase (13 modules replacing monolithic pmssm.py)

### Training Guides
- [Training Guide](doc/TRAINING_GUIDE.md) - Transformer training with train_pmssm.py
- [Parallel Training](doc/PARALLEL_TRAINING.md) - Multi-GPU setup and configuration

### Active Learning Pipelines
- [Transformer AL Plan](doc/active_learning_plan.md) - Detailed design and algorithms ⭐
- [GP Pipeline Reference](doc/gp_pipeline_comparison.md) - GP features, CLI options, models ⭐
- [GP Integration](doc/gp_integration_plan.md) - GP integration progress (completed)

### Reference
- [Logging Info](doc/LOGGING_INFO.md) - Structured logging with structlog
- [Plot Organization](doc/PLOT_ORGANIZATION.md) - Output plot structure

⭐ = Recommended for new users

## License

MIT License - see [LICENSE](LICENSE) for details.
