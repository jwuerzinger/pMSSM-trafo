# pMSSM-trafo

Transformer-based neural networks for predicting dark matter relic density from pMSSM (phenomenological Minimal Supersymmetric Standard Model) parameters.

## Overview

This project trains and compares different neural network architectures to predict the dark matter relic density (Ωh²) from 19 pMSSM input parameters. The models learn to map supersymmetric particle mass parameters to cosmological observables.

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

See [doc/active_learning_plan.md](doc/active_learning_plan.md) for:
- Detailed algorithm description
- Complete CLI reference
- Troubleshooting guide
- Implementation details

## Project Structure

```
pMSSM-trafo/
├── pmssm.py                    # Model definitions and training utilities
├── train_pmssm.py              # Main training script with CLI
├── active_learning.py          # Active learning pipeline with baseline comparison
├── pixi.toml                   # Dependency configuration
├── data/                       # ROOT files with pMSSM data
├── logs/                       # Training logs (timestamped)
├── plots/                      # Output plots (organized by run)
├── active_learning_output/     # Active learning results
├── Run3ModelGen/               # Submodule for pMSSM model generation
├── tests/                      # Unit tests
└── doc/                        # Documentation
    ├── TRAINING_GUIDE.md
    ├── LOGGING_INFO.md
    ├── PARALLEL_TRAINING.md
    ├── transformer_configs.md
    └── active_learning_plan.md # Active learning documentation
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

- [Training Guide](doc/TRAINING_GUIDE.md) - Detailed training instructions
- [Parallel Training](doc/PARALLEL_TRAINING.md) - Multi-GPU setup
- [Logging Info](doc/LOGGING_INFO.md) - Log configuration
- [Transformer Configs](doc/transformer_configs.md) - Model architecture details

## License

MIT License - see [LICENSE](LICENSE) for details.
