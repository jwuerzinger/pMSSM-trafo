# Active Learning Pipeline for pMSSM

## Overview

This document describes the active learning pipeline implemented in `active_learning.py` for iteratively improving the pMSSM relic density prediction model by selecting the most informative training points.

## Concept

Active learning is a machine learning paradigm where the model actively selects which data points to learn from. For expensive physics simulations like pMSSM, this is valuable because:

1. **Simulation Cost**: Each pMSSM point requires running SPheno + micromegas, which is computationally expensive
2. **High-Dimensional Space**: The 19-dimensional parameter space cannot be densely sampled
3. **Efficient Learning**: By selecting points where the model is most uncertain, we maximize information gain per simulation

## Algorithm

### Active Learning Loop

```
for iteration in 1..N:
    1. Train both AL and Baseline models in parallel
       - AL: Trains on current dataset (grows via active learning)
       - Baseline: Trains on random samples from input dataset
    2. For AL model: Generate pool of candidate points in parameter space
    3. Compute predictive uncertainty via MC Dropout
    4. Select top-K points with highest uncertainty
    5. Output points for generation (CSV format)
    6. [If --generate-data] Generate new models via Run3ModelGen
    7. [If --generate-data] Add new data to AL training set for next iteration
    8. Baseline grows by random sampling from input dataset (excluding initial AL samples)
```

### Baseline Comparison

The pipeline trains two models in parallel for fair comparison:

- **Active Learning (AL)**: Intelligently selects uncertain points via MC Dropout
- **Baseline (Random)**: Randomly samples new points from the input dataset

**Key properties**:
- Both models start with identical initial data
- Both models have identical train/val split sizes at each iteration
- AL grows by adding generated points based on uncertainty
- Baseline grows by randomly sampling from the original input dataset (excluding initial AL samples)
- **No data contamination**: Baseline never samples from AL's pool or generated data

## Uncertainty Estimation: MC Dropout

We use Monte Carlo Dropout for uncertainty estimation:

1. **Training**: Model trained with `dropout=0.1`
2. **Inference**: Keep dropout active (`model.train()`)
3. **Sampling**: Run T forward passes (default T=30)
4. **Uncertainty**: Compute variance of predictions

```python
predictions = [model(x) for _ in range(T)]
uncertainty = variance(predictions)
```

This approach:
- Requires only a single model (no ensemble)
- Approximates Bayesian uncertainty
- Simple to implement with existing architecture

## Data Quality and Validation

### Automatic Data Quality Checks

The pipeline automatically checks for data integrity issues at training time:

1. **Duplicate Detection**: Checks for duplicate entries within training and validation sets
2. **Index Overlap**: Verifies no indices appear in both train and val sets
3. **Data Leakage**: Detects identical samples appearing in both train and val sets with different indices
4. **Contamination Prevention**: Ensures baseline model never samples from AL's growing dataset

All checks are logged as warnings if issues are detected.

### Automatic Parameter Validation

- **Candidate Pool Size**: If `n_select > n_candidates`, automatically increases `n_candidates = n_select`
- **Selection Logging**: Reports actual number of points selected vs requested
  - Example: `Selected 1000 most uncertain points (requested: 10000, available: 1000)`

## Usage

### Basic Usage

```bash
python active_learning.py
```

### Testing Mode

```bash
python active_learning.py --testing
```

Uses small data (30 samples, 10 epochs) for quick verification.

### Full Configuration

```bash
python active_learning.py \
    --n-iterations 5 \
    --n-candidates 10000 \
    --n-select 10 \
    --mc-samples 30 \
    --epochs 500 \
    --dropout 0.1 \
    --output-dir active_learning_output
```

### Closed-Loop with Model Generation

```bash
python active_learning.py \
    --n-iterations 5 \
    --n-select 10 \
    --generate-data \
    --output-dir active_learning_output
```

This will:
1. Train the model on initial data
2. Select uncertain points
3. Generate new models using Run3ModelGen (SPheno + micromegas)
4. Add generated data to training set
5. Repeat for N iterations

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--testing` | False | Testing mode with small data |
| `--n-iterations` | 1 | Number of AL iterations |
| `--n-candidates` | 1000 | Candidate pool size |
| `--n-select` | 10 | Points to select per iteration |
| `--mc-samples` | 30 | MC Dropout forward passes |
| `--epochs` | 100 | Training epochs per iteration |
| `--dropout` | 0.1 | Dropout rate for MC Dropout |
| `--n-datasets` | -1 | Number of ROOT files to load |
| `--n-samples` | None | Limit training samples |
| `--output-dir` | active_learning_output | Output directory |
| `--generate-data` | False | Generate new models using Run3ModelGen |

## Output Format

```
active_learning_output/
├── active_learning.log          # Main pipeline log
├── iteration_001/
│   ├── selected_points.csv      # Selected points with uncertainty scores
│   ├── model_checkpoint.pt      # AL model checkpoint
│   ├── al_training.log          # AL model training log
│   ├── baseline_training.log    # Baseline model training log
│   ├── plots/
│   │   ├── al/                  # Active Learning model diagnostics
│   │   │   ├── losses_transformer_tabular.png
│   │   │   ├── transformer_tabular_true_vs_pred_train.png
│   │   │   ├── transformer_tabular_true_vs_pred_validation.png
│   │   │   ├── transformer_tabular_hist_true_vs_pred_train.png
│   │   │   └── transformer_tabular_hist_true_vs_pred_validation.png
│   │   └── baseline/            # Baseline model diagnostics
│   │       ├── losses_transformer_tabular.png
│   │       ├── transformer_tabular_true_vs_pred_train.png
│   │       ├── transformer_tabular_true_vs_pred_validation.png
│   │       ├── transformer_tabular_hist_true_vs_pred_train.png
│   │       └── transformer_tabular_hist_true_vs_pred_validation.png
│   ├── modelgen_config.yaml     # [If --generate-data] Config for Run3ModelGen
│   └── scan/                    # [If --generate-data] Generated model outputs
│       ├── input/               # SLHA input files
│       ├── SPheno/              # SPheno outputs
│       ├── micromegas/          # micromegas outputs (Omega values)
│       └── ntuple.X.X.root      # Combined ROOT ntuple
├── iteration_002/
│   └── ...
├── plots/
│   ├── hist_dataset.png         # Initial dataset distribution
│   └── iteration_metrics.png    # Comparison plot (Loss, R², Dataset Size)
└── summary.json                 # Configuration and all results
```

### CSV Format (selected_points.csv)

```csv
meL,meR,mtauL,mtauR,mqL1,muR,mdR,mqL3,mtR,mbR,M_1,M_2,mu,M_3,At,Ab,Atau,mA,tanb,uncertainty
1234.5,567.8,2000.0,2000.0,4000.0,4000.0,4000.0,890.1,1234.5,678.9,500.0,-800.0,300.0,2500.0,5000.0,1000.0,-500.0,1500.0,25.3,0.0453
...
```

All parameter values are in **original (non-normalized) physical units**.

## pMSSM Parameter Ranges

| Parameter | Range | Description |
|-----------|-------|-------------|
| tanb | [1, 60] | tan(beta) |
| M_1 | [-2000, 2000] | Bino mass (GeV) |
| M_2 | [-2000, 2000] | Wino mass (GeV) |
| M_3 | [1000, 4000] | Gluino mass (GeV) |
| mu | [-2000, 2000] | Higgsino mass (GeV) |
| At | [-8000, 8000] | Top trilinear (GeV) |
| Ab | [-2000, 2000] | Bottom trilinear (GeV) |
| Atau | [-2000, 2000] | Tau trilinear (GeV) |
| mA | [0, 2000] | Pseudoscalar mass (GeV) |
| meL, meR | [0, 2000] | Selectron masses (GeV) |
| mtauL, mtauR | 2000 (fixed) | Stau masses (GeV) |
| mqL1, muR, mdR | 4000 (fixed) | 1st gen squark (GeV) |
| mqL3, mtR, mbR | [0, 2000] | 3rd gen squark (GeV) |

## Integration with Run3ModelGen

The `--generate-data` flag enables automatic model generation using Run3ModelGen:

```bash
python active_learning.py --n-iterations 3 --generate-data
```

This performs the following for each iteration:
1. Reads selected points from `selected_points.csv`
2. Creates a YAML config with `prior: fixed` mode
3. Calls `ModelGenerator` to run SPheno + micromegas
4. Loads the generated ROOT ntuple with new Omega values
5. Augments the training dataset for the next iteration

### Requirements

- Run3ModelGen submodule must be initialized: `git submodule update --init`
- Run3ModelGen must be built (includes SPheno and micromegas):
  ```bash
  cd Run3ModelGen
  pixi run build
  cd ..
  ```

### Manual Generation (Alternative)

If you prefer to run generation separately:

```bash
# After running active learning without --generate-data
cd active_learning_output/iteration_001

# The selected_points.csv can be used with Run3ModelGen
# Copy and modify modelgen_config.yaml if needed
```

## Implementation Details

### Parallel Training

Both AL and Baseline models train in parallel on separate GPUs (if available):
- **GPU 0**: Active Learning model
- **GPU 2**: Baseline model

Sequential training is automatically used if fewer than 2 GPUs are available.

### Data Contamination Prevention

The baseline model is carefully designed to avoid data contamination:

**Iteration 1**: Both models start with the same initial samples (e.g., first 1000 from input dataset)

**Iteration 2+**:
- AL dataset grows with generated data: `X_AL = [initial_samples, generated_samples]`
- Baseline samples from input excluding initial AL samples: `X_Baseline = [initial_samples, random_from_input[1000:]]`

This ensures:
- Fair comparison (both start with same data)
- No contamination (baseline never sees AL's generated data)
- Independent growth patterns (AL via uncertainty, baseline via random)

### Logging Structure

**Main Log** (`active_learning.log`):
- Pipeline configuration
- Iteration progress
- Point selection details
- Model generation status

**Per-Model Logs** (`al_training.log`, `baseline_training.log`):
- Training progress (epoch-by-epoch)
- Data quality checks (duplicates, leakage)
- Final metrics (loss, R² score)
- Diagnostic plot generation

### Metrics Tracking

The pipeline tracks and compares:
- **Train/Validation Loss**: Best MSE loss achieved
- **R² Score**: Coefficient of determination on validation set
- **Dataset Size**: Number of training and validation samples per iteration

Final comparison plot (`iteration_metrics.png`) shows:
1. **Train/Validation Loss**: MSE loss curves for both AL and baseline models
2. **R² Score**: Model performance across iterations
3. **Dataset Size**: Growth of training and validation sets for both models

This allows visual comparison of how active learning improves model performance compared to random sampling as the dataset grows.

## Recent Improvements

### February 2026 Updates

1. **Baseline Comparison**: Added parallel training of random baseline model for fair comparison with active learning
2. **Data Contamination Prevention**: Implemented strict separation between AL and baseline datasets
   - Both models start with identical initial data
   - Baseline samples from input dataset (excluding initial AL samples)
   - AL grows via generated data based on uncertainty
3. **Data Quality Checks**: Automatic detection of:
   - Duplicate entries in train/val sets
   - Index overlaps between train/val
   - Data leakage (identical samples in train and val)
4. **Improved Logging**:
   - Separate logs per model per iteration
   - Accurate reporting of selected vs available points
   - Clear contamination-free sampling messages
5. **Automatic Validation**: Auto-adjusts `n_candidates` if `n_select > n_candidates`
6. **Per-Model Diagnostics**: Separate diagnostic plots for AL and baseline models
7. **Dataset Size Tracking**: New plot showing the growth of training and validation sets across iterations for both AL and baseline models

## Troubleshooting

### Issue: "Selected N points but expected M"

**Symptom**: Log shows different number of selected points than requested
```
Selected 1000 most uncertain points (requested: 10000, available: 1000)
```

**Cause**: `n_select > n_candidates` - trying to select more points than available in candidate pool

**Solution**: The pipeline now automatically handles this by:
1. Auto-adjusting `n_candidates = n_select` if `n_select > n_candidates`
2. Logging the actual vs requested number of points

**Manual fix**: Increase `--n-candidates` to be at least equal to `--n-select`:
```bash
python active_learning.py --n-candidates 10000 --n-select 10000
```

### Issue: Data contamination warnings

**Symptom**: Warnings about duplicate entries or data leakage
```
Found 5 duplicate entries in training set!
Found 2 identical data samples appearing in both train and val sets!
```

**Cause**: Duplicate points in input dataset or incorrect sampling

**Impact**: May affect model performance and uncertainty estimates

**Action**: Review the input dataset for duplicates and verify the data loading pipeline

### Issue: Different train/val sizes between AL and Baseline

**Symptom**: AL and Baseline have different dataset sizes

**Cause**: Bug in split generation (fixed in recent updates)

**Solution**: Both models now use identical train/val splits for fair comparison

## Future Extensions

1. **Alternative Acquisition Functions**: Expected Improvement, entropy-based selection
2. **Ensemble Uncertainty**: Train multiple models for more robust uncertainty
3. **Region-Based Selection**: Focus on specific Omega_h^2 ranges of interest
4. **Parallel Model Generation**: Use Condor for distributed generation
5. **Adaptive Sampling**: Adjust candidate pool based on previous iterations
6. **Cross-validation**: K-fold validation for more robust uncertainty estimates
