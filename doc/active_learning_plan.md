# Active Learning Pipeline for pMSSM

## Overview

This document describes the active learning pipeline implemented in `active_learning.py` for iteratively improving the pMSSM relic density prediction model by selecting the most informative training points.

## Concept

Active learning is a machine learning paradigm where the model actively selects which data points to learn from. For expensive physics simulations like pMSSM, this is valuable because:

1. **Simulation Cost**: Each pMSSM point requires running SPheno + micromegas, which is computationally expensive
2. **High-Dimensional Space**: The 19-dimensional parameter space cannot be densely sampled
3. **Efficient Learning**: By selecting points where the model is most uncertain, we maximize information gain per simulation

## Algorithm

```
for iteration in 1..N:
    1. Train PMSSMTransformerTabular on current training data
    2. Generate pool of candidate points in parameter space
    3. Compute predictive uncertainty via MC Dropout
    4. Select top-K points with highest uncertainty
    5. Output points for generation (CSV format)
    6. [If --generate-data] Generate new models via Run3ModelGen
    7. [If --generate-data] Add new data to training set for next iteration
```

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
├── iteration_001/
│   ├── selected_points.csv      # 10 points with all parameters
│   ├── model_checkpoint.pt      # Trained model weights
│   ├── plots/                   # Diagnostic plots for this iteration
│   │   ├── losses_transformer_tabular.png
│   │   ├── transformer_tabular_true_vs_pred_train.png
│   │   ├── transformer_tabular_true_vs_pred_validation.png
│   │   ├── transformer_tabular_hist_true_vs_pred_train.png
│   │   └── transformer_tabular_hist_true_vs_pred_validation.png
│   ├── modelgen_config.yaml     # [If --generate-data] Config for Run3ModelGen
│   └── scan/                    # [If --generate-data] Generated model outputs
│       ├── input/               # SLHA input files
│       ├── SPheno/              # SPheno outputs
│       ├── micromegas/          # micromegas outputs (Omega values)
│       └── ntuple.X.X.root      # Combined ROOT ntuple
├── iteration_002/
│   └── ...
├── plots/
│   └── hist_dataset.png
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

## Future Extensions

1. **Alternative Acquisition Functions**: Expected Improvement, entropy-based selection
2. **Ensemble Uncertainty**: Train multiple models for more robust uncertainty
3. **Region-Based Selection**: Focus on specific Omega_h^2 ranges of interest
4. **Parallel Model Generation**: Use Condor for distributed generation
5. **Adaptive Sampling**: Adjust candidate pool based on previous iterations
