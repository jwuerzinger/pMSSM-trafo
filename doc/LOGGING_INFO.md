# Logging Configuration

All training output is now logged using `structlog` and written to both the console and a log file.

## Features

1. **Dual Output**: All messages appear on console AND in a log file
2. **Timestamped Logs**: Log files are named with timestamps: `logs/training_YYYYMMDD_HHMMSS.log`
3. **Persistent Records**: Review past training runs anytime by checking the `logs/` directory
4. **Structured Logging**: Using `structlog` for better log management

## Log Contents

The log files contain:
- Training start/end timestamps
- GPU information
- Data loading details (n_datasets, n_samples)
- Training progress (every 100 epochs)
- Model training metrics (Train MSE, Val MSE, Learning Rate)
- Summary information

## Installation

Make sure `structlog` is installed. It's already added to `pixi.toml`:

```bash
# If using pixi
pixi install

# Or manually with pip
pip install structlog
```

## Example Log Output

The output is now in clean, human-readable format:

```
2026-01-22 15:30:00 [info     ] ============================================================
2026-01-22 15:30:00 [info     ] Training started at 2026-01-22 15:30:00
2026-01-22 15:30:00 [info     ] Log file: logs/training_20260122_153000.log
2026-01-22 15:30:00 [info     ] ============================================================
2026-01-22 15:30:01 [info     ] Set device to: cuda
2026-01-22 15:30:01 [info     ] Using GPU: NVIDIA H100
2026-01-22 15:30:02 [info     ] Loading data: n_datasets=-1, n_samples=all
2026-01-22 15:30:10 [info     ]
============================================================
2026-01-22 15:30:10 [info     ] Training Improved PMSSMTransformer
2026-01-22 15:30:10 [info     ] ============================================================
2026-01-22 15:31:00 [info     ] Epoch 000 | Train MSE = 0.245123 | Val MSE = 0.251234 | LR = 3.000000e-04
2026-01-22 15:35:00 [info     ] Epoch 100 | Train MSE = 0.123456 | Val MSE = 0.129876 | LR = 2.987654e-04
...
```

Each log entry includes:
- **Timestamp** (YYYY-MM-DD HH:MM:SS format)
- **Log level** (`[info]`, `[error]`, etc.)
- **Message content**

## Testing

Run the test script to verify logging works:

```bash
python test_logging.py
```

This will create a test log file and verify that output appears in both console and file.
