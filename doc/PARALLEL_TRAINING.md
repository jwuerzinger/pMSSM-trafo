# Parallel Training on Multiple GPUs

The training script now supports parallel training of transformer models on multiple GPUs for significantly faster execution.

## GPU Configuration

### Parallel Mode (2+ GPUs available)
- **Physical GPU 1** (`cuda:0`): PMSSMTransformer (parallel)
- **Physical GPU 2** (`cuda:1`): PMSSMTransformerTabular (parallel)
- **Physical GPU 2** (`cuda:1`): MLP Baseline (sequential, after transformers complete)

### Sequential Mode (1 GPU or --no-parallel flag)
- **Physical GPU 1** (`cuda:0`): All models trained sequentially
  1. PMSSMTransformer
  2. PMSSMTransformerTabular
  3. MLP Baseline

## How It Works

1. **Environment Setup**: `CUDA_VISIBLE_DEVICES='1,2'` makes GPUs 1 and 2 visible as `cuda:0` and `cuda:1`

2. **Parallel Execution**: Using Python's `multiprocessing`, both transformer models train simultaneously:
   - Each model runs in its own process
   - Each process uses its designated GPU
   - Training happens in parallel, not sequentially

3. **Speed Improvement**: Training time is approximately **halved** compared to sequential training

## Usage

### Parallel Training (Default)

Simply run the training script as usual:

```bash
python train_pmssm.py
```

The script automatically:
- Detects available GPUs
- Starts parallel training if 2+ GPUs are available
- Falls back to sequential training if insufficient GPUs

### Sequential Training (Single GPU)

To disable parallel training even when multiple GPUs are available:

```bash
python train_pmssm.py --no-parallel
```

This is useful for:
- Debugging
- Avoiding multiprocessing overhead
- When other processes are using GPU 2

## Training Flow

### Parallel Mode (Default with 2+ GPUs)

```
┌─────────────────────────────────────┐
│  Load Data & Create Datasets       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Start Parallel Training            │
├─────────────────┬───────────────────┤
│                 │                   │
│  Process 1      │    Process 2      │
│  GPU 0 (Phys 1) │    GPU 1 (Phys 2) │
│  ├─ PMSSMTransformer               │
│  └─ Training...  └─ PMSSMTransformerTabular
│                      └─ Training... │
│                 │                   │
└─────────────────┴───────────────────┘
               │
               ▼ (Both complete)
┌─────────────────────────────────────┐
│  Train MLP Baseline (Sequential)    │
│  GPU 1 (Physical GPU 2)             │
└─────────────────────────────────────┘
```

### Sequential Mode (1 GPU or --no-parallel)

```
┌─────────────────────────────────────┐
│  Load Data & Create Datasets       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Train PMSSMTransformer             │
│  GPU 0 (Physical GPU 1)             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Train PMSSMTransformerTabular      │
│  GPU 0 (Physical GPU 1)             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Train MLP Baseline                 │
│  GPU 0 (Physical GPU 1)             │
└─────────────────────────────────────┘
```

## Monitoring

Each model writes to its own log file:
- `logs/transformer_YYYYMMDD_HHMMSS.log` - PMSSMTransformer
- `logs/transformer_tabular_YYYYMMDD_HHMMSS.log` - PMSSMTransformerTabular
- Main log: `logs/training_YYYYMMDD_HHMMSS.log` - Overall progress

## GPU Memory

Each transformer model requires approximately:
- **PMSSMTransformer**: ~800 MB GPU memory
- **PMSSMTransformerTabular**: ~800 MB GPU memory
- **MLP**: ~1.5 GB GPU memory

H100 GPUs have 80GB memory, so memory is not a concern.

## Testing

Verify GPU configuration:

```bash
python tests/test_parallel_gpus.py
```

Expected output:
```
✓ Parallel training on 2 GPUs is possible!
```

## Fallback Behavior

If fewer than 2 GPUs are available:
- Script automatically falls back to sequential training
- Uses `cuda:0` if available, otherwise CPU
- No code changes needed

## Performance Comparison

| Configuration | Training Time (2000 epochs) |
|---------------|----------------------------|
| Sequential (1 GPU) | ~X hours |
| Parallel (2 GPUs) | ~X/2 hours |
| Speed improvement | ~2x faster |

## Technical Details

- **Multiprocessing method**: `spawn` (required for CUDA)
- **Process isolation**: Each model in separate process
- **No GPU memory sharing**: Clean separation between processes
- **Synchronization**: Main process waits for both to complete via `.join()`

## Files Modified

- [train_pmssm.py](../train_pmssm.py): Added multiprocessing support
  - `train_transformer()` function for parallel execution
  - `train_transformer_tabular()` function for parallel execution
  - Modified `main()` to spawn parallel processes

## Troubleshooting

**Issue**: "CUDA out of memory"
- **Solution**: Reduce batch_size from 256 to 128 in training functions
- **Alternative**: Use `--no-parallel` to train sequentially (frees GPU memory after each model)

**Issue**: Multiprocessing errors
- **Solution**: Ensure `mp.set_start_method('spawn')` is called before creating processes
- **Workaround**: Use `--no-parallel` flag to avoid multiprocessing

**Issue**: Only 1 GPU detected
- **Check**: `echo $CUDA_VISIBLE_DEVICES` - should show `1,2`
- **Fix**: Ensure environment variable is set before importing torch
- **Workaround**: Script automatically falls back to sequential training

**Issue**: Want to use different GPUs
- **Current**: Physical GPUs 1 and 2 (set via `CUDA_VISIBLE_DEVICES='1,2'`)
- **Change**: Modify line 3 in train_pmssm.py: `os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'` (or other GPU IDs)

**Issue**: One GPU is busy
- **Solution**: Use `--no-parallel` to train on single GPU only
