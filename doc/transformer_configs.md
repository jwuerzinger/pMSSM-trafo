# Transformer Configuration Recommendations for pMSSM

## Key Changes Made to PMSSMTransformer

1. **Improved initialization**: Scaled positional encodings and CLS token (0.02x)
2. **Pre-normalization**: Using `norm_first=True` for better gradient flow
3. **Richer input embedding**: Added LayerNorm + ReLU after initial embedding
4. **Deeper regression head**: 3-layer MLP instead of single layer
5. **Removed dropout by default**: For small datasets, dropout=0.0 works better

## Recommended Configurations to Test

### Configuration 1: Matching MLP Capacity (Recommended)
```python
model = pmssm.PMSSMTransformer(
    n_params=19,
    d_model=128,           # Increased from 16
    nhead=4,               # Increased from 1
    num_layers=3,          # Increased from 2
    dim_feedforward=512,   # Increased from 64
    dropout=0.0,           # Keep at 0 for small datasets
    use_prenorm=True,      # Pre-normalization
)

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
# Note: Slightly higher LR and lower weight decay
```

### Configuration 2: High Capacity
```python
model = pmssm.PMSSMTransformer(
    d_model=256,
    nhead=8,
    num_layers=4,
    dim_feedforward=1024,
    dropout=0.0,
)

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
```

### Configuration 3: Tabular-Specific Architecture (Best for this task)
```python
# New architecture specifically designed for tabular data
model = pmssm.PMSSMTransformerTabular(
    n_params=19,
    d_model=128,
    nhead=4,
    num_layers=3,
    dim_feedforward=512,
    dropout=0.0,
)

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
```

## Additional Training Improvements

### 1. Learning Rate Scheduling
```python
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# Option A: Cosine annealing
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

# Option B: Reduce on plateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

# Add to training loop:
# scheduler.step()  # For cosine
# scheduler.step(val_loss)  # For plateau
```

### 2. Gradient Clipping
```python
# In training loop, after loss.backward():
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### 3. Warmup
```python
def get_lr_with_warmup(epoch, warmup_epochs=100, base_lr=3e-4, min_lr=1e-6):
    if epoch < warmup_epochs:
        return min_lr + (base_lr - min_lr) * epoch / warmup_epochs
    return base_lr

# In training loop:
for epoch in range(epochs):
    lr = get_lr_with_warmup(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

## Why Transformers Struggle with Tabular Data

1. **No sequence structure**: pMSSM parameters aren't a sequence - there's no inherent order
2. **Heterogeneous features**: Each feature represents a different physical quantity (masses, angles, etc.)
3. **Small feature count**: 19 features is too few to benefit from self-attention
4. **Data efficiency**: Transformers need more data than MLPs to learn effectively

## Expected Results

- **Original transformer**: Val MSE ~ 0.15-0.30
- **Improved PMSSMTransformer (Config 1)**: Val MSE ~ 0.05-0.10
- **PMSSMTransformerTabular**: Val MSE ~ 0.03-0.08
- **MLP baseline**: Val MSE ~ 0.01-0.03

The MLP will likely still outperform because:
1. It's the right architecture for this type of data
2. It directly models feature interactions via learned weights
3. More parameter-efficient for tabular data

## Alternative: Hybrid Approach

If you want to keep exploring transformers, consider a hybrid:

```python
class PMSSMHybrid(nn.Module):
    def __init__(self, n_params=19, d_model=128):
        super().__init__()

        # Initial MLP processing
        self.input_proj = nn.Sequential(
            nn.Linear(n_params, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model * n_params),
        )

        # Reshape and apply transformer
        self.transformer = PMSSMTransformer(...)

        # Or skip transformer and go directly to output
        self.output = nn.Linear(d_model * n_params, 1)
```

This processes the full input vector first before treating elements as tokens.
