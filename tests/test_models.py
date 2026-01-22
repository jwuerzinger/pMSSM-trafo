"""
Quick test to verify all model architectures work correctly.
Run this before the full training to catch any errors early.
"""
import warnings
# Suppress nested tensor warning (expected with norm_first=True)
warnings.filterwarnings('ignore', message='.*enable_nested_tensor.*')

import pmssm
import torch

print("Testing model instantiation...")

# Test 1: Improved PMSSMTransformer
print("\n1. Testing PMSSMTransformer...")
model1 = pmssm.PMSSMTransformer(
    d_model=128,
    nhead=4,
    num_layers=3,
    dim_feedforward=512,
    dropout=0.0,
    use_prenorm=True,
)
x_test = torch.randn(4, 19)  # batch of 4, 19 features
y_pred = model1(x_test)
assert y_pred.shape == (4, 1), f"Expected shape (4, 1), got {y_pred.shape}"
print(f"   ✓ PMSSMTransformer works! Output shape: {y_pred.shape}")
print(f"   ✓ Parameters: {sum(p.numel() for p in model1.parameters()):,}")

# Test 2: PMSSMTransformerTabular
print("\n2. Testing PMSSMTransformerTabular...")
model2 = pmssm.PMSSMTransformerTabular(
    d_model=128,
    nhead=4,
    num_layers=3,
    dim_feedforward=512,
    dropout=0.0,
)
y_pred = model2(x_test)
assert y_pred.shape == (4, 1), f"Expected shape (4, 1), got {y_pred.shape}"
print(f"   ✓ PMSSMTransformerTabular works! Output shape: {y_pred.shape}")
print(f"   ✓ Parameters: {sum(p.numel() for p in model2.parameters()):,}")

# Test 3: PMSSMFeedForward (baseline)
print("\n3. Testing PMSSMFeedForward...")
model3 = pmssm.PMSSMFeedForward(
    d_model=64,
    num_layers=4,
    dim_feedforward=512,
)
y_pred = model3(x_test)
assert y_pred.shape == (4, 1), f"Expected shape (4, 1), got {y_pred.shape}"
print(f"   ✓ PMSSMFeedForward works! Output shape: {y_pred.shape}")
print(f"   ✓ Parameters: {sum(p.numel() for p in model3.parameters()):,}")

# Test 4: Backward pass
print("\n4. Testing backward pass...")
optimizer = torch.optim.AdamW(model1.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

y_true = torch.randn(4, 1)
y_pred = model1(x_test)
loss = criterion(y_pred, y_true)
loss.backward()

# Test gradient clipping
torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm=1.0)
optimizer.step()
print(f"   ✓ Backward pass and gradient clipping work!")
print(f"   ✓ Loss: {loss.item():.6f}")

print("\n" + "="*60)
print("All tests passed! ✓")
print("You can now run: python train_pmssm.py")
print("="*60)
