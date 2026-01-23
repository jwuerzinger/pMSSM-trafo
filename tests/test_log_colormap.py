"""
Test script to verify LogNorm import and usage.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Generate sample data
np.random.seed(42)
x = np.random.randn(1000)
y = x + np.random.randn(1000) * 0.5

print("="*60)
print("Log Colormap Test")
print("="*60)
print("\nTesting LogNorm with hist2d...")

# Create histogram with log scale
plt.figure(figsize=(8, 6))
plt.hist2d(x, y, bins=30, cmap="inferno", norm=LogNorm())
plt.colorbar(label="Counts (log scale)")
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.title("Test 2D Histogram with Log Colormap")

# Save test plot
plt.savefig("test_log_colormap.png", dpi=100)
print("✓ LogNorm imported successfully")
print("✓ hist2d with LogNorm created successfully")
print("✓ Test plot saved to: test_log_colormap.png")
print("\nYou can delete test_log_colormap.png after verifying it looks correct.")
print("="*60)
