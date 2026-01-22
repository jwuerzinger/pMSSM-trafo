"""
Test script to verify command-line option logic works correctly.
"""

def test_option_logic(testing, n_datasets_opt, n_samples_opt):
    """Simulate the logic from train_pmssm.py"""
    # Determine n_datasets: explicit option > testing mode > full data
    if n_datasets_opt is None:
        n_datasets = 3 if testing else -1
    else:
        n_datasets = n_datasets_opt

    # Determine n_samples: explicit option > testing mode > all samples
    if n_samples_opt is None:
        n_samples = 30 if testing else None
    else:
        n_samples = n_samples_opt

    return n_datasets, n_samples

print("Testing command-line option logic:")
print("="*60)

# Test case 1: Default (no flags)
n_d, n_s = test_option_logic(testing=False, n_datasets_opt=None, n_samples_opt=None)
print(f"1. Default: n_datasets={n_d}, n_samples={n_s}")
assert n_d == -1 and n_s is None, "Failed: default should use all data"

# Test case 2: --testing flag
n_d, n_s = test_option_logic(testing=True, n_datasets_opt=None, n_samples_opt=None)
print(f"2. --testing: n_datasets={n_d}, n_samples={n_s}")
assert n_d == 3 and n_s == 30, "Failed: testing should use 3 datasets, 30 samples"

# Test case 3: --n-datasets 5 (overrides testing)
n_d, n_s = test_option_logic(testing=True, n_datasets_opt=5, n_samples_opt=None)
print(f"3. --testing --n-datasets 5: n_datasets={n_d}, n_samples={n_s}")
assert n_d == 5 and n_s == 30, "Failed: explicit n_datasets should override testing"

# Test case 4: --n-samples 100 (overrides testing)
n_d, n_s = test_option_logic(testing=True, n_datasets_opt=None, n_samples_opt=100)
print(f"4. --testing --n-samples 100: n_datasets={n_d}, n_samples={n_s}")
assert n_d == 3 and n_s == 100, "Failed: explicit n_samples should override testing"

# Test case 5: Both explicit options
n_d, n_s = test_option_logic(testing=False, n_datasets_opt=10, n_samples_opt=500)
print(f"5. --n-datasets 10 --n-samples 500: n_datasets={n_d}, n_samples={n_s}")
assert n_d == 10 and n_s == 500, "Failed: both explicit options should be used"

# Test case 6: Explicit options override testing
n_d, n_s = test_option_logic(testing=True, n_datasets_opt=10, n_samples_opt=500)
print(f"6. --testing --n-datasets 10 --n-samples 500: n_datasets={n_d}, n_samples={n_s}")
assert n_d == 10 and n_s == 500, "Failed: explicit options should override testing"

# Test case 7: Use all datasets but limited samples
n_d, n_s = test_option_logic(testing=False, n_datasets_opt=-1, n_samples_opt=1000)
print(f"7. --n-datasets -1 --n-samples 1000: n_datasets={n_d}, n_samples={n_s}")
assert n_d == -1 and n_s == 1000, "Failed: should use all datasets with 1000 samples"

print("="*60)
print("✓ All option logic tests passed!")
print("\nUsage examples:")
print("  python train_pmssm.py                                    # Full training")
print("  python train_pmssm.py --testing                          # Quick test (3 datasets, 30 samples)")
print("  python train_pmssm.py --n-datasets 5                     # 5 datasets, all samples")
print("  python train_pmssm.py --n-samples 1000                   # All datasets, 1000 samples each")
print("  python train_pmssm.py --n-datasets 5 --n-samples 1000   # 5 datasets, 1000 samples each")
