"""
Quick test to verify get_model_name function works correctly.
"""
import pmssm

print("Testing get_model_name function...\n")

# Test 1: PMSSMTransformer
model1 = pmssm.PMSSMTransformer()
name1 = pmssm.get_model_name(model1)
print(f"PMSSMTransformer -> '{name1}'")
assert name1 == "transformer", f"Expected 'transformer', got '{name1}'"

# Test 2: PMSSMTransformerTabular
model2 = pmssm.PMSSMTransformerTabular()
name2 = pmssm.get_model_name(model2)
print(f"PMSSMTransformerTabular -> '{name2}'")
assert name2 == "transformer_tabular", f"Expected 'transformer_tabular', got '{name2}'"

# Test 3: PMSSMFeedForward
model3 = pmssm.PMSSMFeedForward()
name3 = pmssm.get_model_name(model3)
print(f"PMSSMFeedForward -> '{name3}'")
assert name3 == "MLP", f"Expected 'MLP', got '{name3}'"

print("\n✓ All model names correct!")
print("\nExpected plot filenames:")
print(f"  - {name1}_losses.png, {name1}_true_vs_pred_train.png, etc.")
print(f"  - {name2}_losses.png, {name2}_true_vs_pred_train.png, etc.")
print(f"  - {name3}_losses.png, {name3}_true_vs_pred_train.png, etc.")
