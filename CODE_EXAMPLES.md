# Code Examples: SH Optimizer State and Consistency Checks

## Example 1: Understanding _extend_sh_optimizer_state_for_densification

### Scenario
We have 3 gaussians with SH degree 1 (2 coefficients each), and we split the first gaussian (index 0) into 2 new gaussians.

### Before Split
```
gaussians: [g0, g1, g2]
sh_degrees: [1, 1, 1]  (2 coeffs each)
sh_coeffs_flat: [c0_0, c0_1, c1_0, c1_1, c2_0, c2_1]  (6 total)
gauss_offsets: [0, 2, 4]
num_coeffs_per_gauss: [2, 2, 2]

optimizer.state['sh_coeffs']['exp_avg']:
  [m0_0, m0_1, m1_0, m1_1, m2_0, m2_1]
```

### After Split (before optimizer update)
```
gaussians: [g0, g1, g2, g0_new1, g0_new2]  (g0 is then pruned)
sh_degrees: [1, 1, 1, 1, 1]
sh_coeffs_flat: [c0_0, c0_1, c1_0, c1_1, c2_0, c2_1, c0new1_0, c0new1_1, c0new2_0, c0new2_1]
gauss_offsets: [0, 2, 4, 6, 8]
num_coeffs_per_gauss: [2, 2, 2, 2, 2]

optimizer.state['sh_coeffs']['exp_avg'] (uninitialized new parts):
  [m0_0, m0_1, m1_0, m1_1, m2_0, m2_1, 0, 0, 0, 0]
```

### After _extend_sh_optimizer_state_for_densification([0, 0])
The method copies moments from g0 to both new gaussians:
```
optimizer.state['sh_coeffs']['exp_avg']:
  [m0_0, m0_1, m1_0, m1_1, m2_0, m2_1, m0_0, m0_1, m0_0, m0_1]
                                        ↑     ↑     ↑     ↑
                                     copied from g0 to g0_new1 and g0_new2
```

This ensures both new gaussians inherit the training history of their parent.

## Example 2: Consistency Check in Action

### Healthy State
```python
# After training_setup
✓ Gaussian consistency check OK [after training_setup]: 50000 gaussians

# After densify_and_clone (cloning 1234 gaussians)
[densify_and_clone] Cloning 1234 gaussians
✓ Gaussian consistency check OK [after densify_and_clone]: 51234 gaussians

# After densify_and_split (splitting 567 gaussians into 2x = 1134 new)
[densify_and_split] Splitting 567 gaussians (x2)
✓ Gaussian consistency check OK [after densify_and_split]: 52368 gaussians

# After densify_and_prune (pruning 345 gaussians)
[prune_points] Before prune: checking consistency...
✓ Gaussian consistency check OK [before prune_points]: 52368 gaussians
[prune_points] Pruning 345 gaussians, keeping 52023
[prune_points] After prune: checking consistency...
✓ Gaussian consistency check OK [after prune_points]: 52023 gaussians
```

### Detecting a Bug
If there's a bug in the split code that forgets to extend opacity:
```
[densify_and_split] Splitting 567 gaussians (x2)
ERROR: Gaussian consistency check FAILED [after densify_and_split]:
  xyz(51368) != opacity(50801)
  xyz(51368) != sh_degrees(51368)  [OK]
  expected_coeffs(103072) != actual_coeffs(103072)  [OK]
```

This immediately tells us that opacity wasn't properly extended during the split.

## Example 3: Tracking SH Degree Changes During Split

If split creates gaussians with different SH degrees than source:

### Before Split
```
Source gaussian (g0): SH degree 2 (9 coefficients)
  sh_coeffs_flat offset 0-8: [c0_0, c0_1, ..., c0_8]
  exp_avg offset 0-8: [m0_0, m0_1, ..., m0_8]
```

### After Split (if split creates degree 1 gaussians - 4 coefficients)
```
New gaussian (g0_new): SH degree 1 (4 coefficients)
  sh_coeffs_flat offset 100-103: [?, ?, ?, ?]
  exp_avg offset 100-103: [m0_0, m0_1, m0_2, m0_3]  (copied from g0)
                          [m0_4, ..., m0_8]  (not copied, zeroed)
```

The method intelligently copies min(9, 4) = 4 moments, and zeros the rest.

## Example 4: Debugging the "More Gaussians Created" Issue

If you're seeing more gaussians in this repo than in the original despite constant SH degree 3:

1. **Enable verbose logging** - All densify methods now print their actions
2. **Watch the split/clone counts** - Is split creating more than expected?
3. **Check the prune rate** - Is prune removing fewer than expected?
4. **Look for consistency errors** - Do counts mismatch suggest data corruption?

### Sample Investigation Output
```
Iteration 1000:
[densify_and_clone] Cloning 100 gaussians
✓ Gaussian consistency check OK [after densify_and_clone]: 5100 gaussians

[densify_and_split] Splitting 200 gaussians (x2)
✓ Gaussian consistency check OK [after densify_and_split]: 5500 gaussians

[prune_points] Pruning 50 gaussians, keeping 5450
✓ Gaussian consistency check OK [after prune_points]: 5450 gaussians

Iteration 2000:
[densify_and_clone] Cloning 250 gaussians  # higher clone rate?
[densify_and_split] Splitting 350 gaussians (x2)  # more splits?
[prune_points] Pruning 75 gaussians  # lower prune rate?
```

If you see consistently higher clone/split rates or lower prune rates, 
that explains the extra gaussians.

## Example 5: Manual Consistency Check

You can also call the validation method manually during training:

```python
# In your training loop
if iteration % 1000 == 0:
    # Manually check consistency
    gaussian_model._validate_gaussian_consistency(f"[iter {iteration}]")
    
    # Or log gaussian counts
    print(f"Iteration {iteration}: {gaussian_model._xyz.shape[0]} gaussians")
    print(f"  - sh_degrees: {gaussian_model.sh_storage.num_gaussians}")
    print(f"  - sh_coeffs_flat: {gaussian_model.sh_storage.sh_coeffs_flat.shape[0]} total coeffs")
```

This gives you fine-grained control over when validation happens.
