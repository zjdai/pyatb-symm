# 2026-04-28 Cross-Structure Real-Space Covariance Validation Log

## Goal

Re-run the cross-structure HS covariance validation for:

- source: `test_workspace/test-abacus-2/soc`
- target: `test_workspace/test-abacus-4`

with the corrected rule:

- use the real-space rotation that maps the actual Cartesian structure,
- build orbital+spin `D` from that real-space rotation,
- then validate covariance on the same `R` blocks, without introducing an extra `R -> R'` remapping layer.

## Important Debugging Notes

### Wrong attempt 1: atoms-only Procrustes

- A first atoms-only rigid fit gave a very small atom-position error,
  but a very large lattice error.
- Root cause: Bi2Se3 itself has point-group symmetry, so atom coordinates alone admit a symmetry-related false rotation.
- This is not the rotation that maps the full structure orientation.

### Corrected approach

- Use a combined real-space fit:
  - lattice vectors
  - nonzero atomic Cartesian coordinates
- Solve the row-vector orthogonal Procrustes problem
  - `cart_source @ R_row ~= cart_target`
- Convert to the Cartesian rotation convention expected by the D-matrix code:
  - `R_col = R_row^T`

This is the rotation used to build the orbital+spin local representation matrices.

## Validation Output

Report file:

- `/home/zjdai/file-test/pyatb_symm/test_workspace/test-abacus-4/test2_to_test4_realspace_rotation_covariance_report.json`

## Geometric Rotation Quality

The fitted real-space rotation used for `D` gives:

- `det = 0.9999999999999999`
- `max_abs_lattice_error = 9.351880272845392e-05`
- `fro_lattice_error = 1.7146014411770493e-04`
- `max_abs_atom_error = 2.0869794914801787e-04`
- `max_atom_error_norm = 2.2817837937748366e-04`

So the rotation is geometrically consistent with both the lattice and the wrapped atomic structure.

## HS Covariance Results

### Source -> Target

Validation formula:

```text
H_target(R) ?= D(R_cart) H_source(R) D(R_cart)^dagger
S_target(R) ?= D(R_cart) S_source(R) D(R_cart)^dagger
```

Results:

- `HR rel_fro_global = 1.0759581974675406e-05`
- `HR max_abs_diff = 1.819333167396575e-05`
- `SR rel_fro_global = 1.1825975213669202e-05`
- `SR max_abs_diff = 4.597218759620075e-06`
- `common_R_count = 165`
- `only_reference_R_count = 0`
- `only_predicted_R_count = 0`

Worst HR element:

- `R = [0, 0, 0]`
- `(row, col) = (74, 74)`
- `abs_diff = 1.819333167396575e-05`

Worst SR element:

- `R = [0, 0, -1]`
- `(row, col) = (11, 155)`
- `abs_diff = 4.597218759620075e-06`

### Target -> Source

Validation formula:

```text
H_source(R) ?= D(R_cart)^dagger H_target(R) D(R_cart)
S_source(R) ?= D(R_cart)^dagger S_target(R) D(R_cart)
```

Results:

- `HR rel_fro_global = 1.0759616399590502e-05`
- `HR max_abs_diff = 1.8220270389002202e-05`
- `SR rel_fro_global = 1.1826031612177153e-05`
- `SR max_abs_diff = 4.188333590348509e-06`
- `common_R_count = 165`
- `only_reference_R_count = 0`
- `only_predicted_R_count = 0`

The reverse-direction result is numerically consistent with the forward-direction result.

## Wrong-Rotation Control

As a control, re-running the same `source -> target` check with the transposed/wrong rotation gives much worse results:

- `HR rel_fro_global = 0.2810372368264005`
- `HR max_abs_diff = 0.30521995408348407`
- `SR rel_fro_global = 0.6279282790459534`
- `SR max_abs_diff = 0.35113247180691337`

This confirms that the earlier large mismatch was indeed tied to using the wrong Cartesian rotation in the D-matrix construction.

## Conclusion

- The cross-structure validation was re-run with the corrected real-space rotation logic.
- The orbital+spin `D` matrix must be built from the true Cartesian rotation of the structure, not from an incorrectly interpreted lattice-basis transform.
- Under the corrected construction, the `test-abacus-2/soc -> test-abacus-4` HS covariance check improves from order `1e-1` / `1e0` mismatch to order `1e-5`.

## 2026-04-28 Re-validation after main-flow full-completion integration (2 -> 4, with rotation)

To align with the main flow that now completes each `R` block before comparison, an additional re-validation was performed:

- Source structure: `test_workspace/test-abacus-4/verification_tmp/STRU_S2_abs`
  - same 5-atom primitive geometry as `test-abacus-2/soc`, but with absolute orbital paths for direct orbital metadata loading.
- Target structure: `test_workspace/test-abacus-4/verification_tmp/STRU_S1_abs`
  - same orientation as `test-abacus-4`, also with absolute orbital paths.
- HS data remained:
  - source: `test-abacus-2/soc/OUT.ABACUS/data-HR/SR-sparse_SPIN0.csr`
  - target: `test-abacus-4/OUT.ABACUS/data-HR/SR-sparse_SPIN0.csr`

Output file:

- `test_workspace/test-abacus-4/test2_to_test4_realspace_rotation_covariance_report_rerun_20260428_full_matrix_integrated.json`

Results:

- `full_matrix_from_hermitian = true`
- `common_R_count = 165`
- `atom_mapping_max_error = 4.217539478323928e-06`
- `HR rel_fro_global = 1.0759581974911338e-05`
- `SR rel_fro_global = 1.1825975213684855e-05`
- `HR max_abs_diff = 1.8193331674631885e-05`
- `SR max_abs_diff = 4.597218761509189e-06`

Conclusion: after integrating per-`R` full completion into the main pipeline, the `test-abacus-2 -> test-abacus-4` rotation covariance remains in the `1e-5` range and passes.

## 2026-04-28 Main-flow fix and re-validation on `test-abacus-3/pyatb` (full-completion + rotation)

### Root cause

1. `symm_stru.py::_build_atom_mapping()` wrapped `old_pos_in_std_basis` too early, which erased integer supercell-to-primitive shifts (`shift -> [0,0,0]`).  
2. `hs_standardize.py::_assemble_target_dense_blocks()` double-counted rotated contributions when `full_matrix_from_hermitian=True` and `R/-R` partners both existed (effective factor ~2).

### Code changes

- `pyatb-main/src/pyatb/symmetry/symm_stru.py`
  - In `_build_atom_mapping()`, keep unwrapped `old_pos_in_std_basis = cart_old @ inv_std_lattice`; only apply periodic reduction in residual checks.
- `pyatb-main/src/pyatb/symmetry/hs_standardize.py`
  - In `_assemble_target_dense_blocks()`:
    - separate original upper-triangular source blocks from full-reconstructed blocks;
    - when `full_matrix_from_hermitian=True` and `-R` exists, apply `0.5` pair-weight to that `R` contribution to remove duplicate accumulation.
- `pyatb-main/tests/test_character_module.py`
  - Updated supercell->primitive mapping expectation: the second atom shift is now `[1,0,0]` (instead of `[0,0,0]`).

### Regression tests

Command:

```bash
cd /home/zjdai/file-test/pyatb_symm/pyatb-main
pytest -q tests/test_hs_standardize.py tests/test_hs_covariance.py tests/test_character_module.py
```

Result: `40 passed, 4 warnings`.

### Re-run on `test-abacus-3/pyatb`

Command:

```bash
cd /home/zjdai/file-test/pyatb_symm/test_workspace/test-abacus-3/pyatb
pyatb > rerun_after_pair_weight_fix_20260428.log 2>&1
```

`Out/CHARACTER/standardization_summary.json` now shows non-zero integer shifts in `atom_mapping` (e.g. `[-1,-1,0]`, `[-2,-1,-1]`, etc.).

### HS covariance re-check (`HS-symm` vs `test-abacus-2/soc` reference)

Output file:

- `test_workspace/test-abacus-3/pyatb/integrated_covariance_3pyatb_to_2soc_full_report_after_pair_weight_fix_20260428.json`

Key metrics:

- `HR rel_fro_diff = 8.923393464487352e-05`
- `SR rel_fro_diff = 1.1664759139084118e-05`
- `HR max_abs_diff = 0.0053413193317481955`
- `SR max_abs_diff = 4.131167610664163e-06`
- `target_missing_count = 0` (HR/SR)

Conclusion: after the main-flow fixes, `test-abacus-3/pyatb` standardized structure and `HS-symm` return to `1e-4 ~ 1e-5` covariance-error level and pass.
