# 2026-04-28 Dk Boundary / HS Validation Log

## Scope

- Merge ABACUS boundary-point canonicalization behavior into the repository-side symmetry mapping path.
- Unify local orbital+spin rotation construction so `build_dk_matrix()` and `hs_standardize.py` use the same D-matrix convention.
- Re-run unit tests and real-data HS self-symmetry validation.

## Code Changes

- `pyatb-main/src/pyatb/symmetry/Dk_matrix.py`
  - Added `_canonicalize_fractional_coordinates()`.
  - Canonicalized `extract_abacus_basis_metadata()` fractional positions into `[0, 1)`.
  - Canonicalized `find_atom_mapping()` input atom positions before matching.
  - Added shared `build_atom_local_rotation()` for orbital+spin local representation blocks.
  - Updated `build_dk_matrix()` to use the shared local-rotation builder.

- `pyatb-main/src/pyatb/symmetry/hs_standardize.py`
  - Switched `_local_rotation()` to the shared `build_atom_local_rotation(..., passive_basis=True)` path.

- `pyatb-main/tests/test_character_module.py`
  - Added regression test for boundary-equivalent fractional coordinates in `Dk_matrix.find_atom_mapping()`.
  - Added regression test for active/passive local-rotation conventions in `build_atom_local_rotation()`.
  - Updated `hs_standardize._local_rotation()` tests to match the new shared implementation path.

## Root Cause Fixed

- The boundary-equivalent atom case `0 1 0 == 0 0 0` was still unsafe in the repository-side `Dk_matrix.find_atom_mapping()` path if raw fractional coordinates reached that function.
- This produced fake lattice shifts such as `(0, -1, 1)` instead of `(0, 0, 0)`, which then contaminated symmetry-phase / atom-map logic.
- The local D-matrix construction was also split across multiple files. This turn consolidates the orbital+spin local block construction into one shared helper.

## Unit Tests

### Targeted RED/GREEN regression tests

Run:

```bash
conda run -n symm bash -lc 'cd /home/zjdai/file-test/pyatb_symm/pyatb-main && pytest -q tests/test_character_module.py -k "dk_find_atom_mapping_canonicalizes_boundary_equivalent_positions or build_atom_local_rotation_supports_active_and_passive_conventions"'
```

Result:

- `2 passed`

### Relevant symmetry/HS subset

Run:

```bash
conda run -n symm bash -lc 'cd /home/zjdai/file-test/pyatb_symm/pyatb-main && pytest -q tests/test_hs_standardize.py tests/test_character_module.py -k "build_atom_mapping_wraps_boundary_equivalent_positions_without_extra_shift or dk_find_atom_mapping_canonicalizes_boundary_equivalent_positions or build_atom_local_rotation_supports_active_and_passive_conventions or local_rotation_uses_inverse_cartesian_transform or local_rotation_includes_spin_rotation_for_soc or build_dk_matrix_returns_identity_for_single_s_orbital or build_dk_matrix_supports_soc_interleaved_spin_basis or canonicalize_fractional_coordinates_wraps_boundary_points or standardize_sparse_blocks_uses_passive_basis_rotation_order"'
```

Result:

- `9 passed, 25 deselected`

### Full related test files

Run:

```bash
conda run -n symm bash -lc 'cd /home/zjdai/file-test/pyatb_symm/pyatb-main && pytest -q tests/test_hs_standardize.py tests/test_character_module.py'
```

Result:

- `34 passed, 4 warnings`

## Real-Data HS Self-Symmetry Validation

Run:

```bash
conda run -n symm python /tmp/validate_self_symmetry_axis_angle_canon.py
```

### `test_workspace/test-abacus-2/soc`

- `all_HR_passed = True`
- `all_SR_passed = True`
- `max_HR_rel_fro = 7.348293095132547e-09`
- `max_HR_abs = 8.502953857211339e-09`
- `max_SR_rel_fro = 1.408314006298417e-08`
- `max_SR_abs = 7.105726340661533e-09`

### `test_workspace/test-abacus-4`

- `all_HR_passed = True`
- `all_SR_passed = True`
- `max_HR_rel_fro = 1.0461610608790815e-08`
- `max_HR_abs = 1.6186617213421666e-08`
- `max_SR_rel_fro = 1.9900195723509407e-08`
- `max_SR_abs = 1.1202022191791894e-08`

## Conclusion

- The repository now carries the ABACUS boundary-point canonicalization behavior for the symmetry mapping path.
- The local D-matrix construction is shared between k-space representation building and HS standardization.
- With the corrected boundary handling, the HS self-symmetry covariance validation remains passed for both reference datasets at the `1e-8` level.
