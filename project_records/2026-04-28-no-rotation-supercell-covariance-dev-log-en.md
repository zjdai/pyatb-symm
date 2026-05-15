# 2026-04-28 No-Rotation Supercell-to-Primitive Covariance Log

## Goal

Validate whether the following ABACUS datasets satisfy HS covariance in the special case where:

- both structures use the same Cartesian frame,
- no orbital or spin rotation is needed,
- only the integer supercell matrix, atom mapping, and `R -> R'` relabeling should matter.

Validation targets:

- supercell source data:
  - `test_workspace/test-abacus-3/OUT.ABACUS/data-HR-sparse_SPIN0.csr`
  - `test_workspace/test-abacus-3/OUT.ABACUS/data-SR-sparse_SPIN0.csr`
- primitive target data:
  - `test_workspace/test-abacus-4/OUT.ABACUS/data-HR-sparse_SPIN0.csr`
  - `test_workspace/test-abacus-4/OUT.ABACUS/data-SR-sparse_SPIN0.csr`

The structure files used for basis metadata are:

- source: `test_workspace/test-abacus-3/pyatb/STRU`
- target: `test_workspace/test-abacus-4/verification_tmp/STRU_S1_abs`

These copies are used because their orbital paths resolve correctly.

## Main Findings

### 1. The geometric mapping is correct

The inferred integer lattice transform is:

```text
[[ 0, -1,  0],
 [-1,  0,  1],
 [-1,  0, -1]]
```

This matches the random supercell matrix used to build `test-abacus-3`.

Pair-vector validation result:

- `pair_vector_error_max_abs = 7.973655637982802e-07`

So the mapping from `(R, atom_a, atom_b)` to `(R', atom_a', atom_b')` is geometrically correct.

### 2. Supercell-to-primitive validation must not sum duplicate source blocks

The previous `canonicalize_abacus_hs()` path accumulates all mapped source blocks into the same target key.

That is acceptable for primitive-cell reordering / standardization, but not for validating a supercell against a primitive cell.

Why:

- multiple atoms inside the supercell map to the same primitive atom,
- therefore the same target block receives multiple equivalent source samples,
- those samples should be checked for consistency, not added together.

The new validation path instead:

- preserves every mapped source block sample,
- compares each sample directly against the target block,
- separately checks consistency among duplicate source samples belonging to the same target key.

### 3. Duplicate source samples are internally consistent

Sample results:

- `HR duplicate_consistency_max_abs = 1.0000000502143058e-13`
- `SR duplicate_consistency_max_abs = 9.999999506879345e-17`

This means:

- the atom mapping is stable,
- the `R` relabeling formula is stable,
- geometrically equivalent source blocks agree with each other inside the supercell data.

### 4. Direct comparison to the primitive target still shows significant residual differences

Sample results:

- `HR max_abs_diff = 0.33128893712493707`
- `HR rel_fro_diff = 0.0555324466596934`
- `SR max_abs_diff = 0.288381729`
- `SR rel_fro_diff = 0.17911222587872075`

At this point, the following simple failure modes are ruled out:

- wrong integer supercell matrix,
- wrong atom mapping,
- wrong `R` mapping formula,
- simple wrap / boundary pairing mistakes,
- incorrect mixing of duplicate supercell contributions.

If debugging continues, the likely remaining causes are:

- supercell vs primitive DFT-output convention differences,
- ABACUS HS/SR output conventions in supercell / primitive runs,
- additional block-level gauge, phase, or basis-convention differences.

## Code Changes

New module:

- `pyatb-main/src/pyatb/symmetry/hs_covariance.py`

Main entry points:

- `infer_integer_lattice_transform()`
- `collect_no_rotation_block_samples_from_dense_blocks()`
- `validate_no_rotation_block_mapping_from_dense_blocks()`
- `validate_abacus_supercell_to_primitive_no_rotation()`

New tests:

- `pyatb-main/tests/test_hs_covariance.py`

Coverage:

- many-to-one supercell-to-primitive atom mapping,
- duplicate source blocks are kept separate instead of being summed.

## Verification Commands

### Unit tests

```bash
pytest /home/zjdai/file-test/pyatb_symm/pyatb-main/tests/test_hs_standardize.py /home/zjdai/file-test/pyatb_symm/pyatb-main/tests/test_hs_covariance.py -q
```

Result:

```text
7 passed in 0.22s
```

### Sample validation

Report written to:

- `test_workspace/test-abacus-4/test3_to_test4_no_rotation_covariance_report.json`

The JSON report contains:

- the inferred supercell matrix,
- atom mapping,
- block-wise HR/SR errors,
- duplicate-source consistency,
- real-space pair-vector errors.

## Current Status

This round isolated the no-rotation supercell-to-primitive validation into a dedicated path and confirmed:

- the mapping logic itself is correct,
- the remaining residual mismatch is not a simple `R` / atom pairing bug.

The next debugging step should focus directly on the worst blocks reported in `test3_to_test4_no_rotation_covariance_report.json`.

## Follow-up Check: Physical Rewrite of the Boundary Atom

A follow-up check was performed afterward:

- in `test_workspace/test-abacus-4/verification_tmp/STRU_S1_abs`, the boundary atom was physically rewritten
  - from `0.0000000000 1.0000000000 0.0000000000`
  - to `0.0000000000 0.0000000000 0.0000000000`
- then the exact same validation script was run again.

The rerun produced exactly the same metrics:

- `HR max_abs_diff = 0.33128893712493707`
- `HR rel_fro_diff = 0.0555324466596934`
- `SR max_abs_diff = 0.288381729`
- `SR rel_fro_diff = 0.17911222587872075`

This confirms:

- the no-rotation validation path was already handling boundary wrapping correctly,
- physically rewriting the target structure text from `1.0` to `0.0` does not change the result,
- the remaining residual mismatch is not caused by this boundary-standardization issue.

## Follow-up Check Using the Top-Level STRU3 / STRU4 Directly

A stricter follow-up validation was then performed with the following constraints:

- the geometry mapping reads only the top-level structure files:
  - `test_workspace/test-abacus-3/STRU`
  - `test_workspace/test-abacus-4/STRU`
- both structures are wrapped first, so boundary atoms such as `1 -> 0` are canonicalized;
- no coordinate-axis rotation is used;
- single-atom mapping is solved directly from the real-space relation:
  - for each source `(R, atom)`, compute the real-space position `r`,
  - for each target primitive-cell atom at `R = 0`, with position `tau`,
  - solve `(r - tau) = n @ A_target`,
  - where `n = (n1, n2, n3)` is the target-cell `R'`;
- the pair mapping for `(R, atom1, atom2)` is then built from these single-atom mappings and used for HS/SR validation.

To make the top-level `STRU` files directly readable for orbital metadata, two wrapped temporary absolute-path copies were written:

- `test_workspace/test-abacus-4/verification_tmp/STRU3_wrapped_abs`
- `test_workspace/test-abacus-4/verification_tmp/STRU4_wrapped_abs`

Report written to:

- `test_workspace/test-abacus-4/test3_to_test4_top_level_wrap_realspace_report.json`

### Results

Using the ABACUS row-vector lattice convention, the inferred matrix is:

```text
M =
[[ 0, -1,  0],
 [-1,  0,  1],
 [-1,  0, -1]]
```

Numerical checks:

- `max_abs(M_raw - round(M_raw)) = 7.263072743235455e-08`
- `max_abs(A_source - M @ A_target) = 1.2266936479932156e-06`

Single-atom mapping error:

- `single_atom_mapping_max_abs_cart_error = 1.5060593909765885e-06`

Worst single-atom match:

- `old_atom = 10`
- `new_atom = 5`
- `shift = (-1, -1, -1)`
- `max_abs_cart_error = 1.5060593909765885e-06`

HS/SR covariance results:

- `HR max_abs_diff = 0.33128893712493707`
- `HR rel_fro_diff = 0.0555324466596934`
- `SR max_abs_diff = 0.288381729`
- `SR rel_fro_diff = 0.17911222587872075`

These are exactly the same as in the previous no-rotation validation based on the copied structure files.

### Conclusion

This confirms that:

- using the top-level `STRU3/STRU4` directly, with wrap first and then solving `(R', atom')` from the single-atom real-space equation, gives the same result as before;
- the `M` matrix, single-atom mapping, and pair mapping are geometrically correct;
- the remaining HS/SR mismatch is not caused by:
  - not using the top-level structures,
  - missing boundary wrapping,
  - transposing or misreading the row-wise lattice convention,
  - incorrect pairing from `(R, atom)` to `(R', atom')`.

## Re-validation with the Corrected Single-Atom Invariant

A further validation was then run using the corrected invariant explicitly:

```text
R @ A + tau_a = R' @ A' + tau'_a'
```

Here:

- `A` is the lattice matrix from `test-abacus-3/STRU`;
- `A'` is the lattice matrix from `test-abacus-4/STRU`;
- lattice vectors are stored row-by-row following the ABACUS convention;
- `R' = R @ M + shift_a`;
- `tau_a = shift_a @ A' + tau'_a'` is the home-cell atom mapping from the source cell to the target primitive cell.

Report written to:

- `test_workspace/test-abacus-4/test3_to_test4_single_atom_invariance_report.json`

### M matrix and spglib cross-check

The inferred matrix is:

```text
M =
[[ 0, -1,  0],
 [-1,  0,  1],
 [-1,  0, -1]]
```

And:

- `det(M) = 2`
- `volume_ratio_source_over_target = 2.0000004357334773`

spglib cross-check:

- `find_primitive` succeeded;
- primitive atom count `= 5`;
- source atom count `= 10`;
- `source_to_primitive_atom_ratio = 2.0`
- `source_to_primitive_volume_ratio = 1.9999999999810305`

Here spglib is used only to confirm:

- `STRU3` indeed reduces to a 5-atom primitive cell;
- the multiplicity agrees with `det(M)=2`.

It is not used to define the exact orientation of `M`, because spglib may choose another symmetry-equivalent primitive basis.

### Single-atom invariant error

This run explicitly checked 109 source `R` vectors and 10 atoms per `R`, for 1090 single-atom samples in total:

- `single_atom_relation_max_abs_diff = 3.629148395134507e-06`
- `single_atom_relation_sum_abs_diff = 0.0024472054923566217`

Worst sample:

- `source_R = (3, 1, 0)`
- `target_R = (-2, -4, 0)`
- `old_atom = 10`
- `new_atom = 5`
- `shift = (-1, -1, -1)`
- `max_abs_cart_diff = 3.629148395134507e-06`

So the corrected single-atom invariant is satisfied geometrically.

### HS/SR results built from that same mapping

Using the same single-atom mapping and then extending it to pair blocks, the HS/SR results remain:

- `HR max_abs_diff = 0.33128893712493707`
- `HR rel_fro_diff = 0.0555324466596934`
- `SR max_abs_diff = 0.288381729`
- `SR rel_fro_diff = 0.17911222587872075`

### Final conclusion

After re-running the validation explicitly from the corrected single-atom invariant

```text
R @ A + tau_a = R' @ A' + tau'_a'
```

we can further confirm that:

- `M` is correct;
- `R' = R @ M + shift_a` is correct;
- the source-to-target real-space single-atom relation is correct;
- the pair-level geometric mapping derived from it is correct;
- the remaining `HR/SR ~ 1e-1` mismatch is not caused by an incorrect geometric mapping definition.

## Brute-force Block Matching vs Geometric Mapping

An additional direct block-matching analysis was then performed:

- for every nonzero source local block `(R, atom1, atom2)`,
- search the **entire** target block set `(R', atom1', atom2')`,
- and pick the minimum-error target block using

```text
combined_rel = ||dH||_F / ||H_src||_F + ||dS||_F / ||S_src||_F
```

Output:

- `test_workspace/test-abacus-4/block_bruteforce_vs_geometry_report.json`

Analysis script:

- `test_workspace/test-abacus-4/verification_tmp/bruteforce_hs_block_match.py`

### Overall results

- nonzero source blocks: `1056`
- full target block count: `4125`
- brute-force best block still equals the geometric mapping: `986`
- brute-force best block equals the Hermitian counterpart of the geometric key: `0`
- brute-force best block differs from the geometric mapping: `70`

### Two classes among the 70 disagreements

#### Class 1: alternative but equivalent representation

There are `36` blocks in this class.

Their features are:

- the brute-force best key differs from the geometric key,
- but both keys correspond to the same real-space target pair vector.

So this class is not a geometric failure; it is a different but equivalent `(R', atom1', atom2')` representation of the same real-space pair.

All 36 of these blocks come from only two source atom pairs:

- `(atom1, atom2) = (7, 8)` with `18` blocks
- `(atom1, atom2) = (6, 8)` with `18` blocks

#### Class 2: genuinely non-equivalent blocks

There are `34` blocks in this class.

Their features are:

- the brute-force best key differs from the geometric key,
- and the corresponding target pair vectors are also different,
- so they cannot be explained by a simple equivalent relabeling.

All 34 of these blocks come from only two source atom pairs:

- `(atom1, atom2) = (2, 3)` with `18` blocks
- `(atom1, atom2) = (7, 9)` with `16` blocks

### Representative problematic cases

A typical Bi-Bi problematic block:

- source: `(R, atom1, atom2) = ([1, 2, 0], 2, 3)`
- geometric target: `([-1, -1, 2], 2, 1)`
- brute-force best: `([-2, -1, 2], 1, 2)`
- target pair-vector difference: `2.0809890516506924`

Error comparison:

- geometric mapping: `combined_rel = 2.0`
- brute-force best: `combined_rel = 1.3740818927950933`

A typical Se-Se equivalent-relabeling block:

- source: `(R, atom1, atom2) = ([-1, -1, 1], 7, 8)`
- geometric target: `([1, 2, -2], 5, 3)`
- brute-force best: `([0, 1, -3], 3, 4)`

The target pair-vector difference between these two keys is only:

- `3.55e-15`

So this one is an equivalent representation, not a geometric mistake.

### What this tells us

The brute-force search refines the previous conclusion:

- most source blocks still prefer the geometrically mapped target block;
- some Se-Se disagreements are just alternative but equivalent target representations;
- a remaining subset of Bi-Bi and Se-Se blocks prefer target blocks that are neither the geometric key nor the same real-space pair;
- therefore the residual issue is not only about `R` / atom indexing, but also about a subset of target HS/SR blocks being numerically closer to different target blocks.

## 2026-04-28 Addendum: the actual issue is the ABACUS triangular-storage comparison path

In `test_workspace/test-abacus-4/block_mapping_diagnostics_full.json` and
`test_workspace/test-abacus-4/block_mapping_diagnostics_full.txt`, the 70 brute-force disagreement blocks were analyzed further.

New conclusions:

- all `70` disagreements follow the same pattern: the `geometry_target` falls in a `lower` target block, while the brute-force best lands in an `upper` block;
- the summary is `orientation_counter = {'lower->upper': 70}`;
- `36` of them are only equivalent re-representations of the same real-space pair vector;
- the remaining `34` brute-force best blocks are not geometrically correct, they only look numerically closer because the current comparison reads the geometric target block as an effective zero block.

The decisive check is this: for each geometric target block `(R', a', b')`, switch to its ABACUS triangular-storage partner
`(-R', b', a')` and compare after conjugate transpose:

- all `70` blocks recover to `combined_rel < 1e-2`;
- the worst case is only `combined_rel = 2.8898801280926326e-04`;
- for that worst case, the maximum elementwise error is only `2.86803e-06` in `HR` and `2.18e-08` in `SR`.

This shows:

- the geometric mapping formula `R' = R @ M + shift_2 - shift_1` is not invalidated by these 70 disagreements;
- the real problem is that the current ABACUS `HR/SR` comparison uses only the explicit upper-triangular stored block and does not switch to the Hermitian storage partner when the geometric target falls into a lower block;
- therefore the previously observed `combined_rel = 2.0` is mainly an artificial zero-block comparison, not a failure of the geometric mapping itself.

## 2026-04-28 Rerun Addendum (rule mapping first, then nonzero brute-force matching)

Per user request, `test-abacus-3 -> test-abacus-4` was re-run in two explicit stages:

1. Build `(R, atom1, atom2) -> (R', atom1', atom2')` from structure-derived mapping rules.
2. Run brute-force matching over all nonzero source blocks and compare against the geometric mapping.

New output files:

- `test_workspace/test-abacus-4/test3_to_test4_no_rotation_covariance_report_rerun_20260428_step1.json`
- `test_workspace/test-abacus-4/block_bruteforce_vs_geometry_report_rerun_20260428_step2.json`
- `test_workspace/test-abacus-4/block_mapping_diagnostics_full_rerun_20260428_step2.json`
- `test_workspace/test-abacus-4/block_mapping_diagnostics_full_rerun_20260428_step2.txt`

Rerun results:

- Rule-mapping statistics are unchanged:
  - `M = [[0,-1,0],[-1,0,1],[-1,0,-1]]`
  - `HR mapped_block_count = 1056`, `SR mapped_block_count = 968`
  - `HR rel_fro_diff = 0.0555324466596934`
  - `SR rel_fro_diff = 0.17911222587872075`
- Nonzero brute-force matching statistics are unchanged:
  - `best_equals_geometry_count = 986`
  - `other_mismatch_count = 70`
  - `mismatch_count = 70`
- For those 70 mismatches, orientation/partner diagnostics remain:
  - `orientation_counter = {'lower->upper': 70}`
  - `partner_match_summary.all_combined_rel_below_1e-2 = true`
  - `worst_combined_rel = 2.8898801280926326e-04`

Conclusion remains unchanged: the structure-based mapping rules are self-consistent; the 70 mismatches come from ABACUS triangular-storage comparison path (lower-triangle geometric target not switching to Hermitian partner), not from a geometric mapping-formula failure.

## 2026-04-28 Rerun Addendum (complete each R block first, then check translational covariance)

Per user request, we first completed each `R` block into a full matrix (including lower triangle), and then re-validated `test-abacus-3 -> test-abacus-4` translational covariance.

### Code updates

- `pyatb-main/src/pyatb/symmetry/hs_covariance.py`
  - Added `_full_dense_blocks_by_r_from_hermitian_partners(...)`:
    - fills lower-triangle entries for each `R` using `X(R) = X(-R)^dagger`.
  - Added `_dense_blocks_by_r_with_optional_full_reconstruction(...)`.
  - Added `full_matrix_from_hermitian` flag to:
    - `validate_no_rotation_block_mapping_from_xr(...)`
    - `validate_abacus_supercell_to_primitive_no_rotation(...)`

### Unit tests

- `pyatb-main/tests/test_hs_covariance.py`
  - Added `test_full_dense_blocks_are_completed_from_minus_r_partner`
  - Result: `3 passed in 0.22s`

### New report

- `test_workspace/test-abacus-4/test3_to_test4_no_rotation_covariance_report_rerun_20260428_full_matrix.json`
  - `full_matrix_from_hermitian = true`

### Numerical comparison (before vs after full reconstruction)

- HR:
  - before: `mapped_block_count=1056`, `rel_fro_diff=0.0555324466596934`, `max_abs_diff=0.33128893712493707`
  - after: `mapped_block_count=1922`, `rel_fro_diff=8.539412460957883e-05`, `max_abs_diff=3.6012000000029687e-04`
- SR:
  - before: `mapped_block_count=968`, `rel_fro_diff=0.17911222587872075`, `max_abs_diff=0.288381729`
  - after: `mapped_block_count=1806`, `rel_fro_diff=1.6127757588504447e-07`, `max_abs_diff=6.270000000099807e-08`

Conclusion: once lower triangles are completed before comparison, the `test-abacus-3 -> test-abacus-4` translational covariance residuals drop sharply, consistent with the triangular-storage-path diagnosis.

## 2026-04-28 Main-flow integration addendum (standardized structure + Hamiltonian pipeline defaults to full completion)

### Code integration

- `pyatb-main/src/pyatb/symmetry/hs_standardize.py`
  - Added:
    - `_dense_blocks_by_r(...)`
    - `_full_dense_blocks_by_r_from_hermitian_partners(...)`
    - `_dense_blocks_by_r_with_optional_full_reconstruction(...)`
  - `_assemble_target_dense_blocks(...)` now accepts `full_matrix_from_hermitian` and performs per-`R` full completion via `X(R)=X(-R)^dagger` before atom-block remapping.
  - `canonicalize_abacus_hs(...)` now accepts `full_matrix_from_hermitian` (default `True`) and reports the flag in return values.
- `pyatb-main/src/pyatb/symmetry/character.py`
  - In `calculate_character(...)`, the rebuild branch now forwards `analysis_result['full_matrix_from_hermitian']` into `canonicalize_abacus_hs(...)`, default enabled.
- `pyatb-main/src/pyatb/symmetry/symm_stru.py`
  - `StandardizationResult` now includes `full_matrix_from_hermitian`.
  - Standardization result now sets `full_matrix_from_hermitian=True` by default.
  - `standardization_summary.txt` now emits bilingual fields for this flag.

### Unit tests

- `pytest -q tests/test_hs_standardize.py tests/test_hs_covariance.py tests/test_character_module.py`
- Result: `40 passed, 4 warnings`

### Re-validation after integration (3 -> 4, no-rotation supercell mapping)

- Output:
  - `test_workspace/test-abacus-4/test3_to_test4_no_rotation_covariance_report_rerun_20260428_integrated_full.json`
- Result:
  - `full_matrix_from_hermitian = true`
  - `HR rel_fro_diff = 8.539412460957883e-05`
  - `SR rel_fro_diff = 1.6127757588504447e-07`
  - `HR max_abs_diff = 3.6012000000029687e-04`
  - `SR max_abs_diff = 6.270000000099807e-08`

Conclusion: the lower-triangle completion logic is now integrated into the pyatb standardization + Hamiltonian main pipeline and remains validated on `test-abacus-3 -> test-abacus-4`.
