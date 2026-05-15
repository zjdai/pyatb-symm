# 2026-04-28 Optimization round (items 2/3/4 only) - EN

## Scope (as requested)
Do NOT modify `abacus_readHR/abacus_readSR`.
Implement only:
1. Item-2: remove write->readback loop after data symmetrization.
2. Item-3: active nonzero atom-pair cache for symmetry transform.
3. Item-4: incremental/candidate-pair stats in iterative symmetrization loop.

Test case:
- `/home/zjdai/file-test/pyatb_symm/test_workspace/test-abacus-3/pyatb`

## Code changes
### A) Item-2 (no readback loop)
Files:
- `pyatb-main/src/pyatb/symmetry/data_covariance_constraint.py`
- `pyatb-main/src/pyatb/symmetry/character.py`

Details:
- Added in-memory conversion from dense block dictionaries to `multiXR` objects:
  - `_dense_blocks_to_multixr(...)`
  - `build_multixr_from_dense_blocks(...)`
- CHARACTER `data_symmetrize==1` path now:
  - still writes `*-covsymm.csr` outputs,
  - but no longer re-parses those files for TBModel input.
- Important fix applied:
  - internal `hr_blocks` are already in eV units,
  - conversion must NOT divide by `Ry_to_eV` again,
  - otherwise character results drift.
- Added writer-equivalent numeric quantization (`%.16e`) during in-memory conversion to match file round-trip behavior exactly.

### B) Item-3 (active nonzero atom-pair cache)
File:
- `pyatb-main/src/pyatb/symmetry/data_covariance_constraint.py`

Details:
- Context now carries:
  - `pair_row_by_source`
  - source atom-pair labels in `pair_rows`.
- Added:
  - `build_active_pair_index(...)`
  - `merge_active_pair_index(...)`
- `transform_blocks_with_context(...)` accepts `active_pair_index` and only iterates cached active source pairs for each R block.

### C) Item-4 (incremental stats in iterative loop)
File:
- `pyatb-main/src/pyatb/symmetry/data_covariance_constraint.py`

Details:
- Added `compare_block_sets_on_candidate_pairs(...)`.
- In `sequential_symmetrize_hs(...)` inner iterations:
  - build candidate pair set from current active index + touched target pairs,
  - evaluate iteration stats on candidate pairs instead of full dense all-element scan,
  - convergence still uses `HR max_abs <= target` criterion.

## Verification
### 1) Correctness baseline snapshot
Reference snapshot used:
- `Out/CHARACTER_snap_pre_algo`

Reference hashes:
- HR: `cd1494c7d74d41e506949ac35b84a2e67c80da24c0101edf317851279aca1280`
- SR: `0225de408ba1799ba12c2576e1cbb71e8a2caaf6ef66dd8666bd0f59b36784a5`
- band_irrep: `3d517f225f0e14b4a185b8ab0b8f2f689e4d758db871a70c1d9a24349f792888`
- symmetry_character_report: `07fa74f93cd759e1607237d43fbbd33eef9c739613b22545c967727d9ddefd03`
- trace: `d78d62da8256522e5e57992ee9f60313fcb559deacbedf09817b1edba4ae8146`

### 2) Post-change run
Command:
- `pyatb`

Run log:
- `run_opt_234_v3.log`
- elapsed: `57.38 s`
- max RSS: `1661832 KB`

### 3) Exact invariance checks
- `diff -q` between snapshot and current outputs (HS + character + trace): `DIFF_EXIT=0`
- hashes fully identical to reference (all five files).
- no `??` in `band_irrep.txt` / `symmetry_character_report.txt`.

## Performance
Recent reference before this round:
- `run_opt_algo1.log`: `59.89 s`

After this round (2/3/4):
- `run_opt_234_v3.log`: `57.38 s`

Net speedup in this round:
- about `2.51 s` on this test case.
