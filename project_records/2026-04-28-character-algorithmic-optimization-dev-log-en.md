# 2026-04-28 CHARACTER algorithmic optimization log (EN)

## Goal
Apply additional algorithm-level optimizations while keeping:
- HS outputs unchanged,
- character outputs unchanged,
- runtime improved on test case:
  - `/home/zjdai/file-test/pyatb_symm/test_workspace/test-abacus-3/pyatb`

## Profiling snapshot (before this round)
Command:
- `python -m cProfile -o prof_opt_algo.pstats pyatb`

Top hotspots:
- `compare_block_sets`: ~19.10 s cumulative
- `transform_blocks_with_context`: ~18.03 s cumulative
- `sequential_symmetrize_hs`: ~26.95 s cumulative

## Implemented optimizations
### 1) One-pass detailed compare for final worst-element scan
File:
- `pyatb-main/src/pyatb/symmetry/data_covariance_constraint.py`

Changes:
- Added `compare_block_sets_with_detail(...)` to compute:
  - matrix summary (max/mean/rms/rel_fro), and
  - worst element detail
  in a single traversal.
- Replaced final `compare_block_sets + _max_error_element_detail` double-pass with one-pass call.

### 2) Combination cache + pre-sliced irrep tables in character assignment
File:
- `pyatb-main/src/pyatb/symmetry/character_core.py`

Changes:
- Added `_cached_combinations_with_replacement(...)` (module cache).
- In `assign_irrep_combination(...)`:
  - pre-slice irrep character tables once per call using active operation indices,
  - precompute irrep labels once,
  - reuse cached combinations.

### 3) Minor compare loop overhead reduction
File:
- `pyatb-main/src/pyatb/symmetry/data_covariance_constraint.py`

Changes:
- Removed unnecessary key sorting in compare/detail scans (set union iteration), preserving numerical logic.

## Validation steps
### A) Pre-change snapshot
Saved exact baseline output files to:
- `Out/CHARACTER_snap_pre_algo`

Files snapshotted:
- `data-HR-sparse_SPIN0-symm-covsymm.csr`
- `data-SR-sparse_SPIN0-symm-covsymm.csr`
- `band_irrep.txt`
- `symmetry_character_report.txt`
- `trace.txt`

### B) Re-run after optimization
Command:
- `pyatb`

Result:
- `EXIT_CODE=0`
- runtime: `elapsed_sec=59.89`
- max RSS: `1658668 KB`

Reference runtime before this round:
- previous optimized run: `elapsed_sec=62.31`

### C) Exact invariance checks
Compared `Out/CHARACTER_snap_pre_algo` vs current `Out/CHARACTER`:
- `diff -q` on all 5 key files: `DIFF_EXIT=0`
- SHA256 all identical:
  - HR: `cd1494c7d74d41e506949ac35b84a2e67c80da24c0101edf317851279aca1280`
  - SR: `0225de408ba1799ba12c2576e1cbb71e8a2caaf6ef66dd8666bd0f59b36784a5`
  - band_irrep: `3d517f225f0e14b4a185b8ab0b8f2f689e4d758db871a70c1d9a24349f792888`
  - symmetry_character_report: `07fa74f93cd759e1607237d43fbbd33eef9c739613b22545c967727d9ddefd03`
  - trace: `d78d62da8256522e5e57992ee9f60313fcb559deacbedf09817b1edba4ae8146`

Additional check:
- no `??` found in `band_irrep.txt` and `symmetry_character_report.txt`.

## Conclusion
- HS data and character outputs remained byte-identical.
- Runtime improved from `62.31 s` to `59.89 s` on this test.
