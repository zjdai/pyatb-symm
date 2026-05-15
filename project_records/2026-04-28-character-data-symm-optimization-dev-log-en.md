# 2026-04-28 CHARACTER data-symmetrization optimization log (EN)

## Scope
Optimize CHARACTER data-symmetrization path in this order:
1. P0: reuse prebuilt symmetry operation contexts
2. P1: pre-cache atom-pair transform metadata + in-place touched-pair averaging
3. P2: streamline report output format (move full history to JSON)

Test target:
- `/home/zjdai/file-test/pyatb_symm/test_workspace/test-abacus-3/pyatb`
- Input keeps `data_symmetrize = 1`

## Baseline (before optimization)
- Command: `pyatb`
- Runtime: `elapsed_sec=75.67`, `max_rss_kb=1658612`
- Character outputs hash:
  - `band_irrep.txt`: `3d517f225f0e14b4a185b8ab0b8f2f689e4d758db871a70c1d9a24349f792888`
  - `symmetry_character_report.txt`: `07fa74f93cd759e1607237d43fbbd33eef9c739613b22545c967727d9ddefd03`
  - `trace.txt`: `d78d62da8256522e5e57992ee9f60313fcb559deacbedf09817b1edba4ae8146`
- No `??` found in character outputs.

## P0 changes
Files:
- `pyatb-main/src/pyatb/symmetry/data_covariance_constraint.py`
- `pyatb-main/src/pyatb/symmetry/character.py`

Details:
- Added `prepare_operation_contexts(...)`.
- `self_covariance_statistics(...)` now accepts optional prebuilt `operation_contexts`.
- `sequential_symmetrize_hs(...)` now accepts optional prebuilt `operation_contexts`.
- CHARACTER path builds contexts once and reuses for before/after stats + symmetrization.

P0 test:
- Runtime: `elapsed_sec=72.02`, `max_rss_kb=1659556`
- Hash check: character outputs unchanged (all hashes identical to baseline).
- Symmetrization summary key metrics unchanged.

## P1 changes
File:
- `pyatb-main/src/pyatb/symmetry/data_covariance_constraint.py`

Details:
- Prebuilt per-operation atom/pair cache in context (`pair_rows`) to avoid repeated slice/map/shift assembly in inner loops.
- `transform_blocks_with_context(...)` switched to cached pair traversal.
- `average_block_sets_on_touched_pairs(...)` added `in_place` path and grouped updates by `R` to avoid full-dict copy each iteration.
- Sequential symmetrization now uses `in_place=True`.

P1 test:
- Runtime: `elapsed_sec=63.70`, `max_rss_kb=1659860`
- Hash check: character outputs unchanged (all hashes identical to baseline).
- Symmetrization summary key metrics unchanged.

## P2 changes
File:
- `pyatb-main/src/pyatb/symmetry/character.py`

Details:
- `data_symmetrization_report.txt` format improved:
  - keep compact global summary,
  - add per-operation final residual table,
  - include explicit worst residual element summary.
- Full symmetrization history moved to `data_symmetrization_history.json`.

P2 test:
- Runtime: `elapsed_sec=62.31`, `max_rss_kb=1665108`

### Detailed invariance check (P1 vs P2)
Snapshot source:
- `Out/CHARACTER_snap_p1`

Current output:
- `Out/CHARACTER`

Exact checks:
- `diff -q` on 5 key files: exit `0` (exactly identical)
  - `data-HR-sparse_SPIN0-symm-covsymm.csr`
  - `data-SR-sparse_SPIN0-symm-covsymm.csr`
  - `band_irrep.txt`
  - `symmetry_character_report.txt`
  - `trace.txt`
- `sha256` equality confirmed:
  - HR covsymm CSR: `cd1494c7d74d41e506949ac35b84a2e67c80da24c0101edf317851279aca1280`
  - SR covsymm CSR: `0225de408ba1799ba12c2576e1cbb71e8a2caaf6ef66dd8666bd0f59b36784a5`
  - band_irrep: `3d517f225f0e14b4a185b8ab0b8f2f689e4d758db871a70c1d9a24349f792888`
  - symmetry_character_report: `07fa74f93cd759e1607237d43fbbd33eef9c739613b22545c967727d9ddefd03`
  - trace: `d78d62da8256522e5e57992ee9f60313fcb559deacbedf09817b1edba4ae8146`

Conclusion:
- P0/P1 delivered runtime reduction with unchanged physics outputs.
- P2 changed report formatting only; HS and character numerical outputs are bitwise unchanged.
