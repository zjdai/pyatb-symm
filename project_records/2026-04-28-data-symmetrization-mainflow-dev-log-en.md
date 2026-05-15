# 2026-04-28 Data Symmetrization Mainflow Integration (EN)

## Scope
- Integrated non-Hermitian data symmetrization into `CHARACTER` main workflow.
- Added `CHARACTER` input switches for data symmetrization.
- Added before/after covariance metrics output (max/mean) for HR and SR.

## Code Changes
- `pyatb-main/src/pyatb/symmetry/character.py`
  - Added CHARACTER parameters:
    - `data_symmetrize` (0/1)
    - `data_symm_target_max_abs_ry`
    - `data_symm_max_iter_per_operation`
    - `data_symm_nonzero_block_tol`
    - `data_symm_verbose`
  - Added pipeline:
    1) load active STRU/HS,
    2) run covariance symmetrization,
    3) write `*-covsymm.csr`,
    4) rebuild TB from symmetrized HS,
    5) continue character evaluation.
  - Wrote summary file:
    - `Out/CHARACTER/data_symmetrization_report.txt`
    - includes before/after max/mean changes.
  - Added fallback tolerance for irrep assignment:
    - first pass `tol=5e-2`, retry with `tol=1e-1` if unresolved.
- `pyatb-main/src/pyatb/io/default_input.py`
  - Added CHARACTER block defaults for above data symm parameters.
- `pyatb-main/src/pyatb/symmetry/data_covariance_constraint.py`
  - Kept per-operation loop logic (max 5 by default).
  - Removed Hermitization stage from final symmetrization path.
  - Final summary now reports post-symmetrization covariance only.
- `pyatb-main/tests/test_character_input.py`
  - Extended parsing test to cover new CHARACTER data symm inputs.

## Verification
- Syntax:
  - `python -m py_compile` on modified modules passed.
- Unit test:
  - `pytest -q pyatb-main/tests/test_character_input.py` -> `3 passed`.
- Runtime test (`PYTHONPATH=.../pyatb-main/src`):
  - `test_workspace/test-abacus-3/pyatb` with data symm enabled:
    - generated `Out/CHARACTER/data_symmetrization_report.txt`
    - report includes HR/SR max/mean before->after deltas.
  - `test_workspace/test-abacus-2/pyatb` baseline rerun.
- Character table check:
  - `ab3`: `symmetry_character_report.txt` and `band_irrep.txt` `??` count = 0.
  - `ab2`: `symmetry_character_report.txt` and `band_irrep.txt` `??` count = 0.
  - Shared `(kname, band)` keys between ab3 and ab2: mismatch count = 0.

## Notes
- `ab3` and `ab2` have different k-label sets (`ab3` uses `Y`; `ab2` uses `T/L`) due standardized-cell path differences, but shared-label irreps match.
