# Velocity Covariance Test Utilities

This directory collects the velocity-covariance test scripts and notes used for
the current pyatb symmetry validation work.

## Core implementation

Reusable covariance functions live in:

```text
src/pyatb/symmetry/velocity_covariance.py
```

The functions cover:

- wavefunction coefficient transport: `transform_wavefunction_coefficients`
- orbital-basis velocity covariance: `transform_orbital_velocity_matrix`
- transported-gauge band velocity covariance: `transform_band_velocity_matrix`
- band-gauge switch: `switch_band_velocity_gauge`
- direct target-gauge band velocity covariance:
  `transform_band_velocity_to_target_gauge`
- B-switch construction:
  `band_gauge_switch_matrix`

For antiunitary operations, the functions use `nspin=1` or `nspin=4` to build
the proper time-reversal matrix unless the caller says the representation matrix
already includes it.

## Files in this directory

- `test_velocity_covariance_functions.py`: small unit tests for the reusable
  covariance functions.
- `pyatb_test_velocity_core.py`: copy of the pyatb orbital-basis velocity
  covariance tester.
- `check_velocity_symmetry_report.py`: single-case orbital-basis report driver.
- `check_velocity_band_symmetry_report.py`: single-case band-basis report driver.
- `batch_check_velocity_symmetry_627.py`: batch orbital-basis driver for
  `pyatb-6.27` results.
- `batch_check_velocity_band_symmetry_627.py`: batch band-basis driver for
  `pyatb-6.27` results.
- `kp_model.py`: snapshot of `src/pyatb/symmetry/kp/model.py` kept with the
  velocity-covariance notes for cross-checking k.p symmetry-model conventions.
- `velocity_matrix_covariance_formulas.md`: formula summary for wavefunction,
  orbital-basis velocity, and band-basis velocity covariance.
- `band_basis_representation_matrix.md`: detailed note on band-basis
  representation matrices and B-switch.
- `velocity_covariance_memory.md`: compact record of the velocity covariance
  testing conventions.

Run the lightweight function tests from the repository root with:

```bash
python src/pyatb/symmetry/test-velocity/test_velocity_covariance_functions.py
```
