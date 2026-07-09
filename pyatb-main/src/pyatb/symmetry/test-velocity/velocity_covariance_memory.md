# Velocity Covariance Test Notes

This note preserves the conventions used for the velocity matrix covariance checks.

## Core Convention

- Source velocity is computed directly at `k`.
- Target point is:
  - unitary: `k' = k @ inv(R)`
  - antiunitary: `k' = -k @ inv(R)`
- The representation matrix uses the target point phase: `D(g, k')`.
- Cartesian velocity components are mixed with the Cartesian operation matrix `O`.
- Unitariy relation:
  `V_alpha(k') = D(g,k') [sum_beta O_alpha_beta V_beta(k)] D(g,k')^dag`
- Antiunitary relation:
  `V_alpha(k') = D(g,k') U_T [-conj(sum_beta O_alpha_beta V_beta(k))] U_T^dag D(g,k')^dag`

## Matrix Inputs

For `pyatb-6.27` velocity reports, H/S use the CHARACTER-symmetrized matrices:

- `pyatb-6.27/Out/CHARACTER/data-HR-sparse_SPIN0-covsymm.csr`
- `pyatb-6.27/Out/CHARACTER/data-SR-sparse_SPIN0-covsymm.csr`

The position matrix uses the ABACUS raw rR matrix:

- `OUT.ABACUS/data-rR-sparse.csr`
- `rR_unit = Bohr`
- `HR_unit = Ry`

## Scripts

- Single pyatb directory report:
  `workflows/irvsp-pyatb-compare/scripts/check_velocity_symmetry_report.py`
- Batch all pc `pyatb-6.27` reports:
  `workflows/irvsp-pyatb-compare/scripts/batch_check_velocity_symmetry_627.py`

Outputs go under each `pyatb-6.27/check-velocity/` directory and summaries go under:

- `workflows/irvsp-pyatb-compare/results/velocity_symmetry_all_pc_627/`

## Confirmed stru2 Results

For `abacus/stru2/pc/nsoc/pyatb-6.27`:

- all max_abs `4.294770425862e-04`
- all max_rel_fro `1.108636883989e-05`

For `abacus/stru2/pc/soc/pyatb-6.27`:

- all max_abs `4.293222389342e-04`
- all max_rel_fro `1.108423246620e-05`
