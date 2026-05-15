# PYATB Character Workflow Summary (Structure/HS Standardization, Data Symmetrization, Group Representation)

## 0. Scope
This note summarizes the current CHARACTER pipeline under `pyatb-main/src/pyatb/symmetry`, with formulas and implementation mapping for:

1. Structure standardization + HS (HR/SR) standardization
2. Data symmetrization (covariance constraint)
3. Group representation (little-group construction, character calculation, irrep labeling)

Notation follows code conventions:
- Row-lattice matrix: `A` (rows are `a1,a2,a3`)
- Integer lattice transforms: `M`, `B`
- Space-group operation: `g={R|tau}`
- Atom map: `a -> a'` with integer cell shift `w_a`
- Local orbital/spin representation: `D_a`

---

## 1. Structure Standardization and HS Standardization

### 1.1 Three-structure chain and two-stage mapping
The implementation uses three structures:

- `stru1`: original input structure (possibly supercell / non-standard axes)
- `stru2`: `spglib.standardize_cell(..., to_primitive=True, no_idealize=True)`
- `stru3`: `spglib.standardize_cell(..., to_primitive=True, no_idealize=False)`

Entry point:
- `SymmStructureAnalyzer.analyze_nonmagnetic` (`symm_stru.py`)

Key transforms:
- `M12 = round(A1 * A2^{-1})`
- `Q23`: row-vector Procrustes fit from `A2 @ Q23 ~= A3`
- `B23 = round((A2 @ Q23) * A3^{-1})`
- `B13 = M12 @ B23 = round((A1 @ Q23) * A3^{-1})`

Implementation:
- `_fit_row_rotation`, `_round_integer_matrix`, `_compose_two_stage_mapping`

### 1.2 Atom mapping and R mapping
For each atom `a`, solve the final mapping `stru1 -> stru3`:
- index map `pi(a)`
- integer shift `w_a`

For each pair `(R,a,b)`, the mapped R index is

\[
R' = R B_{13} + w_b - w_a
\]

This is implemented by `map_target_r_vector` and used in `canonicalize_abacus_hs`.

### 1.3 Block covariance transform used in HS standardization
For each pair block, the current transform is

\[
H'_{\pi(a)\pi(b)}(R') = D_a^{\dagger} H_{ab}(R) D_b
\]
\[
S'_{\pi(a)\pi(b)}(R') = D_a^{\dagger} S_{ab}(R) D_b
\]

Implementation:
- `hs_standardize.py::_assemble_target_dense_blocks`
- Code form: `rotated = d_a.conj().T @ local_block @ d_b`

Notes:
- `D_a` is built with `build_atom_local_rotation(..., passive_basis=True)`.
- `xyz_axis_transform_cartesian` is passed as `Q23^T` from `symm_stru.py` to keep the passive-basis convention consistent.

### 1.4 Full-matrix reconstruction from Hermitian partners
ABACUS sparse XR stores upper-triangular entries. For robust mapping/covariance checks, the code can reconstruct full dense blocks via `R` / `-R` Hermitian partners before transformation.

Functions:
- `_full_dense_blocks_by_r_from_hermitian_partners`
- `_dense_blocks_by_r_with_optional_full_reconstruction`

### 1.5 Outputs and pipeline handoff
If standardization is required:
- write `STRU-symm` in the working directory
- write standardized HR/SR into `Out/CHARACTER`
- rebuild TB object and continue character workflow on standardized data

Entry:
- `character.py::calculate_character` -> `canonicalize_abacus_hs`

---

## 2. Data Symmetrization (Covariance Constraint)

### 2.1 Objective
Given fixed structure/mapping, reduce numerical covariance violations in HS matrices to improve irrep assignment robustness.

Main controls:
- `data_symmetrize` (0/1)
- `data_symm_target_max_abs_ry` (default `1e-8`)
- `data_symm_max_iter_per_operation` (default 5)
- `data_symm_nonzero_block_tol`

Entry:
- `character.py::calculate_character`
- `data_covariance_constraint.py::sequential_symmetrize_hs`

### 2.2 Per-operation transform
For `g={R_g|tau_g}` under the active convention in this module:

\[
R' = R_g R + (\delta_b - \delta_a)
\]
\[
T_g(H)_{a'b'}(R') = D_a H_{ab}(R) D_b^{\dagger}
\]

where `a->a'` and `delta_a` come from `find_atom_mapping`.

Implementation:
- `_prepare_operation_context`
- `transform_blocks_with_context`

### 2.3 Iterative averaging
For each operation (up to `N` iterations):

\[
H \leftarrow \tfrac{1}{2}(H + T_g(H)),\quad
S \leftarrow \tfrac{1}{2}(S + T_g(S))
\]

Only numerically nonzero atom-pair blocks are touched (`nonzero_block_tol`).

Implementation:
- `build_active_pair_index`
- `average_block_sets_on_touched_pairs`
- `compare_block_sets_on_candidate_pairs`

### 2.4 Final diagnostics
After sequential operation sweeps, the module reports:
- per-operation max/mean/rms/rel_fro
- final worst residual element `(R,row,col)`
- before/after HR/SR covariance statistics

Implementation:
- `self_covariance_statistics`
- `compare_block_sets_with_detail`
- `write_symmetrized_hs`

Note:
- Final Hermitization is currently not enforced in this path.
- `hr_max_abs_threshold_ry` is currently present in API but not used as a hard-stop condition.

---

## 3. Group Representation and Character Pipeline

### 3.1 Little-group construction
Given `k`, little-group test is

\[
\mathbf{k}R^{-1} - \mathbf{k} = \mathbf{G},\quad \mathbf{G}\in\mathbb{Z}^3
\]

Implementation:
- `symm_stru.py::_little_group_operation_indices`

Then `kLittleGroups/kLG_*.data` is resolved with IRVSP-compatible variable-k and fallback matching:
- `KLittleGroupsDB.resolve_kpoint_from_star`

### 3.2 k-point convention
Current logic (per latest requirement):
- use input primitive-cell k-points directly
- no k-point transform during structure standardization

Implementation:
- `symm_stru.py::analyze_nonmagnetic` (`canonical_kpoints = kpoints_direct`)

### 3.3 Building `D_k(g)`
For each operation:

\[
D_k(g) = e^{-i2\pi k\cdot w_a} D_a
\]

- `w_a`: cell shift from atom mapping
- `D_a`: orbital rotation (and spin rotation in SOC)

Implementation:
- `Dk_matrix.py::build_dk_matrix`
- orbital part: `atom_orbital_rotation` / `shell_rotation`
- spin part: `spin_half_matrix_from_cartesian_rotation`

### 3.4 Subspace character calculation
From diagonalization (`C_{nk}`, `E_{nk}`, and overlap `S_k`), for degenerate subspace `D`:

\[
\chi_D(g)=\mathrm{Tr}\left[C_D^{\dagger}S_kD_k(g)C_D\right]
\]

Implementation:
- `character.py::_calculate_character_rows`
- `character_core.py::group_degenerate_bands`
- `character_core.py::calculate_subspace_characters`

### 3.5 Irrep labeling
On active little-group operations, assign by combinational matching:

\[
\chi_{calc}(g) \approx \sum_{\alpha} n_\alpha\chi_\alpha(g),\quad n_\alpha\in\mathbb{N}
\]

Implementation:
- `character_core.py::assign_irrep_combination`
- returns labels like `GM8`, `L3 + L4`, otherwise `??`

### 3.6 Single-valued / double-valued and `Reality`
- `raw_name` with leading `-` denotes double-valued irreps
- `spinful=True` filters to double-valued irreps first
- `reality` is carried from database metadata (typically interpreted by FS class)

Implementation:
- `_filter_irreps_by_spin`
- report generation in `symm_stru.py`

---

## 4. Part II: Function-Level Responsibilities

### 4.1 `symm_stru.py`
- `analyze_nonmagnetic`: top-level orchestrator for nonmagnetic symmetry preprocessing.
- `_standardize_nonmagnetic_cell`: obtains `mapping_atoms` and standardized primitive.
- `_build_atom_mapping`: atom/species mapping with integer shifts across lattices.
- `_build_rotated_atom_mapping`: Hungarian matching after Cartesian-axis rotation.
- `_compose_two_stage_mapping`: compose stage-1 and stage-2 atom maps.
- `_fit_row_rotation`: row-vector Procrustes fit for Cartesian axis relation.
- `_write_standardized_stru`: write `STRU-symm` while preserving pseudo/orbital sections.
- `_build_symmetry_operations`: construct full operation descriptors (R,t,spin,symbol,axis).
- `_resolve_kpoint_records`: resolve each k-point little-group and DB record.

### 4.2 `hs_standardize.py`
- `canonicalize_abacus_hs`: end-to-end HS standardization entry.
- `_assemble_target_dense_blocks`: pair mapping + block rotation (`D^dagger H D`).
- `map_target_r_vector`: implements integral `R` remap formula.
- `_full_dense_blocks_by_r_from_hermitian_partners`: lower-triangle reconstruction.
- `_write_abacus_sparse_xr`: write mapped dense blocks into ABACUS sparse format.

### 4.3 `data_covariance_constraint.py`
- `load_abacus_hs_blocks`: load STRU/HR/SR into dense `R->block` dicts.
- `get_symmetry_operations_from_metadata`: spglib operations for covariance checks.
- `_prepare_operation_context`: precompute atom map, shifts, local D, pair cache.
- `transform_blocks_with_context`: apply one symmetry operation transform.
- `build_active_pair_index` / `merge_active_pair_index`: track nonzero pair workload.
- `sequential_symmetrize_hs`: sequential per-operation iterative averaging.
- `self_covariance_statistics`: aggregate covariance residual metrics.
- `write_symmetrized_hs`: persist symmetrized HR/SR.

### 4.4 `Dk_matrix.py`
- `extract_abacus_basis_metadata`: parse basis/shell/atom offsets from TB/STRU.
- `find_atom_mapping`: atom map and cell shifts for one operation.
- `shell_rotation`: real-spherical-harmonic shell rotation with parity handling.
- `build_atom_local_rotation`: orbital or orbital⊗spin local representation.
- `build_dk_matrix`: full `D_k(g)` in Bloch basis (phase + local transform).

### 4.5 `character_core.py`
- `group_degenerate_bands`: degeneracy partition by energy tolerance.
- `calculate_subspace_characters`: compute `Tr(C^dagger S D C)`.
- `assign_irrep_combination`: combinational irrep assignment on active ops.

### 4.6 `k_little_groups.py`
- `KLittleGroupsDB.load`: parse kLittleGroups database.
- `resolve_kpoint_from_star`: robust k-star matching with fallback rules.
- `irrep_table_characters`: evaluate k-dependent character table entries.

---

## 5. Practical conclusions
1. The structure+HS standardization path now covers supercell-to-primitive plus axis-rotation cases.
2. Data symmetrization effectively suppresses covariance noise and logs detailed residual history.
3. The representation pipeline (little group -> `D_k(g)` -> subspace characters -> irrep labels) is complete and IRVSP-style reportable.
4. k-points now remain exactly as user input primitive-cell k-points.

