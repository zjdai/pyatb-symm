from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from ase import Atoms
from ase.io import read as ase_read

from pyatb import INPUT_PATH, RANK, RUNNING_LOG
from pyatb.constants import Ang_to_Bohr
from pyatb.io.abacus_read_stru import _wrap_fractional_coordinates
from pyatb.symmetry.Dk_matrix import axis_angle_from_cartesian_rotation, spin_half_matrix_from_cartesian_rotation
from pyatb.symmetry.hs_standardize import canonicalize_fractional_coordinates, map_target_r_vector
from pyatb.symmetry.k_little_groups import KLittleGroupsDB, KPointResolution


@dataclass
class SymmetryOperation:
    rotation: np.ndarray
    translation: np.ndarray
    inverse_rotation: np.ndarray
    cart_rotation: np.ndarray
    euler_zyz: np.ndarray
    spin_matrix: np.ndarray
    symbol: str
    description: str
    axis: np.ndarray


@dataclass
class StandardizationResult:
    resolved_group: int
    detected_group: int
    structure_changed: bool
    lattice_changed: bool
    atom_permutation_only: bool
    need_rebuild_hs: bool
    source_stru: str
    target_stru: str
    source_hr: str
    source_sr: str
    target_hr: str
    target_sr: str
    lattice_old: np.ndarray
    lattice_new: np.ndarray
    lattice_transform_fractional: np.ndarray
    xyz_axis_transform_cartesian: np.ndarray
    atom_mapping: list[dict]
    rebuild_reason: str
    full_matrix_from_hermitian: bool


class SymmStructureAnalyzer:
    """Non-magnetic structure symmetry pre-check and little-group table output for CHARACTER module."""

    _SPGLIB_OLD_ERROR_HANDLING_WARNING = "Set OLD_ERROR_HANDLING to false and catch the errors directly."

    def __init__(
        self,
        tb,
        output_path: str,
        emit_standardization_summary: bool | None = None,
        emit_aux_outputs: bool | None = None,
    ) -> None:
        self._tb = tb
        self._output_path = Path(output_path)
        if emit_standardization_summary is None:
            env = os.getenv("PYATB_EMIT_STANDARDIZATION_SUMMARY", "0").strip().lower()
            self._emit_standardization_summary = env in {"1", "true", "yes", "on"}
        else:
            self._emit_standardization_summary = bool(emit_standardization_summary)
        if emit_aux_outputs is None:
            env_aux = os.getenv("PYATB_EMIT_CHARACTER_AUX_OUTPUTS", "0").strip().lower()
            self._emit_aux_outputs = env_aux in {"1", "true", "yes", "on"}
        else:
            self._emit_aux_outputs = bool(emit_aux_outputs)

    def _load_input_stru(self):
        stru_path = Path(INPUT_PATH) / "STRU"
        if not stru_path.exists():
            raise FileNotFoundError(
                f"CHARACTER requires '{stru_path}' to exist. "
                "Please place ABACUS STRU in current pyatb working directory."
            )

        atoms = ase_read(str(stru_path), format="abacus")
        if not isinstance(atoms, Atoms):
            raise ValueError(f"Failed to parse ABACUS STRU via ase: {stru_path}")
        atoms.set_scaled_positions(_wrap_fractional_coordinates(atoms.get_scaled_positions(wrap=False)))
        return atoms, stru_path

    def _standardize_nonmagnetic_cell(self, atoms: Atoms, symm_prec: float):
        try:
            import spglib
        except ModuleNotFoundError as exc:
            raise ImportError("spglib is required for CHARACTER symmetry detection.") from exc

        lattice = np.asarray(atoms.cell.array, dtype=float)
        scaled_positions = np.asarray(atoms.get_scaled_positions(), dtype=float)
        numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
        cell = (lattice, scaled_positions, numbers)

        dataset = spglib.get_symmetry_dataset(cell, symprec=symm_prec, _throw=True)
        if dataset is None:
            raise ValueError("spglib failed to get symmetry dataset for STRU.")

        mapping_cell = self._standardize_cell_without_deprecation_warning(
            spglib,
            cell,
            to_primitive=True,
            no_idealize=True,
            symprec=symm_prec,
        )
        if mapping_cell is None:
            raise ValueError("spglib failed to standardize structure to primitive cell in source Cartesian frame.")

        std_cell = self._standardize_cell_without_deprecation_warning(
            spglib,
            cell,
            to_primitive=True,
            no_idealize=False,
            symprec=symm_prec,
        )
        if std_cell is None:
            raise ValueError("spglib failed to standardize structure to primitive cell.")

        conv_cell = self._standardize_cell_without_deprecation_warning(
            spglib,
            cell,
            to_primitive=False,
            no_idealize=False,
            symprec=symm_prec,
        )
        if conv_cell is None:
            raise ValueError("spglib failed to standardize structure to conventional cell.")

        mapping_lattice, mapping_positions, mapping_numbers = mapping_cell
        std_lattice, std_positions, std_numbers = std_cell
        conv_lattice, conv_positions, conv_numbers = conv_cell
        mapping_atoms = Atoms(
            numbers=np.asarray(mapping_numbers, dtype=int),
            cell=np.asarray(mapping_lattice, dtype=float),
            scaled_positions=np.asarray(mapping_positions, dtype=float),
            pbc=True,
        )
        std_atoms = Atoms(
            numbers=np.asarray(std_numbers, dtype=int),
            cell=np.asarray(std_lattice, dtype=float),
            scaled_positions=np.asarray(std_positions, dtype=float),
            pbc=True,
        )
        conv_atoms = Atoms(
            numbers=np.asarray(conv_numbers, dtype=int),
            cell=np.asarray(conv_lattice, dtype=float),
            scaled_positions=np.asarray(conv_positions, dtype=float),
            pbc=True,
        )
        mapping_atoms.set_scaled_positions(
            _wrap_fractional_coordinates(mapping_atoms.get_scaled_positions(wrap=False))
        )
        std_atoms.set_scaled_positions(_wrap_fractional_coordinates(std_atoms.get_scaled_positions(wrap=False)))
        conv_atoms.set_scaled_positions(_wrap_fractional_coordinates(conv_atoms.get_scaled_positions(wrap=False)))

        std_cell_for_sym = (
            np.asarray(std_lattice, dtype=float),
            np.asarray(std_positions, dtype=float),
            np.asarray(std_numbers, dtype=int),
        )
        sym_dataset = spglib.get_symmetry_dataset(std_cell_for_sym, symprec=symm_prec, _throw=True)
        if sym_dataset is None:
            raise ValueError("spglib failed to get real-space symmetry operations for standardized primitive structure.")
        sym_data = self._symmetry_dict_from_dataset(sym_dataset)

        conv_cell_for_sym = (
            np.asarray(conv_lattice, dtype=float),
            np.asarray(conv_positions, dtype=float),
            np.asarray(conv_numbers, dtype=int),
        )
        conv_sym_dataset = spglib.get_symmetry_dataset(conv_cell_for_sym, symprec=symm_prec, _throw=True)
        if conv_sym_dataset is None:
            raise ValueError("spglib failed to get real-space symmetry operations for standardized conventional structure.")
        conv_sym_data = self._symmetry_dict_from_dataset(conv_sym_dataset)

        return dataset, std_atoms, mapping_atoms, sym_data, conv_atoms, conv_sym_data

    def _standardize_magnetic_cell(self, atoms: Atoms, magnetic_moments: np.ndarray, symm_prec: float):
        try:
            import spglib
        except ModuleNotFoundError as exc:
            raise ImportError("spglib is required for CHARACTER magnetic symmetry detection.") from exc

        lattice = np.asarray(atoms.cell.array, dtype=float)
        scaled_positions = np.asarray(atoms.get_scaled_positions(), dtype=float)
        numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
        mag = np.asarray(magnetic_moments, dtype=float)
        if mag.shape != (len(atoms), 3):
            raise ValueError(
                f"Magnetic moments must have shape ({len(atoms)}, 3), got {mag.shape}."
            )

        cell = (lattice, scaled_positions, numbers)
        mag_cell = (lattice, scaled_positions, numbers, mag)

        unitary_sym_data, detected_unitary_group = self._get_magnetic_unitary_symmetry_data(
            atoms,
            mag,
            symm_prec,
        )

        mapping_cell = self._standardize_cell_without_deprecation_warning(
            spglib,
            cell,
            to_primitive=True,
            no_idealize=True,
            symprec=symm_prec,
        )
        if mapping_cell is None:
            raise ValueError("spglib failed to standardize magnetic structure to primitive cell in source Cartesian frame.")

        std_cell = self._standardize_cell_without_deprecation_warning(
            spglib,
            cell,
            to_primitive=True,
            no_idealize=False,
            symprec=symm_prec,
        )
        if std_cell is None:
            raise ValueError("spglib failed to standardize magnetic structure to primitive cell.")

        mapping_lattice, mapping_positions, mapping_numbers = mapping_cell
        std_lattice, std_positions, std_numbers = std_cell
        mapping_atoms = Atoms(
            numbers=np.asarray(mapping_numbers, dtype=int),
            cell=np.asarray(mapping_lattice, dtype=float),
            scaled_positions=np.asarray(mapping_positions, dtype=float),
            pbc=True,
        )
        std_atoms = Atoms(
            numbers=np.asarray(std_numbers, dtype=int),
            cell=np.asarray(std_lattice, dtype=float),
            scaled_positions=np.asarray(std_positions, dtype=float),
            pbc=True,
        )
        mapping_atoms.set_scaled_positions(
            _wrap_fractional_coordinates(mapping_atoms.get_scaled_positions(wrap=False))
        )
        std_atoms.set_scaled_positions(_wrap_fractional_coordinates(std_atoms.get_scaled_positions(wrap=False)))

        map_tol = max(1e-6, float(symm_prec) * 10.0)
        mapping12 = self._build_atom_mapping(atoms, mapping_atoms, map_tol)
        lattice_old = np.asarray(atoms.cell.array, dtype=float)
        lattice_mapping = np.asarray(mapping_atoms.cell.array, dtype=float)
        lattice_new = np.asarray(std_atoms.cell.array, dtype=float)
        q23_row = self._fit_row_rotation(lattice_mapping, lattice_new)
        mapping23, _, _ = self._build_rotated_atom_mapping(
            mapping_atoms,
            std_atoms,
            q23_row,
            tol=max(5.0e-4, map_tol * 10.0),
        )
        mag_mapping = self._compose_magnetic_moments_two_stage(mapping12, mapping23, mag)

        std_mag = np.zeros((len(std_atoms), 3), dtype=float)
        for atom_idx, vec in enumerate(mag_mapping):
            std_mag[int(atom_idx), :] = np.asarray(vec, dtype=float)
        std_mag = std_mag @ np.asarray(q23_row, dtype=float)

        std_unitary_sym_data, _ = self._get_magnetic_unitary_symmetry_data(
            std_atoms,
            std_mag,
            symm_prec,
        )

        return detected_unitary_group, std_atoms, mapping_atoms, unitary_sym_data, std_unitary_sym_data, std_mag

    @classmethod
    def _standardize_cell_without_deprecation_warning(cls, spglib_module, cell, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=cls._SPGLIB_OLD_ERROR_HANDLING_WARNING,
                category=DeprecationWarning,
                module=r"spglib\.cell",
            )
            return spglib_module.standardize_cell(cell, **kwargs)

    @staticmethod
    def _symmetry_dict_from_dataset(dataset):
        return {
            "rotations": np.asarray(dataset.rotations, dtype=int),
            "translations": np.asarray(dataset.translations, dtype=float),
            "equivalent_atoms": np.asarray(dataset.equivalent_atoms, dtype=int),
        }

    @staticmethod
    def _symmetry_dict_from_magnetic_symmetry(symmetry: dict, *, unitary_only: bool = True):
        rotations = np.asarray(symmetry["rotations"], dtype=int)
        translations = np.asarray(symmetry["translations"], dtype=float)
        time_reversals = np.asarray(symmetry.get("time_reversals", np.zeros(len(rotations), dtype=bool)), dtype=bool)
        if unitary_only:
            mask = ~time_reversals
            rotations = rotations[mask]
            translations = translations[mask]
            time_reversals = time_reversals[mask]
        return {
            "rotations": rotations,
            "translations": translations,
            "time_reversals": time_reversals,
            "equivalent_atoms": np.asarray(symmetry.get("equivalent_atoms", np.arange(0)), dtype=int),
        }

    @staticmethod
    def _get_magnetic_unitary_symmetry_data(atoms: Atoms, magnetic_moments: np.ndarray, symm_prec: float):
        try:
            import spglib
        except ModuleNotFoundError as exc:
            raise ImportError("spglib is required for CHARACTER magnetic symmetry detection.") from exc

        mag = np.asarray(magnetic_moments, dtype=float)
        if mag.shape != (len(atoms), 3):
            raise ValueError(
                f"Magnetic moments must have shape ({len(atoms)}, 3), got {mag.shape}."
            )
        cell = (
            np.asarray(atoms.cell.array, dtype=float),
            np.asarray(atoms.get_scaled_positions(), dtype=float),
            np.asarray(atoms.get_atomic_numbers(), dtype=int),
            mag,
        )
        symmetry = spglib.get_magnetic_symmetry(
            cell,
            symprec=symm_prec,
            mag_symprec=symm_prec,
            is_axial=True,
            with_time_reversal=True,
            _throw=True,
        )
        if symmetry is None:
            raise ValueError("spglib failed to get magnetic symmetry operations for STRU.")

        unitary_data = SymmStructureAnalyzer._symmetry_dict_from_magnetic_symmetry(symmetry, unitary_only=True)
        if unitary_data["rotations"].size == 0:
            raise ValueError("spglib magnetic symmetry returned no unitary operations.")

        sg_type = spglib.get_spacegroup_type_from_symmetry(
            unitary_data["rotations"],
            unitary_data["translations"],
            lattice=np.asarray(atoms.cell.array, dtype=float),
            symprec=symm_prec,
        )
        unitary_group = int(sg_type.number) if sg_type is not None else 0
        return unitary_data, unitary_group

    @staticmethod
    def _get_symmetry_data(atoms: Atoms, symm_prec: float):
        try:
            import spglib
        except ModuleNotFoundError as exc:
            raise ImportError("spglib is required for CHARACTER symmetry detection.") from exc

        cell = (
            np.asarray(atoms.cell.array, dtype=float),
            np.asarray(atoms.get_scaled_positions(), dtype=float),
            np.asarray(atoms.get_atomic_numbers(), dtype=int),
        )
        sym_dataset = spglib.get_symmetry_dataset(cell, symprec=symm_prec, _throw=True)
        if sym_dataset is None:
            raise ValueError("spglib failed to get symmetry operations for source STRU.")
        return SymmStructureAnalyzer._symmetry_dict_from_dataset(sym_dataset)

    @staticmethod
    def _resolve_spacegroup(requested_group, detected_group):
        if requested_group == "auto":
            return int(detected_group)
        if int(requested_group) != int(detected_group):
            raise ValueError(
                f"CHARACTER.group mismatch: input={requested_group}, detected={detected_group}."
            )
        return int(requested_group)

    @classmethod
    def _operation_in_set(cls, operation: SymmetryOperation, candidates: list[SymmetryOperation]) -> bool:
        for candidate in candidates:
            if cls._rotation_match(operation.rotation, candidate.rotation) and cls._translation_match(
                operation.translation, candidate.translation
            ):
                return True
        return False

    @classmethod
    def _requested_group_is_unitary_subgroup(
        cls,
        requested_operations: list[SymmetryOperation],
        unitary_operations: list[SymmetryOperation],
    ) -> bool:
        return all(cls._operation_in_set(operation, unitary_operations) for operation in requested_operations)

    @staticmethod
    def _build_atom_mapping(original_atoms: Atoms, std_atoms: Atoms, tol: float):
        original_pos = canonicalize_fractional_coordinates(
            np.asarray(original_atoms.get_scaled_positions(wrap=False), dtype=float),
            tol=tol,
        )
        original_num = np.asarray(original_atoms.get_atomic_numbers(), dtype=int)
        original_lattice = np.asarray(original_atoms.cell.array, dtype=float)
        std_pos = canonicalize_fractional_coordinates(
            np.asarray(std_atoms.get_scaled_positions(wrap=False), dtype=float),
            tol=tol,
        )
        std_num = np.asarray(std_atoms.get_atomic_numbers(), dtype=int)
        std_lattice = np.asarray(std_atoms.cell.array, dtype=float)
        inv_std_lattice = np.linalg.inv(std_lattice)

        mapping = []
        for i, (old_pos, old_num) in enumerate(zip(original_pos, original_num, strict=True)):
            cart_old = np.asarray(old_pos, dtype=float) @ original_lattice
            # Keep the unwrapped fractional coordinate in the standardized basis so
            # integer cell-shift information is preserved for supercell->primitive mapping.
            old_pos_in_std_basis = cart_old @ inv_std_lattice
            best = None
            for j, (new_pos, new_num) in enumerate(zip(std_pos, std_num, strict=True)):
                if old_num != new_num:
                    continue
                shift = np.rint(old_pos_in_std_basis - new_pos).astype(int)
                residual = old_pos_in_std_basis - (new_pos + shift)
                residual -= np.rint(residual)
                err = np.max(np.abs(residual))
                if best is None or err < best[0]:
                    best = (err, j, shift)

            if best is None or best[0] > tol:
                raise ValueError(f"Failed to map atom {i + 1} to standardized primitive structure.")

            mapping.append((i, best[1], best[2]))

        return mapping

    @staticmethod
    def _fit_row_rotation(lattice_from: np.ndarray, lattice_to: np.ndarray) -> np.ndarray:
        """Solve row-vector Procrustes rotation: lattice_from @ Q ~= lattice_to."""
        h = np.asarray(lattice_from, dtype=float).T @ np.asarray(lattice_to, dtype=float)
        u, _, vt = np.linalg.svd(h)
        q = u @ vt
        if np.linalg.det(q) < 0.0:
            u[:, -1] *= -1.0
            q = u @ vt
        return q

    @staticmethod
    def _round_integer_matrix(matrix: np.ndarray, tol: float = 1.0e-4) -> np.ndarray:
        raw = np.asarray(matrix, dtype=float)
        rounded = np.rint(raw).astype(int)
        if float(np.max(np.abs(raw - rounded))) > tol:
            raise ValueError(
                f"Failed to round matrix to integer form within tolerance, max_err={float(np.max(np.abs(raw - rounded)))}."
            )
        return rounded

    @staticmethod
    def _build_rotated_atom_mapping(
        source_atoms: Atoms,
        target_atoms: Atoms,
        cart_rotation_row: np.ndarray,
        tol: float,
    ) -> tuple[list[tuple[int, int, np.ndarray]], np.ndarray, float]:
        """Map source->target when source Cartesian vectors are first right-multiplied by cart_rotation_row."""
        source_cart = np.asarray(source_atoms.positions, dtype=float) @ np.asarray(cart_rotation_row, dtype=float)
        target_cart = np.asarray(target_atoms.positions, dtype=float)
        source_numbers = np.asarray(source_atoms.get_atomic_numbers(), dtype=int)
        target_numbers = np.asarray(target_atoms.get_atomic_numbers(), dtype=int)
        target_lattice = np.asarray(target_atoms.cell.array, dtype=float)
        inv_target_lattice = np.linalg.inv(target_lattice)

        n_source = int(len(source_numbers))
        n_target = int(len(target_numbers))
        if n_source != n_target:
            raise ValueError("Rotated atom mapping expects equal atom counts between source and target primitive cells.")

        ref_old = 0
        ref_species = int(source_numbers[ref_old])
        ref_candidates = [idx for idx, z in enumerate(target_numbers) if int(z) == ref_species]
        if not ref_candidates:
            raise ValueError("Failed to find a reference target atom with the same species for rotated mapping.")

        best = None
        for ref_target in ref_candidates:
            raw_ref = source_cart[ref_old] - target_cart[ref_target]
            big = 1.0e8
            cost = np.full((n_source, n_target), big, dtype=float)
            shift_table: dict[tuple[int, int], np.ndarray] = {}

            for i_old in range(n_source):
                z_old = int(source_numbers[i_old])
                for i_new in range(n_target):
                    if z_old != int(target_numbers[i_new]):
                        continue
                    raw = source_cart[i_old] - target_cart[i_new]
                    delta = raw - raw_ref
                    shift = np.rint(delta @ inv_target_lattice).astype(int)
                    residual = delta - shift @ target_lattice
                    err = float(np.max(np.abs(residual)))
                    cost[i_old, i_new] = err
                    shift_table[(i_old, i_new)] = shift

            row_ind, col_ind = linear_sum_assignment(cost)
            if not np.array_equal(row_ind, np.arange(n_source)):
                raise ValueError("Unexpected row assignment shape in rotated atom mapping.")

            max_err = float(np.max(cost[row_ind, col_ind]))
            if best is None or max_err < best["max_err"]:
                best = {
                    "assignment": [int(idx) for idx in col_ind.tolist()],
                    "shift_table": shift_table,
                    "max_err": max_err,
                    "ref_target": int(ref_target),
                    "raw_ref": np.asarray(raw_ref, dtype=float),
                }

        if best is None or best["max_err"] > tol:
            raise ValueError(f"Failed rotated atom mapping, max_err={None if best is None else best['max_err']}.")

        mapping = []
        for old_idx, new_idx in enumerate(best["assignment"]):
            shift = np.asarray(best["shift_table"][(old_idx, new_idx)], dtype=int)
            mapping.append((int(old_idx), int(new_idx), shift))

        translation = target_cart[int(best["ref_target"])] - source_cart[ref_old]
        return mapping, np.asarray(translation, dtype=float), float(best["max_err"])

    @staticmethod
    def _compose_two_stage_mapping(
        mapping12: list[tuple[int, int, np.ndarray]],
        mapping23: list[tuple[int, int, np.ndarray]],
        b23_integer: np.ndarray,
    ) -> list[tuple[int, int, np.ndarray]]:
        map23_by_old = {int(old_idx): (int(new_idx), np.asarray(shift, dtype=int)) for old_idx, new_idx, shift in mapping23}
        composed: list[tuple[int, int, np.ndarray]] = []

        for old_idx, mid_idx, shift12 in mapping12:
            if int(mid_idx) not in map23_by_old:
                raise ValueError(f"Failed to compose mapping: missing stage-2 atom for mid atom {int(mid_idx) + 1}.")
            new_idx, shift23 = map23_by_old[int(mid_idx)]
            shift13 = np.rint(np.asarray(shift12, dtype=float) @ np.asarray(b23_integer, dtype=float) + shift23).astype(int)
            composed.append((int(old_idx), int(new_idx), shift13))

        return composed

    @staticmethod
    def _compose_magnetic_moments_two_stage(
        mapping12: list[tuple[int, int, np.ndarray]],
        mapping23: list[tuple[int, int, np.ndarray]],
        magnetic_moments: np.ndarray,
    ) -> np.ndarray:
        mag = np.asarray(magnetic_moments, dtype=float)
        if mag.ndim != 2 or mag.shape[1] != 3:
            raise ValueError(f"Magnetic moments must have shape (N, 3), got {mag.shape}.")

        map12_by_old = {int(old_idx): int(mid_idx) for old_idx, mid_idx, _ in mapping12}
        map23_by_mid = {int(mid_idx): int(new_idx) for mid_idx, new_idx, _ in mapping23}
        result = np.zeros((len(mapping23), 3), dtype=float)

        for old_idx in range(mag.shape[0]):
            if int(old_idx) not in map12_by_old:
                raise ValueError(f"Missing stage-1 atom mapping for magnetic moment atom {int(old_idx) + 1}.")
            mid_idx = map12_by_old[int(old_idx)]
            if int(mid_idx) not in map23_by_mid:
                raise ValueError(f"Missing stage-2 atom mapping for magnetic moment atom {int(old_idx) + 1}.")
            new_idx = map23_by_mid[int(mid_idx)]
            result[int(new_idx), :] = mag[int(old_idx), :]

        return result

    @staticmethod
    def _detect_lattice_change(source_atoms: Atoms, std_atoms: Atoms, tol: float) -> bool:
        delta = np.asarray(source_atoms.cell.array, dtype=float) - np.asarray(std_atoms.cell.array, dtype=float)
        return bool(np.max(np.abs(delta)) > tol)

    @staticmethod
    def _mapping_is_permutation_only(atom_mapping: list[tuple[int, int, np.ndarray]], tol: float) -> bool:
        del tol
        return all(np.all(np.asarray(shift, dtype=int) == 0) for _, _, shift in atom_mapping)

    @staticmethod
    def _match_by_global_fractional_origin_shift(source_atoms: Atoms, target_atoms: Atoms, tol: float):
        """Return whether two same-lattice structures differ only by a global origin shift."""
        source_lattice = np.asarray(source_atoms.cell.array, dtype=float)
        target_lattice = np.asarray(target_atoms.cell.array, dtype=float)
        if len(source_atoms) != len(target_atoms):
            return False, np.zeros(3, dtype=float), float("inf")
        if float(np.max(np.abs(source_lattice - target_lattice))) > tol:
            return False, np.zeros(3, dtype=float), float("inf")

        source_numbers = np.asarray(source_atoms.get_atomic_numbers(), dtype=int)
        target_numbers = np.asarray(target_atoms.get_atomic_numbers(), dtype=int)
        source_pos = canonicalize_fractional_coordinates(
            np.asarray(source_atoms.get_scaled_positions(wrap=True), dtype=float),
            tol=tol,
        )
        target_pos = canonicalize_fractional_coordinates(
            np.asarray(target_atoms.get_scaled_positions(wrap=True), dtype=float),
            tol=tol,
        )
        if len(source_numbers) == 0:
            return True, np.zeros(3, dtype=float), 0.0

        ref_species = int(source_numbers[0])
        candidate_targets = [idx for idx, number in enumerate(target_numbers) if int(number) == ref_species]
        if not candidate_targets:
            return False, np.zeros(3, dtype=float), float("inf")

        best_shift = np.zeros(3, dtype=float)
        best_err = float("inf")
        big = 1.0e8
        for target_ref in candidate_targets:
            shift = np.asarray(source_pos[0] - target_pos[target_ref], dtype=float)
            cost = np.full((len(source_numbers), len(target_numbers)), big, dtype=float)
            for i, source_number in enumerate(source_numbers):
                for j, target_number in enumerate(target_numbers):
                    if int(source_number) != int(target_number):
                        continue
                    residual = source_pos[i] - (target_pos[j] + shift)
                    residual -= np.rint(residual)
                    cost[i, j] = float(np.max(np.abs(residual)))

            row_ind, col_ind = linear_sum_assignment(cost)
            if not np.array_equal(row_ind, np.arange(len(source_numbers))):
                continue
            max_err = float(np.max(cost[row_ind, col_ind]))
            if max_err < best_err:
                best_err = max_err
                best_shift = shift - np.floor(shift)

        return bool(best_err <= tol), best_shift, best_err

    @staticmethod
    def _lattice_transform_fractional(lattice_old: np.ndarray, lattice_new: np.ndarray) -> np.ndarray:
        return np.asarray(lattice_old, dtype=float) @ np.linalg.inv(np.asarray(lattice_new, dtype=float))

    @staticmethod
    def _canonicalize_kpoints(kpoints_direct: np.ndarray | None, lattice_transform_fractional: np.ndarray) -> np.ndarray:
        if kpoints_direct is None:
            return np.zeros((0, 3), dtype=float)
        kpoints = np.asarray(kpoints_direct, dtype=float)
        if kpoints.size == 0:
            return np.zeros((0, 3), dtype=float)
        return kpoints @ np.linalg.inv(np.asarray(lattice_transform_fractional, dtype=float)).T

    @staticmethod
    def _atom_mapping_to_dicts(atom_mapping: list[tuple[int, int, np.ndarray]]) -> list[dict]:
        mapping = []
        for old_idx, new_idx, shift in atom_mapping:
            mapping.append(
                {
                    "old_atom": int(old_idx),
                    "new_atom": int(new_idx),
                    "shift": np.asarray(shift, dtype=int),
                }
            )
        return mapping

    @staticmethod
    def _jsonable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {key: SymmStructureAnalyzer._jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [SymmStructureAnalyzer._jsonable(item) for item in value]
        return value

    @staticmethod
    def _next_nonempty_line(lines: list[str], start: int) -> tuple[int, str]:
        idx = int(start)
        while idx < len(lines):
            line = lines[idx].strip()
            if line:
                return idx, line
            idx += 1
        raise ValueError("Unexpected end of STRU file while parsing section.")

    @classmethod
    def _write_standardized_stru(cls, source_stru_path: Path, std_atoms: Atoms, target_stru_path: Path) -> None:
        lines = source_stru_path.read_text(encoding="utf-8").splitlines()
        headers = ["ATOMIC_SPECIES", "NUMERICAL_ORBITAL", "LATTICE_CONSTANT", "LATTICE_VECTORS", "ATOMIC_POSITIONS"]

        section_index: dict[str, int] = {}
        for header in headers:
            for idx, raw in enumerate(lines):
                if raw.strip().upper() == header:
                    section_index[header] = idx
                    break
            else:
                raise ValueError(f"Missing '{header}' section in source STRU: {source_stru_path}")

        species_lines = [
            raw.strip()
            for raw in lines[section_index["ATOMIC_SPECIES"] + 1 : section_index["NUMERICAL_ORBITAL"]]
            if raw.strip()
        ]
        orbital_lines = [
            raw.strip()
            for raw in lines[section_index["NUMERICAL_ORBITAL"] + 1 : section_index["LATTICE_CONSTANT"]]
            if raw.strip()
        ]

        _, lattice_constant_line = cls._next_nonempty_line(lines, section_index["LATTICE_CONSTANT"] + 1)
        lattice_constant_bohr = float(lattice_constant_line.split()[0])
        lattice_scale_ang = lattice_constant_bohr / float(Ang_to_Bohr)
        if lattice_scale_ang <= 0.0:
            raise ValueError(f"Invalid lattice constant in source STRU: {source_stru_path}")

        species_order = [line.split()[0] for line in species_lines]
        position_meta: dict[str, str] = {}
        idx, coord_type = cls._next_nonempty_line(lines, section_index["ATOMIC_POSITIONS"] + 1)
        idx += 1
        while idx < len(lines):
            try:
                idx, species_name = cls._next_nonempty_line(lines, idx)
            except ValueError:
                break
            idx, mag_line = cls._next_nonempty_line(lines, idx + 1)
            idx, atom_count_line = cls._next_nonempty_line(lines, idx + 1)
            atom_count = int(atom_count_line.split()[0])
            position_meta[species_name] = mag_line
            idx += atom_count + 1

        scaled_positions = np.asarray(std_atoms.get_scaled_positions(), dtype=float)
        scaled_positions -= np.floor(scaled_positions)
        symbols = list(std_atoms.get_chemical_symbols())
        lattice_vectors = np.asarray(std_atoms.cell.array, dtype=float) / lattice_scale_ang

        with target_stru_path.open("w", encoding="utf-8") as handle:
            handle.write("ATOMIC_SPECIES\n")
            for line in species_lines:
                handle.write(f"{line}\n")
            handle.write("\nNUMERICAL_ORBITAL\n")
            for line in orbital_lines:
                handle.write(f"{line}\n")
            handle.write("\nLATTICE_CONSTANT\n")
            handle.write(f"{lattice_constant_bohr:.7f}\n")
            handle.write("\nLATTICE_VECTORS\n")
            for row in lattice_vectors:
                handle.write(f"{row[0]:18.10f}{row[1]:18.10f}{row[2]:18.10f}\n")
            handle.write("\nATOMIC_POSITIONS\n")
            handle.write(f"{coord_type}\n\n")

            for species in species_order:
                indices = [atom_index for atom_index, symbol in enumerate(symbols) if symbol == species]
                if not indices:
                    continue
                handle.write(f"{species}\n")
                handle.write(f"{position_meta.get(species, '0.0')}\n")
                handle.write(f"{len(indices)}\n")
                for atom_index in indices:
                    frac = scaled_positions[atom_index]
                    handle.write(
                        f"{frac[0]:16.10f}{frac[1]:16.10f}{frac[2]:16.10f} 0 0 0\n"
                    )
                handle.write("\n")

    def _finalize_standardization_result(
        self,
        *,
        detected_group: int,
        resolved_group: int,
        lattice_changed: bool,
        atom_permutation_only: bool,
        source_stru: str,
        source_hr: str,
        source_sr: str,
        atom_mapping: list[dict],
        lattice_old: np.ndarray,
        lattice_new: np.ndarray,
        lattice_transform_fractional: np.ndarray,
        xyz_axis_transform_cartesian: np.ndarray,
        rebuild_reason: str,
        full_matrix_from_hermitian: bool = True,
    ) -> dict:
        need_rebuild_hs = bool(lattice_changed or not atom_permutation_only)
        target_stru = "STRU-symm" if need_rebuild_hs else str(source_stru)
        target_hr = (
            str((self._output_path / "data-HR-sparse_SPIN0-symm.csr").resolve())
            if need_rebuild_hs
            else str(source_hr)
        )
        target_sr = (
            str((self._output_path / "data-SR-sparse_SPIN0-symm.csr").resolve())
            if need_rebuild_hs
            else str(source_sr)
        )

        result = StandardizationResult(
            resolved_group=int(resolved_group),
            detected_group=int(detected_group),
            structure_changed=bool(need_rebuild_hs),
            lattice_changed=bool(lattice_changed),
            atom_permutation_only=bool(atom_permutation_only),
            need_rebuild_hs=bool(need_rebuild_hs),
            source_stru=str(source_stru),
            target_stru=target_stru,
            source_hr=str(source_hr),
            source_sr=str(source_sr),
            target_hr=target_hr,
            target_sr=target_sr,
            lattice_old=np.asarray(lattice_old, dtype=float),
            lattice_new=np.asarray(lattice_new, dtype=float),
            lattice_transform_fractional=np.asarray(lattice_transform_fractional, dtype=float),
            xyz_axis_transform_cartesian=np.asarray(xyz_axis_transform_cartesian, dtype=float),
            atom_mapping=atom_mapping,
            rebuild_reason=str(rebuild_reason),
            full_matrix_from_hermitian=bool(full_matrix_from_hermitian),
        )
        return {key: self._jsonable(value) for key, value in result.__dict__.items()}

    @staticmethod
    def _operation_order(rot: np.ndarray, max_order: int = 12) -> int:
        cur = np.eye(3, dtype=int)
        for n in range(1, max_order + 1):
            cur = cur @ rot
            if np.array_equal(cur, np.eye(3, dtype=int)):
                return n
        return max_order

    @staticmethod
    def _build_op_symbol_and_desc(rotation: np.ndarray):
        det = int(round(np.linalg.det(rotation)))
        proper = -rotation if det < 0 else rotation
        order = SymmStructureAnalyzer._operation_order(np.asarray(proper, dtype=int))

        if det > 0:
            if order == 1:
                return "E", "unity op."
            return f"C{order}", f"{int(round(360 / order))}-degree rotation"

        if order == 1:
            return "I", "inverse op."
        return f"IC{order}", f"{int(round(360 / order))}-degree rotation times inversion"

    @staticmethod
    def _matrix_to_euler_zyz(rot: np.ndarray, tol: float = 1.0e-10) -> np.ndarray:
        mat = np.asarray(rot, dtype=float)
        c = float(np.clip(mat[2, 2], -1.0, 1.0))
        beta = float(np.arccos(c))

        if abs(np.sin(beta)) > tol:
            alpha = float(np.arctan2(mat[1, 2], mat[0, 2]))
            gamma = float(np.arctan2(mat[2, 1], -mat[2, 0]))
            return np.array([alpha, beta, gamma], dtype=float)

        if c > 0.0:
            alpha = float(np.arctan2(mat[1, 0], mat[0, 0]))
            return np.array([alpha, 0.0, 0.0], dtype=float)

        alpha = float(np.arctan2(mat[0, 1], mat[0, 0]))
        return np.array([alpha, np.pi, 0.0], dtype=float)

    @staticmethod
    def _build_symmetry_operations(std_atoms: Atoms, sym_data) -> list[SymmetryOperation]:
        lattice = np.asarray(std_atoms.cell.array, dtype=float)
        col_lattice = lattice.T
        inv_col_lattice = np.linalg.inv(col_lattice)

        ops: list[SymmetryOperation] = []

        for rot, trans in zip(sym_data["rotations"], sym_data["translations"], strict=True):
            rot_i = np.asarray(rot, dtype=int)
            trans_f = np.asarray(trans, dtype=float)

            inv_rot = np.asarray(np.rint(np.linalg.inv(rot_i)).astype(int), dtype=int)

            cart_rot = col_lattice @ rot_i @ inv_col_lattice
            proper_cart = -cart_rot if np.linalg.det(cart_rot) < 0 else cart_rot

            euler = SymmStructureAnalyzer._matrix_to_euler_zyz(proper_cart)

            spin = spin_half_matrix_from_cartesian_rotation(proper_cart)
            symbol, desc = SymmStructureAnalyzer._build_op_symbol_and_desc(rot_i)
            axis, _, _ = axis_angle_from_cartesian_rotation(proper_cart)
            axis = SymmStructureAnalyzer._canonicalize_axis(axis)

            ops.append(
                SymmetryOperation(
                    rotation=rot_i,
                    translation=trans_f,
                    inverse_rotation=inv_rot,
                    cart_rotation=cart_rot,
                    euler_zyz=euler,
                    spin_matrix=spin,
                    symbol=symbol,
                    description=desc,
                    axis=axis,
                )
            )

        return ops

    @staticmethod
    def _phase_factor(k_direct: np.ndarray, tau_direct: np.ndarray) -> complex:
        phase = -2.0 * np.pi * float(np.dot(k_direct, tau_direct))
        return np.exp(1j * phase)

    @staticmethod
    def _format_complex(z: complex, tol: float = 5.0e-6) -> str:
        real = 0.0 if abs(float(np.real(z))) < tol else float(np.real(z))
        imag = 0.0 if abs(float(np.imag(z))) < tol else float(np.imag(z))
        return f"{real:.2f}{imag:+.2f}i"

    @staticmethod
    def _format_spin_row(u00: complex, u01: complex) -> str:
        return f" ({u00.real:6.3f}{u00.imag:6.3f})({u01.real:6.3f}{u01.imag:6.3f})"

    @staticmethod
    def _canonicalize_axis(axis: np.ndarray, tol: float = 1.0e-8) -> np.ndarray:
        axis = np.asarray(axis, dtype=float)
        norm = float(np.linalg.norm(axis))
        if norm <= tol:
            return np.array([1.0, 0.0, 0.0], dtype=float)
        axis = axis / norm
        for value in axis:
            if abs(value) > tol:
                if value < 0.0:
                    axis = -axis
                break
        axis[np.abs(axis) < tol] = 0.0
        return axis

    @classmethod
    def _signed_rotation_angle(cls, cart_rotation: np.ndarray, axis: np.ndarray) -> float:
        axis = cls._canonicalize_axis(axis)
        ref = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(float(np.dot(ref, axis))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=float)
        u = ref - axis * float(np.dot(ref, axis))
        u_norm = float(np.linalg.norm(u))
        if u_norm <= 1.0e-12:
            return 0.0
        u = u / u_norm
        v = np.cross(axis, u)
        ru = np.asarray(cart_rotation, dtype=float) @ u
        return float(np.arctan2(np.dot(ru, v), np.dot(ru, u)))

    @staticmethod
    def _rotation_order_from_symbol(symbol: str) -> int:
        if symbol == "E" or symbol == "I":
            return 1
        if symbol.startswith("IC"):
            return int(symbol[2:])
        if symbol.startswith("C"):
            return int(symbol[1:])
        return 0

    @classmethod
    def _operation_sort_key(cls, op: SymmetryOperation):
        symbol = op.symbol
        if symbol == "E":
            family = 0
        elif symbol.startswith("C"):
            family = 1
        elif symbol == "I":
            family = 2
        elif symbol.startswith("IC"):
            family = 3
        else:
            family = 4

        order = cls._rotation_order_from_symbol(symbol)
        axis = cls._canonicalize_axis(op.axis)
        azimuth = float(np.arctan2(axis[1], axis[0]))
        signed_angle = cls._signed_rotation_angle(op.cart_rotation, axis)
        axis_bucket = 0 if abs(axis[2]) > 0.9 else 1

        return (
            family,
            -order,
            axis_bucket,
            -signed_angle if order > 2 else 0.0,
            -azimuth if order == 2 else 0.0,
            round(axis[0], 8),
            round(axis[1], 8),
            round(axis[2], 8),
            tuple(np.asarray(op.rotation, dtype=int).reshape(-1).tolist()),
        )

    @classmethod
    def _sort_operations_irvsp_like(cls, operations: list[SymmetryOperation]) -> list[SymmetryOperation]:
        return sorted(operations, key=cls._operation_sort_key)

    @staticmethod
    def _rotation_match(a: np.ndarray, b: np.ndarray) -> bool:
        return bool(np.array_equal(np.asarray(a, dtype=int), np.asarray(b, dtype=int)))

    @staticmethod
    def _translation_match(a: np.ndarray, b: np.ndarray, tol: float = 1e-6) -> bool:
        diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
        diff -= np.rint(diff)
        return float(np.max(np.abs(diff))) <= tol

    @staticmethod
    def _translation_diff_mod_lattice(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
        return diff - np.rint(diff)

    @staticmethod
    def _format_translation(vec: np.ndarray) -> str:
        arr = np.asarray(vec, dtype=float)
        return "(" + ", ".join(f"{x:.8f}" for x in arr) + ")"

    @staticmethod
    def _database_operation_candidates(db_op, db: KLittleGroupsDB):
        candidates = [
            (
                np.asarray(db_op.rotation, dtype=int),
                np.asarray(db_op.translation, dtype=float),
                "database",
            )
        ]

        kc2p = getattr(db, "kc2p", None)
        if kc2p is None:
            return candidates

        transform = np.asarray(kc2p, dtype=float)
        if np.allclose(transform, np.eye(3), atol=1.0e-12):
            return candidates

        inv_transform = np.linalg.inv(transform)
        rotation = inv_transform @ np.asarray(db_op.rotation, dtype=float) @ transform
        rounded_rotation = np.rint(rotation).astype(int)
        if not np.allclose(rotation, rounded_rotation, atol=1.0e-8):
            return candidates

        translation = inv_transform @ np.asarray(db_op.translation, dtype=float)
        translation -= np.floor(translation)
        candidates.append((rounded_rotation, translation, "database-conventional-to-primitive"))

        # Some centered-lattice kLittleGroups tables differ from spglib only by
        # how screw/glide translations are attached to symmetry-equivalent
        # rotations in the primitive setting.  Keep strict matching, but allow
        # the translation candidates from other database operations that carry
        # the same transformed rotation.
        for other in getattr(db, "symops", []):
            other_rotation = inv_transform @ np.asarray(other.rotation, dtype=float) @ transform
            other_rounded = np.rint(other_rotation).astype(int)
            if not np.allclose(other_rotation, other_rounded, atol=1.0e-8):
                continue
            if not np.array_equal(other_rounded, rounded_rotation):
                continue
            other_translation = inv_transform @ np.asarray(other.translation, dtype=float)
            other_translation -= np.floor(other_translation)
            if not any(
                np.array_equal(other_rounded, np.asarray(rot, dtype=int))
                and np.allclose(other_translation, np.asarray(tau, dtype=float), atol=1.0e-12)
                for rot, tau, _label in candidates
            ):
                candidates.append(
                    (
                        other_rounded,
                        other_translation,
                        "database-conventional-to-primitive-same-rotation",
                    )
                )
        return candidates

    @classmethod
    def _solve_origin_shift_from_operations(
        cls,
        operations: list[SymmetryOperation],
        db: KLittleGroupsDB,
        tol: float = 1.0e-6,
    ) -> np.ndarray | None:
        n_target = min(len(operations), db.doubnum // 2, len(getattr(db, "symops", [])))
        if n_target == 0:
            return np.zeros(3, dtype=float)

        matched_rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        used: set[int] = set()
        for db_idx in range(n_target):
            db_op = db.symops[db_idx]
            candidates = cls._database_operation_candidates(db_op, db)
            match = None
            match_db_tau = None
            match_rot = None
            for db_rotation, db_translation, _label in candidates:
                for op_idx, op in enumerate(operations):
                    if op_idx in used:
                        continue
                    if cls._rotation_match(op.rotation, db_rotation):
                        match = op
                        match_rot = np.asarray(op.rotation, dtype=float)
                        match_db_tau = np.asarray(db_translation, dtype=float)
                        used.add(op_idx)
                        break
                if match is not None:
                    break
            if match is None or match_db_tau is None or match_rot is None:
                return None
            matched_rows.append((match_rot, np.asarray(match.translation, dtype=float), match_db_tau))

        grid = np.arange(24, dtype=float) / 24.0
        for sx in grid:
            for sy in grid:
                for sz in grid:
                    shift = np.array([sx, sy, sz], dtype=float)
                    ok = True
                    for rot, tau_std, tau_db in matched_rows:
                        # Atomic positions are stored and shifted as row vectors.
                        # Under x_new = x_old + shift, the Seitz translation in
                        # row-vector convention transforms as
                        # tau_new = tau_old + shift @ (I - R^T).
                        shifted_tau = tau_std + shift @ (np.eye(3, dtype=float) - rot.T)
                        diff = shifted_tau - tau_db
                        diff -= np.rint(diff)
                        if float(np.max(np.abs(diff))) > tol:
                            ok = False
                            break
                    if ok:
                        return shift
        return None

    @classmethod
    def _solve_conventional_origin_shift_from_database(
        cls,
        conventional_operations: list[SymmetryOperation],
        db: KLittleGroupsDB,
        tol: float = 1.0e-6,
    ) -> np.ndarray | None:
        n_target = min(len(getattr(db, "symops", [])), db.doubnum // 2)
        if n_target == 0:
            return np.zeros(3, dtype=float)

        grid = np.arange(24, dtype=float) / 24.0
        identity = np.eye(3, dtype=float)
        for sx in grid:
            for sy in grid:
                for sz in grid:
                    shift = np.array([sx, sy, sz], dtype=float)
                    used: set[int] = set()
                    ok = True
                    for db_idx in range(n_target):
                        db_op = db.symops[db_idx]
                        match_index = None
                        for op_idx, op in enumerate(conventional_operations):
                            if op_idx in used:
                                continue
                            if not cls._rotation_match(op.rotation, db_op.rotation):
                                continue
                            shifted_tau = (
                                np.asarray(op.translation, dtype=float)
                                + shift @ (identity - np.asarray(op.rotation, dtype=float).T)
                            )
                            if cls._translation_match(shifted_tau, db_op.translation, tol=tol):
                                match_index = op_idx
                                break
                        if match_index is None:
                            ok = False
                            break
                        used.add(match_index)
                    if ok:
                        return shift
        return None

    @staticmethod
    def _conventional_origin_shift_to_primitive(shift_conventional: np.ndarray, db: KLittleGroupsDB) -> np.ndarray:
        transform = getattr(db, "kc2p", None)
        if transform is None:
            return np.asarray(shift_conventional, dtype=float)
        return np.linalg.inv(np.asarray(transform, dtype=float)) @ np.asarray(shift_conventional, dtype=float)

    def _reorder_operations_with_database(self, operations: list[SymmetryOperation], db: KLittleGroupsDB):
        n_target = min(len(operations), db.doubnum // 2)
        used: set[int] = set()
        order: list[int] = []
        warnings: list[str] = []

        for db_idx in range(n_target):
            db_op = db.symops[db_idx]
            matched = None
            matched_label = "database"
            candidates = self._database_operation_candidates(db_op, db)

            for db_rotation, db_translation, label in candidates:
                for i, op in enumerate(operations):
                    if i in used:
                        continue
                    if self._rotation_match(op.rotation, db_rotation) and self._translation_match(
                        op.translation, db_translation
                    ):
                        matched = i
                        matched_label = label
                        break
                if matched is not None:
                    break

            if matched is None:
                same_rotation = [
                    (i, op, db_translation, label)
                    for db_rotation, db_translation, label in candidates
                    for i, op in enumerate(operations)
                    if i not in used and self._rotation_match(op.rotation, db_rotation)
                ]
                if same_rotation:
                    candidate_lines = []
                    for i, op, db_translation, label in same_rotation:
                        diff = self._translation_diff_mod_lattice(op.translation, db_translation)
                        candidate_lines.append(
                            "  "
                            f"candidate #{i + 1}: tau={self._format_translation(op.translation)}, "
                            f"database_basis={label}, diff_mod_lattice={self._format_translation(diff)}"
                        )
                    raise ValueError(
                        "Failed to align symmetry operation with kLittleGroups database: "
                        f"op #{db_idx + 1} has matching rotation but translation mismatch.\n"
                        f"Database tau={self._format_translation(db_op.translation)}\n"
                        + "\n".join(candidate_lines)
                    )
                raise ValueError(
                    "Failed to align symmetry operation with kLittleGroups database: "
                    f"op #{db_idx + 1} has no matching rotation in spglib operations."
                )

            used.add(matched)
            order.append(matched)
            if matched_label != "database":
                warnings.append(f"op #{db_idx + 1} matched after {matched_label} basis conversion")

        for i in range(len(operations)):
            if i not in used:
                order.append(i)

        reordered = [operations[i] for i in order]
        return reordered, warnings

    def _database_alignment_summary(self, operations: list[SymmetryOperation], db: KLittleGroupsDB):
        n_target = min(len(operations), db.doubnum // 2)
        used: set[int] = set()
        details: list[str] = []
        matched_all = True

        for db_idx in range(n_target):
            db_op = db.symops[db_idx]
            candidates = self._database_operation_candidates(db_op, db)
            matched = None
            matched_label = "database"
            for db_rotation, db_translation, label in candidates:
                for i, op in enumerate(operations):
                    if i in used:
                        continue
                    if self._rotation_match(op.rotation, db_rotation) and self._translation_match(
                        op.translation, db_translation
                    ):
                        matched = i
                        matched_label = label
                        break
                if matched is not None:
                    break

            if matched is not None:
                used.add(matched)
                if matched_label == "database":
                    details.append(f"op #{db_idx + 1}: exact match")
                else:
                    details.append(f"op #{db_idx + 1}: match after {matched_label} basis conversion")
                continue

            matched_all = False
            same_rotation = [
                (i, op, db_translation, label)
                for db_rotation, db_translation, label in candidates
                for i, op in enumerate(operations)
                if i not in used and self._rotation_match(op.rotation, db_rotation)
            ]
            if same_rotation:
                for i, op, db_translation, label in same_rotation:
                    diff = self._translation_diff_mod_lattice(op.translation, db_translation)
                    details.append(
                        f"op #{db_idx + 1}: translation mismatch; "
                        f"spglib op #{i + 1}, basis={label}, "
                        f"spglib tau={self._format_translation(op.translation)}, "
                        f"database tau={self._format_translation(db_translation)}, "
                        f"diff_mod_lattice={self._format_translation(diff)}"
                    )
            else:
                details.append(f"op #{db_idx + 1}: no matching rotation in spglib operations")

        return matched_all, details

    def _reorder_operations_with_database_rotations(self, operations: list[SymmetryOperation], db: KLittleGroupsDB):
        n_target = min(len(operations), db.doubnum // 2)
        used: set[int] = set()
        order: list[int] = []
        warnings: list[str] = []

        for db_idx in range(n_target):
            db_op = db.symops[db_idx]
            matched = None
            matched_label = "database"
            matched_translation = np.asarray(db_op.translation, dtype=float)
            candidates = self._database_operation_candidates(db_op, db)

            for db_rotation, db_translation, label in candidates:
                for i, op in enumerate(operations):
                    if i in used:
                        continue
                    if self._rotation_match(op.rotation, db_rotation):
                        matched = i
                        matched_label = label
                        matched_translation = np.asarray(db_translation, dtype=float)
                        break
                if matched is not None:
                    break

            if matched is None:
                raise ValueError(
                    "Failed to align symmetry operation rotations with kLittleGroups database: "
                    f"op #{db_idx + 1} has no matching rotation in spglib operations."
                )

            used.add(matched)
            order.append(matched)
            op = operations[matched]
            if not self._translation_match(op.translation, matched_translation):
                diff = self._translation_diff_mod_lattice(op.translation, matched_translation)
                warnings.append(
                    "strict operation alignment failed; used rotation-only database order for "
                    f"op #{db_idx + 1} ({matched_label}). "
                    f"source tau={self._format_translation(op.translation)}, "
                    f"database tau={self._format_translation(matched_translation)}, "
                    f"diff_mod_lattice={self._format_translation(diff)}"
                )
            elif matched_label != "database":
                warnings.append(f"op #{db_idx + 1} matched by rotation after {matched_label} basis conversion")

        for i in range(len(operations)):
            if i not in used:
                order.append(i)

        reordered = [operations[i] for i in order]
        return reordered, warnings

    @staticmethod
    def _little_group_operation_indices(k_direct: np.ndarray, operations: list[SymmetryOperation], tol: float = 1e-5):
        k = np.asarray(k_direct, dtype=float)
        lkg = []
        for i, op in enumerate(operations, start=1):
            wkr = k @ op.inverse_rotation - k
            test = np.abs(np.rint(wkr) - wkr).sum()
            if test <= tol:
                lkg.append(i)
        return lkg

    @staticmethod
    def _cornwell_condition_satisfied(
        k_direct: np.ndarray,
        operations: list[SymmetryOperation],
        little_group_indices: list[int],
        tol: float = 1.0e-12,
    ) -> bool:
        """IRVSP CRWCND test for whether the little group is ordinary point-group-like."""
        k = np.asarray(k_direct, dtype=float).reshape(3)
        for i_one_based in little_group_indices:
            op_i = operations[int(i_one_based) - 1]
            rot_i = np.asarray(op_i.rotation, dtype=int)
            for j_one_based in little_group_indices:
                tau_j = np.asarray(operations[int(j_one_based) - 1].translation, dtype=float).reshape(3)
                rt = rot_i @ tau_j - tau_j
                phase = -2.0 * np.pi * float(np.dot(k, rt))
                diff = (np.cos(phase) - 1.0) ** 2 + np.sin(phase) ** 2
                if diff > tol:
                    return False
        return True

    @classmethod
    def _match_operation_by_seitz(
        cls,
        rotation: np.ndarray,
        translation: np.ndarray,
        operations: list[SymmetryOperation],
    ) -> int | None:
        target_rotation = np.asarray(rotation, dtype=int)
        target_translation = np.asarray(translation, dtype=float)
        for idx, op in enumerate(operations):
            if not np.array_equal(np.asarray(op.rotation, dtype=int), target_rotation):
                continue
            if cls._translation_match(op.translation, target_translation):
                return idx
        return None

    @classmethod
    def _representative_table_operation_indices(
        cls,
        operations: list[SymmetryOperation],
        active_operation_indices: list[int],
        rotation_index: int,
    ) -> list[int]:
        """Map current-k little-group operations to representative-k table columns.

        resolve_kpoint_from_star matches a user k point to a database representative
        by k_rep = k @ S, where S is the inverse-rotation matrix of operation
        rotation_index.  The D(k,g) matrices still use the current-k operations,
        while irrep tables are tabulated at k_rep.  Therefore the table column for
        {R_g|t_g} is the full Seitz conjugation p g p^-1, where p is the
        real-space operation whose inverse rotation is S.
        """
        if not active_operation_indices:
            return []
        if rotation_index <= 0 or rotation_index > len(operations):
            return list(active_operation_indices)

        star_operation = operations[rotation_index - 1]
        star_rotation = np.asarray(star_operation.rotation, dtype=int)
        star_translation = np.asarray(star_operation.translation, dtype=float)
        if np.array_equal(star_rotation, np.eye(3, dtype=int)) and cls._translation_match(
            star_translation,
            np.zeros(3, dtype=float),
        ):
            return list(active_operation_indices)
        inverse_star_rotation = np.rint(np.linalg.inv(star_rotation)).astype(int)

        mapped: list[int] = []
        for op_idx in active_operation_indices:
            operation = operations[int(op_idx)]
            rotation = np.asarray(operation.rotation, dtype=int)
            translation = np.asarray(operation.translation, dtype=float)
            representative_rotation = star_rotation @ rotation @ inverse_star_rotation
            representative_rotation = np.rint(representative_rotation).astype(int)
            representative_translation = (
                star_translation
                + star_rotation @ translation
                - representative_rotation @ star_translation
            )
            matched = cls._match_operation_by_seitz(
                representative_rotation,
                representative_translation,
                operations,
            )
            if matched is None:
                raise ValueError(
                    "Failed to map k-point little-group operation to kLittleGroups table column: "
                    f"source_op={int(op_idx) + 1}, star_op={int(rotation_index)}, "
                    f"target_rotation={representative_rotation.tolist()}, "
                    f"target_translation={cls._format_translation(representative_translation)}"
                )
            mapped.append(int(matched))
        return mapped

    @staticmethod
    def _operation_indices_are_table_active(entry, indices: list[int]) -> bool:
        irreps = list(getattr(entry, "irreps", []))
        if not irreps:
            return True
        active_ops = np.asarray(getattr(irreps[0], "active_ops", []), dtype=bool).reshape(-1)
        if active_ops.size == 0:
            return True
        for idx in indices:
            if int(idx) < 0 or int(idx) >= active_ops.size or not bool(active_ops[int(idx)]):
                return False
        return True

    @staticmethod
    def _inactive_table_operation_indices(entry, indices: list[int]) -> list[int]:
        irreps = list(getattr(entry, "irreps", []))
        if not irreps:
            return []
        active_ops = np.asarray(getattr(irreps[0], "active_ops", []), dtype=bool).reshape(-1)
        if active_ops.size == 0:
            return []
        inactive = []
        for idx in indices:
            if int(idx) < 0 or int(idx) >= active_ops.size or not bool(active_ops[int(idx)]):
                inactive.append(int(idx))
        return inactive

    @staticmethod
    def _frac_diff_mod1(a: np.ndarray, b: np.ndarray) -> float:
        delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
        delta -= np.rint(delta)
        return float(np.sum(np.abs(delta)))

    @staticmethod
    def _shift_negative_to_unit_interval(k_prim: np.ndarray) -> np.ndarray:
        k = np.asarray(k_prim, dtype=float).copy()
        for i in range(3):
            if k[i] < 0.0 and abs(k[i]) > 1.0e-6:
                k[i] += 1.0
        return k

    @staticmethod
    def _collect_matrix_rows(mat: np.ndarray):
        m = np.asarray(mat)
        return [m[0, :], m[1, :], m[2, :]]

    @staticmethod
    def _database_primitive_lattice_from_spglib_dataset(dataset, db: KLittleGroupsDB) -> np.ndarray:
        if not hasattr(dataset, "std_lattice"):
            raise AttributeError("spglib dataset does not provide std_lattice for database-basis k-point mapping.")
        if not hasattr(db, "kc2p"):
            return np.asarray(dataset.std_lattice, dtype=float)
        std_lattice = np.asarray(dataset.std_lattice, dtype=float)
        # Row-vector convention:
        # k_prim = k_conv @ kc2p  =>  A_prim = kc2p^T @ A_conv
        return np.asarray(db.kc2p, dtype=float).T @ std_lattice

    @staticmethod
    def _kpoint_current_to_database_primitive(current_lattice: np.ndarray, database_lattice: np.ndarray) -> np.ndarray:
        current_lattice = np.asarray(current_lattice, dtype=float)
        database_lattice = np.asarray(database_lattice, dtype=float)
        current_recip = np.linalg.inv(current_lattice).T
        database_recip = np.linalg.inv(database_lattice).T
        return current_recip @ np.linalg.inv(database_recip)

    @classmethod
    def _align_operations_to_reference(
        cls,
        reference_ops: list[SymmetryOperation],
        candidate_ops: list[SymmetryOperation],
        *,
        origin_shift: np.ndarray | None = None,
        tol: float = 1.0e-6,
    ) -> list[SymmetryOperation]:
        remaining = list(candidate_ops)
        aligned: list[SymmetryOperation] = []
        shift = np.zeros(3, dtype=float) if origin_shift is None else np.asarray(origin_shift, dtype=float)

        for ref in reference_ops:
            match_index = None
            rotation_candidates = [
                idx for idx, op in enumerate(remaining) if cls._rotation_match(ref.rotation, op.rotation)
            ]
            if len(rotation_candidates) == 1:
                match_index = rotation_candidates[0]
            else:
                for idx in rotation_candidates:
                    op = remaining[idx]
                    predicted_translation = (
                        np.asarray(op.translation, dtype=float)
                        + shift @ (np.eye(3, dtype=float) - np.asarray(op.rotation, dtype=float).T)
                    )
                    if cls._translation_match(ref.translation, predicted_translation, tol=tol):
                        match_index = idx
                        break
            if match_index is None:
                raise ValueError("Failed to align source symmetry operations with standardized operation order.")
            aligned.append(remaining.pop(match_index))

        return aligned

    def _resolve_kpoint_records(
        self,
        operations: list[SymmetryOperation],
        db: KLittleGroupsDB,
        kpoints_direct: np.ndarray,
        current_to_db_prim: np.ndarray | None = None,
    ):
        records = []
        inverse_rotations = [np.asarray(op.inverse_rotation, dtype=int) for op in operations]
        has_inversion = any(op.symbol == "I" for op in operations)

        for ik, k in enumerate(np.asarray(kpoints_direct, dtype=float), start=1):
            lkg = self._little_group_operation_indices(k, operations)
            active_structure_ops = [idx - 1 for idx in lkg if idx <= len(operations)]
            active_ops = active_structure_ops

            if not hasattr(db, "_reference_kpoint_matches"):
                try:
                    resolution = db.resolve_kpoint_from_star(
                        k,
                        inverse_rotations,
                        has_inversion=has_inversion,
                        little_group_size=len(lkg),
                        detected_ops=lkg,
                        current_to_db_prim=current_to_db_prim,
                    )
                except TypeError as exc:
                    if "current_to_db_prim" not in str(exc):
                        raise
                    resolution = db.resolve_kpoint_from_star(
                        k,
                        inverse_rotations,
                        has_inversion=has_inversion,
                        little_group_size=len(lkg),
                        detected_ops=lkg,
                    )
                active_db_ops = [
                    int(idx) for idx in resolution.entry.little_group_ops if int(idx) < len(operations)
                ]
                table_ops = self._representative_table_operation_indices(
                    operations,
                    active_ops,
                    int(getattr(resolution, "rotation_index", 1)),
                )
                if not self._operation_indices_are_table_active(resolution.entry, table_ops):
                    inactive_ops = self._inactive_table_operation_indices(resolution.entry, table_ops)
                    raise ValueError(
                        "Star-rotated little-group operation mapping is not active in kLittleGroups table. "
                        "This indicates an inconsistent k-point representative or symmetry-operation alignment. "
                        f"k={np.asarray(k, dtype=float).tolist()}, "
                        f"representative_k={np.asarray(resolution.rotated_k_prim, dtype=float).tolist()}, "
                        f"k_name={resolution.entry.name}, "
                        f"star_op={int(getattr(resolution, 'rotation_index', 0))}, "
                        f"current_ops={[int(idx) + 1 for idx in active_ops]}, "
                        f"table_ops={[int(idx) + 1 for idx in table_ops]}, "
                        f"inactive_table_ops={[int(idx) + 1 for idx in inactive_ops]}, "
                        f"database_active_ops={[int(idx) + 1 for idx in active_db_ops]}"
                    )
                resolution.cornwell_satisfied = self._cornwell_condition_satisfied(k, operations, lkg)
                records.append(
                    {
                        "k_index": ik,
                        "k_direct": np.asarray(k, dtype=float),
                        "little_group_indices": lkg,
                        "active_operation_indices": active_ops,
                        "table_operation_indices": table_ops,
                        "database_operation_indices": active_db_ops,
                        "resolution": resolution,
                        "k_name": resolution.entry.name,
                        "cornwell_satisfied": resolution.cornwell_satisfied,
                        "mapped_k_direct": np.asarray(
                            getattr(resolution, "mapped_k_prim", np.asarray(k, dtype=float)),
                            dtype=float,
                        ),
                    }
                )
                continue

            k_current = np.asarray(k, dtype=float)
            if current_to_db_prim is None:
                mapped_k_prim = k_current.copy()
            else:
                mapped_k_prim = k_current @ np.asarray(current_to_db_prim, dtype=float)
                mapped_k_prim -= np.floor(mapped_k_prim)

            candidate_specs: list[tuple[int, np.ndarray]] = [
                (idx + 1, np.asarray(inv_rot, dtype=int))
                for idx, inv_rot in enumerate(inverse_rotations)
            ]
            if not has_inversion:
                candidate_specs.append((0, -np.eye(3, dtype=int)))

            scored_candidates = []
            for star_index, inv_rot in candidate_specs:
                rotated = self._shift_negative_to_unit_interval(mapped_k_prim @ inv_rot)
                matches = db._reference_kpoint_matches(rotated, tol=1.0e-5)
                if not matches:
                    continue
                for entry, varnum, k_conv, entry_index, fracdiff in matches:
                    resolution = KPointResolution(
                        entry=entry,
                        entry_index=int(entry_index),
                        k_conv=np.asarray(k_conv, dtype=float),
                        rotation_index=int(star_index),
                        variable_count=int(varnum),
                        mapped_k_prim=np.asarray(mapped_k_prim, dtype=float).copy(),
                        rotated_k_prim=np.asarray(rotated, dtype=float).copy(),
                    )
                    table_ops = self._representative_table_operation_indices(
                        operations,
                        active_ops,
                        int(star_index),
                    )
                    table_active = self._operation_indices_are_table_active(resolution.entry, table_ops)
                    active_db_ops = [
                        int(idx)
                        for idx in resolution.entry.little_group_ops
                        if int(idx) < len(operations)
                    ]
                    entry_set = {int(idx) + 1 for idx in active_db_ops}
                    table_set = {int(idx) + 1 for idx in table_ops}
                    score = (
                        0 if table_active else 1,
                        len(entry_set.symmetric_difference(table_set)),
                        abs(len(entry_set) - len(table_set)),
                        int(getattr(resolution, "variable_count", 9999)),
                        float(fracdiff),
                        int(getattr(resolution, "entry_index", 10**9)),
                        int(star_index if star_index >= 0 else 10**6),
                    )
                    scored_candidates.append((score, resolution, table_ops, active_db_ops))

            if not scored_candidates:
                raise ValueError(
                    "Nonsymmorphic kpoint is NOT found for primitive k="
                    f"{np.asarray(k, dtype=float).tolist()}."
                )

            scored_candidates.sort(key=lambda item: item[0])
            score, resolution, table_ops, active_db_ops = scored_candidates[0]

            if score[0] != 0:
                inactive_ops = self._inactive_table_operation_indices(resolution.entry, table_ops)
                raise ValueError(
                    "Star-rotated little-group operation mapping is not active in kLittleGroups table. "
                    "This indicates an inconsistent k-point representative or symmetry-operation alignment. "
                    f"k={np.asarray(k, dtype=float).tolist()}, "
                    f"representative_k={np.asarray(resolution.rotated_k_prim, dtype=float).tolist()}, "
                    f"k_name={resolution.entry.name}, "
                    f"star_op={int(getattr(resolution, 'rotation_index', 0))}, "
                    f"current_ops={[int(idx) + 1 for idx in active_ops]}, "
                    f"table_ops={[int(idx) + 1 for idx in table_ops]}, "
                    f"inactive_table_ops={[int(idx) + 1 for idx in inactive_ops]}, "
                    f"database_active_ops={[int(idx) + 1 for idx in active_db_ops]}"
                )

            resolution.cornwell_satisfied = self._cornwell_condition_satisfied(k, operations, lkg)
            records.append(
                {
                    "k_index": ik,
                    "k_direct": np.asarray(k, dtype=float),
                    "little_group_indices": lkg,
                    "active_operation_indices": active_ops,
                    "table_operation_indices": table_ops,
                    "database_operation_indices": active_db_ops,
                    "resolution": resolution,
                    "k_name": resolution.entry.name,
                    "cornwell_satisfied": resolution.cornwell_satisfied,
                    "mapped_k_direct": np.asarray(
                        getattr(resolution, "mapped_k_prim", np.asarray(k, dtype=float)),
                        dtype=float,
                    ),
                }
            )
        return records

    def _write_transformations(
        self,
        fp,
        source_atoms: Atoms,
        std_atoms: Atoms,
        spgfile: Path,
        standardization_result: dict,
    ):
        lattice_old = np.asarray(source_atoms.cell.array, dtype=float)
        lattice_new = np.asarray(std_atoms.cell.array, dtype=float)
        reciprocal_new = np.linalg.inv(lattice_new).T

        fp.write("Transformations:\n")
        fp.write("Original lattice vectors\n")
        for i in range(3):
            fp.write(
                f"a{i + 1} {lattice_old[i, 0]:16.8f}{lattice_old[i, 1]:16.8f}{lattice_old[i, 2]:16.8f}\n"
            )

        fp.write("\nSymmetrized lattice vectors\n")
        for i in range(3):
            fp.write(
                f"a{i + 1} {lattice_new[i, 0]:16.8f}{lattice_new[i, 1]:16.8f}{lattice_new[i, 2]:16.8f}\n"
            )

        fp.write("\nReciprocal lattice vectors of symmetrized structure\n")
        for i in range(3):
            fp.write(
                f"b{i + 1} {reciprocal_new[i, 0]:16.8f}{reciprocal_new[i, 1]:16.8f}{reciprocal_new[i, 2]:16.8f}\n"
            )
        fp.write("\n")

        if bool(standardization_result.get("need_rebuild_hs")):
            lattice_transform = np.asarray(
                standardization_result.get("lattice_transform_fractional", np.eye(3)),
                dtype=float,
            )
            axis_rotation = np.asarray(
                standardization_result.get("xyz_axis_transform_cartesian", np.eye(3)),
                dtype=float,
            )
            fp.write("Supercell transform matrix M\n")
            for i in range(3):
                fp.write(
                    f"M{i + 1} {lattice_transform[i, 0]:16.8f}{lattice_transform[i, 1]:16.8f}{lattice_transform[i, 2]:16.8f}\n"
                )
            fp.write("\nRotation matrix Q (Cartesian)\n")
            for i in range(3):
                fp.write(
                    f"Q{i + 1} {axis_rotation[i, 0]:16.8f}{axis_rotation[i, 1]:16.8f}{axis_rotation[i, 2]:16.8f}\n"
                )
            fp.write("\n")

        fp.write("Little group file :\n")
        fp.write(f"{spgfile.resolve()}\n\n")

    def _write_symmetry_operations(self, fp, operations: list[SymmetryOperation]):
        fp.write("SYMMETRY OPERATIONS Pi={Ri|taui+tm}\n")
        fp.write("  Ri     taui   inv(Ri) Ri(Cartesian coord)  Eulers angles  spin transf.\n\n")

        for i, op in enumerate(operations, start=1):
            r_rows = self._collect_matrix_rows(op.rotation)
            inv_rows = self._collect_matrix_rows(op.inverse_rotation)
            c_rows = self._collect_matrix_rows(op.cart_rotation)
            eul = np.asarray(op.euler_zyz, dtype=float)
            u = op.spin_matrix

            fp.write(
                f" {int(r_rows[0][0]):2d} {int(r_rows[0][1]):2d} {int(r_rows[0][2]):2d}"
                f" {op.translation[0]:7.3f}"
                f" {int(inv_rows[0][0]):3d} {int(inv_rows[0][1]):2d} {int(inv_rows[0][2]):2d}"
                f" {c_rows[0][0]:7.3f}{c_rows[0][1]:7.3f}{c_rows[0][2]:7.3f} {eul[0]:7.3f}\n"
            )
            fp.write(
                f" {int(r_rows[1][0]):2d} {int(r_rows[1][1]):2d} {int(r_rows[1][2]):2d}"
                f" {op.translation[1]:7.3f}"
                f" {int(inv_rows[1][0]):3d} {int(inv_rows[1][1]):2d} {int(inv_rows[1][2]):2d}"
                f" {c_rows[1][0]:7.3f}{c_rows[1][1]:7.3f}{c_rows[1][2]:7.3f} {eul[1]:7.3f}"
                f" {self._format_spin_row(u[0, 0], u[0, 1])}\n"
            )
            fp.write(
                f" {int(r_rows[2][0]):2d} {int(r_rows[2][1]):2d} {int(r_rows[2][2]):2d}"
                f" {op.translation[2]:7.3f}"
                f" {int(inv_rows[2][0]):3d} {int(inv_rows[2][1]):2d} {int(inv_rows[2][2]):2d}"
                f" {c_rows[2][0]:7.3f}{c_rows[2][1]:7.3f}{c_rows[2][2]:7.3f} {eul[2]:7.3f}"
                f" {self._format_spin_row(u[1, 0], u[1, 1])}\n"
            )

            fp.write(f" {i:3d} ({op.symbol})\n")
            fp.write(f"{op.description}\n")
            order = self._rotation_order_from_symbol(op.symbol)
            if order > 2:
                sense = "clockwise" if self._signed_rotation_angle(op.cart_rotation, op.axis) > 0.0 else "counterclockwise"
                fp.write(f"{sense} rotation through ({op.axis[0]:6.3f}, {op.axis[1]:6.3f}, {op.axis[2]:6.3f})\n\n")
            else:
                fp.write(f"rotation through ({op.axis[0]:6.3f}, {op.axis[1]:6.3f}, {op.axis[2]:6.3f})\n\n")

    def _write_k_little_group_table(self, fp, operations: list[SymmetryOperation], db: KLittleGroupsDB, kpoints_direct: np.ndarray):
        for record in self._resolve_kpoint_records(operations, db, kpoints_direct):
            ik = int(record["k_index"])
            k = np.asarray(record["k_direct"], dtype=float)
            resolution = record["resolution"]
            match = resolution.entry
            match_index = getattr(resolution, "entry_index", 0)
            active_db_ops = list(record["active_operation_indices"])
            table_db_ops = list(record.get("table_operation_indices", active_db_ops))
            if len(table_db_ops) != len(active_db_ops):
                table_db_ops = active_db_ops

            fp.write("\n********************************************************************************\n\n")
            fp.write(f"knum = {ik:2d}    kname= \n")
            fp.write(f"k = {k[0]:.6f} {k[1]:.6f} {k[2]:.6f}\n\n")

            fp.write(f"       The k-point name is {match.name:<3s}\n")
            fp.write(f"       {len(active_db_ops)} symmetry operations (module lattice translations) in space group {db.path.stem.split('_')[-1]}\n")
            fp.write("\n")
            fp.write(
                f"{match_index:5d}    {match.name:<3s}: kname {resolution.k_conv[0]:9.2f}{resolution.k_conv[1]:5.2f}{resolution.k_conv[2]:5.2f} :  given in the conventional basis\n"
            )
            fp.write(
                f"{match.antisym:5d} : the existence of antiunitary symmetries. 1-exist; 0-no\n"
            )
            fp.write(" Reality")
            for idx in active_db_ops:
                fp.write(f"{idx + 1:12d}")
            fp.write("\n")

            for ir_index, ir in enumerate(match.irreps):
                display_name = ir.raw_name[1:] if ir.raw_name.startswith("-") else ir.raw_name
                traces = db.irrep_table_characters(resolution, ir)
                fp.write(f"{ir.reality:5d}   {display_name:<6s}")
                for idx in table_db_ops:
                    fp.write(f"{self._format_complex(traces[idx]):>12s}")
                fp.write("\n")

                next_is_double = ir_index + 1 < len(match.irreps) and match.irreps[ir_index + 1].raw_name.startswith("-")
                current_is_single = not ir.raw_name.startswith("-")
                if current_is_single and next_is_double:
                    fp.write("        ")
                    fp.write("-" * (12 * len(active_db_ops)))
                    fp.write("\n")

            fp.write("\n")
            col_element = 10
            col_ops = 14
            col_axis = 32
            fp.write(
                f"{'element':^{col_element}s}"
                f"{'symmetry ops':^{col_ops}s}"
                f"{'main axes':^{col_axis}s}\n"
            )
            for idx in active_db_ops:
                op = operations[idx]
                axis_text = f"({op.axis[0]:7.3f}, {op.axis[1]:7.3f}, {op.axis[2]:7.3f})"
                fp.write(
                    f"{op.symbol:^{col_element}s}"
                    f"{str(idx + 1):^{col_ops}s}"
                    f"{axis_text:^{col_axis}s}\n"
                )

    def analyze_nonmagnetic(self, requested_group, symm_prec: float, kpoints_direct=None):
        source_atoms, source_path = self._load_input_stru()
        dataset, std_atoms, mapping_atoms, sym_data, conv_atoms, conv_sym_data = self._standardize_nonmagnetic_cell(
            source_atoms,
            symm_prec,
        )
        source_sym_data = self._get_symmetry_data(source_atoms, symm_prec)

        detected_group = int(dataset.number)
        resolved_group = self._resolve_spacegroup(requested_group, detected_group)

        map_tol = max(1e-6, float(symm_prec) * 10.0)
        mapping12 = self._build_atom_mapping(source_atoms, mapping_atoms, map_tol)
        lattice_old = np.asarray(source_atoms.cell.array, dtype=float)
        lattice_mapping = np.asarray(mapping_atoms.cell.array, dtype=float)
        lattice_new = np.asarray(std_atoms.cell.array, dtype=float)
        q23_row = self._fit_row_rotation(lattice_mapping, lattice_new)
        b23 = self._round_integer_matrix(
            (lattice_mapping @ q23_row) @ np.linalg.inv(lattice_new),
            tol=max(1.0e-4, map_tol * 10.0),
        )
        mapping23, _, _ = self._build_rotated_atom_mapping(
            mapping_atoms,
            std_atoms,
            q23_row,
            tol=max(5.0e-4, map_tol * 10.0),
        )
        atom_mapping = self._compose_two_stage_mapping(mapping12, mapping23, b23)

        m12 = self._round_integer_matrix(
            lattice_old @ np.linalg.inv(lattice_mapping),
            tol=max(1.0e-4, map_tol * 10.0),
        )
        b13_direct = self._round_integer_matrix(
            (lattice_old @ q23_row) @ np.linalg.inv(lattice_new),
            tol=max(1.0e-4, map_tol * 10.0),
        )
        b13_chain = np.asarray(m12 @ b23, dtype=int)
        if not np.array_equal(b13_chain, b13_direct):
            raise ValueError(
                "Inconsistent lattice transform from source to standardized structure: "
                f"chain={b13_chain.tolist()}, direct={b13_direct.tolist()}"
            )

        lattice_changed = self._detect_lattice_change(source_atoms, std_atoms, map_tol) or len(source_atoms) != len(std_atoms)
        permutation_only = False if lattice_changed else self._mapping_is_permutation_only(atom_mapping, map_tol)
        origin_shift_only = False
        source_to_std_origin_shift = np.zeros(3, dtype=float)
        if not lattice_changed and not permutation_only:
            origin_shift_only, source_to_std_origin_shift, _ = self._match_by_global_fractional_origin_shift(
                source_atoms,
                std_atoms,
                map_tol,
            )
        atom_permutation_only = bool(permutation_only or origin_shift_only)
        lattice_transform_fractional = np.asarray(b13_chain, dtype=float)
        # hs_standardize uses D^\dagger H D with passive_basis=True; pass Q^T so local D is built from Q.
        xyz_axis_transform_cartesian = np.asarray(q23_row.T, dtype=float)
        if kpoints_direct is None:
            canonical_kpoints = np.zeros((0, 3), dtype=float)
        else:
            canonical_kpoints = np.asarray(kpoints_direct, dtype=float)

        if lattice_changed:
            rebuild_reason = "lattice-changed"
        elif permutation_only:
            rebuild_reason = "reuse-original"
        elif origin_shift_only:
            rebuild_reason = "reuse-original-origin-shift"
        else:
            rebuild_reason = "atom-position-changed"

        standardization_result = self._finalize_standardization_result(
            detected_group=detected_group,
            resolved_group=resolved_group,
            lattice_changed=lattice_changed,
            atom_permutation_only=atom_permutation_only,
            source_stru=source_path.name,
            source_hr="data-HR-sparse_SPIN0.csr",
            source_sr="data-SR-sparse_SPIN0.csr",
            atom_mapping=self._atom_mapping_to_dicts(atom_mapping),
            lattice_old=lattice_old,
            lattice_new=lattice_new,
            lattice_transform_fractional=lattice_transform_fractional,
            xyz_axis_transform_cartesian=xyz_axis_transform_cartesian,
            rebuild_reason=rebuild_reason,
            full_matrix_from_hermitian=True,
        )

        operations = self._sort_operations_irvsp_like(self._build_symmetry_operations(std_atoms, sym_data))
        conventional_operations = self._sort_operations_irvsp_like(
            self._build_symmetry_operations(conv_atoms, conv_sym_data)
        )
        source_operations = self._sort_operations_irvsp_like(self._build_symmetry_operations(source_atoms, source_sym_data))

        klg_dir = Path(__file__).resolve().parents[1] / "kLittleGroups"
        spgfile = klg_dir / f"kLG_{resolved_group}.data"
        if not spgfile.exists():
            raise FileNotFoundError(f"kLittleGroups file not found: {spgfile}")

        db = KLittleGroupsDB.load(spgfile)
        try:
            database_primitive_lattice = self._database_primitive_lattice_from_spglib_dataset(dataset, db)
            current_to_db_prim = self._kpoint_current_to_database_primitive(lattice_new, database_primitive_lattice)
        except AttributeError:
            current_to_db_prim = np.eye(3, dtype=float)
        align_tol = max(1.0e-6, map_tol * 10.0)
        database_origin_shift = np.zeros(3, dtype=float)
        database_aligned_operations = operations
        database_alignment_warnings: list[str] = []
        solved_origin_shift = self._solve_origin_shift_from_operations(operations, db, tol=align_tol)
        if solved_origin_shift is not None and float(np.max(np.abs(solved_origin_shift))) > align_tol:
            database_origin_shift = np.asarray(solved_origin_shift, dtype=float)
            database_alignment_warnings.append(
                "spglib operations matched kLittleGroups after primitive fractional origin shift "
                f"{self._format_translation(database_origin_shift)}"
            )
        elif solved_origin_shift is None:
            conventional_origin_shift = self._solve_conventional_origin_shift_from_database(
                conventional_operations,
                db,
                tol=align_tol,
            )
            if (
                conventional_origin_shift is not None
                and float(np.max(np.abs(conventional_origin_shift))) > align_tol
            ):
                database_origin_shift = self._conventional_origin_shift_to_primitive(conventional_origin_shift, db)
                database_alignment_warnings.append(
                    "spglib operations matched kLittleGroups after conventional fractional origin shift "
                    f"{self._format_translation(conventional_origin_shift)} "
                    f"(primitive {self._format_translation(database_origin_shift)})"
                )
        if float(np.max(np.abs(database_origin_shift))) > align_tol:
            shifted_std_atoms = std_atoms.copy()
            shifted_pos = (
                np.asarray(shifted_std_atoms.get_scaled_positions(wrap=False), dtype=float)
                + database_origin_shift
            )
            shifted_std_atoms.set_scaled_positions(_wrap_fractional_coordinates(shifted_pos))
            shifted_sym_data = self._get_symmetry_data(shifted_std_atoms, symm_prec)
            database_aligned_operations = self._sort_operations_irvsp_like(
                self._build_symmetry_operations(shifted_std_atoms, shifted_sym_data)
            )
        if bool(standardization_result["need_rebuild_hs"]):
            reordered_ops, reorder_warnings = self._reorder_operations_with_database(database_aligned_operations, db)
            reorder_warnings = database_alignment_warnings + reorder_warnings
            aligned_source_ops = list(reordered_ops)
            match_summary_ok, match_summary_details = self._database_alignment_summary(database_aligned_operations, db)
        else:
            try:
                reordered_ops, reorder_warnings = self._reorder_operations_with_database(source_operations, db)
                aligned_source_ops = list(reordered_ops)
                match_summary_ok, match_summary_details = self._database_alignment_summary(source_operations, db)
            except ValueError as source_align_error:
                reordered_ops, reorder_warnings = self._reorder_operations_with_database(database_aligned_operations, db)
                reorder_warnings = database_alignment_warnings + reorder_warnings
                aligned_source_ops = self._align_operations_to_reference(
                    reordered_ops,
                    source_operations,
                    origin_shift=source_to_std_origin_shift + database_origin_shift,
                    tol=align_tol,
                )
                del source_align_error
                match_summary_ok, match_summary_details = self._database_alignment_summary(database_aligned_operations, db)
        kpoint_records = []
        if canonical_kpoints.size > 0:
            kpoint_records = self._resolve_kpoint_records(
                reordered_ops,
                db,
                canonical_kpoints,
                current_to_db_prim=current_to_db_prim,
            )

        if RANK == 0:
            target_stru_path = Path(INPUT_PATH) / str(standardization_result["target_stru"])
            if bool(standardization_result["need_rebuild_hs"]):
                self._write_standardized_stru(source_path, std_atoms, target_stru_path)

            summary_path = self._output_path / "symmetry_summary.txt"
            standardization_summary_txt = self._output_path / "standardization_summary.txt"
            standardization_summary_json = self._output_path / "standardization_summary.json"
            lattice_old_path = self._output_path / "lattice_old.txt"
            lattice_new_path = self._output_path / "lattice_new.txt"
            lattice_transform_path = self._output_path / "lattice_transform_matrix.txt"
            xyz_transform_path = self._output_path / "xyz_axis_transform_matrix.txt"
            ops_path = self._output_path / "real_space_symmetry_ops.txt"
            atom_map_path = self._output_path / "atom_mapping.txt"
            r_atom_map_path = self._output_path / "R_block_mapping.txt"
            report_path = self._output_path / "symmetry_character_report.txt"
            aux_output_paths = [
                summary_path,
                lattice_old_path,
                lattice_new_path,
                lattice_transform_path,
                xyz_transform_path,
                ops_path,
                atom_map_path,
                r_atom_map_path,
            ]

            if self._emit_aux_outputs:
                with summary_path.open("w", encoding="utf-8") as f:
                    f.write(f"source_stru: {source_path}\n")
                    f.write(f"standardized_stru: {target_stru_path}\n")
                    f.write(f"structure_changed: {standardization_result['structure_changed']}\n")
                    f.write(f"detected_group: {detected_group}\n")
                    f.write(f"resolved_group: {resolved_group}\n")
                    f.write(f"operation_count: {len(reordered_ops)}\n")
                    f.write(f"spgfile: {spgfile.resolve()}\n")
                    if reorder_warnings:
                        f.write("reorder_warnings:\n")
                        for w in reorder_warnings:
                            f.write(f"  - {w}\n")

                np.savetxt(lattice_old_path, lattice_old, fmt="% .12f")
                np.savetxt(lattice_new_path, lattice_new, fmt="% .12f")
                np.savetxt(lattice_transform_path, lattice_transform_fractional, fmt="% .12f")
                np.savetxt(xyz_transform_path, xyz_axis_transform_cartesian, fmt="% .12f")

                with ops_path.open("w", encoding="utf-8") as f:
                    for i, op in enumerate(reordered_ops, start=1):
                        f.write(f"op {i}\n")
                        for row in np.asarray(op.rotation, dtype=int):
                            f.write(f"  {row[0]:2d} {row[1]:2d} {row[2]:2d}\n")
                        f.write(
                            f"  t {op.translation[0]: .10f} {op.translation[1]: .10f} {op.translation[2]: .10f}\n"
                        )

                with atom_map_path.open("w", encoding="utf-8") as f:
                    f.write(f"structure_changed: {standardization_result['structure_changed']}\n")
                    for old_idx, new_idx, shift in atom_mapping:
                        f.write(
                            "R %4d %4d %4d atom%d ------> R %4d %4d %4d atom%d\n"
                            % (shift[0], shift[1], shift[2], old_idx + 1, 0, 0, 0, new_idx + 1)
                        )

                r_vectors = np.asarray(getattr(self._tb, "R_direct_coor", np.zeros((0, 3), dtype=int)), dtype=int)
                with r_atom_map_path.open("w", encoding="utf-8") as f:
                    if r_vectors.size == 0:
                        f.write("No R vectors available from TB Hamiltonian data.\n")
                    else:
                        for r in r_vectors:
                            for old_idx, new_idx, shift in atom_mapping:
                                target_r = map_target_r_vector(
                                    lattice_transform_fractional,
                                    np.asarray(r, dtype=int),
                                    np.zeros(3, dtype=int),
                                    np.asarray(shift, dtype=int),
                                )
                                f.write(
                                    "R %4d %4d %4d atom%d ------> R %4d %4d %4d atom%d\n"
                                    % (
                                        r[0],
                                        r[1],
                                        r[2],
                                        old_idx + 1,
                                        target_r[0],
                                        target_r[1],
                                        target_r[2],
                                        new_idx + 1,
                                    )
                                )
            else:
                for p in aux_output_paths:
                    if p.exists():
                        p.unlink()

            if self._emit_standardization_summary:
                with standardization_summary_txt.open("w", encoding="utf-8") as f:
                    f.write(f"lattice_changed: {standardization_result['lattice_changed']}\n")
                    f.write(f"晶格是否改变: {standardization_result['lattice_changed']}\n")
                    f.write(f"atom_permutation_only: {standardization_result['atom_permutation_only']}\n")
                    f.write(f"是否仅原子互换: {standardization_result['atom_permutation_only']}\n")
                    f.write(f"need_rebuild_hs: {standardization_result['need_rebuild_hs']}\n")
                    f.write(f"是否需要重建HS: {standardization_result['need_rebuild_hs']}\n")
                    f.write(f"rebuild_reason: {standardization_result['rebuild_reason']}\n")
                    f.write(f"重建原因: {standardization_result['rebuild_reason']}\n")
                    f.write(f"full_matrix_from_hermitian: {standardization_result['full_matrix_from_hermitian']}\n")
                    f.write(
                        f"是否补齐每个R的完整矩阵: {standardization_result['full_matrix_from_hermitian']}\n"
                    )
                    f.write(f"source_stru: {standardization_result['source_stru']}\n")
                    f.write(f"target_stru: {standardization_result['target_stru']}\n")
                    f.write(f"source_hr: {standardization_result['source_hr']}\n")
                    f.write(f"target_hr: {standardization_result['target_hr']}\n")
                    f.write(f"source_sr: {standardization_result['source_sr']}\n")
                    f.write(f"target_sr: {standardization_result['target_sr']}\n")

                standardization_summary_json.write_text(
                    json.dumps(standardization_result, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            else:
                for p in (standardization_summary_txt, standardization_summary_json):
                    if p.exists():
                        p.unlink()

            with report_path.open("w", encoding="utf-8") as f:
                f.write("Symmetry operation alignment summary:\n")
                if match_summary_ok:
                    f.write("  - All spglib symmetry operations fully match the kLittleGroups table.\n")
                for detail in match_summary_details:
                    f.write(f"  - {detail}\n")
                if reorder_warnings:
                    for warning in reorder_warnings:
                        f.write(f"  - reorder note: {warning}\n")
                f.write("\n")
                self._write_transformations(f, source_atoms, std_atoms, spgfile, standardization_result)
                self._write_symmetry_operations(f, reordered_ops)
                if canonical_kpoints.size > 0:
                    self._write_k_little_group_table(f, reordered_ops, db, canonical_kpoints)

            with open(RUNNING_LOG, "a", encoding="utf-8") as f:
                f.write("\nSymmetry detection summary ==> \n")
                f.write(f" >> structure_standardized : {int(bool(standardization_result['need_rebuild_hs']))}\n")
                f.write(f" >> structure_changed : {standardization_result['structure_changed']}\n")
                f.write(f" >> detected_group    : {detected_group}\n")
                f.write(f" >> resolved_group    : {resolved_group}\n")
                f.write(f" >> operations        : {len(reordered_ops)}\n")
                f.write(f" >> database_match_ok : {int(match_summary_ok)}\n")
                f.write(f" >> standardized_stru : {target_stru_path.resolve()}\n")
                f.write(f" >> standardized_hr   : {(Path(standardization_result['target_hr']).resolve())}\n")
                f.write(f" >> standardized_sr   : {(Path(standardization_result['target_sr']).resolve())}\n")
                f.write(f" >> spgfile           : {spgfile.resolve()}\n")
                f.write(f" >> report_file       : {report_path.resolve()}\n")

        return {
            **standardization_result,
            "operation_count": len(reordered_ops),
            "spgfile": str(spgfile.resolve()),
            "operations": reordered_ops,
            "source_operations": aligned_source_ops,
            "kpoint_records": kpoint_records,
            "canonical_kpoints_direct": canonical_kpoints,
            "kpoint_current_to_db_prim": current_to_db_prim,
        }

    def analyze_magnetic(
        self,
        requested_group,
        symm_prec: float,
        magnetic_moments: np.ndarray,
        kpoints_direct=None,
    ):
        if requested_group == "auto":
            raise ValueError("CHARACTER.mag_tag=1 requires an explicit CHARACTER.group space-group number.")

        source_atoms, source_path = self._load_input_stru()
        magnetic_moments = np.asarray(magnetic_moments, dtype=float)
        if magnetic_moments.shape != (len(source_atoms), 3):
            raise ValueError(
                f"CHARACTER.mag must provide at least 3 values per atom; expected shape ({len(source_atoms)}, 3), "
                f"got {magnetic_moments.shape}."
            )

        detected_unitary_group, std_atoms, mapping_atoms, source_unitary_sym_data, std_unitary_sym_data, std_mag = self._standardize_magnetic_cell(
            source_atoms,
            magnetic_moments,
            symm_prec,
        )
        resolved_group = int(requested_group)
        klg_dir = Path(__file__).resolve().parents[1] / "kLittleGroups"
        spgfile = klg_dir / f"kLG_{resolved_group}.data"
        if not spgfile.exists():
            raise FileNotFoundError(f"kLittleGroups file not found for requested magnetic unitary group: {spgfile}")

        db = KLittleGroupsDB.load(spgfile)
        if hasattr(db, "kc2p"):
            dataset_like = type(
                "DatasetLike",
                (),
                {"std_lattice": np.asarray(db.kc2p, dtype=float) @ np.asarray(std_atoms.cell.array, dtype=float)},
            )()
            database_primitive_lattice = self._database_primitive_lattice_from_spglib_dataset(dataset_like, db)
            current_to_db_prim = self._kpoint_current_to_database_primitive(
                np.asarray(std_atoms.cell.array, dtype=float),
                database_primitive_lattice,
            )
        else:
            current_to_db_prim = np.eye(3, dtype=float)
        map_tol = max(1e-6, float(symm_prec) * 10.0)
        mapping12 = self._build_atom_mapping(source_atoms, mapping_atoms, map_tol)
        lattice_old = np.asarray(source_atoms.cell.array, dtype=float)
        lattice_mapping = np.asarray(mapping_atoms.cell.array, dtype=float)
        lattice_new = np.asarray(std_atoms.cell.array, dtype=float)
        q23_row = self._fit_row_rotation(lattice_mapping, lattice_new)
        b23 = self._round_integer_matrix(
            (lattice_mapping @ q23_row) @ np.linalg.inv(lattice_new),
            tol=max(1.0e-4, map_tol * 10.0),
        )
        mapping23, _, _ = self._build_rotated_atom_mapping(
            mapping_atoms,
            std_atoms,
            q23_row,
            tol=max(5.0e-4, map_tol * 10.0),
        )
        atom_mapping = self._compose_two_stage_mapping(mapping12, mapping23, b23)
        m12 = self._round_integer_matrix(
            lattice_old @ np.linalg.inv(lattice_mapping),
            tol=max(1.0e-4, map_tol * 10.0),
        )
        b13_direct = self._round_integer_matrix(
            (lattice_old @ q23_row) @ np.linalg.inv(lattice_new),
            tol=max(1.0e-4, map_tol * 10.0),
        )
        b13_chain = np.asarray(m12 @ b23, dtype=int)
        if not np.array_equal(b13_chain, b13_direct):
            raise ValueError(
                "Inconsistent lattice transform from magnetic source to standardized structure: "
                f"chain={b13_chain.tolist()}, direct={b13_direct.tolist()}"
            )

        lattice_changed = self._detect_lattice_change(source_atoms, std_atoms, map_tol) or len(source_atoms) != len(std_atoms)
        permutation_only = False if lattice_changed else self._mapping_is_permutation_only(atom_mapping, map_tol)
        origin_shift_only = False
        source_to_std_origin_shift = np.zeros(3, dtype=float)
        if not lattice_changed and not permutation_only:
            origin_shift_only, source_to_std_origin_shift, _ = self._match_by_global_fractional_origin_shift(
                source_atoms,
                std_atoms,
                map_tol,
            )
        atom_permutation_only = bool(permutation_only or origin_shift_only)
        lattice_transform_fractional = np.asarray(b13_chain, dtype=float)
        xyz_axis_transform_cartesian = np.asarray(q23_row.T, dtype=float)
        if lattice_changed:
            rebuild_reason = "magnetic-lattice-changed"
        elif permutation_only:
            rebuild_reason = "magnetic-reuse-original"
        elif origin_shift_only:
            rebuild_reason = "magnetic-reuse-original-origin-shift"
        else:
            rebuild_reason = "magnetic-atom-position-changed"

        std_unitary_ops = self._sort_operations_irvsp_like(self._build_symmetry_operations(std_atoms, std_unitary_sym_data))
        source_unitary_ops = self._sort_operations_irvsp_like(self._build_symmetry_operations(source_atoms, source_unitary_sym_data))
        origin_shift = self._solve_origin_shift_from_operations(std_unitary_ops, db, tol=max(1.0e-6, map_tol * 10.0))
        if origin_shift is not None and float(np.max(np.abs(origin_shift))) > max(1.0e-6, map_tol * 10.0):
            shifted_std_atoms = std_atoms.copy()
            shifted_pos = np.asarray(shifted_std_atoms.get_scaled_positions(wrap=False), dtype=float) + np.asarray(origin_shift, dtype=float)
            shifted_std_atoms.set_scaled_positions(_wrap_fractional_coordinates(shifted_pos))
            std_atoms = shifted_std_atoms
            std_unitary_sym_data, _ = self._get_magnetic_unitary_symmetry_data(
                std_atoms,
                std_mag,
                symm_prec,
            )
            std_unitary_ops = self._sort_operations_irvsp_like(self._build_symmetry_operations(std_atoms, std_unitary_sym_data))
        try:
            required_operation_count = int(db.doubnum // 2)
            reordered_ops_all, reorder_warnings = self._reorder_operations_with_database(std_unitary_ops, db)
            match_summary_ok, match_summary_details = self._database_alignment_summary(std_unitary_ops, db)
            if len(reordered_ops_all) < required_operation_count:
                raise ValueError(
                    "Requested space group has more operations than the detected magnetic unitary group."
                )
            reordered_ops = reordered_ops_all[:required_operation_count]
            requested_is_subgroup = True
        except ValueError as strict_align_error:
            reordered_ops = []
            reorder_warnings = []
            requested_is_subgroup = False
            message = (
                "Magnetic symmetry operation alignment with kLittleGroups database failed: "
                f"{strict_align_error}"
            )
            if RANK == 0:
                report_path = self._output_path / "symmetry_character_report.txt"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                with report_path.open("w", encoding="utf-8") as fp:
                    fp.write("Magnetic symmetry detection:\n")
                    fp.write(f"Unitary operations form space group {detected_unitary_group}\n")
                    fp.write(f"User requested space group {resolved_group}\n")
                    fp.write(f"Error: {message}\n")
                with open(RUNNING_LOG, "a", encoding="utf-8") as fp:
                    fp.write("\nMagnetic symmetry detection summary ==> \n")
                    fp.write(f" >> unitary_detected_group : {detected_unitary_group}\n")
                    fp.write(f" >> requested_group        : {resolved_group}\n")
                    fp.write(" >> requested_is_subgroup  : 0\n")
                    fp.write(f" >> strict_align_error     : {strict_align_error}\n")
            raise ValueError(message) from strict_align_error
        if not requested_is_subgroup:
            message = (
                "Magnetic unitary symmetry group check failed: "
                f"unitary operations form space group {detected_unitary_group}, "
                f"requested group {resolved_group} is not identical to or a subgroup of the detected unitary group."
            )
            if RANK == 0:
                report_path = self._output_path / "symmetry_character_report.txt"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                with report_path.open("w", encoding="utf-8") as fp:
                    fp.write("Magnetic symmetry detection:\n")
                    fp.write(f"Unitary operations form space group {detected_unitary_group}\n")
                    fp.write(f"User requested space group {resolved_group}\n")
                    fp.write("User requested group is not identical to or a subgroup of the unitary operation group.\n")
                    fp.write(f"Error: {message}\n")
                with open(RUNNING_LOG, "a", encoding="utf-8") as fp:
                    fp.write("\nMagnetic symmetry detection summary ==> \n")
                    fp.write(f" >> unitary_detected_group : {detected_unitary_group}\n")
                    fp.write(f" >> requested_group        : {resolved_group}\n")
                    fp.write(" >> requested_is_subgroup  : 0\n")
            raise ValueError(message)

        if bool(lattice_changed or not atom_permutation_only):
            aligned_source_ops = list(reordered_ops)
        else:
            aligned_source_ops = self._align_operations_to_reference(
                reordered_ops,
                source_unitary_ops,
                origin_shift=source_to_std_origin_shift,
                tol=max(1.0e-6, map_tol * 10.0),
            )

        canonical_kpoints = (
            np.zeros((0, 3), dtype=float)
            if kpoints_direct is None
            else np.asarray(kpoints_direct, dtype=float)
        )
        kpoint_records = []
        if canonical_kpoints.size > 0:
            kpoint_records = self._resolve_kpoint_records(
                reordered_ops,
                db,
                canonical_kpoints,
                current_to_db_prim=current_to_db_prim,
            )

        standardization_result = self._finalize_standardization_result(
            detected_group=detected_unitary_group,
            resolved_group=resolved_group,
            lattice_changed=lattice_changed,
            atom_permutation_only=atom_permutation_only,
            source_stru=source_path.name,
            source_hr="data-HR-sparse_SPIN0.csr",
            source_sr="data-SR-sparse_SPIN0.csr",
            atom_mapping=self._atom_mapping_to_dicts(atom_mapping),
            lattice_old=lattice_old,
            lattice_new=lattice_new,
            lattice_transform_fractional=lattice_transform_fractional,
            xyz_axis_transform_cartesian=xyz_axis_transform_cartesian,
            rebuild_reason=rebuild_reason,
            full_matrix_from_hermitian=True,
        )

        if RANK == 0:
            report_path = self._output_path / "symmetry_character_report.txt"
            with report_path.open("w", encoding="utf-8") as f:
                f.write("Magnetic symmetry detection:\n")
                f.write(f"Unitary operations form space group {detected_unitary_group}\n")
                f.write(f"User requested space group {resolved_group}\n")
                f.write("User requested group is identical to or a subgroup of the unitary operation group.\n\n")
                if match_summary_ok:
                    f.write("Symmetry operation alignment summary:\n")
                    f.write("  - All spglib symmetry operations fully match the kLittleGroups table.\n")
                    for detail in match_summary_details:
                        f.write(f"  - {detail}\n")
                    f.write("\n")
                else:
                    f.write("Symmetry operation alignment summary:\n")
                    for detail in match_summary_details:
                        f.write(f"  - {detail}\n")
                    if reorder_warnings:
                        for warning in reorder_warnings:
                            f.write(f"  - reorder note: {warning}\n")
                    f.write("\n")
                self._write_transformations(f, source_atoms, std_atoms, spgfile, standardization_result)
                self._write_symmetry_operations(f, reordered_ops)
                if canonical_kpoints.size > 0:
                    self._write_k_little_group_table(f, reordered_ops, db, canonical_kpoints)

            with open(RUNNING_LOG, "a", encoding="utf-8") as f:
                f.write("\nMagnetic symmetry detection summary ==> \n")
                f.write(f" >> unitary_detected_group : {detected_unitary_group}\n")
                f.write(f" >> requested_group        : {resolved_group}\n")
                f.write(" >> requested_is_subgroup  : 1\n")
                f.write(f" >> structure_standardized : {int(bool(standardization_result['need_rebuild_hs']))}\n")
                f.write(f" >> operations             : {len(reordered_ops)}\n")
                f.write(f" >> database_match_ok      : {int(match_summary_ok)}\n")
                f.write(f" >> spgfile                : {spgfile.resolve()}\n")
                f.write(f" >> report_file            : {report_path.resolve()}\n")

        return {
            **standardization_result,
            "magnetic": True,
            "unitary_detected_group": int(detected_unitary_group),
            "requested_group_is_unitary_subgroup": True,
            "operation_count": len(reordered_ops),
            "spgfile": str(spgfile.resolve()),
            "operations": reordered_ops,
            "source_operations": aligned_source_ops,
            "kpoint_records": kpoint_records,
            "canonical_kpoints_direct": canonical_kpoints,
            "kpoint_current_to_db_prim": current_to_db_prim,
        }
