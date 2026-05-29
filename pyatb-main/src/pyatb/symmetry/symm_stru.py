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
from pyatb.symmetry.Dk_matrix import (
    axis_angle_from_cartesian_rotation,
    canonicalize_irvsp_spin_group_signs,
    spin_half_matrix_from_cartesian_rotation,
)
from pyatb.symmetry.hs_standardize import canonicalize_fractional_coordinates, map_target_r_vector
from pyatb.symmetry.k_little_groups import KLittleGroupsDB
from pyatb.symmetry.kpoint_little_group import KPointLittleGroupMixin
from pyatb.symmetry.structure_mapping import (
    build_structure_mapping,
    structure_mapping_summary,
    structure_mapping_to_dicts,
    structure_mapping_to_tuples,
)
from pyatb.symmetry.symmetry_report import SymmetryReportMixin


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


class SymmStructureAnalyzer(KPointLittleGroupMixin, SymmetryReportMixin):
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
    def _operation_sets_match(
        cls,
        left: list[SymmetryOperation],
        right: list[SymmetryOperation],
    ) -> bool:
        if len(left) != len(right):
            return False
        return all(cls._operation_in_set(operation, right) for operation in left) and all(
            cls._operation_in_set(operation, left) for operation in right
        )

    @classmethod
    def _operations_after_origin_shift(
        cls,
        operations: list[SymmetryOperation],
        shift: np.ndarray,
    ) -> list[SymmetryOperation]:
        shift_vec = np.asarray(shift, dtype=float).reshape(3)
        identity = np.eye(3, dtype=float)
        shifted: list[SymmetryOperation] = []
        for op in operations:
            rotation = np.asarray(op.rotation, dtype=float)
            translation = np.asarray(op.translation, dtype=float) + shift_vec @ (identity - rotation.T)
            translation = canonicalize_fractional_coordinates(translation, tol=1.0e-8)
            shifted.append(
                SymmetryOperation(
                    rotation=np.asarray(op.rotation, dtype=int),
                    translation=np.asarray(translation, dtype=float),
                    inverse_rotation=np.asarray(op.inverse_rotation, dtype=int),
                    cart_rotation=np.asarray(op.cart_rotation, dtype=float),
                    euler_zyz=np.asarray(op.euler_zyz, dtype=float),
                    spin_matrix=np.asarray(op.spin_matrix, dtype=complex),
                    symbol=str(op.symbol),
                    description=str(op.description),
                    axis=np.asarray(op.axis, dtype=float),
                )
            )
        return shifted

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

        scaled_positions = _wrap_fractional_coordinates(np.asarray(std_atoms.get_scaled_positions(wrap=False), dtype=float))
        scaled_positions[np.isclose(scaled_positions, 1.0, atol=1.0e-8)] = 0.0
        scaled_positions[np.isclose(scaled_positions, 0.0, atol=1.0e-8)] = 0.0
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
        force_rebuild_hs: bool = False,
    ) -> dict:
        need_rebuild_hs = bool(force_rebuild_hs or lattice_changed or not atom_permutation_only)
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

        if ops:
            spin_matrices = canonicalize_irvsp_spin_group_signs(
                [operation.rotation for operation in ops],
                [operation.spin_matrix for operation in ops],
            )
            for operation, spin in zip(ops, spin_matrices, strict=True):
                operation.spin_matrix = spin

        return ops

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
        arr = np.asarray(vec, dtype=float).reshape(-1)
        if arr.size == 0:
            return "()"
        return "(" + f"{arr[0]: .8f}" + "".join(f", {x:.8f}" for x in arr[1:]) + ")"

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

    @staticmethod
    def _apply_database_spin_convention(operations: list[SymmetryOperation], db: KLittleGroupsDB) -> None:
        n_target = min(len(operations), int(getattr(db, "doubnum", 0)) // 2, len(getattr(db, "symops", [])))
        for idx in range(n_target):
            spin = np.asarray(getattr(db.symops[idx], "spin", np.eye(2, dtype=complex)), dtype=complex)
            if spin.shape != (2, 2) or float(np.linalg.norm(spin)) <= 1.0e-12:
                continue
            operations[idx].spin_matrix = spin.copy()

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
        lattice_old = np.asarray(source_atoms.cell.array, dtype=float)
        lattice_mapping = np.asarray(mapping_atoms.cell.array, dtype=float)
        lattice_new = np.asarray(std_atoms.cell.array, dtype=float)
        q23_row = self._fit_row_rotation(lattice_mapping, lattice_new)
        b23 = self._round_integer_matrix(
            (lattice_mapping @ q23_row) @ np.linalg.inv(lattice_new),
            tol=max(1.0e-4, map_tol * 10.0),
        )

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
        idealized_mapping_tol = max(5.0e-3, 5.0e-4, map_tol * 10.0)
        mapping_result = build_structure_mapping(
            source_atoms,
            std_atoms,
            rotation_matrix=q23_row,
            supercell_matrix=b13_chain,
            fractional_translation=np.zeros(3, dtype=float),
            tol=idealized_mapping_tol,
            lattice_tol=max(1.0e-4, map_tol * 10.0),
        )
        atom_mapping = structure_mapping_to_tuples(mapping_result)

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
        database_alignment_notes: list[str] = []
        solved_origin_shift = self._solve_origin_shift_from_operations(operations, db, tol=align_tol)
        if solved_origin_shift is not None and float(np.max(np.abs(solved_origin_shift))) > align_tol:
            database_origin_shift = np.asarray(solved_origin_shift, dtype=float)
            database_alignment_warnings.append(
                "spglib operations matched kLittleGroups after primitive fractional origin shift "
                f"{self._format_translation(database_origin_shift)}"
            )
            database_alignment_notes.extend(
                [
                    "spglib operations matched kLittleGroups after primitive fractional origin shift",
                    f"primitive  basis: {self._format_translation(database_origin_shift)}",
                ]
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
                database_alignment_notes.extend(
                    [
                        "spglib operations matched kLittleGroups after conventional fractional origin shift",
                        f"convention basis: {self._format_translation(conventional_origin_shift)}",
                        f"primitive  basis: {self._format_translation(database_origin_shift)}",
                    ]
                )
        database_origin_shift_applied = float(np.max(np.abs(database_origin_shift))) > align_tol
        if database_origin_shift_applied:
            expected_shifted_operations = self._operations_after_origin_shift(operations, database_origin_shift)
            shifted_std_atoms = std_atoms.copy()
            shifted_pos = (
                np.asarray(shifted_std_atoms.get_scaled_positions(wrap=False), dtype=float)
                + database_origin_shift
            )
            shifted_std_atoms.set_scaled_positions(_wrap_fractional_coordinates(shifted_pos))
            shifted_sym_data = self._get_symmetry_data(shifted_std_atoms, symm_prec)
            shifted_operations = self._sort_operations_irvsp_like(
                self._build_symmetry_operations(shifted_std_atoms, shifted_sym_data)
            )
            if not self._operation_sets_match(expected_shifted_operations, shifted_operations):
                raise ValueError(
                    "Shifted standardized primitive structure failed spglib symmetry consistency check: "
                    "symmetry operations reported by spglib do not match the origin-shifted operations."
                )
            database_alignment_notes.append(
                "shifted standardized primitive structure rechecked by spglib without additional standardization"
            )
            std_atoms = shifted_std_atoms
            sym_data = shifted_sym_data
            operations = shifted_operations
            database_aligned_operations = shifted_operations
            origin_shift_mapping_tol = max(1.0e-8, float(symm_prec))
            mapping_result = build_structure_mapping(
                source_atoms,
                std_atoms,
                rotation_matrix=q23_row,
                supercell_matrix=b13_chain,
                fractional_translation=database_origin_shift,
                tol=origin_shift_mapping_tol,
                lattice_tol=max(1.0e-4, map_tol * 10.0),
            )
            atom_mapping = structure_mapping_to_tuples(mapping_result)
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
            if rebuild_reason in {"reuse-original", "reuse-original-origin-shift"}:
                rebuild_reason = "database-origin-shift"
            elif "database-origin-shift" not in rebuild_reason:
                rebuild_reason = f"{rebuild_reason}+database-origin-shift"

        force_rebuild_hs = bool(origin_shift_only or database_origin_shift_applied)
        standardization_result = self._finalize_standardization_result(
            detected_group=detected_group,
            resolved_group=resolved_group,
            lattice_changed=lattice_changed,
            atom_permutation_only=atom_permutation_only,
            source_stru=source_path.name,
            source_hr="data-HR-sparse_SPIN0.csr",
            source_sr="data-SR-sparse_SPIN0.csr",
            atom_mapping=structure_mapping_to_dicts(mapping_result),
            lattice_old=lattice_old,
            lattice_new=lattice_new,
            lattice_transform_fractional=lattice_transform_fractional,
            xyz_axis_transform_cartesian=xyz_axis_transform_cartesian,
            rebuild_reason=rebuild_reason,
            full_matrix_from_hermitian=True,
            force_rebuild_hs=force_rebuild_hs,
        )
        standardization_result["structure_mapping"] = structure_mapping_summary(mapping_result)

        if bool(standardization_result["need_rebuild_hs"]):
            operation_order_reference = database_aligned_operations
            reordered_ops, reorder_warnings = self._reorder_operations_with_database(database_aligned_operations, db)
            reorder_warnings = database_alignment_warnings + reorder_warnings
            aligned_source_ops = self._align_operations_to_reference(
                reordered_ops,
                operations,
                origin_shift=database_origin_shift,
                tol=align_tol,
            )
            match_summary_ok, match_summary_details = self._database_alignment_summary(database_aligned_operations, db)
        else:
            try:
                operation_order_reference = source_operations
                reordered_ops, reorder_warnings = self._reorder_operations_with_database(source_operations, db)
                aligned_source_ops = list(reordered_ops)
                match_summary_ok, match_summary_details = self._database_alignment_summary(source_operations, db)
            except ValueError as source_align_error:
                operation_order_reference = database_aligned_operations
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
        self._apply_database_spin_convention(reordered_ops, db)
        self._apply_database_spin_convention(aligned_source_ops, db)

        origin_redefined_for_report = self._origin_shift_applied(
            source_to_std_origin_shift,
            database_origin_shift,
            tol=align_tol,
        )
        phase_from_source_operations = origin_redefined_for_report and not bool(
            standardization_result["need_rebuild_hs"]
        )
        kpoint_records = []
        if canonical_kpoints.size > 0:
            kpoint_records = self._resolve_kpoint_records(
                reordered_ops,
                db,
                canonical_kpoints,
                current_to_db_prim=current_to_db_prim,
                phase_from_source_operations=phase_from_source_operations,
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

            operation_basis_conversion = self._operation_basis_conversion_applied(
                current_to_db_prim,
                match_summary_details,
                reorder_warnings,
            )
            symmetry_operations_reordered = self._operations_sequence_reordered(
                operation_order_reference,
                reordered_ops,
            )
            structure_atoms_reordered = self._atom_mapping_reordered(standardization_result)
            origin_redefined = origin_redefined_for_report
            with report_path.open("w", encoding="utf-8") as f:
                self._write_report_header(
                    f,
                    detected_group=detected_group,
                    resolved_group=resolved_group,
                    dataset=dataset,
                    db=db,
                    current_to_db_prim=current_to_db_prim,
                    standardization_result=standardization_result,
                    operation_basis_conversion=operation_basis_conversion,
                    symmetry_operations_reordered=symmetry_operations_reordered,
                    structure_atoms_reordered=structure_atoms_reordered,
                    origin_redefined=origin_redefined,
                    spgfile=spgfile,
                    database_alignment_notes=database_alignment_notes,
                    source_to_standard_origin_shift=source_to_std_origin_shift,
                    database_origin_shift=database_origin_shift,
                )
                self._write_transformations(f, source_atoms, std_atoms, spgfile, standardization_result)
                self._write_symmetry_operations(f, aligned_source_ops)
                if canonical_kpoints.size > 0:
                    self._write_k_little_group_table(
                        f,
                        reordered_ops,
                        db,
                        canonical_kpoints,
                        phase_operations=aligned_source_ops,
                        phase_from_source_operations=phase_from_source_operations,
                        current_to_db_prim=current_to_db_prim,
                    )

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
        lattice_old = np.asarray(source_atoms.cell.array, dtype=float)
        lattice_mapping = np.asarray(mapping_atoms.cell.array, dtype=float)
        lattice_new = np.asarray(std_atoms.cell.array, dtype=float)
        q23_row = self._fit_row_rotation(lattice_mapping, lattice_new)
        b23 = self._round_integer_matrix(
            (lattice_mapping @ q23_row) @ np.linalg.inv(lattice_new),
            tol=max(1.0e-4, map_tol * 10.0),
        )
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
        mapping_result = build_structure_mapping(
            source_atoms,
            std_atoms,
            rotation_matrix=q23_row,
            supercell_matrix=b13_chain,
            fractional_translation=np.zeros(3, dtype=float),
            tol=max(5.0e-4, map_tol * 10.0),
            lattice_tol=max(1.0e-4, map_tol * 10.0),
        )
        atom_mapping = structure_mapping_to_tuples(mapping_result)

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
        origin_shift_applied = bool(
            origin_shift is not None and float(np.max(np.abs(origin_shift))) > max(1.0e-6, map_tol * 10.0)
        )
        if origin_shift_applied:
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
            mapping_result = build_structure_mapping(
                source_atoms,
                std_atoms,
                rotation_matrix=q23_row,
                supercell_matrix=b13_chain,
                fractional_translation=np.asarray(origin_shift, dtype=float),
                tol=max(5.0e-4, map_tol * 10.0),
                lattice_tol=max(1.0e-4, map_tol * 10.0),
            )
            atom_mapping = structure_mapping_to_tuples(mapping_result)
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
            if rebuild_reason in {"magnetic-reuse-original", "magnetic-reuse-original-origin-shift"}:
                rebuild_reason = "magnetic-database-origin-shift"
            elif "magnetic-database-origin-shift" not in rebuild_reason:
                rebuild_reason = f"{rebuild_reason}+magnetic-database-origin-shift"
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
        phase_from_source_operations = self._origin_shift_applied(
            source_to_std_origin_shift,
            origin_shift,
            tol=max(1.0e-6, map_tol * 10.0),
        )
        kpoint_records = []
        if canonical_kpoints.size > 0:
            kpoint_records = self._resolve_kpoint_records(
                reordered_ops,
                db,
                canonical_kpoints,
                current_to_db_prim=current_to_db_prim,
                phase_from_source_operations=phase_from_source_operations,
            )

        standardization_result = self._finalize_standardization_result(
            detected_group=detected_unitary_group,
            resolved_group=resolved_group,
            lattice_changed=lattice_changed,
            atom_permutation_only=atom_permutation_only,
            source_stru=source_path.name,
            source_hr="data-HR-sparse_SPIN0.csr",
            source_sr="data-SR-sparse_SPIN0.csr",
            atom_mapping=structure_mapping_to_dicts(mapping_result),
            lattice_old=lattice_old,
            lattice_new=lattice_new,
            lattice_transform_fractional=lattice_transform_fractional,
            xyz_axis_transform_cartesian=xyz_axis_transform_cartesian,
            rebuild_reason=rebuild_reason,
            full_matrix_from_hermitian=True,
            force_rebuild_hs=bool(origin_shift_only or origin_shift_applied),
        )
        standardization_result["structure_mapping"] = structure_mapping_summary(mapping_result)

        if RANK == 0:
            report_path = self._output_path / "symmetry_character_report.txt"
            operation_basis_conversion = self._operation_basis_conversion_applied(
                current_to_db_prim,
                match_summary_details,
                reorder_warnings,
            )
            operation_order_reference = std_unitary_ops[: len(reordered_ops)]
            symmetry_operations_reordered = self._operations_sequence_reordered(
                operation_order_reference,
                reordered_ops,
            )
            structure_atoms_reordered = self._atom_mapping_reordered(standardization_result)
            origin_redefined = phase_from_source_operations
            database_alignment_notes = []
            if origin_shift is not None and float(np.max(np.abs(origin_shift))) > max(1.0e-6, map_tol * 10.0):
                database_alignment_notes.extend(
                    [
                        "spglib operations matched kLittleGroups after primitive fractional origin shift",
                        f"primitive  basis: {self._format_translation(origin_shift)}",
                    ]
                )
            with report_path.open("w", encoding="utf-8") as f:
                self._write_report_header(
                    f,
                    detected_group=detected_unitary_group,
                    resolved_group=resolved_group,
                    dataset={},
                    db=db,
                    current_to_db_prim=current_to_db_prim,
                    standardization_result=standardization_result,
                    operation_basis_conversion=operation_basis_conversion,
                    symmetry_operations_reordered=symmetry_operations_reordered,
                    structure_atoms_reordered=structure_atoms_reordered,
                    origin_redefined=origin_redefined,
                    spgfile=spgfile,
                    database_alignment_notes=database_alignment_notes,
                    source_to_standard_origin_shift=source_to_std_origin_shift,
                    database_origin_shift=origin_shift,
                )
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
                self._write_symmetry_operations(f, source_unitary_ops)
                if canonical_kpoints.size > 0:
                    self._write_k_little_group_table(
                        f,
                        reordered_ops,
                        db,
                        canonical_kpoints,
                        phase_operations=aligned_source_ops,
                        phase_from_source_operations=phase_from_source_operations,
                    )

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
