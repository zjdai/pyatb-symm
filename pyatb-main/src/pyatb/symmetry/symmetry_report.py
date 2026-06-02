from __future__ import annotations

from pathlib import Path

import numpy as np


class SymmetryReportMixin:
    """Report and diagnostic output helpers for CHARACTER symmetry preprocessing."""

    @staticmethod
    def _yes_no(value) -> str:
        return "yes" if bool(value) else "no"

    @staticmethod
    def _format_report_path(path_value, default: str = "") -> str:
        if path_value is None:
            return default
        text = str(path_value)
        if not text:
            return default
        try:
            return str(Path(text).resolve())
        except (OSError, RuntimeError):
            return text

    @staticmethod
    def _format_summary_error(value, unit: str = "") -> str:
        if value is None:
            return "pending"
        suffix = f" {unit}" if unit else ""
        return f"{float(value):.6e}{suffix}"

    @staticmethod
    def _atom_shift_or_changed(source_atoms, std_atoms, standardization_result: dict, tol: float = 1.0e-8) -> bool:
        if SymmetryReportMixin._atom_mapping_reordered(standardization_result):
            return True
        for item in standardization_result.get("atom_mapping", []):
            try:
                shift = np.asarray(item.get("shift", [0, 0, 0]), dtype=float).reshape(-1)
            except AttributeError:
                continue
            if shift.size > 0 and float(np.max(np.abs(shift))) > tol:
                return True

        source_numbers = np.asarray(source_atoms.get_atomic_numbers(), dtype=int)
        std_numbers = np.asarray(std_atoms.get_atomic_numbers(), dtype=int)
        if source_numbers.shape != std_numbers.shape or not np.array_equal(source_numbers, std_numbers):
            return True

        source_scaled = np.asarray(source_atoms.get_scaled_positions(wrap=False), dtype=float)
        std_scaled = np.asarray(std_atoms.get_scaled_positions(wrap=False), dtype=float)
        if source_scaled.shape != std_scaled.shape:
            return True
        delta = source_scaled - std_scaled
        delta -= np.rint(delta)
        return bool(float(np.max(np.abs(delta))) > tol) if delta.size else False

    @classmethod
    def _write_structure_covariance_summary(
        cls,
        fp,
        standardization_result: dict,
        *,
        hamiltonian_error=None,
        overlap_error=None,
    ) -> None:
        mapping_summary = standardization_result.get("structure_mapping")
        if isinstance(mapping_summary, dict):
            lattice_error = mapping_summary.get("max_lattice_error", 0.0)
            atom_error = mapping_summary.get("max_atom_error", 0.0)
        else:
            lattice_error = None
            atom_error = None
        fp.write(
            "Symmetry operation lattice matching max error: "
            f"{cls._format_summary_error(lattice_error, 'Angstrom')}\n"
        )
        fp.write(
            "Symmetry operation atom matching max error: "
            f"{cls._format_summary_error(atom_error, 'Angstrom')}\n"
        )
        fp.write(
            "Symmetry operation Hamiltonian max error: "
            f"{cls._format_summary_error(hamiltonian_error, 'eV')}\n"
        )
        fp.write(
            "Symmetry operation overlap max error: "
            f"{cls._format_summary_error(overlap_error)}\n"
        )

    def _write_transformations(
        self,
        fp,
        source_atoms,
        std_atoms,
        _spgfile: Path,
        standardization_result: dict,
    ):
        lattice_old = np.asarray(source_atoms.cell.array, dtype=float)
        lattice_new = np.asarray(std_atoms.cell.array, dtype=float)
        reciprocal_new = np.linalg.inv(lattice_new).T
        lattice_transform = np.asarray(
            standardization_result.get("lattice_transform_fractional", np.eye(3)),
            dtype=float,
        )
        axis_rotation = np.asarray(
            standardization_result.get("xyz_axis_transform_cartesian", np.eye(3)),
            dtype=float,
        )
        stru_unfolding = not self._matrix_is_identity(lattice_transform)
        stru_rotation = not self._matrix_is_identity(axis_rotation)

        atom_shift_changed = self._atom_shift_or_changed(source_atoms, std_atoms, standardization_result)
        block_separator = "-----------------------------------------------------\n"

        fp.write("Structure Standardization summary:\n")
        fp.write(f"stru unfolding: {self._yes_no(stru_unfolding)}\n")
        fp.write(f"stru rotation: {self._yes_no(stru_rotation)}\n")
        fp.write(f"atom shift/changed: {self._yes_no(atom_shift_changed)}\n\n")

        fp.write("Original lattice vectors\n")
        for i in range(3):
            fp.write(
                f"a{i + 1} {lattice_old[i, 0]:16.8f}{lattice_old[i, 1]:16.8f}{lattice_old[i, 2]:16.8f}\n"
            )
        fp.write(block_separator)

        fp.write("Symmetrized lattice vectors\n")
        for i in range(3):
            fp.write(
                f"a{i + 1} {lattice_new[i, 0]:16.8f}{lattice_new[i, 1]:16.8f}{lattice_new[i, 2]:16.8f}\n"
            )
        fp.write(block_separator)

        fp.write("Supercell transform matrix M\n")
        for i in range(3):
            fp.write(
                f"M{i + 1} {lattice_transform[i, 0]:16.8f}{lattice_transform[i, 1]:16.8f}{lattice_transform[i, 2]:16.8f}\n"
            )
        fp.write(block_separator)

        fp.write("Rotation matrix Q (Cartesian)\n")
        for i in range(3):
            fp.write(
                f"Q{i + 1} {axis_rotation[i, 0]:16.8f}{axis_rotation[i, 1]:16.8f}{axis_rotation[i, 2]:16.8f}\n"
            )
        fp.write(block_separator)

        fp.write("Reciprocal lattice vectors of symmetrized structure\n")
        for i in range(3):
            fp.write(
                f"b{i + 1} {reciprocal_new[i, 0]:16.8f}{reciprocal_new[i, 1]:16.8f}{reciprocal_new[i, 2]:16.8f}\n"
            )

        self._write_report_separator(fp)

    @staticmethod
    def _collect_matrix_rows(mat: np.ndarray):
        m = np.asarray(mat)
        return [m[0, :], m[1, :], m[2, :]]

    @staticmethod
    def _dataset_value(dataset, key: str, default=None):
        if isinstance(dataset, dict):
            return dataset.get(key, default)
        return getattr(dataset, key, default)

    @staticmethod
    def _matrix_is_identity(matrix, tol: float = 1.0e-8) -> bool:
        arr = np.asarray(matrix, dtype=float)
        if arr.shape != (3, 3):
            return False
        return bool(np.allclose(arr, np.eye(3), atol=tol))

    @staticmethod
    def _origin_shift_applied(*shifts, tol: float = 1.0e-8) -> bool:
        for shift in shifts:
            if shift is None:
                continue
            arr = np.asarray(shift, dtype=float).reshape(-1)
            if arr.size >= 3 and float(np.max(np.abs(arr[:3]))) > tol:
                return True
        return False

    @staticmethod
    def _atom_mapping_reordered(standardization_result: dict) -> bool:
        for item in standardization_result.get("atom_mapping", []):
            try:
                if int(item.get("old_atom", -1)) != int(item.get("new_atom", -1)):
                    return True
            except AttributeError:
                continue
        return False

    @classmethod
    def _operations_sequence_reordered(
        cls,
        reference_operations: list[SymmetryOperation],
        reordered_operations: list[SymmetryOperation],
    ) -> bool:
        if len(reference_operations) != len(reordered_operations):
            return True
        for ref, current in zip(reference_operations, reordered_operations):
            if not cls._rotation_match(ref.rotation, current.rotation):
                return True
            if not cls._translation_match(ref.translation, current.translation):
                return True
        return False

    @classmethod
    def _operation_basis_conversion_applied(
        cls,
        current_to_db_prim: np.ndarray | None,
        match_summary_details: list[str] | None,
        reorder_warnings: list[str] | None,
    ) -> bool:
        if current_to_db_prim is not None and not cls._matrix_is_identity(current_to_db_prim):
            return True
        text = "\n".join((match_summary_details or []) + (reorder_warnings or [])).lower()
        return "basis conversion" in text or "conventional-to-primitive" in text

    @classmethod
    def _write_report_matrix(cls, fp, title: str, matrix) -> None:
        fp.write(f"{title}\n")
        arr = np.asarray(matrix, dtype=float)
        for i in range(3):
            fp.write(f"  {arr[i, 0]:16.8f}{arr[i, 1]:16.8f}{arr[i, 2]:16.8f}\n")

    @staticmethod
    def _write_report_separator(fp) -> None:
        fp.write("*********************************************************************************************\n")

    @classmethod
    def _write_character_report_title(cls, fp) -> None:
        width = len("*********************************************************************************************")
        cls._write_report_separator(fp)
        cls._write_report_separator(fp)
        fp.write(f"{'CHARACTER   CALCULATION':^{width}}".rstrip() + "\n")
        cls._write_report_separator(fp)
        cls._write_report_separator(fp)

    @classmethod
    def _write_report_header(
        cls,
        fp,
        *,
        detected_group: int,
        resolved_group: int,
        dataset,
        db,
        current_to_db_prim: np.ndarray | None,
        standardization_result: dict,
        operation_basis_conversion: bool,
        symmetry_operations_reordered: bool,
        structure_atoms_reordered: bool,
        origin_redefined: bool,
        spgfile: Path | None = None,
        database_alignment_notes: list[str] | None = None,
        source_to_standard_origin_shift=None,
        database_origin_shift=None,
        data_symmetrized: bool = False,
        calculated_stru=None,
        calculated_hr=None,
        calculated_sr=None,
        hamiltonian_error=None,
        overlap_error=None,
    ) -> None:
        symbol = cls._dataset_value(dataset, "international", "")
        hall = cls._dataset_value(dataset, "hall", "")
        symbol_text = f" ({symbol})" if symbol else ""
        hall_text = f", Hall symbol {hall}" if hall else ""
        init_stru = standardization_result.get("source_stru", "STRU")
        init_hr = standardization_result.get("source_hr", "data-HR-sparse_SPIN0.csr")
        init_sr = standardization_result.get("source_sr", "data-SR-sparse_SPIN0.csr")
        calc_stru = calculated_stru or standardization_result.get("target_stru", init_stru)
        calc_hr = calculated_hr or standardization_result.get("target_hr", init_hr)
        calc_sr = calculated_sr or standardization_result.get("target_sr", init_sr)

        cls._write_character_report_title(fp)
        fp.write(f"Spglib determined space group No. {int(detected_group)}{symbol_text}{hall_text}.\n")
        fp.write(f"The kLittleGroups character table number is No. {int(resolved_group)}.\n")
        fp.write(f"The CHARACTER.group number used by pyatb is No. {int(resolved_group)}.\n")
        fp.write(f"kLittleGroups space-group symbol: {db.spacegroup_symbol}\n")
        cls._write_report_matrix(
            fp,
            "Conventional-to-primitive cell transformation matrix in kLittleGroups convention:",
            db.kc2p,
        )
        cls._write_report_matrix(
            fp,
            "Primitive-to-conventional cell transformation matrix in kLittleGroups convention:",
            db.p2c,
        )
        if spgfile is not None:
            fp.write("Little group file :\n")
            fp.write(f"{Path(spgfile).resolve()}\n")
        cls._write_report_separator(fp)
        cls._write_report_separator(fp)
        fp.write("Structure and data covariance summary:\n")
        fp.write(f"Structure standardized: {cls._yes_no(standardization_result.get('need_rebuild_hs'))}\n")
        fp.write(f"Data symmetrized: {cls._yes_no(data_symmetrized)}\n")
        cls._write_structure_covariance_summary(
            fp,
            standardization_result,
            hamiltonian_error=hamiltonian_error,
            overlap_error=overlap_error,
        )
        fp.write(f"init stru file: {cls._format_report_path(init_stru)}\n")
        fp.write(f"init hr file: {cls._format_report_path(init_hr)}\n")
        fp.write(f"init sr file: {cls._format_report_path(init_sr)}\n")
        fp.write(f"calculated stru: {cls._format_report_path(calc_stru)}\n")
        fp.write(f"calculated hr file: {cls._format_report_path(calc_hr)}\n")
        fp.write(f"calculated sr file: {cls._format_report_path(calc_sr)}\n")
        cls._write_report_separator(fp)

    def _write_symmetry_operations(self, fp, operations: list[SymmetryOperation]):
        separator = "---------------------------------------------------------------------------------------\n"
        self._write_report_separator(fp)
        fp.write("Symmetry operations Pi={Ri|taui+tm}   note: defined in Symmetrized Stru\n")
        fp.write(separator)

        for i, op in enumerate(operations, start=1):
            r_rows = self._collect_matrix_rows(op.rotation)
            inv_rows = self._collect_matrix_rows(op.inverse_rotation)
            c_rows = self._collect_matrix_rows(op.cart_rotation)
            u = op.spin_matrix
            order = self._rotation_order_from_symbol(op.symbol)
            angle = 0.0 if order <= 1 else 360.0 / float(order)
            if order > 2:
                signed_angle = self._signed_rotation_angle(op.cart_rotation, op.axis)
                sense = "clockwise" if signed_angle > 0.0 else "counterclockwise"
            else:
                sense = "clockwise"

            fp.write(f"{i} ({op.symbol}): {op.description}\n")
            fp.write(f"main axes: ({op.axis[0]:6.3f}, {op.axis[1]:6.3f}, {op.axis[2]:6.3f})\n")
            fp.write(f"{sense} rotation degree: {angle:.0f}-degree\n")
            fp.write("    Ri       taui   inv(Ri)   Ri(Cartesian coord)         spin matrix\n")
            fp.write(
                f" {int(r_rows[0][0]):2d} {int(r_rows[0][1]):2d} {int(r_rows[0][2]):2d}"
                f" {op.translation[0]:7.3f}"
                f" {int(inv_rows[0][0]):3d} {int(inv_rows[0][1]):2d} {int(inv_rows[0][2]):2d}"
                f" {c_rows[0][0]:7.3f}{c_rows[0][1]:7.3f}{c_rows[0][2]:7.3f}\n"
            )
            fp.write(
                f" {int(r_rows[1][0]):2d} {int(r_rows[1][1]):2d} {int(r_rows[1][2]):2d}"
                f" {op.translation[1]:7.3f}"
                f" {int(inv_rows[1][0]):3d} {int(inv_rows[1][1]):2d} {int(inv_rows[1][2]):2d}"
                f" {c_rows[1][0]:7.3f}{c_rows[1][1]:7.3f}{c_rows[1][2]:7.3f}   "
                f" {self._format_spin_row(u[0, 0], u[0, 1])}\n"
            )
            fp.write(
                f" {int(r_rows[2][0]):2d} {int(r_rows[2][1]):2d} {int(r_rows[2][2]):2d}"
                f" {op.translation[2]:7.3f}"
                f" {int(inv_rows[2][0]):3d} {int(inv_rows[2][1]):2d} {int(inv_rows[2][2]):2d}"
                f" {c_rows[2][0]:7.3f}{c_rows[2][1]:7.3f}{c_rows[2][2]:7.3f}   "
                f" {self._format_spin_row(u[1, 0], u[1, 1])}\n"
            )
            fp.write(separator)
