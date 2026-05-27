from __future__ import annotations

from pathlib import Path

import numpy as np


class SymmetryReportMixin:
    """Report and diagnostic output helpers for CHARACTER symmetry preprocessing."""

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
            mapping_summary = standardization_result.get("structure_mapping")
            if isinstance(mapping_summary, dict):
                fp.write("\nStructure mapping validation\n")
                fp.write(f"matrix_relation: {mapping_summary.get('matrix_relation', 'unknown')}\n")
                fp.write(
                    "max_lattice_error: "
                    f"{float(mapping_summary.get('max_lattice_error', 0.0)):.6e}\n"
                )
                fp.write(
                    "max_atom_error: "
                    f"{float(mapping_summary.get('max_atom_error', 0.0)):.6e}\n"
                )
                fp.write(
                    "mean_atom_error: "
                    f"{float(mapping_summary.get('mean_atom_error', 0.0)):.6e}\n"
                )
                fp.write(
                    "max_fractional_error: "
                    f"{float(mapping_summary.get('max_fractional_error', 0.0)):.6e}\n"
                )
                fp.write(
                    "global_shift: "
                    f"{self._format_translation(mapping_summary.get('global_shift', np.zeros(3)))}\n"
                )
                fp.write(
                    "fractional_translation_input: "
                    f"{self._format_translation(mapping_summary.get('fractional_translation', np.zeros(3)))}\n"
                )
            fp.write("\n")

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
    ) -> None:
        symbol = cls._dataset_value(dataset, "international", "")
        hall = cls._dataset_value(dataset, "hall", "")
        symbol_text = f" ({symbol})" if symbol else ""
        hall_text = f", Hall symbol {hall}" if hall else ""
        fp.write("Preprocessing flags:\n")
        fp.write(
            "operation_pc_to_sc_conversion = "
            f"{'yes' if operation_basis_conversion else 'no'} ; "
            "whether symmetry operations required a basis conversion to match the kLittleGroups setting.\n"
        )
        fp.write(
            "symmetry_operations_reordered = "
            f"{'yes' if symmetry_operations_reordered else 'no'} ; "
            "whether detected symmetry operations were reordered to match the kLittleGroups table order.\n"
        )
        fp.write(
            "structure_atoms_reordered = "
            f"{'yes' if structure_atoms_reordered else 'no'} ; "
            "whether atom indices were reordered when mapping the input STRU to the standardized structure.\n"
        )
        fp.write(
            "origin_redefined = "
            f"{'yes' if origin_redefined else 'no'} ; "
            "whether a fractional origin shift was used to align the structure or symmetry operations.\n"
        )
        for note in database_alignment_notes or []:
            fp.write(f"{note}\n")
        fp.write(f"structure_standardized: {'yes' if standardization_result.get('need_rebuild_hs') else 'no'}\n")
        fp.write(f"standardization_rebuild_reason: {standardization_result.get('rebuild_reason', 'unknown')}\n")
        fp.write(f"active_stru: {standardization_result.get('target_stru', 'STRU')}\n")
        fp.write(f"active_hr: {standardization_result.get('target_hr', 'data-HR-sparse_SPIN0.csr')}\n")
        fp.write(f"active_sr: {standardization_result.get('target_sr', 'data-SR-sparse_SPIN0.csr')}\n")
        if source_to_standard_origin_shift is not None:
            fp.write(
                "source_to_active_origin_shift: "
                f"{cls._format_translation(source_to_standard_origin_shift)}\n"
            )
        if database_origin_shift is not None:
            fp.write(
                "database_origin_shift_for_klg_alignment: "
                f"{cls._format_translation(database_origin_shift)}\n"
            )
        cls._write_report_separator(fp)
        fp.write(f"spglib determined space group No. {int(detected_group)}{symbol_text}{hall_text}.\n")
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

    def _write_symmetry_operations(self, fp, operations: list[SymmetryOperation]):
        separator = "---------------------------------------------------------------------------------------\n"
        fp.write("Symmetry operations Pi={Ri|taui+tm}\n")
        fp.write("note: primitive basis, active CHARACTER structure\n")
        fp.write(separator)

        for i, op in enumerate(operations, start=1):
            r_rows = self._collect_matrix_rows(op.rotation)
            inv_rows = self._collect_matrix_rows(op.inverse_rotation)
            c_rows = self._collect_matrix_rows(op.cart_rotation)
            eul = np.asarray(op.euler_zyz, dtype=float)
            u = op.spin_matrix

            fp.write(f"{i} ({op.symbol}): {op.description}\n")
            order = self._rotation_order_from_symbol(op.symbol)
            if order > 2:
                sense = "clockwise" if self._signed_rotation_angle(op.cart_rotation, op.axis) > 0.0 else "counterclockwise"
                fp.write(f"{sense} rotation through ({op.axis[0]:6.3f}, {op.axis[1]:6.3f}, {op.axis[2]:6.3f})\n")
            else:
                fp.write(f"rotation through ({op.axis[0]:6.3f}, {op.axis[1]:6.3f}, {op.axis[2]:6.3f})\n")
            fp.write("    Ri       taui   inv(Ri)   Ri(Cartesian coord)  Eulers angles       spin transf.\n")
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
            fp.write(separator)
