from __future__ import annotations

import numpy as np

from pyatb.symmetry.k_little_groups import KPointResolution


class KPointLittleGroupMixin:
    """k-point, k-star, little-group, and kLittleGroups table helpers."""

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
    def _representative_table_operation_indices(
        cls,
        operations: list[SymmetryOperation],
        active_operation_indices: list[int],
        rotation_index: int,
    ) -> list[int]:
        """Return table columns for the actual k-star CHARACTER calculation."""
        return list(active_operation_indices)

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
    def _representative_k_direct(resolution, fallback: np.ndarray) -> np.ndarray:
        value = getattr(resolution, "rotated_k_prim", None)
        if value is None:
            value = getattr(resolution, "mapped_k_prim", None)
        if value is None:
            value = fallback
        return np.asarray(value, dtype=float)

    @staticmethod
    def _table_phase_k_direct(resolution, db, current_to_db_prim: np.ndarray | None = None) -> np.ndarray:
        k_db = np.asarray(resolution.k_conv, dtype=float) @ np.asarray(db.kc2p, dtype=float)
        if current_to_db_prim is None:
            return np.asarray(k_db, dtype=float)
        transform = np.asarray(current_to_db_prim, dtype=float)
        if np.allclose(transform, np.eye(3), atol=1.0e-12):
            return np.asarray(k_db, dtype=float)
        return np.asarray(k_db @ np.linalg.inv(transform), dtype=float)

    @staticmethod
    def _table_operation_translations_primitive(
        db,
        table_operation_indices,
        current_to_db_prim: np.ndarray | None = None,
    ) -> np.ndarray:
        indices = np.asarray(table_operation_indices, dtype=int).reshape(-1)
        translations = np.zeros((indices.size, 3), dtype=float)
        kc2p = getattr(db, "kc2p", None)
        conv_to_prim = None
        if kc2p is not None:
            transform = np.asarray(kc2p, dtype=float)
            if not np.allclose(transform, np.eye(3), atol=1.0e-12):
                conv_to_prim = np.linalg.inv(transform)
        current_transform = None
        if current_to_db_prim is not None:
            transform = np.asarray(current_to_db_prim, dtype=float)
            if not np.allclose(transform, np.eye(3), atol=1.0e-12):
                current_transform = transform

        for pos, idx in enumerate(indices):
            if int(idx) < 0 or int(idx) >= len(getattr(db, "symops", [])):
                continue
            tau = np.asarray(db.symops[int(idx)].translation, dtype=float).reshape(3)
            if conv_to_prim is not None:
                tau = conv_to_prim @ tau
            if current_transform is not None:
                tau = current_transform @ tau
            translations[pos, :] = tau
        return translations

    @classmethod
    def _validate_representative_little_group(
        cls,
        *,
        k_direct: np.ndarray,
        resolution,
        active_operation_indices: list[int],
        table_operation_indices: list[int],
        database_operation_indices: list[int],
    ) -> None:
        table_set = {int(idx) for idx in table_operation_indices}
        database_set = {int(idx) for idx in database_operation_indices}
        if table_set == database_set:
            return
        missing_in_table = sorted(database_set - table_set)
        extra_in_table = sorted(table_set - database_set)
        raise ValueError(
            "Representative k-point little-group operation mismatch with kLittleGroups table. "
            "The CHARACTER calculation must use the k-star representative and its table little group. "
            f"k={np.asarray(k_direct, dtype=float).tolist()}, "
            f"representative_k={cls._representative_k_direct(resolution, k_direct).tolist()}, "
            f"k_name={resolution.entry.name}, "
            f"star_op={int(getattr(resolution, 'rotation_index', 0))}, "
            f"current_ops={[int(idx) + 1 for idx in active_operation_indices]}, "
            f"mapped_table_ops={[int(idx) + 1 for idx in table_operation_indices]}, "
            f"database_active_ops={[int(idx) + 1 for idx in database_operation_indices]}, "
            f"missing_in_mapped_table={[int(idx) + 1 for idx in missing_in_table]}, "
            f"extra_in_mapped_table={[int(idx) + 1 for idx in extra_in_table]}"
        )

    @staticmethod
    def _shift_negative_to_unit_interval(k_prim: np.ndarray) -> np.ndarray:
        k = np.asarray(k_prim, dtype=float).copy()
        k -= np.floor(k)
        k[np.isclose(k, 1.0, atol=1.0e-6)] = 0.0
        k[np.isclose(k, 0.0, atol=1.0e-6)] = 0.0
        return k

    @staticmethod
    def _database_primitive_lattice_from_spglib_dataset(dataset, db) -> np.ndarray:
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
        db,
        kpoints_direct: np.ndarray,
        current_to_db_prim: np.ndarray | None = None,
        phase_from_source_operations: bool = False,
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
                    active_db_ops,
                    int(getattr(resolution, "rotation_index", 1)),
                )
                if not self._operation_indices_are_table_active(resolution.entry, table_ops):
                    inactive_ops = self._inactive_table_operation_indices(resolution.entry, table_ops)
                    raise ValueError(
                        "k-star little-group operation selection is not active in kLittleGroups table. "
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
                self._validate_representative_little_group(
                    k_direct=k,
                    resolution=resolution,
                    active_operation_indices=active_ops,
                    table_operation_indices=table_ops,
                    database_operation_indices=active_db_ops,
                )
                character_k_direct = self._representative_k_direct(resolution, k)
                phase_k_direct = self._table_phase_k_direct(
                    resolution,
                    db,
                    current_to_db_prim=current_to_db_prim,
                )
                table_operation_translations = self._table_operation_translations_primitive(
                    db,
                    table_ops,
                    current_to_db_prim=current_to_db_prim,
                )
                character_operation_indices = list(active_db_ops)
                resolution.cornwell_satisfied = self._cornwell_condition_satisfied(
                    character_k_direct,
                    operations,
                    [int(idx) + 1 for idx in character_operation_indices],
                )
                resolution.phase_from_source_operations = bool(phase_from_source_operations)
                records.append(
                    {
                        "k_index": ik,
                        "k_direct": np.asarray(k, dtype=float),
                        "little_group_indices": lkg,
                        "active_operation_indices": character_operation_indices,
                        "current_active_operation_indices": active_ops,
                        "table_operation_indices": table_ops,
                        "database_operation_indices": active_db_ops,
                        "character_k_direct": character_k_direct,
                        "phase_k_direct": phase_k_direct,
                        "table_operation_translations": table_operation_translations,
                        "character_operation_indices": character_operation_indices,
                        "resolution": resolution,
                        "k_name": resolution.entry.name,
                        "cornwell_satisfied": resolution.cornwell_satisfied,
                        "phase_from_source_operations": resolution.phase_from_source_operations,
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
                mapped_k_prim = self._shift_negative_to_unit_interval(mapped_k_prim)

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
                    effective_star_index = int(star_index)
                    effective_rotated = np.asarray(rotated, dtype=float).copy()
                    effective_fracdiff = float(fracdiff)
                    resolution = KPointResolution(
                        entry=entry,
                        entry_index=int(entry_index),
                        k_conv=np.asarray(k_conv, dtype=float),
                        rotation_index=effective_star_index,
                        variable_count=int(varnum),
                        mapped_k_prim=np.asarray(mapped_k_prim, dtype=float).copy(),
                        rotated_k_prim=effective_rotated.copy(),
                    )
                    active_db_ops = [
                        int(idx)
                        for idx in resolution.entry.little_group_ops
                        if int(idx) < len(operations)
                    ]
                    if effective_star_index > 1 and effective_star_index - 1 in active_db_ops:
                        identity_matches = db._reference_kpoint_matches(mapped_k_prim, tol=1.0e-5)
                        for (
                            identity_entry,
                            identity_varnum,
                            identity_k_conv,
                            identity_entry_index,
                            identity_fracdiff,
                        ) in identity_matches:
                            if int(identity_entry_index) != int(entry_index):
                                continue
                            resolution = KPointResolution(
                                entry=identity_entry,
                                entry_index=int(identity_entry_index),
                                k_conv=np.asarray(identity_k_conv, dtype=float),
                                rotation_index=1,
                                variable_count=int(identity_varnum),
                                mapped_k_prim=np.asarray(mapped_k_prim, dtype=float).copy(),
                                rotated_k_prim=np.asarray(mapped_k_prim, dtype=float).copy(),
                            )
                            effective_star_index = 1
                            effective_rotated = np.asarray(mapped_k_prim, dtype=float).copy()
                            effective_fracdiff = float(identity_fracdiff)
                            break
                    active_db_ops = [
                        int(idx)
                        for idx in resolution.entry.little_group_ops
                        if int(idx) < len(operations)
                    ]
                    table_ops = self._representative_table_operation_indices(
                        operations,
                        active_db_ops,
                        effective_star_index,
                    )
                    table_active = self._operation_indices_are_table_active(resolution.entry, table_ops)
                    entry_set = {int(idx) + 1 for idx in active_db_ops}
                    table_set = {int(idx) + 1 for idx in table_ops}
                    score = (
                        0 if table_active else 1,
                        abs(len(entry_set) - len(table_set)),
                        abs(len(active_db_ops) - len(active_ops)),
                        int(getattr(resolution, "variable_count", 9999)),
                        effective_fracdiff,
                        int(getattr(resolution, "entry_index", 10**9)),
                        int(effective_star_index if effective_star_index >= 0 else 10**6),
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
                    "k-star little-group operation selection is not active in kLittleGroups table. "
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

            self._validate_representative_little_group(
                k_direct=k,
                resolution=resolution,
                active_operation_indices=active_ops,
                table_operation_indices=table_ops,
                database_operation_indices=active_db_ops,
            )
            character_k_direct = self._representative_k_direct(resolution, k)
            phase_k_direct = self._table_phase_k_direct(
                resolution,
                db,
                current_to_db_prim=current_to_db_prim,
            )
            table_operation_translations = self._table_operation_translations_primitive(
                db,
                table_ops,
                current_to_db_prim=current_to_db_prim,
            )
            character_operation_indices = list(active_db_ops)
            resolution.cornwell_satisfied = self._cornwell_condition_satisfied(
                character_k_direct,
                operations,
                [int(idx) + 1 for idx in character_operation_indices],
            )
            resolution.phase_from_source_operations = bool(phase_from_source_operations)
            records.append(
                {
                    "k_index": ik,
                    "k_direct": np.asarray(k, dtype=float),
                    "little_group_indices": lkg,
                    "active_operation_indices": character_operation_indices,
                    "current_active_operation_indices": active_ops,
                    "table_operation_indices": table_ops,
                    "database_operation_indices": active_db_ops,
                    "character_k_direct": character_k_direct,
                    "phase_k_direct": phase_k_direct,
                    "table_operation_translations": table_operation_translations,
                    "character_operation_indices": character_operation_indices,
                    "resolution": resolution,
                    "k_name": resolution.entry.name,
                    "cornwell_satisfied": resolution.cornwell_satisfied,
                    "phase_from_source_operations": resolution.phase_from_source_operations,
                    "mapped_k_direct": np.asarray(
                        getattr(resolution, "mapped_k_prim", np.asarray(k, dtype=float)),
                        dtype=float,
                    ),
                }
            )
        return records

    def _write_k_little_group_table(
        self,
        fp,
        operations: list[SymmetryOperation],
        db,
        kpoints_direct: np.ndarray,
        *,
        phase_operations: list[SymmetryOperation] | None = None,
        phase_from_source_operations: bool = False,
        current_to_db_prim: np.ndarray | None = None,
    ):
        table_phase_operations = operations if phase_operations is None else phase_operations
        for record in self._resolve_kpoint_records(
            operations,
            db,
            kpoints_direct,
            current_to_db_prim=current_to_db_prim,
            phase_from_source_operations=phase_from_source_operations,
        ):
            ik = int(record["k_index"])
            k = np.asarray(record["k_direct"], dtype=float)
            resolution = record["resolution"]
            match = resolution.entry
            active_db_ops = list(record["active_operation_indices"])
            table_db_ops = list(record.get("table_operation_indices", active_db_ops))
            if len(table_db_ops) != len(active_db_ops):
                table_db_ops = active_db_ops

            fp.write("\n********************************************************************************\n\n")
            fp.write(f"knum = {ik:2d}    k = {k[0]:.6f} {k[1]:.6f} {k[2]:.6f}\n")
            fp.write(f"The k-point name is {match.name:<3s}\n")

            star_op_index = int(getattr(resolution, "rotation_index", 1))
            if star_op_index == 0:
                star_matrix = -np.eye(3, dtype=int)
                fp.write("The k-point is transformed by inversion-equivalent operation (-I) to k-star\n")
                fp.write("Star rotation matrix applied to k:\n")
                for row in star_matrix:
                    fp.write(f"  {int(row[0]):3d}{int(row[1]):3d}{int(row[2]):3d}\n")
            elif 1 <= star_op_index <= len(operations):
                star_op = operations[star_op_index - 1]
                star_matrix = np.asarray(star_op.inverse_rotation, dtype=int)
                if self._matrix_is_identity(star_matrix):
                    fp.write("The k-point is transformed by Identity operation to k-star\n")
                else:
                    fp.write(
                        "The k-point is transformed by "
                        f"symmetry operation #{star_op_index} ({star_op.symbol}) to k-star\n"
                    )
                    fp.write("Star rotation matrix applied to k (R^-1):\n")
                    for row in star_matrix:
                        fp.write(f"  {int(row[0]):3d}{int(row[1]):3d}{int(row[2]):3d}\n")
            else:
                fp.write(f"The k-point is transformed by unknown star operation #{star_op_index} to k-star\n")

            k_star_prim = np.asarray(
                getattr(resolution, "rotated_k_prim", getattr(resolution, "mapped_k_prim", k)),
                dtype=float,
            )
            k_star_conv = np.asarray(resolution.k_conv, dtype=float)
            fp.write(f"Primitive    basis  {k_star_prim[0]: .6f} {k_star_prim[1]: .6f} {k_star_prim[2]: .6f}\n")
            fp.write(f"Conventional basis  {k_star_conv[0]: .6f} {k_star_conv[1]: .6f} {k_star_conv[2]: .6f}\n")
            display_db_ops = list(record.get("database_operation_indices", table_db_ops))
            cornwell_ok = bool(getattr(resolution, "cornwell_satisfied", True))
            fp.write(f"Cornwell condition: {cornwell_ok}\n")
            if match.irreps:
                phase_tags = np.asarray(match.irreps[0].phase_kinds, dtype=int).reshape(-1)
                fp.write("Phase_kind")
                for idx in display_db_ops:
                    value = 0
                    if 0 <= int(idx) < phase_tags.size:
                        value = 1 if int(phase_tags[int(idx)]) == 2 else 0
                    fp.write(f"{value:4d}")
                fp.write("\n")

            fp.write(
                f"{len(display_db_ops)} symmetry operations (module lattice translations) "
                f"in space group {db.path.stem.split('_')[-1]}\n"
            )
            fp.write(
                f"{match.antisym:5d} : the existence of antiunitary symmetries. 1-exist; 0-no\n"
            )
            fp.write(" Reality")
            for idx in display_db_ops:
                fp.write(f"{idx + 1:12d}")
            fp.write("\n")

            table_operation_translations = record.get("table_operation_translations")
            table_phase_k_direct = record.get("phase_k_direct", k_star_prim)
            for ir_index, ir in enumerate(match.irreps):
                display_name = ir.raw_name[1:] if ir.raw_name.startswith("-") else ir.raw_name
                traces = db.irrep_table_character_slice(
                    resolution,
                    ir,
                    display_db_ops,
                    display_db_ops,
                    phase_k_direct=table_phase_k_direct,
                    phase_operations=table_phase_operations,
                    table_operation_translations=table_operation_translations,
                )
                fp.write(f"{ir.reality:5d}   {display_name:<6s}")
                for value in traces:
                    fp.write(f"{self._format_complex(value):>12s}")
                fp.write("\n")

                next_is_double = ir_index + 1 < len(match.irreps) and match.irreps[ir_index + 1].raw_name.startswith("-")
                current_is_single = not ir.raw_name.startswith("-")
                if current_is_single and next_is_double:
                    fp.write("        ")
                    fp.write("-" * (12 * len(display_db_ops)))
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
            for idx in display_db_ops:
                op = operations[idx]
                axis_text = f"({op.axis[0]:7.3f}, {op.axis[1]:7.3f}, {op.axis[2]:7.3f})"
                fp.write(
                    f"{op.symbol:^{col_element}s}"
                    f"{str(idx + 1):^{col_ops}s}"
                    f"{axis_text:^{col_axis}s}\n"
                )
