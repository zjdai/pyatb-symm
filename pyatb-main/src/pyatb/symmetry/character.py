from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
import re
import warnings
import traceback

import numpy as np

from pyatb import OUTPUT_PATH, RANK, RUNNING_LOG
from pyatb.parallel import COMM, SIZE
from pyatb.kpt import kpoint_generator
from pyatb.tb.tb import tb as TBModel
from pyatb.io.abacus_read_xr import abacus_readHR, abacus_readSR
from pyatb.symmetry.Dk_matrix import build_dk_matrix, spin_half_matrix_from_cartesian_rotation
from pyatb.symmetry.character_core import (
    assign_irrep_combination,
    calculate_subspace_characters,
    group_degenerate_bands,
)
from pyatb.symmetry.hs_standardize import canonicalize_abacus_hs
from pyatb.symmetry.symm_stru import SymmStructureAnalyzer
from pyatb.symmetry.data_covariance_constraint import (
    get_symmetry_operations_from_metadata,
    load_abacus_hs_blocks,
    prepare_operation_contexts,
    self_covariance_statistics,
    sequential_symmetrize_hs,
    write_symmetrized_hs,
)


class Character:
    _COVARIANCE_ERROR_ABORT_THRESHOLD = 1.0e-1
    _COVARIANCE_WARNING_THRESHOLD = 1.0e-5

    def __init__(self, tb, **kwargs) -> None:
        self._tb = tb
        self._max_kpoint_num = tb.max_kpoint_num
        self._kpoint_mode = None
        self._k_generator = None
        self.output_path = os.path.join(OUTPUT_PATH, "CHARACTER")
        self._emit_character_aux_outputs = (
            os.getenv("PYATB_EMIT_CHARACTER_AUX_OUTPUTS", "0").strip().lower() in {"1", "true", "yes", "on"}
        )
        self._emit_data_symm_aux_outputs = (
            os.getenv("PYATB_EMIT_DATA_SYMM_AUX_OUTPUTS", "0").strip().lower() in {"1", "true", "yes", "on"}
        )

        if RANK == 0:
            if os.path.exists(self.output_path):
                shutil.rmtree(self.output_path)
            os.mkdir(self.output_path)

            with open(RUNNING_LOG, "a") as f:
                f.write("\n")
                f.write("\n------------------------------------------------------")
                f.write("\n|                                                    |")
                f.write("\n|                     CHARACTER                      |")
                f.write("\n|                                                    |")
                f.write("\n------------------------------------------------------")
                f.write("\n\n")

        COMM.Barrier()

    @staticmethod
    def _collect_all_kpoints(generator) -> np.ndarray:
        all_k = []
        for batch in generator:
            all_k.append(np.asarray(batch, dtype=float))
        if not all_k:
            return np.zeros((0, 3), dtype=float)
        return np.vstack(all_k)

    @staticmethod
    def _format_complex(value: complex) -> str:
        return f"{value.real: .6f}{value.imag:+.6f}i"

    @staticmethod
    def _clean_real(value: float, tol: float = 5.0e-6) -> float:
        value = float(value)
        return 0.0 if abs(value) < tol else value

    @classmethod
    def _format_trace_complex(cls, value: complex) -> str:
        return f"{cls._clean_real(value.real):12.6f}{cls._clean_real(value.imag):12.6f}"

    @classmethod
    def _format_report_complex(cls, value: complex) -> str:
        return f"{cls._clean_real(value.real, tol=5.0e-3): .2f}{cls._clean_real(value.imag, tol=5.0e-3):+.2f}i"

    @staticmethod
    def _operation_get(operation, key: str, default=None):
        if isinstance(operation, dict):
            return operation.get(key, default)
        return getattr(operation, key, default)

    def _operation_spin_matrix(self, operation) -> np.ndarray:
        spin = self._operation_get(operation, "spin_matrix")
        if spin is not None:
            return np.asarray(spin, dtype=complex)

        cart_rotation = self._operation_get(operation, "cart_rotation")
        if cart_rotation is None:
            return np.eye(2, dtype=complex)
        return spin_half_matrix_from_cartesian_rotation(np.asarray(cart_rotation, dtype=float))

    @staticmethod
    def _group_rows_by_k(rows: list[dict]) -> dict[int, list[dict]]:
        grouped: dict[int, list[dict]] = {}
        for row in rows:
            grouped.setdefault(int(row["k_index"]), []).append(row)
        for entries in grouped.values():
            entries.sort(key=lambda item: int(item["start_band"]))
        return grouped

    @staticmethod
    def _record_character_kpoint(record: dict, fallback: np.ndarray) -> np.ndarray:
        value = record.get("character_k_direct")
        if value is None:
            resolution = record.get("resolution")
            if resolution is not None:
                value = getattr(resolution, "rotated_k_prim", None)
                if value is None:
                    value = getattr(resolution, "mapped_k_prim", None)
        if value is None:
            value = fallback
        return np.asarray(value, dtype=float).reshape(3)

    @staticmethod
    def _analysis_operations_to_covariance_operations(operations: list) -> list[dict]:
        converted: list[dict] = []
        for index, operation in enumerate(operations, start=1):
            if isinstance(operation, dict):
                rotation = np.asarray(operation.get("rotation", np.eye(3, dtype=int)), dtype=int)
                translation = np.asarray(operation.get("translation", np.zeros(3, dtype=float)), dtype=float)
                cart_rotation = np.asarray(operation.get("cart_rotation", np.eye(3, dtype=float)), dtype=float)
            else:
                rotation = np.asarray(getattr(operation, "rotation", np.eye(3, dtype=int)), dtype=int)
                translation = np.asarray(getattr(operation, "translation", np.zeros(3, dtype=float)), dtype=float)
                cart_rotation = np.asarray(getattr(operation, "cart_rotation", np.eye(3, dtype=float)), dtype=float)
            converted.append(
                {
                    "index": int(index),
                    "rotation": rotation,
                    "translation": translation,
                    "cart_rotation": cart_rotation,
                }
            )
        return converted

    @classmethod
    def _max_covariance_error(cls, hr_stats: dict, sr_stats: dict) -> float:
        return float(
            max(
                float(hr_stats.get("global_max_abs", 0.0)),
                float(sr_stats.get("global_max_abs", 0.0)),
            )
        )

    @classmethod
    def _validate_covariance_statistics(
        cls,
        hr_stats: dict,
        sr_stats: dict,
        *,
        running_log_path: str | Path,
        stage_label: str,
        suggest_data_symmetrize: bool = True,
    ) -> None:
        max_error = cls._max_covariance_error(hr_stats, sr_stats)
        hr_max = float(hr_stats.get("global_max_abs", 0.0))
        hr_mean = float(hr_stats.get("mean_abs_over_operations", 0.0))
        sr_max = float(sr_stats.get("global_max_abs", 0.0))
        sr_mean = float(sr_stats.get("mean_abs_over_operations", 0.0))

        if RANK == 0:
            with open(running_log_path, "a", encoding="utf-8") as fp:
                fp.write(f"\nData Covariance Check ({stage_label})\n")
                fp.write(f"HR max/mean = {hr_max:.6e}/{hr_mean:.6e}\n")
                fp.write(f"SR max/mean = {sr_max:.6e}/{sr_mean:.6e}\n")
                fp.write(f"combined_max = {max_error:.6e}\n")

        if max_error > cls._COVARIANCE_ERROR_ABORT_THRESHOLD:
            raise ValueError("哈密顿数据或者交叠矩阵数据在对称性操作下误差过大，请检查结构和数据一致性")

        if suggest_data_symmetrize and cls._COVARIANCE_WARNING_THRESHOLD < max_error < cls._COVARIANCE_ERROR_ABORT_THRESHOLD:
            message = (
                "warning: covariance of the input HR/SR data is not high enough; "
                "set data_symmetrize = 1 to avoid irrep-character assignment exceeding tolerance"
            )
            if RANK == 0:
                with open(running_log_path, "a", encoding="utf-8") as fp:
                    fp.write(f"{message}\n")
            warnings.warn(message, UserWarning, stacklevel=2)

    def _write_trace_output(self, analysis_result: dict, rows: list[dict], occ_band: int) -> None:
        trace_path = Path(self.output_path) / "trace.txt"
        source_operations = analysis_result.get("source_operations") or analysis_result.get("operations") or []
        kpoint_records = analysis_result.get("kpoint_records") or []
        rows_by_k = self._group_rows_by_k(rows)

        with trace_path.open("w", encoding="utf-8") as handle:
            handle.write(f"{int(occ_band):3d}\n")
            handle.write(f"{1:3d}\n")
            handle.write(f"{len(source_operations):3d}\n")

            for operation in source_operations:
                rotation = np.asarray(self._operation_get(operation, "rotation", np.eye(3, dtype=int)), dtype=int)
                translation = np.asarray(self._operation_get(operation, "translation", np.zeros(3, dtype=float)), dtype=float)
                spin = self._operation_spin_matrix(operation)
                row = (
                    f"{rotation[0,0]:3d}{rotation[0,1]:3d}{rotation[0,2]:3d}"
                    f"{rotation[1,0]:3d}{rotation[1,1]:3d}{rotation[1,2]:3d}"
                    f"{rotation[2,0]:3d}{rotation[2,1]:3d}{rotation[2,2]:3d}"
                    f"{self._clean_real(translation[0]):12.6f}{self._clean_real(translation[1]):12.6f}{self._clean_real(translation[2]):12.6f}"
                    f"{self._clean_real(spin[0,0].real):12.6f}{self._clean_real(spin[0,0].imag):12.6f}"
                    f"{self._clean_real(spin[0,1].real):12.6f}{self._clean_real(spin[0,1].imag):12.6f}"
                    f"{self._clean_real(spin[1,0].real):12.6f}{self._clean_real(spin[1,0].imag):12.6f}"
                    f"{self._clean_real(spin[1,1].real):12.6f}{self._clean_real(spin[1,1].imag):12.6f}"
                )
                handle.write(f"{row}\n")

            handle.write(f"{len(kpoint_records):3d}\n")
            for record in kpoint_records:
                k_direct = np.asarray(record.get("k_direct", np.zeros(3, dtype=float)), dtype=float)
                handle.write(
                    f"{self._clean_real(k_direct[0]):12.6f}{self._clean_real(k_direct[1]):12.6f}{self._clean_real(k_direct[2]):12.6f}\n"
                )

            for record in kpoint_records:
                active_operation_indices = list(record.get("active_operation_indices", []))
                k_rows = rows_by_k.get(int(record["k_index"]), [])
                handle.write(f"{len(active_operation_indices):3d}\n")
                handle.write("".join(f"{op_index + 1:5d}" for op_index in active_operation_indices) + "\n")
                for row in k_rows:
                    output = f"{int(row['start_band']):3d}{int(row['degeneracy']):3d}{float(row['energy']):12.6f}"
                    for character in row["characters"]:
                        output += self._format_trace_complex(character)
                    handle.write(f"{output}\n")

    def _write_band_irrep_output(self, rows: list[dict]) -> None:
        output = Path(self.output_path) / "band_irrep.txt"
        rows_by_k = self._group_rows_by_k(rows)
        col_band = 8
        col_deg = 10
        col_energy = 16
        col_irrep = 24
        with output.open("w", encoding="utf-8") as handle:
            for k_index in sorted(rows_by_k):
                k_rows = rows_by_k[k_index]
                k_name = str(k_rows[0].get("k_name", "")).strip()
                handle.write(f"knum={int(k_index):3d}   kname={k_name}\n")
                handle.write(
                    f"{'band':<{col_band}s}"
                    f"{'degency':<{col_deg}s}"
                    f"{'energy':<{col_energy}s}"
                    f"{'irrrp':<{col_irrep}s}\n"
                )
                for row in k_rows:
                    handle.write(
                        f"{str(int(row['start_band'])):<{col_band}s}"
                        f"{str(int(row['degeneracy'])):<{col_deg}s}"
                        f"{float(row['energy']):<{col_energy}.6f}"
                        f"{str(row['irrep']):<{col_irrep}s}\n"
                    )
                handle.write("\n")

    def _format_report_table(self, rows: list[dict]) -> str:
        if not rows:
            return ""

        active_operation_indices = list(rows[0]["active_operation_indices"])
        header_labels = rows[0].get("active_operation_labels")
        if not header_labels or len(header_labels) != len(active_operation_indices):
            header_labels = ["E" if op_index == 0 else str(op_index + 1) for op_index in active_operation_indices]

        col_band = 8
        col_deg = 12
        col_eig = 14
        col_widths = [max(13, len(str(label)) + 2) for label in header_labels]
        lines = [
            "",
            f"{'band':^{col_band}s}{'degeneracy':^{col_deg}s}{'eigval':^{col_eig}s}"
            + "".join(f"{str(label):^{width}s}" for label, width in zip(header_labels, col_widths, strict=True))
            + " ",
        ]
        for row in rows:
            text = (
                f"{str(int(row['start_band'])):^{col_band}s}"
                f"{str(int(row['degeneracy'])):^{col_deg}s}"
                f"{float(row['energy']):^{col_eig}.6f}"
            )
            for character, width in zip(row["characters"], col_widths, strict=True):
                text += f"{self._format_report_complex(character):>{width}s}"
            text += f" ={row['irrep']}  "
            lines.append(text)
        return "\n".join(lines) + "\n"

    def _append_character_report(self, rows: list[dict]) -> None:
        report_path = Path(self.output_path) / "symmetry_character_report.txt"
        rows_by_k = self._group_rows_by_k(rows)

        if not report_path.exists():
            with report_path.open("w", encoding="utf-8") as handle:
                for k_index in sorted(rows_by_k):
                    handle.write(f"knum = {k_index:2d}    kname= \n")
                    handle.write(self._format_report_table(rows_by_k[k_index]))
            return

        lines = report_path.read_text(encoding="utf-8").splitlines(keepends=True)
        updated_lines: list[str] = []
        current_k_index: int | None = None
        inserted_for: set[int] = set()

        for line in lines:
            is_separator = line.strip() == "*" * 80
            if is_separator and current_k_index in rows_by_k and current_k_index not in inserted_for:
                updated_lines.append(self._format_report_table(rows_by_k[current_k_index]))
                inserted_for.add(current_k_index)
            updated_lines.append(line)

            match = re.match(r"\s*knum\s*=\s*(\d+)", line)
            if match:
                current_k_index = int(match.group(1))

        if current_k_index in rows_by_k and current_k_index not in inserted_for:
            updated_lines.append(self._format_report_table(rows_by_k[current_k_index]))
            inserted_for.add(current_k_index)

        missing = [k_index for k_index in sorted(rows_by_k) if k_index not in inserted_for]
        if missing:
            updated_lines.append("\n")
            for k_index in missing:
                updated_lines.append(f"knum = {k_index:2d}    kname= \n")
                updated_lines.append(self._format_report_table(rows_by_k[k_index]))

        report_path.write_text("".join(updated_lines), encoding="utf-8")

    def _set_tb_from_hs_files(
        self,
        active_stru_path: Path,
        active_hr_path: Path,
        active_sr_path: Path,
        HR_unit: str,
        lattice_constant: float,
        lattice_vector: np.ndarray,
    ) -> None:
        if int(self._tb.nspin) == 2:
            raise ValueError("CHARACTER does not support nspin=2.")

        hr_obj = abacus_readHR(int(self._tb.nspin), str(active_hr_path), str(HR_unit))
        sr_obj = abacus_readSR(int(self._tb.nspin), str(active_sr_path))
        active_tb = TBModel(
            int(self._tb.nspin),
            float(lattice_constant),
            np.asarray(lattice_vector, dtype=float),
            self._max_kpoint_num,
        )
        active_tb.set_solver_HSR(hr_obj, sr_obj, bool(getattr(self._tb, "HSR_iSsparse", False)))
        active_tb.read_stru(str(active_stru_path), need_orb=True)
        self._tb = active_tb

    def _calculate_character_rows(
        self,
        kpoints_direct: np.ndarray,
        analysis_result: dict,
        band_array: np.ndarray,
        symm_prec: float = 1.0e-6,
    ) -> list[dict]:
        if kpoints_direct.size == 0:
            return []

        source_operations = analysis_result.get("source_operations") or analysis_result.get("operations") or []
        kpoint_records = analysis_result.get("kpoint_records") or []
        if not source_operations or not kpoint_records:
            return []

        requested_start = int(band_array[0]) - 1
        requested_stop = int(band_array[1]) - 1
        local_records = [
            record
            for record in kpoint_records
            if (int(record["k_index"]) - 1) % max(int(SIZE), 1) == int(RANK)
        ]
        if local_records:
            local_kpoints = np.vstack(
                [
                    self._record_character_kpoint(
                        record,
                        np.asarray(kpoints_direct, dtype=float)[int(record["k_index"]) - 1],
                    )
                    for record in local_records
                ]
            )
            eigenvectors, eigenvalues = self._tb.tb_solver.diago_H(local_kpoints)
            overlaps = self._tb.tb_solver.get_Sk(local_kpoints)
        else:
            eigenvectors = eigenvalues = overlaps = None

        rows: list[dict] = []

        for local_pos, record in enumerate(local_records):
            k_index = int(record["k_index"]) - 1
            active_operation_indices = list(record.get("active_operation_indices", []))
            if not active_operation_indices:
                continue
            table_operation_indices = list(record.get("table_operation_indices", active_operation_indices))
            character_operation_indices = list(
                record.get("character_operation_indices")
                or record.get("database_operation_indices")
                or table_operation_indices
            )
            if not character_operation_indices:
                continue
            character_k_direct = self._record_character_kpoint(record, kpoints_direct[k_index])
            active_operation_labels = [str(op_index + 1) for op_index in character_operation_indices]

            op_matrices = [
                build_dk_matrix(
                    self._tb,
                    character_k_direct,
                    source_operations[op_index],
                    map_tol=float(symm_prec),
                )
                for op_index in character_operation_indices
            ]
            groups = group_degenerate_bands(eigenvalues[local_pos], tol=5.0e-4)

            for group_start, group_stop in groups:
                if group_stop < requested_start or group_start > requested_stop:
                    continue
                characters = calculate_subspace_characters(
                    eigenvectors=eigenvectors[local_pos],
                    overlap=overlaps[local_pos],
                    operation_matrices=op_matrices,
                    band_range=(group_start, group_stop),
                )
                max_irrep_terms = max(4, int(group_stop) - int(group_start) + 1)
                try:
                    irrep = assign_irrep_combination(
                        characters,
                        record["resolution"],
                        character_operation_indices,
                        max_terms=max_irrep_terms,
                        tol=5.0e-2,
                        spinful=int(getattr(self._tb, "nspin", 1)) == 4,
                        table_operation_indices=character_operation_indices,
                        phase_k_direct=record.get("phase_k_direct", kpoints_direct[k_index]),
                        phase_operations=source_operations,
                        table_operation_translations=record.get("table_operation_translations"),
                    )
                except Exception:
                    try:
                        irrep = assign_irrep_combination(
                            characters,
                            record["resolution"],
                            character_operation_indices,
                            max_terms=max_irrep_terms,
                            tol=1.0e-1,
                            spinful=int(getattr(self._tb, "nspin", 1)) == 4,
                            table_operation_indices=character_operation_indices,
                            phase_k_direct=record.get("phase_k_direct", kpoints_direct[k_index]),
                            phase_operations=source_operations,
                            table_operation_translations=record.get("table_operation_translations"),
                        )
                    except Exception:
                        irrep = "??"

                rows.append(
                    {
                        "k_index": k_index + 1,
                        "k_name": record.get("k_name", ""),
                        "start_band": group_start + 1,
                        "degeneracy": group_stop - group_start + 1,
                        "energy": float(np.mean(eigenvalues[local_pos, group_start : group_stop + 1])),
                        "active_operation_indices": character_operation_indices,
                        "table_operation_indices": character_operation_indices,
                        "active_operation_labels": active_operation_labels,
                        "characters": characters,
                        "irrep": irrep,
                    }
                )

        gathered_rows = COMM.gather(rows, root=0)
        if RANK != 0:
            return []

        merged_rows = [row for rank_rows in gathered_rows for row in rank_rows]
        merged_rows.sort(key=lambda row: (int(row["k_index"]), int(row["start_band"])))
        return merged_rows

    def set_k_mp(
        self,
        mp_grid,
        k_start=np.array([0.0, 0.0, 0.0], dtype=float),
        k_vect1=np.array([1.0, 0.0, 0.0], dtype=float),
        k_vect2=np.array([0.0, 1.0, 0.0], dtype=float),
        k_vect3=np.array([0.0, 0.0, 1.0], dtype=float),
        **kwargs,
    ) -> None:
        mp_grid = np.asarray(mp_grid, dtype=int)
        k_start = np.asarray(k_start, dtype=float)
        k_vect1 = np.asarray(k_vect1, dtype=float)
        k_vect2 = np.asarray(k_vect2, dtype=float)
        k_vect3 = np.asarray(k_vect3, dtype=float)

        self._kpoint_mode = "mp"
        self._k_generator = kpoint_generator.mp_generator(
            self._max_kpoint_num,
            k_start,
            k_vect1,
            k_vect2,
            k_vect3,
            mp_grid,
        )

    def set_k_line(self, high_symmetry_kpoint, kpoint_num_in_line, kpoint_label, **kwargs) -> None:
        high_symmetry_kpoint = np.asarray(high_symmetry_kpoint, dtype=float)
        kpoint_num_in_line = np.asarray(kpoint_num_in_line, dtype=int)
        kpoint_label = np.asarray(kpoint_label, dtype=str)

        self._kpoint_mode = "line"
        self._k_generator = kpoint_generator.line_generator(
            self._max_kpoint_num,
            high_symmetry_kpoint,
            kpoint_num_in_line,
        )

    def set_k_direct(self, kpoint_direct_coor, **kwargs) -> None:
        kpoint_direct_coor = np.asarray(kpoint_direct_coor, dtype=float)

        self._kpoint_mode = "direct"
        self._k_generator = kpoint_generator.array_generater(self._max_kpoint_num, kpoint_direct_coor)

    def _validate_kpoint_parameters(self, kpoint_mode, **kwargs):
        normalized = {}

        if kpoint_mode == "direct":
            if "kpoint_direct_coor" not in kwargs:
                raise ValueError("CHARACTER direct mode requires kpoint_direct_coor.")

            kpoint_direct_coor = np.asarray(kwargs["kpoint_direct_coor"], dtype=float)
            if kpoint_direct_coor.ndim != 2 or kpoint_direct_coor.shape[1] != 3 or kpoint_direct_coor.shape[0] == 0:
                raise ValueError("CHARACTER.kpoint_direct_coor must have shape (N, 3) with N > 0.")

            kpoint_num = kwargs.get("kpoint_num")
            if kpoint_num is not None and int(kpoint_num) != kpoint_direct_coor.shape[0]:
                raise ValueError("CHARACTER.kpoint_num must match the number of direct k points.")

            normalized["kpoint_direct_coor"] = kpoint_direct_coor
            return normalized

        if kpoint_mode == "line":
            required_keys = ["high_symmetry_kpoint", "kpoint_num_in_line", "kpoint_label"]
            for key in required_keys:
                if key not in kwargs:
                    raise ValueError(f"CHARACTER line mode requires {key}.")

            high_symmetry_kpoint = np.asarray(kwargs["high_symmetry_kpoint"], dtype=float)
            kpoint_num_in_line = np.asarray(kwargs["kpoint_num_in_line"], dtype=int)
            kpoint_label = np.asarray(kwargs["kpoint_label"], dtype=str)

            if high_symmetry_kpoint.ndim != 2 or high_symmetry_kpoint.shape[1] != 3 or high_symmetry_kpoint.shape[0] < 2:
                raise ValueError("CHARACTER.high_symmetry_kpoint must have shape (N, 3) with N >= 2.")
            if kpoint_num_in_line.ndim != 1 or kpoint_num_in_line.shape[0] != high_symmetry_kpoint.shape[0]:
                raise ValueError("CHARACTER.kpoint_num_in_line must align with high_symmetry_kpoint.")
            if np.any(kpoint_num_in_line <= 0):
                raise ValueError("CHARACTER.kpoint_num_in_line must contain positive integers.")
            if kpoint_label.ndim != 1 or kpoint_label.shape[0] != high_symmetry_kpoint.shape[0]:
                raise ValueError("CHARACTER.kpoint_label must align with high_symmetry_kpoint.")

            normalized["high_symmetry_kpoint"] = high_symmetry_kpoint
            normalized["kpoint_num_in_line"] = kpoint_num_in_line
            normalized["kpoint_label"] = kpoint_label
            return normalized

        if "mp_grid" not in kwargs:
            raise ValueError("CHARACTER mp mode requires mp_grid.")

        mp_grid = np.asarray(kwargs["mp_grid"], dtype=int)
        if mp_grid.shape != (3,) or np.any(mp_grid <= 0):
            raise ValueError("CHARACTER.mp_grid must contain exactly three positive integers.")

        normalized["mp_grid"] = mp_grid
        normalized["k_start"] = np.asarray(kwargs.get("k_start", [0.0, 0.0, 0.0]), dtype=float)
        normalized["k_vect1"] = np.asarray(kwargs.get("k_vect1", [1.0, 0.0, 0.0]), dtype=float)
        normalized["k_vect2"] = np.asarray(kwargs.get("k_vect2", [0.0, 1.0, 0.0]), dtype=float)
        normalized["k_vect3"] = np.asarray(kwargs.get("k_vect3", [0.0, 0.0, 1.0]), dtype=float)

        for key in ["k_start", "k_vect1", "k_vect2", "k_vect3"]:
            if normalized[key].shape != (3,):
                raise ValueError(f"CHARACTER.{key} must contain exactly three float values.")

        return normalized

    def _validate_parameters(self, kpoint_mode, group, symm_prec, occ_band, band, mag_tag, mag, **kwargs):
        if kpoint_mode not in ["direct", "line", "mp"]:
            raise ValueError("CHARACTER.kpoint_mode must be one of 'direct', 'line', or 'mp'.")

        if group != "auto" and (not isinstance(group, (int, np.integer)) or group <= 0):
            raise ValueError("CHARACTER.group must be 'auto' or a positive integer.")

        if symm_prec <= 0:
            raise ValueError("CHARACTER.symm_prec must be greater than 0.")

        if occ_band <= 0:
            raise ValueError("CHARACTER.occ_band must be a positive integer.")

        band_array = np.array(band, dtype=int)
        if band_array.shape != (2,) or np.any(band_array <= 0):
            raise ValueError("CHARACTER.band must contain exactly two positive integers.")

        if band_array[0] > band_array[1]:
            raise ValueError("CHARACTER.band must satisfy band[0] <= band[1].")

        if mag_tag not in [0, 1]:
            raise ValueError("CHARACTER.mag_tag must be 0 or 1.")

        if isinstance(mag, (list, tuple)):
            mag = np.asarray(mag, dtype=float)

        if isinstance(mag, np.ndarray):
            if mag.ndim != 1:
                raise ValueError("CHARACTER.mag must be 'auto' or a flat float list.")
        elif mag != "auto":
            raise ValueError("CHARACTER.mag must be 'auto' or a flat float list.")

        kpoint_parameters = self._validate_kpoint_parameters(kpoint_mode, **kwargs)
        return band_array, kpoint_parameters

    @staticmethod
    def _tb_atom_count(tb) -> int:
        atom_count = 0
        for atom_type in getattr(tb, "stru_atom", []) or []:
            if hasattr(atom_type, "atom_num"):
                atom_count += int(getattr(atom_type, "atom_num"))
            elif hasattr(atom_type, "cartesian_coor"):
                atom_count += int(len(getattr(atom_type, "cartesian_coor")))
        return int(atom_count)

    def _prepare_magnetic_moments(self, mag) -> np.ndarray:
        if isinstance(mag, str) and mag == "auto":
            raise ValueError("CHARACTER.mag_tag=1 requires explicit CHARACTER.mag values.")

        values = np.asarray(mag, dtype=float).reshape(-1)
        atom_count = self._tb_atom_count(self._tb)
        if atom_count <= 0:
            raise ValueError("Failed to determine atom count from STRU for CHARACTER.mag validation.")

        required = 3 * atom_count
        if values.size < required:
            raise ValueError(
                f"CHARACTER.mag_tag=1 requires at least {required} magnetic moment values "
                f"for {atom_count} atoms; got {values.size}."
            )
        return np.asarray(values[:required], dtype=float).reshape(atom_count, 3)

    def _prepare_character_on_root(
        self,
        stru_file,
        group,
        symm_prec,
        mag_tag,
        mag,
        HR_route,
        SR_route,
        HR_unit,
        data_symmetrize,
        data_symm_target_max_abs_ry,
        data_symm_max_iter_per_operation,
        data_symm_nonzero_block_tol,
        data_symm_verbose,
    ) -> dict:
        kpoints_direct = self._collect_all_kpoints(self._k_generator)
        analyzer = SymmStructureAnalyzer(self._tb, self.output_path)
        if mag_tag == 1:
            magnetic_moments = self._prepare_magnetic_moments(mag)
            analysis_result = analyzer.analyze_magnetic(
                group,
                symm_prec,
                magnetic_moments=magnetic_moments,
                kpoints_direct=kpoints_direct,
            )
        else:
            analysis_result = analyzer.analyze_nonmagnetic(group, symm_prec, kpoints_direct=kpoints_direct)
        effective_kpoints = np.asarray(analysis_result.get("canonical_kpoints_direct", kpoints_direct), dtype=float)
        active_stru_path = Path(stru_file)
        active_hr_path = Path(HR_route or "data-HR-sparse_SPIN0.csr")
        active_sr_path = Path(SR_route or "data-SR-sparse_SPIN0.csr")
        structure_standardized = bool(analysis_result.get("need_rebuild_hs"))
        active_lattice_constant = float(self._tb.lattice_constant)
        active_lattice_vector = np.asarray(self._tb.lattice_vector, dtype=float)

        if RANK == 0:
            with open(RUNNING_LOG, "a", encoding="utf-8") as fp:
                fp.write("\nStructure Standardization (CHARACTER)\n")
                fp.write(f"enabled = {int(structure_standardized)}\n")

        if analysis_result.get("need_rebuild_hs"):
            hr_source = HR_route or analysis_result.get("source_hr", "data-HR-sparse_SPIN0.csr")
            sr_source = SR_route or analysis_result.get("source_sr", "data-SR-sparse_SPIN0.csr")
            full_matrix_from_hermitian = bool(analysis_result.get("full_matrix_from_hermitian", True))
            hs_symmetry_operations = analysis_result.get("source_operations") or analysis_result.get("operations") or []
            if RANK == 0:
                canonicalize_abacus_hs(
                    tb=self._tb,
                    target_stru_path=Path(analysis_result["target_stru"]),
                    hr_route=hr_source,
                    sr_route=sr_source,
                    hr_unit=HR_unit,
                    atom_mapping=analysis_result["atom_mapping"],
                    lattice_new=np.asarray(analysis_result["lattice_new"], dtype=float),
                    lattice_transform_fractional=np.asarray(analysis_result["lattice_transform_fractional"], dtype=float),
                    xyz_axis_transform_cartesian=np.asarray(analysis_result["xyz_axis_transform_cartesian"], dtype=float),
                    output_hr_path=Path(analysis_result["target_hr"]),
                    output_sr_path=Path(analysis_result["target_sr"]),
                    mapping_output_path=(Path(self.output_path) / "R_block_mapping.txt") if self._emit_character_aux_outputs else None,
                    full_matrix_from_hermitian=full_matrix_from_hermitian,
                    symmetry_operations=hs_symmetry_operations,
                    symmetry_error_threshold=self._COVARIANCE_ERROR_ABORT_THRESHOLD,
                    symmetry_report_path=Path(self.output_path) / "hs_standardize_symmetry_check.json",
                    symmetry_map_tol=float(symm_prec),
                )

            active_stru_path = Path(analysis_result["target_stru"])
            active_lattice_vector = np.asarray(analysis_result["lattice_new"], dtype=float) / active_lattice_constant
            active_hr_path = Path(analysis_result["target_hr"])
            active_sr_path = Path(analysis_result["target_sr"])

            if RANK == 0:
                with open(RUNNING_LOG, "a", encoding="utf-8") as fp:
                    fp.write(f"standardized_stru = {active_stru_path.resolve()}\n")
                    fp.write(f"standardized_hr   = {active_hr_path.resolve()}\n")
                    fp.write(f"standardized_sr   = {active_sr_path.resolve()}\n")
        elif RANK == 0:
            with open(RUNNING_LOG, "a", encoding="utf-8") as fp:
                fp.write(f"active_stru = {active_stru_path.resolve()}\n")
                fp.write(f"active_hr   = {active_hr_path.resolve()}\n")
                fp.write(f"active_sr   = {active_sr_path.resolve()}\n")

        metadata, hr_blocks, sr_blocks = load_abacus_hs_blocks(
            stru_path=active_stru_path,
            hr_path=active_hr_path,
            sr_path=active_sr_path,
            nspin=int(self._tb.nspin),
            hr_unit=str(HR_unit),
        )
        analysis_source_operations = analysis_result.get("source_operations") or analysis_result.get("operations") or []
        if mag_tag == 1 and analysis_source_operations:
            covariance_operations = self._analysis_operations_to_covariance_operations(analysis_source_operations)
        else:
            covariance_operations = get_symmetry_operations_from_metadata(metadata, symprec=float(symm_prec))
        operation_contexts = prepare_operation_contexts(
            metadata,
            covariance_operations,
            map_tol=float(symm_prec),
        )
        before_hr = self_covariance_statistics(
            hr_blocks,
            metadata,
            covariance_operations,
            map_tol=float(symm_prec),
            nonzero_block_tol=float(data_symm_nonzero_block_tol),
            operation_contexts=operation_contexts,
        )
        before_sr = self_covariance_statistics(
            sr_blocks,
            metadata,
            covariance_operations,
            map_tol=float(symm_prec),
            nonzero_block_tol=float(data_symm_nonzero_block_tol),
            operation_contexts=operation_contexts,
        )
        self._validate_covariance_statistics(
            before_hr,
            before_sr,
            running_log_path=RUNNING_LOG,
            stage_label="before data symmetrization",
            suggest_data_symmetrize=(data_symmetrize == 0),
        )

        if data_symmetrize == 1:
            operations = covariance_operations
            hr_symm, sr_symm, symm_history = sequential_symmetrize_hs(
                hr_blocks=hr_blocks,
                sr_blocks=sr_blocks,
                metadata=metadata,
                operations=operations,
                hr_max_abs_threshold_ry=1.0e-3,
                operation_target_max_abs_ry=float(data_symm_target_max_abs_ry),
                max_iter_per_operation=int(data_symm_max_iter_per_operation),
                map_tol=float(symm_prec),
                nonzero_block_tol=float(data_symm_nonzero_block_tol),
                verbose=bool(data_symm_verbose),
                operation_contexts=operation_contexts,
            )
            after_hr = self_covariance_statistics(
                hr_symm,
                metadata,
                operations,
                map_tol=float(symm_prec),
                nonzero_block_tol=float(data_symm_nonzero_block_tol),
                operation_contexts=operation_contexts,
            )
            after_sr = self_covariance_statistics(
                sr_symm,
                metadata,
                operations,
                map_tol=float(symm_prec),
                nonzero_block_tol=float(data_symm_nonzero_block_tol),
                operation_contexts=operation_contexts,
            )
            self._validate_covariance_statistics(
                after_hr,
                after_sr,
                running_log_path=RUNNING_LOG,
                stage_label="after data symmetrization",
                suggest_data_symmetrize=False,
            )

            cov_hr_path = Path(self.output_path) / f"{active_hr_path.stem}-covsymm.csr"
            cov_sr_path = Path(self.output_path) / f"{active_sr_path.stem}-covsymm.csr"
            if RANK == 0:
                write_symmetrized_hs(
                    hr_blocks=hr_symm,
                    sr_blocks=sr_symm,
                    output_hr_path=cov_hr_path,
                    output_sr_path=cov_sr_path,
                    basis_num=int(metadata.basis_num),
                    nspin=int(self._tb.nspin),
                    hr_unit=str(HR_unit),
                )

            symm_report_path = Path(self.output_path) / "data_symmetrization_report.txt"
            hr_max_before = float(before_hr["global_max_abs"])
            hr_mean_before = float(before_hr["mean_abs_over_operations"])
            hr_max_after = float(after_hr["global_max_abs"])
            hr_mean_after = float(after_hr["mean_abs_over_operations"])
            sr_max_before = float(before_sr["global_max_abs"])
            sr_mean_before = float(before_sr["mean_abs_over_operations"])
            sr_max_after = float(after_sr["global_max_abs"])
            sr_mean_after = float(after_sr["mean_abs_over_operations"])
            symm_history_json_path = Path(self.output_path) / "data_symmetrization_history.json"

            if RANK == 0:
                if self._emit_data_symm_aux_outputs:
                    if symm_history:
                        with symm_history_json_path.open("w", encoding="utf-8") as jfp:
                            json.dump(symm_history, jfp, ensure_ascii=False, indent=2)

                    with symm_report_path.open("w", encoding="utf-8") as fp:
                        fp.write("Data Symmetrization Summary\n")
                        fp.write(f"source_stru = {active_stru_path}\n")
                        fp.write(f"source_hr   = {active_hr_path}\n")
                        fp.write(f"source_sr   = {active_sr_path}\n")
                        fp.write(f"output_hr   = {cov_hr_path}\n")
                        fp.write(f"output_sr   = {cov_sr_path}\n")
                        fp.write(f"target_max_abs_ry = {data_symm_target_max_abs_ry:.12e}\n")
                        fp.write(f"max_iter_per_operation = {data_symm_max_iter_per_operation}\n")
                        fp.write(f"nonzero_block_tol = {data_symm_nonzero_block_tol:.12e}\n")
                        fp.write(
                            f"HR before: max={hr_max_before:.12e}, mean={hr_mean_before:.12e}\n"
                            f"HR after : max={hr_max_after:.12e}, mean={hr_mean_after:.12e}\n"
                            f"HR delta : max={hr_max_after-hr_max_before:.12e}, mean={hr_mean_after-hr_mean_before:.12e}\n"
                        )
                        fp.write(
                            f"SR before: max={sr_max_before:.12e}, mean={sr_mean_before:.12e}\n"
                            f"SR after : max={sr_max_after:.12e}, mean={sr_mean_after:.12e}\n"
                            f"SR delta : max={sr_max_after-sr_max_before:.12e}, mean={sr_mean_after-sr_mean_before:.12e}\n"
                        )
                        if symm_history:
                            fp.write(f"history_entries = {len(symm_history)}\n")
                            fp.write(f"history_json = {symm_history_json_path}\n")
                            fp.write(
                                "\nPer-operation final residual (after local iteration)\n"
                                "op_idx  op_pos  iters  converged  HR_max_abs      HR_mean_abs     SR_max_abs      SR_mean_abs\n"
                            )
                            for row in symm_history:
                                if "iterations" not in row:
                                    continue
                                final_iter = row["iterations"][-1]
                                hr_stat = final_iter["HR"]
                                sr_stat = final_iter["SR"]
                                fp.write(
                                    f"{int(row['operation_index']):5d}  "
                                    f"{int(row.get('operation_position', 0)):6d}  "
                                    f"{len(row['iterations']):5d}  "
                                    f"{'Y' if bool(row.get('converged', False)) else 'N':9s}  "
                                    f"{float(hr_stat['max_abs']):.3e}  "
                                    f"{float(hr_stat['mean_abs']):.3e}  "
                                    f"{float(sr_stat['max_abs']):.3e}  "
                                    f"{float(sr_stat['mean_abs']):.3e}\n"
                                )

                            final_summary = symm_history[-1].get("final_symmetry_error", {}) if isinstance(symm_history[-1], dict) else {}
                            worst = final_summary.get("worst_error_element_in_symmetry_scan")
                            if worst is not None:
                                detail = worst.get("detail", {})
                                fp.write("\nFinal worst HR residual in symmetry scan\n")
                                fp.write(
                                    f"operation_index = {int(worst.get('operation_index', -1))}\n"
                                    f"max_abs = {float(worst.get('max_abs', 0.0)):.12e}\n"
                                    f"R = {detail.get('R', [0, 0, 0])}, row = {int(detail.get('row', 1))}, col = {int(detail.get('col', 1))}\n"
                                    f"diff = ({float(detail.get('diff_real', 0.0)):.12e}, {float(detail.get('diff_imag', 0.0)):.12e})\n"
                                )
                else:
                    for path in (symm_history_json_path, symm_report_path):
                        if path.exists():
                            path.unlink()

                with open(RUNNING_LOG, "a", encoding="utf-8") as fp:
                    fp.write("\nData Symmetrization (CHARACTER)\n")
                    fp.write("enabled = 1\n")
                    fp.write(
                        f"HR max/mean before -> after : {hr_max_before:.6e}/{hr_mean_before:.6e} -> "
                        f"{hr_max_after:.6e}/{hr_mean_after:.6e}\n"
                    )
                    fp.write(
                        f"SR max/mean before -> after : {sr_max_before:.6e}/{sr_mean_before:.6e} -> "
                        f"{sr_max_after:.6e}/{sr_mean_after:.6e}\n"
                    )
                    fp.write(f"symmetrized_hr = {cov_hr_path.resolve()}\n")
                    fp.write(f"symmetrized_sr = {cov_sr_path.resolve()}\n")

            active_hr_path = cov_hr_path
            active_sr_path = cov_sr_path

        elif RANK == 0:
            with open(RUNNING_LOG, "a", encoding="utf-8") as fp:
                fp.write("\nData Symmetrization (CHARACTER)\n")
                fp.write("enabled = 0\n")

        return {
            "analysis_result": analysis_result,
            "effective_kpoints": np.asarray(effective_kpoints, dtype=float),
            "active_stru_path": str(active_stru_path.resolve()),
            "active_hr_path": str(active_hr_path.resolve()),
            "active_sr_path": str(active_sr_path.resolve()),
            "lattice_constant": float(active_lattice_constant),
            "lattice_vector": np.asarray(active_lattice_vector, dtype=float),
        }

    def calculate_character(
        self,
        stru_file="STRU",
        kpoint_mode="direct",
        group="auto",
        symm_prec=1e-5,
        occ_band=1,
        band=(1, 1),
        mag_tag=0,
        mag="auto",
        HR_route=None,
        SR_route="data-SR-sparse_SPIN0.csr",
        HR_unit="Ry",
        data_symmetrize=0,
        data_symm_target_max_abs_ry=1.0e-8,
        data_symm_max_iter_per_operation=5,
        data_symm_nonzero_block_tol=1.0e-9,
        data_symm_verbose=0,
        **kwargs,
    ) -> None:
        data_symmetrize = int(data_symmetrize)
        if data_symmetrize not in (0, 1):
            raise ValueError("CHARACTER.data_symmetrize must be 0 or 1.")
        data_symm_max_iter_per_operation = int(data_symm_max_iter_per_operation)
        if data_symm_max_iter_per_operation <= 0:
            raise ValueError("CHARACTER.data_symm_max_iter_per_operation must be a positive integer.")
        data_symm_target_max_abs_ry = float(data_symm_target_max_abs_ry)
        if data_symm_target_max_abs_ry <= 0.0:
            raise ValueError("CHARACTER.data_symm_target_max_abs_ry must be greater than 0.")
        data_symm_nonzero_block_tol = float(data_symm_nonzero_block_tol)
        if data_symm_nonzero_block_tol <= 0.0:
            raise ValueError("CHARACTER.data_symm_nonzero_block_tol must be greater than 0.")
        data_symm_verbose = int(data_symm_verbose)
        if data_symm_verbose not in (0, 1):
            raise ValueError("CHARACTER.data_symm_verbose must be 0 or 1.")

        _, kpoint_parameters = self._validate_parameters(
            kpoint_mode=kpoint_mode,
            group=group,
            symm_prec=symm_prec,
            occ_band=occ_band,
            band=band,
            mag_tag=mag_tag,
            mag=mag,
            **kwargs,
        )

        if kpoint_mode == "mp":
            self.set_k_mp(**kpoint_parameters)
        elif kpoint_mode == "line":
            self.set_k_line(**kpoint_parameters)
        else:
            self.set_k_direct(**kpoint_parameters)

        if mag_tag == 1 and group == "auto":
            raise ValueError("CHARACTER.mag_tag=1 requires an explicit CHARACTER.group space-group number.")

        self._tb.read_stru(stru_file, need_orb=True)
        COMM.Barrier()

        try:
            preprocess_payload = self._prepare_character_on_root(
                stru_file=stru_file,
                group=group,
                symm_prec=symm_prec,
                mag_tag=mag_tag,
                mag=mag,
                HR_route=HR_route,
                SR_route=SR_route,
                HR_unit=HR_unit,
                data_symmetrize=data_symmetrize,
                data_symm_target_max_abs_ry=data_symm_target_max_abs_ry,
                data_symm_max_iter_per_operation=data_symm_max_iter_per_operation,
                data_symm_nonzero_block_tol=data_symm_nonzero_block_tol,
                data_symm_verbose=data_symm_verbose,
            )
            preprocess_payload["ok"] = True
        except Exception:
            preprocess_payload = {
                "ok": False,
                "traceback": traceback.format_exc(),
            }

        preprocess_payload = COMM.bcast(preprocess_payload if RANK == 0 else None, root=0)
        if not preprocess_payload.get("ok", False):
            raise RuntimeError("CHARACTER root preprocessing failed:\n" + str(preprocess_payload.get("traceback", "")))

        analysis_result = preprocess_payload["analysis_result"]
        effective_kpoints = np.asarray(preprocess_payload["effective_kpoints"], dtype=float)
        active_stru_path = Path(preprocess_payload["active_stru_path"])
        active_hr_path = Path(preprocess_payload["active_hr_path"])
        active_sr_path = Path(preprocess_payload["active_sr_path"])
        self._set_tb_from_hs_files(
            active_stru_path=active_stru_path,
            active_hr_path=active_hr_path,
            active_sr_path=active_sr_path,
            HR_unit=str(HR_unit),
            lattice_constant=float(preprocess_payload["lattice_constant"]),
            lattice_vector=np.asarray(preprocess_payload["lattice_vector"], dtype=float),
        )
        COMM.Barrier()

        character_rows = self._calculate_character_rows(
            effective_kpoints,
            analysis_result,
            np.asarray(band, dtype=int),
            symm_prec=float(symm_prec),
        )

        if RANK == 0 and character_rows:
            self._write_trace_output(analysis_result, character_rows, occ_band)
            self._write_band_irrep_output(character_rows)
            self._append_character_report(character_rows)
