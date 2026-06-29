#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path("/public/home/hlx/zjdai/file-test/26.5.14-test-character")
PYATB_ROOT = Path("/public/home/hlx/zjdai/software/pyatb-symm/pyatb-main")
PYATB_BUILD = PYATB_ROOT / "build/lib.linux-x86_64-cpython-310"
PYATB_SRC = PYATB_ROOT / "src"
PSEUDO_ORBITAL_DIR = Path(
    "/public/home/hlx/zjdai/file-test/24.12.5-2d-magnetism/cell-relax/cell_relax-residul"
)


def _prepare_imports() -> None:
    sys.path.insert(0, str(PYATB_BUILD))
    import pyatb.symmetry  # noqa: PLC0415

    pyatb.symmetry.__path__.insert(0, str(PYATB_SRC / "pyatb/symmetry"))


_prepare_imports()

from pyatb.symmetry.character_core import group_degenerate_bands  # noqa: E402
from pyatb.symmetry.test_velocity import (  # noqa: E402
    KPointData,
    VelocityCovarianceTester,
    _k_key,
    _k_equivalent,
    _mapped_kpoint,
    _wrap_k,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, complex):
        z = complex(value)
        return [float(z.real), float(z.imag)]
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def _complex_pair(value: complex) -> list[float]:
    z = complex(value)
    return [float(z.real), float(z.imag)]


def _format_complex(value: complex) -> str:
    z = complex(value)
    sign = "+" if z.imag >= 0 else "-"
    return f"{z.real:.12e}{sign}{abs(z.imag):.12e}j"


def _format_vec(values: Sequence[float] | np.ndarray) -> str:
    return " ".join(f"{float(value):.10f}" for value in np.asarray(values, dtype=float).reshape(-1))


def _safe_name(prefix: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.+-]+", "_", prefix.strip())
    text = text.replace("-", "m").replace("+", "p").replace(".", "p")
    return text.strip("_") or "velocity_band"


def _read_input(path: Path) -> tuple[list[np.ndarray], tuple[int, int], int | None]:
    lines = path.read_text(encoding="utf-8").splitlines()
    kpoints: list[np.ndarray] = []
    band: tuple[int, int] | None = None
    occ_band: int | None = None
    in_kpoints = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if parts[0] == "occ_band" and len(parts) >= 2:
            occ_band = int(parts[1])
        if parts[0] == "band" and len(parts) >= 3:
            band = (int(parts[1]), int(parts[2]))
        if stripped == "kpoint_direct_coor":
            in_kpoints = True
            continue
        if in_kpoints:
            if len(parts) < 3:
                in_kpoints = False
                continue
            try:
                kpoints.append(np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float))
            except ValueError:
                in_kpoints = False
                continue
    if not kpoints:
        raise ValueError(f"No kpoints found in {path}.")
    if band is None:
        raise ValueError(f"No CHARACTER band range found in {path}.")
    return kpoints, band, occ_band


def _selected_kpoints(args: argparse.Namespace, input_kpoints: list[np.ndarray]) -> list[tuple[int, np.ndarray]]:
    if args.kpoint is not None:
        return [(0, np.asarray(args.kpoint, dtype=float))]
    if args.k_index is not None:
        k_index = int(args.k_index)
        if k_index < 1 or k_index > len(input_kpoints):
            raise ValueError(f"k-index must be in 1..{len(input_kpoints)}, got {k_index}.")
        return [(k_index, np.asarray(input_kpoints[k_index - 1], dtype=float))]
    return [(index, np.asarray(kpoint, dtype=float)) for index, kpoint in enumerate(input_kpoints, start=1)]


def _active_blocks(
    energies: np.ndarray,
    band_range: tuple[int, int],
    energy_tol: float,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    band_start, band_stop = band_range
    band_start0 = int(band_start) - 1
    band_stop0 = int(band_stop) - 1
    active: list[tuple[int, int]] = []
    skipped_partial: list[tuple[int, int]] = []
    for start, stop in group_degenerate_bands(np.asarray(energies, dtype=float), tol=float(energy_tol)):
        if stop < band_start0 or start > band_stop0:
            continue
        if start < band_start0 or stop > band_stop0:
            skipped_partial.append((start, stop))
            continue
        active.append((start, stop))
    return active, skipped_partial


def _target_group_matches(target_energies: np.ndarray, block: tuple[int, int], energy_tol: float) -> bool:
    start, stop = block
    for target_start, target_stop in group_degenerate_bands(np.asarray(target_energies, dtype=float), tol=float(energy_tol)):
        if target_start <= start <= target_stop or target_start <= stop <= target_stop:
            return bool(target_start == start and target_stop == stop)
    return False


def _exact_k_key(k_direct: Sequence[float] | np.ndarray) -> tuple[float, float, float]:
    return tuple(float(value) for value in np.round(np.asarray(k_direct, dtype=float).reshape(3), 10).tolist())


def _matrix_worst(error: np.ndarray, value_matrix: np.ndarray, reference_matrix: np.ndarray) -> dict[str, Any]:
    err = np.asarray(error, dtype=complex)
    value = np.asarray(value_matrix, dtype=complex)
    ref = np.asarray(reference_matrix, dtype=complex)
    if err.size == 0:
        return {
            "max_abs": 0.0,
            "index": [],
            "value": [0.0, 0.0],
            "reference": [0.0, 0.0],
            "difference": [0.0, 0.0],
        }
    unraveled = np.unravel_index(int(np.argmax(np.abs(err))), err.shape)
    return {
        "max_abs": float(np.abs(err[unraveled])),
        "index": [int(item + 1) for item in unraveled],
        "value": _complex_pair(value[unraveled]),
        "reference": _complex_pair(ref[unraveled]),
        "difference": _complex_pair(err[unraveled]),
    }


def _array_error(
    reference: np.ndarray,
    predicted: np.ndarray,
    *,
    directions: Sequence[str] = ("x", "y", "z"),
) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=complex)
    pred = np.asarray(predicted, dtype=complex)
    if ref.shape != pred.shape:
        raise ValueError(f"Cannot compare arrays with different shapes: {ref.shape} vs {pred.shape}.")
    diff = pred - ref
    abs_diff = np.abs(diff)
    max_abs = float(np.max(abs_diff)) if abs_diff.size else 0.0
    mean_abs = float(np.mean(abs_diff)) if abs_diff.size else 0.0
    rms_abs = float(np.sqrt(np.mean(abs_diff**2))) if abs_diff.size else 0.0
    ref_norm = float(np.linalg.norm(ref.reshape(-1)))
    rel_fro = float(np.linalg.norm(diff.reshape(-1)) / ref_norm) if ref_norm > 0.0 else 0.0

    if abs_diff.size:
        worst_index = np.unravel_index(int(np.argmax(abs_diff.reshape(-1))), abs_diff.shape)
        worst_ref = complex(ref[worst_index])
        worst_pred = complex(pred[worst_index])
        worst_diff = complex(diff[worst_index])
    else:
        worst_index = tuple(0 for _ in ref.shape)
        worst_ref = worst_pred = worst_diff = 0.0 + 0.0j

    direction_errors = []
    for direction_index, direction_name in enumerate(directions):
        if direction_index >= diff.shape[0]:
            break
        direction_diff = diff[direction_index]
        direction_ref = ref[direction_index]
        direction_abs = np.abs(direction_diff)
        direction_ref_norm = float(np.linalg.norm(direction_ref.reshape(-1)))
        direction_errors.append(
            {
                "direction": str(direction_name),
                "max_abs": float(np.max(direction_abs)) if direction_abs.size else 0.0,
                "mean_abs": float(np.mean(direction_abs)) if direction_abs.size else 0.0,
                "rms_abs": float(np.sqrt(np.mean(direction_abs**2))) if direction_abs.size else 0.0,
                "rel_fro": (
                    float(np.linalg.norm(direction_diff.reshape(-1)) / direction_ref_norm)
                    if direction_ref_norm > 0.0
                    else 0.0
                ),
                "element_count": int(direction_diff.size),
            }
        )

    worst_direction_index = int(worst_index[0]) if worst_index else 0
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rms_abs": rms_abs,
        "rel_fro": rel_fro,
        "element_count": int(diff.size),
        "worst_direction": (
            str(directions[worst_direction_index])
            if worst_direction_index < len(directions)
            else str(worst_direction_index)
        ),
        "worst_index_1_based": [int(item + 1) for item in worst_index],
        "worst_reference": _complex_pair(worst_ref),
        "worst_predicted": _complex_pair(worst_pred),
        "worst_difference": _complex_pair(worst_diff),
        "direction_errors": direction_errors,
    }


def _mix_velocity(velocity: np.ndarray, cart_rotation: np.ndarray) -> np.ndarray:
    velocity = np.asarray(velocity, dtype=complex)
    rotation = np.asarray(cart_rotation, dtype=float)
    mixed = np.zeros_like(velocity, dtype=complex)
    for alpha in range(3):
        for beta in range(3):
            coeff = float(rotation[alpha, beta])
            if abs(coeff) > 1.0e-14:
                mixed[alpha] += coeff * velocity[beta]
    return mixed


def _project_velocity(eigenvectors: np.ndarray, velocity: np.ndarray, start: int, stop: int) -> np.ndarray:
    subspace = np.asarray(eigenvectors[:, start : stop + 1], dtype=complex)
    return np.asarray(
        [subspace.conj().T @ np.asarray(velocity[alpha], dtype=complex) @ subspace for alpha in range(3)],
        dtype=complex,
    )


def _project_velocity_full(eigenvectors: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    vectors = np.asarray(eigenvectors, dtype=complex)
    return np.asarray(
        [vectors.conj().T @ np.asarray(velocity[alpha], dtype=complex) @ vectors for alpha in range(3)],
        dtype=complex,
    )


def _band_transform_matrix(
    *,
    operation: dict[str, Any],
    source_eigenvectors: np.ndarray,
    target_eigenvectors: np.ndarray,
    target_overlap: np.ndarray,
    orbital_operator: np.ndarray,
    start: int,
    stop: int,
) -> np.ndarray:
    source_block = np.asarray(source_eigenvectors[:, start : stop + 1], dtype=complex)
    target_block = np.asarray(target_eigenvectors[:, start : stop + 1], dtype=complex)
    transformed_source = source_block.conj() if bool(operation.get("time_reversal", False)) else source_block
    return (
        target_block.conj().T
        @ np.asarray(target_overlap, dtype=complex)
        @ np.asarray(orbital_operator, dtype=complex)
        @ transformed_source
    )


def _record_for_block(
    *,
    source_k_index: int,
    source_k: np.ndarray,
    target_k: np.ndarray,
    relation: str,
    operation: dict[str, Any],
    source: Any,
    target: Any,
    target_overlap: np.ndarray,
    orbital_operator: np.ndarray,
    block: tuple[int, int],
    energy_tol: float,
    include_matrices: bool,
) -> dict[str, Any]:
    start, stop = block
    target_band_velocity = _project_velocity(target.eigenvectors, target.velocity_basis, start, stop)
    source_band_velocity_raw_full = _project_velocity_full(source.eigenvectors, source.velocity_basis)
    mixed_source_velocity = _mix_velocity(source.velocity_basis, np.asarray(operation["cart_rotation"], dtype=float))
    source_band_velocity = _project_velocity(source.eigenvectors, mixed_source_velocity, start, stop)
    rotated_source_band_velocity = _mix_velocity(
        source_band_velocity_raw_full[:, start : stop + 1, start : stop + 1],
        np.asarray(operation["cart_rotation"], dtype=float),
    )
    if bool(operation.get("time_reversal", False)):
        source_band_velocity = -np.conj(source_band_velocity)
        rotated_source_band_velocity = -np.conj(rotated_source_band_velocity)

    band_transform = _band_transform_matrix(
        operation=operation,
        source_eigenvectors=source.eigenvectors,
        target_eigenvectors=target.eigenvectors,
        target_overlap=target_overlap,
        orbital_operator=orbital_operator,
        start=start,
        stop=stop,
    )
    # B maps source band-gauge vectors into the target band gauge.  The report's
    # U is defined as B^dag so the checked form is V_target = U^dag V_source U.
    u_matrix = band_transform.conj().T
    predicted_target = np.asarray(
        [u_matrix.conj().T @ source_band_velocity[alpha] @ u_matrix for alpha in range(3)],
        dtype=complex,
    )

    direct_diag_reference = np.diagonal(target_band_velocity, axis1=1, axis2=2)
    direct_diag_predicted = np.diagonal(source_band_velocity, axis1=1, axis2=2)
    transported_diag_reference = np.diagonal(rotated_source_band_velocity, axis1=1, axis2=2)
    transported_diag_predicted = np.diagonal(source_band_velocity, axis1=1, axis2=2)
    trace_reference = np.trace(target_band_velocity, axis1=1, axis2=2)
    trace_predicted = np.trace(source_band_velocity, axis1=1, axis2=2)

    identity = np.eye(band_transform.shape[0], dtype=complex)
    b_unitarity = band_transform.conj().T @ band_transform
    u_unitarity = u_matrix.conj().T @ u_matrix

    source_energy = np.asarray(source.eigenvalues[start : stop + 1], dtype=float)
    target_energy = np.asarray(target.eigenvalues[start : stop + 1], dtype=float)
    record: dict[str, Any] = {
        "source_k_index": int(source_k_index),
        "source_k_direct": np.asarray(source_k, dtype=float).tolist(),
        "source_k_wrapped": _wrap_k(source_k).tolist(),
        "target_k_direct": np.asarray(target_k, dtype=float).tolist(),
        "target_k_wrapped": _wrap_k(target_k).tolist(),
        "relation": str(relation),
        "operation_index": int(operation["index"]),
        "operation_kind": str(operation["kind"]),
        "time_reversal": bool(operation.get("time_reversal", False)),
        "rotation": np.asarray(operation["rotation"], dtype=int).tolist(),
        "translation": np.asarray(operation["translation"], dtype=float).tolist(),
        "cart_rotation": np.asarray(operation["cart_rotation"], dtype=float).tolist(),
        "band_start": int(start + 1),
        "band_stop": int(stop + 1),
        "block_size": int(stop - start + 1),
        "source_energy_min": float(np.min(source_energy)),
        "source_energy_max": float(np.max(source_energy)),
        "target_energy_min": float(np.min(target_energy)),
        "target_energy_max": float(np.max(target_energy)),
        "energy_max_abs_difference": float(np.max(np.abs(source_energy - target_energy))),
        "target_degenerate_block_matches_source": bool(
            _target_group_matches(np.asarray(target.eigenvalues, dtype=float), block, energy_tol)
        ),
        "transported_matrix_rotation_error": _array_error(rotated_source_band_velocity, source_band_velocity),
        "transported_diagonal_rotation_error": _array_error(
            transported_diag_reference,
            transported_diag_predicted,
        ),
        "direct_diagonal_error_before_U": _array_error(direct_diag_reference, direct_diag_predicted),
        "block_trace_error_before_U": _array_error(trace_reference, trace_predicted),
        "basis_switch_velocity_error": _array_error(target_band_velocity, predicted_target),
        "velocity_error_after_U": _array_error(target_band_velocity, predicted_target),
        "B_unitarity_error": _matrix_worst(b_unitarity - identity, b_unitarity, identity),
        "U_unitarity_error": _matrix_worst(u_unitarity - identity, u_unitarity, identity),
    }
    if include_matrices:
        record["B_band_transform"] = band_transform
        record["U_matrix"] = u_matrix
        record["source_band_velocity_transformed"] = source_band_velocity
        record["source_band_velocity_rotated_from_band"] = rotated_source_band_velocity
        record["target_band_velocity"] = target_band_velocity
        record["predicted_target_band_velocity_after_B_switch"] = predicted_target
        record["predicted_target_band_velocity_after_U"] = predicted_target
    return record


def _summary_record(record: dict[str, Any] | None, error_key: str) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        key: record[key]
        for key in (
            "source_k_index",
            "source_k_direct",
            "target_k_direct",
            "target_k_wrapped",
            "relation",
            "operation_index",
            "operation_kind",
            "time_reversal",
            "band_start",
            "band_stop",
            "block_size",
            "energy_max_abs_difference",
            "target_degenerate_block_matches_source",
            error_key,
        )
    }


def _worst(records: Iterable[dict[str, Any]], error_key: str) -> dict[str, Any] | None:
    items = list(records)
    if not items:
        return None
    return max(items, key=lambda item: float(item[error_key]["max_abs"]))


def _summaries_by_relation(records: list[dict[str, Any]], error_key: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for relation in ("little_group", "k_star"):
        relation_records = [record for record in records if record["relation"] == relation]
        result[relation] = _summary_record(_worst(relation_records, error_key), error_key)
    return result


def _precompute_kpoint_data(
    tester: VelocityCovarianceTester,
    selected: list[tuple[int, np.ndarray]],
    args: argparse.Namespace,
) -> int:
    if int(args.cache_chunk_size) <= 0:
        return 0

    unique: dict[tuple[float, float, float], np.ndarray] = {}
    for _source_k_index, source_k in selected:
        source = np.asarray(source_k, dtype=float)
        unique.setdefault(_k_key(source), source)
        for operation in tester.operations:
            if args.operation_kind != "all" and str(operation["kind"]) != args.operation_kind:
                continue
            target = _mapped_kpoint(
                source,
                np.asarray(operation["rotation"], dtype=int),
                time_reversal=bool(operation.get("time_reversal", False)),
            )
            relation = "little_group" if _k_equivalent(target, source, float(args.k_tol)) else "k_star"
            if args.relation != "all" and relation != args.relation:
                continue
            unique.setdefault(_k_key(target), np.asarray(target, dtype=float))

    items = list(unique.items())
    chunk_size = int(args.cache_chunk_size)
    for start in range(0, len(items), chunk_size):
        chunk = items[start : start + chunk_size]
        k_array = np.asarray([item[1] for item in chunk], dtype=float)
        eigenvectors, eigenvalues = tester.tb.tb_solver.diago_H(k_array)
        velocity_basis = tester.tb.tb_solver.get_velocity_basis_k(k_array)
        for local_index, (key, k_direct) in enumerate(chunk):
            tester._k_cache[key] = KPointData(
                k_direct=np.asarray(k_direct, dtype=float),
                eigenvectors=np.asarray(eigenvectors[local_index], dtype=complex),
                eigenvalues=np.asarray(eigenvalues[local_index], dtype=float),
                velocity_basis=np.asarray(velocity_basis[local_index], dtype=complex),
            )
    return len(items)


def _build_tester(
    pyatb_dir: Path,
    kpoints: list[np.ndarray],
    nspin: int,
    args: argparse.Namespace,
) -> VelocityCovarianceTester:
    char_dir = pyatb_dir / "Out" / "CHARACTER"
    mode_dir = pyatb_dir.parent
    return VelocityCovarianceTester(
        pyatb_dir / "STRU",
        hr_path=char_dir / "data-HR-sparse_SPIN0-covsymm.csr",
        sr_path=char_dir / "data-SR-sparse_SPIN0-covsymm.csr",
        rR_path=mode_dir / "OUT.ABACUS" / "data-rR-sparse.csr",
        nspin=int(nspin),
        soc=bool(int(nspin) == 4),
        kpoints=[np.asarray(kpoint, dtype=float).tolist() for kpoint in kpoints],
        hr_unit=str(args.hr_unit),
        rR_unit=str(args.rR_unit),
        is_sparse=bool(args.is_sparse),
        max_kpoint_num=int(args.max_kpoint_num),
        pseudo_dir=args.pseudo_orbital_dir,
        orbital_dir=args.pseudo_orbital_dir,
        with_time_reversal=not bool(args.no_time_reversal),
        symprec=float(args.symprec),
        mag_symprec=float(args.mag_symprec) if args.mag_symprec is not None else None,
        k_tol=float(args.k_tol),
        map_tol=float(args.map_tol),
        is_axial=args.is_axial,
    )


def _write_tsv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "source_k_index",
        "source_k_direct",
        "target_k_wrapped",
        "relation",
        "operation_index",
        "operation_kind",
        "band_start",
        "band_stop",
        "block_size",
        "energy_max_abs_difference",
        "target_degenerate_block_matches_source",
        "transported_diag_rotation_max_abs",
        "direct_diag_max_abs_before_U",
        "block_trace_max_abs_before_U",
        "basis_switch_velocity_max_abs",
        "basis_switch_velocity_rel_fro",
        "B_unitarity_max_abs",
        "U_unitarity_max_abs",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "source_k_index": record["source_k_index"],
                    "source_k_direct": _format_vec(record["source_k_direct"]),
                    "target_k_wrapped": _format_vec(record["target_k_wrapped"]),
                    "relation": record["relation"],
                    "operation_index": record["operation_index"],
                    "operation_kind": record["operation_kind"],
                    "band_start": record["band_start"],
                    "band_stop": record["band_stop"],
                    "block_size": record["block_size"],
                    "energy_max_abs_difference": f"{float(record['energy_max_abs_difference']):.12e}",
                    "target_degenerate_block_matches_source": str(
                        bool(record["target_degenerate_block_matches_source"])
                    ).lower(),
                    "transported_diag_rotation_max_abs": (
                        f"{float(record['transported_diagonal_rotation_error']['max_abs']):.12e}"
                    ),
                    "direct_diag_max_abs_before_U": (
                        f"{float(record['direct_diagonal_error_before_U']['max_abs']):.12e}"
                    ),
                    "block_trace_max_abs_before_U": (
                        f"{float(record['block_trace_error_before_U']['max_abs']):.12e}"
                    ),
                    "basis_switch_velocity_max_abs": (
                        f"{float(record['basis_switch_velocity_error']['max_abs']):.12e}"
                    ),
                    "basis_switch_velocity_rel_fro": (
                        f"{float(record['basis_switch_velocity_error']['rel_fro']):.12e}"
                    ),
                    "B_unitarity_max_abs": f"{float(record['B_unitarity_error']['max_abs']):.12e}",
                    "U_unitarity_max_abs": f"{float(record['U_unitarity_error']['max_abs']):.12e}",
                }
            )


def _append_worst(lines: list[str], title: str, record: dict[str, Any] | None, error_key: str) -> None:
    if record is None:
        lines.append(f"- {title}: NA")
        return
    error = record[error_key]
    lines.append(f"## {title}")
    lines.append(f"- source_k_index: {record['source_k_index']}")
    lines.append(f"- source_k_direct: {_format_vec(record['source_k_direct'])}")
    lines.append(f"- target_k_direct: {_format_vec(record['target_k_direct'])}")
    lines.append(f"- target_k_wrapped: {_format_vec(record['target_k_wrapped'])}")
    lines.append(f"- relation: {record['relation']}")
    lines.append(f"- operation: {record['operation_index']} {record['operation_kind']}")
    lines.append(f"- bands: {record['band_start']}-{record['band_stop']} block_size {record['block_size']}")
    lines.append(f"- energy_max_abs_difference: {float(record['energy_max_abs_difference']):.12e}")
    lines.append(f"- target_degenerate_block_matches_source: {record['target_degenerate_block_matches_source']}")
    lines.append(f"- max_abs: {float(error['max_abs']):.12e}")
    if "rel_fro" in error:
        lines.append(f"- rel_fro: {float(error['rel_fro']):.12e}")
    lines.append(f"- worst_direction: {error.get('worst_direction', 'NA')}")
    lines.append(f"- worst_index_1_based: {error.get('worst_index_1_based', error.get('index', []))}")
    if "worst_reference" in error:
        lines.append(f"- reference: {_format_complex(complex(*error['worst_reference']))}")
        lines.append(f"- predicted: {_format_complex(complex(*error['worst_predicted']))}")
        lines.append(f"- difference: {_format_complex(complex(*error['worst_difference']))}")
    else:
        lines.append(f"- value: {_format_complex(complex(*error['value']))}")
        lines.append(f"- reference: {_format_complex(complex(*error['reference']))}")
        lines.append(f"- difference: {_format_complex(complex(*error['difference']))}")
    lines.append("")


def _write_text(path: Path, report: dict[str, Any]) -> None:
    inputs = report["inputs"]
    summary = report["summary"]
    lines: list[str] = []
    lines.append("# Band-Basis Velocity Covariance Check")
    lines.append("")
    lines.append("## Data Sources")
    for key in ("pyatb_dir", "input_path", "structure_path", "hr_path", "sr_path", "rR_path"):
        lines.append(f"- {key}: {inputs[key]}")
    lines.append("- H/S source: CHARACTER symmetrized covsymm CSR matrices")
    lines.append("- rR source: ABACUS raw position matrix")
    lines.append("")
    lines.append("## Convention")
    for key, value in report["convention"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- selected_k_count: {inputs['selected_k_count']}")
    lines.append(f"- k_selection: {inputs['k_selection']}")
    lines.append(f"- relation_filter: {inputs['relation_filter']}")
    lines.append(f"- operation_filter: {inputs['operation_filter']}")
    lines.append(f"- band range: {inputs['character_band_range_1_based'][0]} {inputs['character_band_range_1_based'][1]}")
    lines.append(f"- energy_tol: {float(inputs['energy_tol']):.12e}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- record_count: {summary['record_count']}")
    lines.append(f"- relation_counts: {summary['relation_counts']}")
    lines.append(f"- operation_kind_counts: {summary['operation_kind_counts']}")
    lines.append(f"- source_k_count_with_records: {summary['source_k_count_with_records']}")
    lines.append(f"- partial_edge_blocks_skipped: {summary['partial_edge_blocks_skipped']}")
    lines.append(f"- target_block_mismatch_count: {summary['target_block_mismatch_count']}")
    lines.append(f"- cached_unique_kpoint_count: {summary['cached_unique_kpoint_count']}")
    lines.append("")
    _append_worst(
        lines,
        "Worst Transported Matrix Rotation Error",
        summary["worst_transported_matrix_rotation"],
        "transported_matrix_rotation_error",
    )
    _append_worst(
        lines,
        "Worst Transported Diagonal Rotation Error",
        summary["worst_transported_diagonal_rotation"],
        "transported_diagonal_rotation_error",
    )
    _append_worst(
        lines,
        "Worst Direct Diagonal Error Before B Switch",
        summary["worst_direct_diagonal_before_U"],
        "direct_diagonal_error_before_U",
    )
    _append_worst(
        lines,
        "Worst Nondegenerate Direct Diagonal Error Before B Switch",
        summary["worst_direct_diagonal_before_U_nondegenerate"],
        "direct_diagonal_error_before_U",
    )
    _append_worst(
        lines,
        "Worst Degenerate Direct Diagonal Error Before B Switch",
        summary["worst_direct_diagonal_before_U_degenerate"],
        "direct_diagonal_error_before_U",
    )
    _append_worst(
        lines,
        "Worst Block Trace Error Before B Switch",
        summary["worst_block_trace_before_U"],
        "block_trace_error_before_U",
    )
    _append_worst(
        lines,
        "Worst Basis Switch Velocity Matrix Error",
        summary["worst_basis_switch_velocity"],
        "basis_switch_velocity_error",
    )
    _append_worst(
        lines,
        "Worst B Unitarity Error",
        summary["worst_B_unitarity"],
        "B_unitarity_error",
    )
    _append_worst(
        lines,
        "Worst U Unitarity Error",
        summary["worst_U_unitarity"],
        "U_unitarity_error",
    )
    lines.append("## Worst Errors By Relation")
    for error_key in (
        "direct_diagonal_error_before_U",
        "block_trace_error_before_U",
        "basis_switch_velocity_error",
        "B_unitarity_error",
    ):
        lines.append(f"### {error_key}")
        for relation, record in summary["worst_by_relation"][error_key].items():
            if record is None:
                lines.append(f"- {relation}: NA")
                continue
            error = record[error_key]
            lines.append(
                f"- {relation}: max_abs={float(error['max_abs']):.12e} "
                f"k={record['source_k_index']} op={record['operation_index']} "
                f"{record['operation_kind']} bands={record['band_start']}-{record['band_stop']}"
            )
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    pyatb_dir = Path(args.pyatb_dir).resolve()
    output_dir = Path(args.output_dir or pyatb_dir / "check-velocity-band").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_kpoints, band_range, occ_band = _read_input(pyatb_dir / "Input")
    selected = _selected_kpoints(args, input_kpoints)
    tester = _build_tester(pyatb_dir, [kpoint for _, kpoint in selected], int(args.nspin), args)
    cached_unique_kpoint_count = _precompute_kpoint_data(tester, selected, args)

    records: list[dict[str, Any]] = []
    skipped_partial: list[dict[str, Any]] = []
    overlap_cache: dict[tuple[float, float, float], np.ndarray] = {}
    op_matrix_cache: dict[tuple[int, tuple[float, float, float]], np.ndarray] = {}

    for source_k_index, source_k in selected:
        source = tester._kpoint_data(source_k)
        active_blocks, partial_blocks = _active_blocks(source.eigenvalues, band_range, float(args.energy_tol))
        for start, stop in partial_blocks:
            skipped_partial.append(
                {
                    "source_k_index": int(source_k_index),
                    "source_k_direct": np.asarray(source_k, dtype=float).tolist(),
                    "band_start": int(start + 1),
                    "band_stop": int(stop + 1),
                }
            )

        for operation in tester.operations:
            target_k = _mapped_kpoint(
                source_k,
                np.asarray(operation["rotation"], dtype=int),
                time_reversal=bool(operation.get("time_reversal", False)),
            )
            relation = "little_group" if _k_equivalent(target_k, source_k, float(args.k_tol)) else "k_star"
            if args.relation != "all" and relation != args.relation:
                continue
            if args.operation_kind != "all" and str(operation["kind"]) != args.operation_kind:
                continue

            if relation == "little_group":
                target = source
                representation_k = target_k
            else:
                target = tester._kpoint_data(target_k)
                representation_k = target_k

            target_exact_key = _exact_k_key(representation_k)
            target_overlap = overlap_cache.get(target_exact_key)
            if target_overlap is None:
                target_overlap = np.asarray(
                    tester.tb.tb_solver.get_Sk(np.asarray([representation_k], dtype=float))[0],
                    dtype=complex,
                )
                overlap_cache[target_exact_key] = target_overlap

            op_key = (int(operation["index"]), target_exact_key)
            orbital_operator = op_matrix_cache.get(op_key)
            if orbital_operator is None:
                orbital_operator = tester._operator_matrix(operation, representation_k)
                op_matrix_cache[op_key] = orbital_operator

            for block in active_blocks:
                records.append(
                    _record_for_block(
                        source_k_index=source_k_index,
                        source_k=source_k,
                        target_k=target_k,
                        relation=relation,
                        operation=operation,
                        source=source,
                        target=target,
                        target_overlap=target_overlap,
                        orbital_operator=orbital_operator,
                        block=block,
                        energy_tol=float(args.energy_tol),
                        include_matrices=bool(args.include_matrices),
                    )
                )

    relation_counts = Counter(record["relation"] for record in records)
    operation_kind_counts = Counter(record["operation_kind"] for record in records)
    target_block_mismatch_count = sum(
        1 for record in records if not bool(record["target_degenerate_block_matches_source"])
    )
    source_k_count = len({record["source_k_index"] for record in records})
    summary = {
        "record_count": int(len(records)),
        "relation_counts": dict(relation_counts),
        "operation_kind_counts": dict(operation_kind_counts),
        "source_k_count_with_records": int(source_k_count),
        "partial_edge_blocks_skipped": int(len(skipped_partial)),
        "target_block_mismatch_count": int(target_block_mismatch_count),
        "cached_unique_kpoint_count": int(cached_unique_kpoint_count),
        "worst_transported_matrix_rotation": _summary_record(
            _worst(records, "transported_matrix_rotation_error"),
            "transported_matrix_rotation_error",
        ),
        "worst_transported_diagonal_rotation": _summary_record(
            _worst(records, "transported_diagonal_rotation_error"),
            "transported_diagonal_rotation_error",
        ),
        "worst_direct_diagonal_before_U": _summary_record(
            _worst(records, "direct_diagonal_error_before_U"),
            "direct_diagonal_error_before_U",
        ),
        "worst_direct_diagonal_before_U_nondegenerate": _summary_record(
            _worst(
                (record for record in records if int(record["block_size"]) == 1),
                "direct_diagonal_error_before_U",
            ),
            "direct_diagonal_error_before_U",
        ),
        "worst_direct_diagonal_before_U_degenerate": _summary_record(
            _worst(
                (record for record in records if int(record["block_size"]) > 1),
                "direct_diagonal_error_before_U",
            ),
            "direct_diagonal_error_before_U",
        ),
        "worst_block_trace_before_U": _summary_record(
            _worst(records, "block_trace_error_before_U"),
            "block_trace_error_before_U",
        ),
        "worst_basis_switch_velocity": _summary_record(
            _worst(records, "basis_switch_velocity_error"),
            "basis_switch_velocity_error",
        ),
        "worst_velocity_after_U": _summary_record(_worst(records, "velocity_error_after_U"), "velocity_error_after_U"),
        "worst_B_unitarity": _summary_record(_worst(records, "B_unitarity_error"), "B_unitarity_error"),
        "worst_U_unitarity": _summary_record(_worst(records, "U_unitarity_error"), "U_unitarity_error"),
        "worst_by_relation": {
            "transported_diagonal_rotation_error": _summaries_by_relation(
                records,
                "transported_diagonal_rotation_error",
            ),
            "direct_diagonal_error_before_U": _summaries_by_relation(records, "direct_diagonal_error_before_U"),
            "block_trace_error_before_U": _summaries_by_relation(records, "block_trace_error_before_U"),
            "basis_switch_velocity_error": _summaries_by_relation(records, "basis_switch_velocity_error"),
            "B_unitarity_error": _summaries_by_relation(records, "B_unitarity_error"),
        },
    }

    if args.kpoint is not None:
        k_selection = "explicit_kpoint"
        default_prefix = "velocity_band_k_" + "_".join(f"{float(x):.6g}" for x in args.kpoint)
    elif args.k_index is not None:
        k_selection = f"input_k_index_{int(args.k_index)}"
        default_prefix = f"velocity_band_{args.relation}_k{int(args.k_index)}"
    else:
        k_selection = "all_input_kpoints"
        default_prefix = f"velocity_band_{args.relation}_all_kpoints"
    prefix = _safe_name(args.output_prefix or default_prefix)

    report = {
        "convention": {
            "test_scope": "band-basis velocity covariance using little_group and k_star operations",
            "k_mapping_unitary": "k' = k @ inv(R)",
            "k_mapping_antiunitary": "k' = -k @ inv(R)",
            "orbital_D_matrix": "D_orb(g,k') is tester._operator_matrix(operation,target_k); antiunitary includes U_T",
            "transported_gauge_velocity_unitary": (
                "V_tr,alpha(k') = sum_beta O_alpha_beta V_beta(k) in the symmetry-transported band gauge"
            ),
            "transported_gauge_velocity_antiunitary": (
                "V_tr,alpha(k') = -conj(sum_beta O_alpha_beta V_beta(k)) in the symmetry-transported band gauge"
            ),
            "folded_little_group_operator": (
                "for little_group records, the band basis is folded back to source C(k); "
                "D_orb still uses the mapped target phase k' and no target-k diagonalization is used"
            ),
            "band_transform_unitary": (
                "B=C_target^dag S(k') D_orb(g,k') C(k); for little_group C_target is source C(k), "
                "for k_star C_target is direct-diagonalization C(k')"
            ),
            "band_transform_antiunitary": (
                "B=C_target^dag S(k') D_orb(g,k') U_T C(k)^*; for little_group C_target is source C(k), "
                "for k_star C_target is direct-diagonalization C(k')"
            ),
            "B_direction": "C_tr(k') = C_target(k') B, so target-gauge velocity is B V_tr(k') B^dag",
            "reported_U": "U=B^dag, equivalent legacy form: V_target_band = U^dag V_source_trans_band U",
            "source_velocity_unitary": "V_source_trans,alpha=C(k)^dag [sum_beta O_alpha_beta V_beta(k)] C(k)",
            "source_velocity_antiunitary": "V_source_trans,alpha=-conj(C(k)^dag [sum_beta O_alpha_beta V_beta(k)] C(k))",
            "target_velocity": "V_target_band,alpha=C(k')^dag V_alpha(k') C(k')",
            "transported_rotation_error": (
                "checks C^dag[sum O V_orb]C against sum O[C^dag V_orb C] inside the source band block"
            ),
            "direct_diagonal_before_U": (
                "diag(V_target_band)-diag(V_tr_band), checked before B switch; "
                "individual entries are gauge-invariant only for nondegenerate blocks"
            ),
            "block_trace_before_U": "trace(V_target_band)-trace(V_tr_band), gauge-invariant within a degenerate block",
            "basis_switch_velocity": "full matrix difference V_target_band - B V_tr_band B^dag",
            "matrix_indexing": "reported rows/columns are 1-based within each degenerate band block",
        },
        "inputs": {
            "pyatb_dir": str(pyatb_dir),
            "input_path": str(pyatb_dir / "Input"),
            "structure_path": str(pyatb_dir / "STRU"),
            "hr_path": str(pyatb_dir / "Out" / "CHARACTER" / "data-HR-sparse_SPIN0-covsymm.csr"),
            "sr_path": str(pyatb_dir / "Out" / "CHARACTER" / "data-SR-sparse_SPIN0-covsymm.csr"),
            "rR_path": str(pyatb_dir.parent / "OUT.ABACUS" / "data-rR-sparse.csr"),
            "nspin": int(args.nspin),
            "selected_k_count": int(len(selected)),
            "k_selection": k_selection,
            "selected_kpoints": [
                {"source_k_index": int(index), "source_k_direct": np.asarray(kpoint, dtype=float).tolist()}
                for index, kpoint in selected
            ],
            "relation_filter": str(args.relation),
            "operation_filter": str(args.operation_kind),
            "character_band_range_1_based": [int(band_range[0]), int(band_range[1])],
            "occ_band": occ_band,
            "energy_tol": float(args.energy_tol),
            "symprec": float(args.symprec),
            "k_tol": float(args.k_tol),
            "map_tol": float(args.map_tol),
            "hr_unit": str(args.hr_unit),
            "rR_unit": str(args.rR_unit),
        },
        "basis": {
            "basis_num": int(tester.metadata.basis_num),
            "spin_factor": int(tester.metadata.spin_factor),
            "operation_count": int(len(tester.operations)),
        },
        "skipped_partial_blocks": skipped_partial,
        "summary": summary,
        "records": records,
    }

    json_path = output_dir / f"{prefix}_records.json"
    tsv_path = output_dir / f"{prefix}_summary.tsv"
    txt_path = output_dir / f"{prefix}_report.txt"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    _write_tsv(tsv_path, records)
    _write_text(txt_path, report)

    print(f"Wrote {txt_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {tsv_path}")
    worst_velocity = summary["worst_basis_switch_velocity"]
    if worst_velocity is not None:
        error = worst_velocity["basis_switch_velocity_error"]
        print(
            "worst_basis_switch_velocity_max_abs="
            f"{float(error['max_abs']):.12e} "
            f"rel_fro={float(error['rel_fro']):.12e} "
            f"k={worst_velocity['source_k_index']} "
            f"op={worst_velocity['operation_index']} "
            f"bands={worst_velocity['band_start']}-{worst_velocity['band_stop']}"
        )
    worst_diag = summary["worst_direct_diagonal_before_U"]
    if worst_diag is not None:
        print(
            "worst_direct_diagonal_before_B_switch_max_abs="
            f"{float(worst_diag['direct_diagonal_error_before_U']['max_abs']):.12e} "
            f"k={worst_diag['source_k_index']} "
            f"op={worst_diag['operation_index']} "
            f"bands={worst_diag['band_start']}-{worst_diag['band_stop']}"
        )
    worst_trace = summary["worst_block_trace_before_U"]
    if worst_trace is not None:
        print(
            "worst_block_trace_before_B_switch_max_abs="
            f"{float(worst_trace['block_trace_error_before_U']['max_abs']):.12e} "
            f"k={worst_trace['source_k_index']} "
            f"op={worst_trace['operation_index']} "
            f"bands={worst_trace['band_start']}-{worst_trace['band_stop']}"
        )
    worst_transport = summary["worst_transported_diagonal_rotation"]
    if worst_transport is not None:
        print(
            "worst_transported_diagonal_rotation_max_abs="
            f"{float(worst_transport['transported_diagonal_rotation_error']['max_abs']):.12e} "
            f"k={worst_transport['source_k_index']} "
            f"op={worst_transport['operation_index']} "
            f"bands={worst_transport['band_start']}-{worst_transport['band_stop']}"
        )
    print(f"relation_counts={dict(relation_counts)}")
    print(f"operation_kind_counts={dict(operation_kind_counts)}")
    print(f"target_block_mismatch_count={target_block_mismatch_count}")
    print(f"cached_unique_kpoint_count={cached_unique_kpoint_count}")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check velocity covariance in the Hamiltonian band eigenbasis.")
    parser.add_argument(
        "--pyatb-dir",
        default=str(PROJECT_ROOT / "abacus/stru2/pc/nsoc/pyatb-6.27"),
        help="pyatb-6.27 directory.",
    )
    parser.add_argument("--output-dir", help="Output directory. Default: <pyatb-dir>/check-velocity-band")
    parser.add_argument("--output-prefix", help="Output file prefix.")
    parser.add_argument("--k-index", type=int, help="1-based k-point index from pyatb Input. Default: all k points.")
    parser.add_argument("--kpoint", nargs=3, type=float, metavar=("KX", "KY", "KZ"), help="Explicit direct source k point.")
    parser.add_argument("--relation", choices=("all", "little_group", "k_star"), default="all")
    parser.add_argument("--operation-kind", choices=("all", "unitary", "antiunitary"), default="all")
    parser.add_argument("--nspin", type=int, default=1, choices=(1, 4))
    parser.add_argument("--hr-unit", default="Ry")
    parser.add_argument("--rR-unit", default="Bohr")
    parser.add_argument("--is-sparse", action="store_true")
    parser.add_argument("--energy-tol", type=float, default=5.0e-4)
    parser.add_argument("--symprec", type=float, default=1.0e-5)
    parser.add_argument("--mag-symprec", type=float, default=None)
    parser.add_argument("--k-tol", type=float, default=1.0e-6)
    parser.add_argument("--map-tol", type=float, default=1.0e-6)
    parser.add_argument("--max-kpoint-num", type=int, default=8000)
    parser.add_argument("--cache-chunk-size", type=int, default=64, help="Precompute k data in chunks; <=0 disables.")
    parser.add_argument("--pseudo-orbital-dir", default=str(PSEUDO_ORBITAL_DIR))
    parser.add_argument("--no-time-reversal", action="store_true")
    parser.add_argument("--is-axial", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--include-matrices", action="store_true", help="Store matrices in the JSON records.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    run(_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
