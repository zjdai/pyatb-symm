#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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

from pyatb.symmetry.test_velocity import (  # noqa: E402
    KPointData,
    VelocityCovarianceTester,
    _k_key,
    _mapped_kpoint,
    _wrap_k,
)


def _read_character_kpoints(input_path: Path) -> list[np.ndarray]:
    lines = input_path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.strip() != "kpoint_direct_coor":
            continue
        kpoints: list[np.ndarray] = []
        for row in lines[index + 1 :]:
            parts = row.split()
            if len(parts) < 3:
                break
            try:
                kpoints.append(np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float))
            except ValueError:
                break
        if not kpoints:
            raise ValueError(f"No kpoints found after kpoint_direct_coor in {input_path}")
        return kpoints
    raise ValueError(f"Failed to find kpoint_direct_coor in {input_path}")


def _fmt_vec(values: Any, digits: int = 10) -> str:
    return ",".join(f"{float(x):.{digits}f}" for x in np.asarray(values, dtype=float).reshape(-1))


def _fmt_float(value: float) -> str:
    return f"{float(value):.12e}"


def _fmt_rotation(rotation: Any) -> str:
    mat = np.asarray(rotation, dtype=int)
    return ";".join(",".join(str(int(x)) for x in row) for row in mat)


def _k_name_map(kpoints: list[np.ndarray]) -> dict[tuple[float, float, float], list[str]]:
    names: dict[tuple[float, float, float], list[str]] = {}
    for index, kpoint in enumerate(kpoints, start=1):
        names.setdefault(_k_key(kpoint, decimals=6), []).append(f"K{index:03d}")
    return names


def _record_max(records: list[dict[str, Any]], operation_kind: str) -> dict[str, Any] | None:
    filtered = [record for record in records if str(record["operation_kind"]) == operation_kind]
    if not filtered:
        return None
    return max(filtered, key=lambda record: float(record["velocity_error"]["max_abs"]))


def _operation_lines(operation_records: list[dict[str, Any]]) -> list[str]:
    lines = ["all_symmetry_operators:"]
    seen: set[int] = set()
    operations = []
    for record in operation_records:
        op_index = int(record["operation_index"])
        if op_index in seen:
            continue
        seen.add(op_index)
        operations.append(record)
    operations.sort(key=lambda record: int(record["operation_index"]))

    for record in operations:
        lines.append(
            f"  op {int(record['operation_index']):02d}: "
            f"kind={record['operation_kind']} "
            f"time_reversal={str(bool(record['time_reversal'])).lower()} "
            f"rotation={_fmt_rotation(record['rotation'])} "
            f"translation={_fmt_vec(record['translation'], 12)}"
        )
    return lines


def _append_max_line(
    lines: list[str],
    label: str,
    record: dict[str, Any] | None,
) -> None:
    if record is None:
        lines.append(f"{label}: NA")
        return
    error = record["velocity_error"]
    lines.append(
        f"{label}: {_fmt_float(float(error['max_abs']))}   "
        f"operation_index {int(record['operation_index'])}"
    )


def _append_k_block(
    lines: list[str],
    *,
    k_index: int,
    source_k: np.ndarray,
    records: list[dict[str, Any]],
    names_by_k: dict[tuple[float, float, float], list[str]],
) -> None:
    source_key = _k_key(source_k, decimals=6)
    source_names = names_by_k.get(source_key, [f"K{k_index:03d}"])
    little = [record for record in records if record["relation"] == "little_group"]
    k_star = [record for record in records if record["relation"] == "k_star"]

    lines.append("")
    lines.append("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    lines.append(f"k_index  {int(k_index)}")
    lines.append(f"k_name: {'/'.join(source_names)}")
    lines.append(f"source_k_direct:   {_fmt_vec(source_k)}")
    lines.append(f"source_k_wrapped:  {_fmt_vec(_wrap_k(source_k))}")
    lines.append(
        "little_group_operator: "
        + (" ".join(str(int(record["operation_index"])) for record in little) if little else "none")
    )
    _append_max_line(lines, "little_group_velocity_max_abs_unitary", _record_max(little, "unitary"))
    _append_max_line(lines, "little_group_velocity_max_abs_antiunitary", _record_max(little, "antiunitary"))
    _append_max_line(lines, "all_related_velocity_max_abs_unitary", _record_max(records, "unitary"))
    _append_max_line(lines, "all_related_velocity_max_abs_antiunitary", _record_max(records, "antiunitary"))

    groups: dict[tuple[float, float, float], list[dict[str, Any]]] = {}
    for record in k_star:
        groups.setdefault(_k_key(record["target_k_wrapped"], decimals=6), []).append(record)

    lines.append("kstar:")
    if not groups:
        lines.append("  none")
        return
    for group_index, (target_key, target_records) in enumerate(sorted(groups.items()), start=1):
        target_records = sorted(target_records, key=lambda record: int(record["operation_index"]))
        target_names = names_by_k.get(target_key, [f"KSTAR_{group_index:03d}"])
        target_direct = target_records[0]["target_k_direct"]
        target_wrapped = target_records[0]["target_k_wrapped"]
        lines.append(
            f"  kstar_{group_index}: "
            f"name={'/'.join(target_names)} "
            f"target_k_direct={_fmt_vec(target_direct)} "
            f"target_k_wrapped={_fmt_vec(target_wrapped)}"
        )
        lines.append(
            "    operator: "
            + " ".join(str(int(record["operation_index"])) for record in target_records)
        )
        _append_max_line(
            lines,
            "    velocity_max_abs_unitary",
            _record_max(target_records, "unitary"),
        )
        _append_max_line(
            lines,
            "    velocity_max_abs_antiunitary",
            _record_max(target_records, "antiunitary"),
        )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _precompute_kpoint_data(tester: VelocityCovarianceTester, chunk_size: int = 128) -> int:
    unique: dict[tuple[float, float, float], np.ndarray] = {}
    for source_k in tester.kpoints:
        source = np.asarray(source_k, dtype=float)
        unique.setdefault(_k_key(source), source)
        for operation in tester.operations:
            target = _mapped_kpoint(
                source,
                np.asarray(operation["rotation"], dtype=int),
                time_reversal=bool(operation.get("time_reversal", False)),
            )
            unique.setdefault(_k_key(target), np.asarray(target, dtype=float))

    items = list(unique.items())
    for start in range(0, len(items), int(chunk_size)):
        chunk = items[start : start + int(chunk_size)]
        k_array = np.asarray([item[1] for item in chunk], dtype=float)
        eigenvectors, eigenvalues = tester.tb.tb_solver.diago_H(k_array)
        velocity_basis = tester.tb.tb_solver.get_velocity_basis_k(k_array)
        for local_index, (key, k_direct) in enumerate(chunk):
            tester._k_cache[key] = KPointData(
                k_direct=np.asarray(k_direct, dtype=float),
                eigenvectors=np.empty((0, 0), dtype=complex),
                eigenvalues=np.asarray(eigenvalues[local_index], dtype=float),
                velocity_basis=np.asarray(velocity_basis[local_index], dtype=complex),
            )
    return len(items)


def run(args: argparse.Namespace) -> dict[str, Any]:
    pyatb_dir = Path(args.pyatb_dir).resolve()
    check_dir = pyatb_dir / "check-velocity"
    check_dir.mkdir(parents=True, exist_ok=True)

    input_path = pyatb_dir / "Input"
    structure_path = pyatb_dir / "STRU"
    char_dir = pyatb_dir / "Out" / "CHARACTER"
    calc_dir = pyatb_dir.parent
    hr_path = char_dir / "data-HR-sparse_SPIN0-covsymm.csr"
    sr_path = char_dir / "data-SR-sparse_SPIN0-covsymm.csr"
    rr_path = calc_dir / "OUT.ABACUS" / "data-rR-sparse.csr"

    kpoints = _read_character_kpoints(input_path)
    tester = VelocityCovarianceTester(
        structure_path,
        hr_path=hr_path,
        sr_path=sr_path,
        rR_path=rr_path,
        nspin=int(args.nspin),
        soc=bool(int(args.nspin) == 4),
        kpoints=kpoints,
        hr_unit=str(args.hr_unit),
        rR_unit=str(args.rR_unit),
        is_sparse=bool(args.is_sparse),
        pseudo_dir=args.pseudo_orbital_dir,
        orbital_dir=args.pseudo_orbital_dir,
        with_time_reversal=True,
        symprec=float(args.symprec),
        mag_symprec=float(args.mag_symprec) if args.mag_symprec is not None else None,
        k_tol=float(args.k_tol),
        map_tol=float(args.map_tol),
    )
    unique_kpoint_count = _precompute_kpoint_data(tester, chunk_size=int(args.cache_chunk_size))
    report = tester.run()

    names_by_k = _k_name_map(kpoints)
    records_by_k: dict[int, list[dict[str, Any]]] = {}
    for record in report["records"]:
        records_by_k.setdefault(int(record["k_index"]), []).append(record)

    lines: list[str] = [
        "Velocity symmetry covariance report",
        f"Case: {pyatb_dir}",
        "Matrix inputs:",
        f"  HR_matrix: {hr_path}",
        f"  SR_matrix: {sr_path}",
        f"  rR_matrix: {rr_path}",
        f"  HR_unit: {args.hr_unit}",
        f"  rR_unit: {args.rR_unit}",
        "Convention:",
        "  unitary:     V_alpha(k') = D(g,k') [sum_beta O_alpha_beta V_beta(k)] D(g,k')^dag",
        "  antiunitary: V_alpha(k') = D(g,k') U_T [-conj(sum_beta O_alpha_beta V_beta(k))] U_T^dag D(g,k')^dag",
        "  k mapping:   unitary k'=k@inv(R), antiunitary k'=-k@inv(R)",
        "  D_phase_k:   target_k_direct",
        "",
    ]
    lines.extend(_operation_lines(report["records"]))
    for k_index, source_k in enumerate(kpoints, start=1):
        _append_k_block(
            lines,
            k_index=k_index,
            source_k=source_k,
            records=records_by_k.get(k_index, []),
            names_by_k=names_by_k,
        )

    summary_lines = [
        "",
        "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%",
        "summary:",
        f"  kpoint_count: {len(kpoints)}",
        f"  operation_count: {int(report['operation_count'])}",
        f"  unique_kpoint_count: {int(unique_kpoint_count)}",
    ]
    for name in ("all", "little_group", "k_star", "unitary", "antiunitary"):
        row = report["summary"][name]
        summary_lines.append(
            f"  {name}: count={int(row['count'])} "
            f"max_abs={_fmt_float(float(row['global_max_abs']))} "
            f"max_rel_fro={_fmt_float(float(row['max_rel_fro_over_records']))}"
        )
    lines.extend(summary_lines)

    output_txt = check_dir / "velocity_symmetry_report.txt"
    output_json = check_dir / "velocity_symmetry_report.records.json"
    output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output_json.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    return {
        "output_txt": output_txt,
        "output_json": output_json,
        "summary": report["summary"],
        "kpoint_count": len(kpoints),
        "operation_count": int(report["operation_count"]),
        "unique_kpoint_count": int(unique_kpoint_count),
        "inputs": {
            "HR_matrix": hr_path,
            "SR_matrix": sr_path,
            "rR_matrix": rr_path,
            "HR_unit": args.hr_unit,
            "rR_unit": args.rR_unit,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check velocity covariance for all CHARACTER k-points.")
    parser.add_argument(
        "--pyatb-dir",
        default=str(PROJECT_ROOT / "abacus/stru2/pc/nsoc/pyatb-6.27"),
    )
    parser.add_argument("--nspin", type=int, default=1)
    parser.add_argument("--hr-unit", default="Ry")
    parser.add_argument("--rR-unit", default="Bohr")
    parser.add_argument("--is-sparse", action="store_true")
    parser.add_argument("--symprec", type=float, default=1.0e-5)
    parser.add_argument("--mag-symprec", type=float, default=None)
    parser.add_argument("--k-tol", type=float, default=1.0e-6)
    parser.add_argument("--map-tol", type=float, default=1.0e-6)
    parser.add_argument("--cache-chunk-size", type=int, default=128)
    parser.add_argument("--pseudo-orbital-dir", default=str(PSEUDO_ORBITAL_DIR))
    args = parser.parse_args()

    result = run(args)
    print(f"Wrote {result['output_txt']}")
    print(f"Wrote {result['output_json']}")
    print(
        f"kpoint_count={result['kpoint_count']} "
        f"operation_count={result['operation_count']} "
        f"unique_kpoint_count={result['unique_kpoint_count']}"
    )
    for key, value in result["inputs"].items():
        print(f"{key}: {value}")
    for name in ("all", "little_group", "k_star", "unitary", "antiunitary"):
        row = result["summary"][name]
        print(
            f"{name}: count={int(row['count'])} "
            f"max_abs={float(row['global_max_abs']):.12e} "
            f"max_rel_fro={float(row['max_rel_fro_over_records']):.12e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
