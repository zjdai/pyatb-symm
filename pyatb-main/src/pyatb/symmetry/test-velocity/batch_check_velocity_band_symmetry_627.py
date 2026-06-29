#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
ABACUS_ROOT = ROOT / "abacus"
RESULTS_DIR = ROOT / "workflows" / "irvsp-pyatb-compare" / "results" / "velocity_band_symmetry_all_pc_627"
PYTHON = "/public/home/hlx/zjdai/software/miniforge3/envs/pyatb-symm/bin/python"
CHECK_SCRIPT = SCRIPT_DIR / "check_velocity_band_symmetry_report.py"
MODES = ("nsoc", "soc")
OUTPUT_PREFIX = "velocity_band_all_kpoints"


@dataclass(frozen=True)
class Target:
    structure: str
    mode: str
    pyatb_dir: str
    nspin: int


@dataclass
class Row:
    structure: str
    mode: str
    pyatb_dir: str
    status: str
    returncode: int
    elapsed_s: float
    record_count: int = 0
    source_k_count: int = 0
    little_group_count: int = 0
    k_star_count: int = 0
    unitary_count: int = 0
    antiunitary_count: int = 0
    target_block_mismatch_count: int = 0
    partial_edge_blocks_skipped: int = 0
    cached_unique_kpoint_count: int = 0
    transported_diag_max_abs: float = 0.0
    direct_diag_max_abs: float = 0.0
    direct_diag_nondeg_max_abs: float = 0.0
    block_trace_max_abs: float = 0.0
    basis_switch_max_abs: float = 0.0
    basis_switch_rel_fro: float = 0.0
    b_unitarity_max_abs: float = 0.0
    worst_relation: str = ""
    worst_operation_kind: str = ""
    worst_k_index: int = 0
    worst_operation_index: int = 0
    worst_band_start: int = 0
    worst_band_stop: int = 0
    message: str = ""


def structure_key(path_or_name: str | Path) -> tuple[int, str]:
    name = Path(path_or_name).name if isinstance(path_or_name, Path) else str(path_or_name)
    match = re.fullmatch(r"stru(\d+)", name)
    return (int(match.group(1)), name) if match else (10**9, name)


def parse_selected(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    selected: set[str] = set()
    for value in values:
        for item in re.split(r"[\s,]+", str(value).strip()):
            if not item:
                continue
            selected.add(item if item.startswith("stru") else f"stru{item}")
    return selected


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def discover_targets(selected: set[str] | None, excluded: set[str], modes: tuple[str, ...]) -> list[Target]:
    targets: list[Target] = []
    for structure_dir in sorted(ABACUS_ROOT.glob("stru*"), key=structure_key):
        if not structure_dir.is_dir():
            continue
        if selected is not None and structure_dir.name not in selected:
            continue
        if structure_dir.name in excluded:
            continue
        for mode in modes:
            pyatb_dir = structure_dir / "pc" / mode / "pyatb-6.27"
            char_dir = pyatb_dir / "Out" / "CHARACTER"
            required = [
                pyatb_dir / "Input",
                pyatb_dir / "STRU",
                char_dir / "data-HR-sparse_SPIN0-covsymm.csr",
                char_dir / "data-SR-sparse_SPIN0-covsymm.csr",
                structure_dir / "pc" / mode / "OUT.ABACUS" / "data-rR-sparse.csr",
            ]
            if all(path.is_file() for path in required):
                targets.append(Target(structure_dir.name, mode, rel(pyatb_dir), 1 if mode == "nsoc" else 4))
    return targets


def summary_path(target: Target) -> Path:
    return ROOT / target.pyatb_dir / "check-velocity-band" / f"{OUTPUT_PREFIX}_records.json"


def _error_max(summary: dict[str, Any], key: str) -> float:
    item = summary.get(key)
    if not isinstance(item, dict):
        return 0.0
    for value in item.values():
        if isinstance(value, dict) and "max_abs" in value:
            return float(value.get("max_abs", 0.0))
    return 0.0


def row_from_summary(target: Target, status: str, returncode: int, elapsed_s: float, message: str = "") -> Row:
    row = Row(
        structure=target.structure,
        mode=target.mode,
        pyatb_dir=target.pyatb_dir,
        status=status,
        returncode=int(returncode),
        elapsed_s=float(elapsed_s),
        message=message,
    )
    path = summary_path(target)
    if not path.is_file():
        return row
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        row.message = f"failed to read summary json: {exc}"
        return row

    summary = payload.get("summary", {}) or {}
    row.record_count = int(summary.get("record_count", 0))
    row.source_k_count = int(summary.get("source_k_count_with_records", 0))
    relation_counts = summary.get("relation_counts", {}) or {}
    operation_counts = summary.get("operation_kind_counts", {}) or {}
    row.little_group_count = int(relation_counts.get("little_group", 0))
    row.k_star_count = int(relation_counts.get("k_star", 0))
    row.unitary_count = int(operation_counts.get("unitary", 0))
    row.antiunitary_count = int(operation_counts.get("antiunitary", 0))
    row.target_block_mismatch_count = int(summary.get("target_block_mismatch_count", 0))
    row.partial_edge_blocks_skipped = int(summary.get("partial_edge_blocks_skipped", 0))
    row.cached_unique_kpoint_count = int(summary.get("cached_unique_kpoint_count", 0))

    row.transported_diag_max_abs = _error_max(summary, "worst_transported_diagonal_rotation")
    row.direct_diag_max_abs = _error_max(summary, "worst_direct_diagonal_before_U")
    row.direct_diag_nondeg_max_abs = _error_max(summary, "worst_direct_diagonal_before_U_nondegenerate")
    row.block_trace_max_abs = _error_max(summary, "worst_block_trace_before_U")
    row.b_unitarity_max_abs = _error_max(summary, "worst_B_unitarity")

    worst = summary.get("worst_basis_switch_velocity")
    if isinstance(worst, dict):
        error = worst.get("basis_switch_velocity_error", {}) or {}
        row.basis_switch_max_abs = float(error.get("max_abs", 0.0))
        row.basis_switch_rel_fro = float(error.get("rel_fro", 0.0))
        row.worst_relation = str(worst.get("relation", ""))
        row.worst_operation_kind = str(worst.get("operation_kind", ""))
        row.worst_k_index = int(worst.get("source_k_index", 0))
        row.worst_operation_index = int(worst.get("operation_index", 0))
        row.worst_band_start = int(worst.get("band_start", 0))
        row.worst_band_stop = int(worst.get("band_stop", 0))
    return row


def write_rows(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(Row("", "", "", "", 0, 0.0)).keys()), delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def run_one(target: Target, args: argparse.Namespace) -> Row:
    if args.skip_existing and summary_path(target).is_file():
        return row_from_summary(target, "existing", 0, 0.0)

    cmd = [
        PYTHON,
        str(CHECK_SCRIPT),
        "--pyatb-dir",
        target.pyatb_dir,
        "--nspin",
        str(target.nspin),
        "--relation",
        "all",
        "--operation-kind",
        "all",
        "--output-prefix",
        OUTPUT_PREFIX,
        "--cache-chunk-size",
        str(args.cache_chunk_size),
        "--max-kpoint-num",
        str(args.max_kpoint_num),
    ]
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = str(args.omp)
    env.setdefault("MPI_NUM_PROCS", "1")

    start = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    captured: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        captured.append(line.rstrip("\n"))
        print(f"    {line}", end="", flush=True)
    returncode = proc.wait()
    elapsed = time.time() - start

    log_path = ROOT / target.pyatb_dir / "check-velocity-band" / f"{OUTPUT_PREFIX}_batch.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(captured) + "\n", encoding="utf-8")
    if returncode == 0:
        return row_from_summary(target, "ok", returncode, elapsed)
    message = captured[-1] if captured else "command failed"
    return row_from_summary(target, "failed", returncode, elapsed, message=message)


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch check band-basis velocity covariance for pc pyatb-6.27 targets.")
    parser.add_argument("--structures", nargs="*", help="Optional structure list, e.g. 1 2 stru3.")
    parser.add_argument("--exclude-structures", nargs="*", default=[], help="Structures to skip.")
    parser.add_argument("--modes", nargs="*", choices=MODES, default=list(MODES))
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-after", help="Skip targets through this structure/mode, e.g. stru10:soc.")
    parser.add_argument("--omp", type=int, default=20)
    parser.add_argument("--cache-chunk-size", type=int, default=64)
    parser.add_argument("--max-kpoint-num", type=int, default=8000)
    parser.add_argument("--output", default=str(RESULTS_DIR / "velocity_band_symmetry_batch_summary.tsv"))
    args = parser.parse_args()

    selected = parse_selected(args.structures)
    excluded = parse_selected(args.exclude_structures) or set()
    targets = discover_targets(selected, excluded, tuple(args.modes))
    if args.start_after:
        marker = str(args.start_after)
        filtered: list[Target] = []
        seen_marker = False
        for target in targets:
            label = f"{target.structure}:{target.mode}"
            if not seen_marker:
                if label == marker:
                    seen_marker = True
                continue
            filtered.append(target)
        targets = filtered
    if args.limit > 0:
        targets = targets[: int(args.limit)]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "velocity_band_symmetry_targets.json").write_text(
        json.dumps([asdict(target) for target in targets], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    output = Path(args.output)
    rows: list[Row] = []
    print(
        f"target_count={len(targets)} mpi=1 omp={args.omp} skip_existing={args.skip_existing} "
        f"output={output}",
        flush=True,
    )
    for index, target in enumerate(targets, start=1):
        print(f"[{index}/{len(targets)}] {target.structure} {target.mode} {target.pyatb_dir}", flush=True)
        try:
            row = run_one(target, args)
        except Exception as exc:
            row = Row(
                target.structure,
                target.mode,
                target.pyatb_dir,
                "exception",
                1,
                0.0,
                message=repr(exc),
            )
        rows.append(row)
        write_rows(output, rows)
        print(
            f"  RESULT {row.status} basis_switch={row.basis_switch_max_abs:.12e} "
            f"diag={row.direct_diag_max_abs:.12e} trace={row.block_trace_max_abs:.12e} "
            f"Bunit={row.b_unitarity_max_abs:.12e} relation={row.worst_relation or '-'} "
            f"kind={row.worst_operation_kind or '-'} k={row.worst_k_index} "
            f"op={row.worst_operation_index} bands={row.worst_band_start}-{row.worst_band_stop} "
            f"records={row.record_count} mismatch={row.target_block_mismatch_count} "
            f"elapsed={row.elapsed_s:.1f}s",
            flush=True,
        )

    write_rows(output, rows)
    failed = [row for row in rows if row.status not in {"ok", "existing"}]
    print(f"completed={len(rows)} failed={len(failed)} summary={output}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
