#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
ABACUS_ROOT = ROOT / "abacus"
RESULTS_DIR = ROOT / "workflows" / "irvsp-pyatb-compare" / "results" / "velocity_symmetry_all_pc_627"
PYTHON = "/public/home/hlx/zjdai/software/miniforge3/envs/pyatb-symm/bin/python"
CHECK_SCRIPT = SCRIPT_DIR / "check_velocity_symmetry_report.py"
MODES = ("nsoc", "soc")


@dataclass(frozen=True)
class Target:
    structure: str
    mode: str
    pyatb_dir: Path
    nspin: int


@dataclass
class Row:
    structure: str
    mode: str
    pyatb_dir: str
    status: str
    returncode: int
    elapsed_s: float
    kpoint_count: int = 0
    operation_count: int = 0
    unique_kpoint_count: int = 0
    all_count: int = 0
    all_max_abs: float = 0.0
    all_max_rel_fro: float = 0.0
    little_group_count: int = 0
    little_group_max_abs: float = 0.0
    little_group_max_rel_fro: float = 0.0
    k_star_count: int = 0
    k_star_max_abs: float = 0.0
    k_star_max_rel_fro: float = 0.0
    unitary_count: int = 0
    unitary_max_abs: float = 0.0
    unitary_max_rel_fro: float = 0.0
    antiunitary_count: int = 0
    antiunitary_max_abs: float = 0.0
    antiunitary_max_rel_fro: float = 0.0
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
        for item in re.split(r"[\s,]+", value.strip()):
            if not item:
                continue
            selected.add(item if item.startswith("stru") else f"stru{item}")
    return selected


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def discover_targets(selected: set[str] | None, modes: tuple[str, ...]) -> list[Target]:
    targets: list[Target] = []
    for structure_dir in sorted(ABACUS_ROOT.glob("stru*"), key=structure_key):
        if not structure_dir.is_dir():
            continue
        if selected is not None and structure_dir.name not in selected:
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
            if not all(path.is_file() for path in required):
                continue
            targets.append(Target(structure_dir.name, mode, pyatb_dir, 1 if mode == "nsoc" else 4))
    return targets


def parse_summary(stdout: str, target: Target, elapsed: float, returncode: int) -> Row:
    row = Row(
        structure=target.structure,
        mode=target.mode,
        pyatb_dir=rel(target.pyatb_dir),
        status="ok" if returncode == 0 else "failed",
        returncode=int(returncode),
        elapsed_s=float(elapsed),
    )
    if returncode != 0:
        row.message = stdout.strip().splitlines()[-1] if stdout.strip() else "command failed"
        return row

    for line in stdout.splitlines():
        if line.startswith("kpoint_count="):
            parts = dict(part.split("=", 1) for part in line.split() if "=" in part)
            row.kpoint_count = int(parts.get("kpoint_count", 0))
            row.operation_count = int(parts.get("operation_count", 0))
            row.unique_kpoint_count = int(parts.get("unique_kpoint_count", 0))
            continue
        match = re.match(
            r"^(all|little_group|k_star|unitary|antiunitary): count=(\d+) "
            r"max_abs=([0-9.eE+-]+) max_rel_fro=([0-9.eE+-]+)",
            line,
        )
        if not match:
            continue
        name, count, max_abs, max_rel = match.groups()
        setattr(row, f"{name}_count", int(count))
        setattr(row, f"{name}_max_abs", float(max_abs))
        setattr(row, f"{name}_max_rel_fro", float(max_rel))
    return row


def existing_row(target: Target) -> Row | None:
    report_json = target.pyatb_dir / "check-velocity" / "velocity_symmetry_report.records.json"
    if not report_json.is_file():
        return None
    try:
        payload = json.loads(report_json.read_text(encoding="utf-8"))
    except Exception as exc:
        return Row(target.structure, target.mode, rel(target.pyatb_dir), "invalid_existing", 1, 0.0, message=str(exc))
    summary = payload.get("summary", {})
    row = Row(target.structure, target.mode, rel(target.pyatb_dir), "existing", 0, 0.0)
    row.operation_count = int(payload.get("operation_count", 0))
    k_indices = {int(item.get("k_index", 0)) for item in payload.get("records", [])}
    row.kpoint_count = len(k_indices)
    for name in ("all", "little_group", "k_star", "unitary", "antiunitary"):
        item = summary.get(name, {})
        setattr(row, f"{name}_count", int(item.get("count", 0)))
        setattr(row, f"{name}_max_abs", float(item.get("global_max_abs", 0.0)))
        setattr(row, f"{name}_max_rel_fro", float(item.get("max_rel_fro_over_records", 0.0)))
    return row


def write_rows(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(Row("", "", "", "", 0, 0.0)).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def run_one(target: Target, args: argparse.Namespace) -> Row:
    if args.skip_existing:
        row = existing_row(target)
        if row is not None and row.status == "existing":
            return row

    cmd = [
        PYTHON,
        str(CHECK_SCRIPT),
        "--pyatb-dir",
        str(target.pyatb_dir),
        "--nspin",
        str(target.nspin),
        "--cache-chunk-size",
        str(args.cache_chunk_size),
    ]
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.timeout_s if args.timeout_s > 0 else None,
    )
    elapsed = time.time() - start
    log_path = target.pyatb_dir / "check-velocity" / "batch_check_velocity.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout, encoding="utf-8")
    return parse_summary(proc.stdout, target, elapsed, proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run velocity covariance reports for pc pyatb-6.27 targets.")
    parser.add_argument("--structures", nargs="*", help="Optional structure list, e.g. 1 2 stru3.")
    parser.add_argument("--modes", nargs="*", choices=MODES, default=list(MODES))
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-chunk-size", type=int, default=32)
    parser.add_argument("--timeout-s", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default=str(RESULTS_DIR / "velocity_symmetry_pc_627_summary.tsv"))
    args = parser.parse_args()

    selected = parse_selected(args.structures)
    targets = discover_targets(selected, tuple(args.modes))
    if args.limit > 0:
        targets = targets[: int(args.limit)]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "targets.json").write_text(
        json.dumps([asdict(target) for target in targets], indent=2, default=str),
        encoding="utf-8",
    )
    rows: list[Row] = []
    output = Path(args.output)
    print(f"target_count={len(targets)} output={output}", flush=True)
    for index, target in enumerate(targets, start=1):
        print(f"[{index}/{len(targets)}] {target.structure} {target.mode}", flush=True)
        try:
            row = run_one(target, args)
        except subprocess.TimeoutExpired as exc:
            row = Row(target.structure, target.mode, rel(target.pyatb_dir), "timeout", 124, float(args.timeout_s), message=str(exc))
        except Exception as exc:
            row = Row(target.structure, target.mode, rel(target.pyatb_dir), "exception", 1, 0.0, message=repr(exc))
        rows.append(row)
        write_rows(output, rows)
        print(
            f"  {row.status} max_abs={row.all_max_abs:.6e} max_rel={row.all_max_rel_fro:.6e} elapsed={row.elapsed_s:.1f}s",
            flush=True,
        )
    write_rows(output, rows)
    failed = [row for row in rows if row.status not in {"ok", "existing"}]
    print(f"completed={len(rows)} failed={len(failed)} summary={output}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
