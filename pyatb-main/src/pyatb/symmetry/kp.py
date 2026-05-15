from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np

from pyatb import OUTPUT_PATH, RANK, RUNNING_LOG
from pyatb.symmetry.character import Character


@dataclass(frozen=True)
class KPointBandSelection:
    """One kp target: a direct-coordinate k point and its band range."""

    kpoint: Sequence[float]
    band: Sequence[int]
    label: str | None = None
    occ_band: int | None = None

    def __post_init__(self) -> None:
        kpoint = tuple(float(value) for value in self.kpoint)
        if len(kpoint) != 3:
            raise ValueError("kp selection kpoint must contain exactly three direct coordinates.")

        band = tuple(int(value) for value in self.band)
        if len(band) != 2 or band[0] <= 0 or band[1] <= 0 or band[0] > band[1]:
            raise ValueError("kp selection band must be two positive integers with band[0] <= band[1].")

        if self.occ_band is not None and int(self.occ_band) <= 0:
            raise ValueError("kp selection occ_band must be a positive integer.")

        object.__setattr__(self, "kpoint", kpoint)
        object.__setattr__(self, "band", band)
        if self.occ_band is not None:
            object.__setattr__(self, "occ_band", int(self.occ_band))

    @property
    def band_start(self) -> int:
        return int(self.band[0])

    @property
    def band_stop(self) -> int:
        return int(self.band[1])


def _coerce_selection(selection: KPointBandSelection | Mapping[str, Any]) -> KPointBandSelection:
    if isinstance(selection, KPointBandSelection):
        return selection
    if isinstance(selection, Mapping):
        try:
            kpoint = selection["kpoint"]
            band = selection["band"]
        except KeyError as exc:
            raise ValueError("kp selection mapping must contain 'kpoint' and 'band'.") from exc
        return KPointBandSelection(
            kpoint=kpoint,
            band=band,
            label=selection.get("label"),
            occ_band=selection.get("occ_band"),
        )
    raise TypeError("kp selection must be a KPointBandSelection or a mapping.")


def _normalize_selections(
    selections: Sequence[KPointBandSelection | Mapping[str, Any]],
) -> list[KPointBandSelection]:
    normalized = [_coerce_selection(selection) for selection in selections]
    if not normalized:
        raise ValueError("kp calculation requires at least one kpoint/band selection.")
    return normalized


def _parse_band_irrep_file(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"CHARACTER band irrep output was not found: {path}")

    rows: list[dict[str, Any]] = []
    current_k_index: int | None = None
    current_k_name = ""

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("knum="):
            left, _, right = stripped.partition("kname=")
            current_k_name = right.strip()
            current_k_index = int(left.replace("knum=", "").strip())
            continue
        if stripped.lower().startswith("band"):
            continue

        parts = stripped.split()
        if len(parts) < 4:
            continue
        if current_k_index is None:
            raise ValueError(f"Malformed CHARACTER band irrep output without k header: {path}")
        rows.append(
            {
                "k_index": current_k_index,
                "k_name": current_k_name,
                "band": int(parts[0]),
                "degeneracy": int(parts[1]),
                "energy": float(parts[2]),
                "irrep": " ".join(parts[3:]),
            }
        )

    return rows


def calculate_kpoint_irreps(
    tb,
    selections: Sequence[KPointBandSelection | Mapping[str, Any]],
    *,
    stru_file: str = "STRU",
    group: str | int = "auto",
    symm_prec: float = 1.0e-5,
    occ_band: int | None = None,
    mag_tag: int = 0,
    mag: str | Sequence[float] = "auto",
    archive_output_path: str | Path | None = None,
    **character_kwargs,
) -> list[dict[str, Any]]:
    """Compute little-group irreps for selected k points and per-k-point bands.

    This is the first kp workflow layer. It delegates the actual character
    calculation to :class:`pyatb.symmetry.character.Character`, then parses the
    existing ``band_irrep.txt`` output into a Python structure.
    """

    normalized = _normalize_selections(selections)
    results: list[dict[str, Any]] = []
    archive_root = Path(archive_output_path) if archive_output_path is not None else None
    if archive_root is not None and RANK == 0:
        archive_root.mkdir(parents=True, exist_ok=True)

    for selection_index, selection in enumerate(normalized, start=1):
        selection_occ_band = int(occ_band if occ_band is not None else selection.occ_band or selection.band_stop)
        calculator = Character(tb)
        calculator.calculate_character(
            stru_file=stru_file,
            kpoint_mode="direct",
            kpoint_num=1,
            kpoint_direct_coor=np.asarray([selection.kpoint], dtype=float),
            group=group,
            symm_prec=float(symm_prec),
            occ_band=selection_occ_band,
            band=(selection.band_start, selection.band_stop),
            mag_tag=int(mag_tag),
            mag=mag,
            **character_kwargs,
        )
        output_path = Path(calculator.output_path)
        rows = _parse_band_irrep_file(output_path / "band_irrep.txt")
        archive_path = None
        if archive_root is not None and RANK == 0:
            archive_path = archive_root / f"selection_{selection_index:03d}"
            if archive_path.exists():
                shutil.rmtree(archive_path)
            shutil.copytree(output_path, archive_path)
        results.append(
            {
                "selection_index": selection_index,
                "label": selection.label,
                "kpoint": [float(value) for value in selection.kpoint],
                "band": [selection.band_start, selection.band_stop],
                "occ_band": selection_occ_band,
                "output_path": str(output_path),
                "archive_path": str(archive_path) if archive_path is not None else None,
                "rows": rows,
            }
        )

    return results


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _write_kp_band_irrep_summary(results: Sequence[Mapping[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            label = result.get("label") or ""
            kpoint = result["kpoint"]
            band = result["band"]
            handle.write(
                f"selection={int(result['selection_index']):3d}   label={label}\n"
                f"kpoint={float(kpoint[0]): .8f} {float(kpoint[1]): .8f} {float(kpoint[2]): .8f}   "
                f"band={int(band[0])} {int(band[1])}   occ_band={int(result['occ_band'])}\n"
                "band    degency   energy          irrrp\n"
            )
            for row in result["rows"]:
                handle.write(
                    f"{int(row['band']):<8d}"
                    f"{int(row['degeneracy']):<10d}"
                    f"{float(row['energy']):<16.6f}"
                    f"{str(row['irrep']):<24s}\n"
                )
            handle.write("\n")


def calculate_kp_irreps(
    tb,
    *,
    kpoint_mode: str = "direct",
    kpoint_num: int | None = None,
    kpoint_direct_coor: Sequence[Sequence[float]],
    band: Sequence[Sequence[int]],
    label: Sequence[str] | None = None,
    output_path: str | Path | None = None,
    stru_file: str = "STRU",
    group: str | int = "auto",
    symm_prec: float = 1.0e-5,
    occ_band: int | None = None,
    mag_tag: int = 0,
    mag: str | Sequence[float] = "auto",
    **character_kwargs,
) -> list[dict[str, Any]]:
    """Main-flow KP entry point that writes summary output under ``Out/KP``."""

    if kpoint_mode != "direct":
        raise ValueError("KP currently supports only direct k-point mode.")

    kpoints = np.asarray(kpoint_direct_coor, dtype=float)
    if kpoints.ndim != 2 or kpoints.shape[1] != 3 or kpoints.shape[0] == 0:
        raise ValueError("KP.kpoint_direct_coor must have shape (N, 3) with N > 0.")
    if kpoint_num is not None and int(kpoint_num) != kpoints.shape[0]:
        raise ValueError("KP.kpoint_num must match the number of direct k points.")

    bands = np.asarray(band, dtype=int)
    if bands.shape != (kpoints.shape[0], 2):
        raise ValueError("KP.band must contain one band_start band_stop pair for each k point.")

    labels = list(label) if label is not None else [None] * kpoints.shape[0]
    if len(labels) != kpoints.shape[0]:
        raise ValueError("KP.label must match the number of direct k points.")

    selections = [
        KPointBandSelection(kpoint=kpoints[index], band=bands[index], label=labels[index])
        for index in range(kpoints.shape[0])
    ]
    kp_output_path = Path(output_path) if output_path is not None else Path(OUTPUT_PATH) / "KP"
    if RANK == 0:
        if kp_output_path.exists():
            shutil.rmtree(kp_output_path)
        kp_output_path.mkdir(parents=True, exist_ok=True)

    effective_occ_band = None if occ_band is None or int(occ_band) <= 0 else int(occ_band)
    results = calculate_kpoint_irreps(
        tb,
        selections,
        stru_file=stru_file,
        group=group,
        symm_prec=symm_prec,
        occ_band=effective_occ_band,
        mag_tag=mag_tag,
        mag=mag,
        archive_output_path=kp_output_path,
        **character_kwargs,
    )

    if RANK == 0:
        (kp_output_path / "kp_irrep.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        _write_kp_band_irrep_summary(results, kp_output_path / "band_irrep.txt")
        with open(RUNNING_LOG, "a", encoding="utf-8") as handle:
            handle.write("\nKP Irreducible Representation Summary\n")
            handle.write(f"output_path = {kp_output_path.resolve()}\n")
            handle.write(f"selection_count = {len(results)}\n")

    return results


__all__ = [
    "KPointBandSelection",
    "calculate_kpoint_irreps",
    "calculate_kp_irreps",
]
