from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pyatb.constants import Ang_to_Bohr
from pyatb.io.abacus_read_xr import abacus_readHR, abacus_readSR
from pyatb.symmetry.hs_standardize import (
    _atom_slice,
    _build_metadata_from_stru,
    _iter_dense_xr_blocks,
    build_standardization_atom_mapping,
    map_target_r_vector,
)


def read_stru_lattice_vector(stru_path: str | Path) -> np.ndarray:
    path = Path(stru_path)
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    stripped = [line.strip() for line in raw_lines]
    scale = float(stripped[stripped.index("LATTICE_CONSTANT") + 1]) / Ang_to_Bohr
    start = stripped.index("LATTICE_VECTORS") + 1
    lattice = np.array(
        [[float(value) for value in stripped[start + row].split()[:3]] for row in range(3)],
        dtype=float,
    )
    return lattice * scale


def infer_integer_lattice_transform(
    source_lattice_vector: np.ndarray,
    target_lattice_vector: np.ndarray,
    tol: float = 1.0e-6,
) -> np.ndarray:
    raw = np.asarray(source_lattice_vector, dtype=float) @ np.linalg.inv(np.asarray(target_lattice_vector, dtype=float))
    rounded = np.rint(raw).astype(int)
    if float(np.max(np.abs(raw - rounded))) > tol:
        raise ValueError("Failed to infer an integer lattice transform between source and target structures.")
    return rounded


def _dense_blocks_by_r(xr) -> dict[tuple[int, int, int], np.ndarray]:
    return {tuple(int(value) for value in r_vector.tolist()): dense for r_vector, dense in _iter_dense_xr_blocks(xr)}


def _full_dense_blocks_by_r_from_hermitian_partners(
    dense_by_r: dict[tuple[int, int, int], np.ndarray],
) -> dict[tuple[int, int, int], np.ndarray]:
    full_by_r: dict[tuple[int, int, int], np.ndarray] = {}
    for r_key, upper_dense in dense_by_r.items():
        partner_key = tuple(int(-value) for value in r_key)
        partner_upper = dense_by_r.get(partner_key)
        full_dense = np.asarray(upper_dense, dtype=complex).copy()
        if partner_upper is None:
            full_by_r[r_key] = full_dense
            continue
        nrow = full_dense.shape[0]
        lower_indices = np.tril_indices(nrow, k=-1)
        full_dense[lower_indices] = np.conj(np.asarray(partner_upper, dtype=complex).T[lower_indices])
        full_by_r[r_key] = full_dense
    return full_by_r


def _dense_blocks_by_r_with_optional_full_reconstruction(
    xr,
    full_matrix_from_hermitian: bool = False,
) -> dict[tuple[int, int, int], np.ndarray]:
    dense_by_r = _dense_blocks_by_r(xr)
    if not bool(full_matrix_from_hermitian):
        return dense_by_r
    return _full_dense_blocks_by_r_from_hermitian_partners(dense_by_r)


def collect_no_rotation_block_samples_from_dense_blocks(
    source_dense_by_r: dict[tuple[int, int, int], np.ndarray],
    source_metadata,
    target_metadata,
    atom_mapping: list[dict],
    lattice_transform_fractional: np.ndarray,
    zero_tol: float = 1.0e-12,
) -> dict[tuple[tuple[int, int, int], int, int], list[dict]]:
    mapping_by_old = {int(item["old_atom"]): item for item in atom_mapping}
    grouped: dict[tuple[tuple[int, int, int], int, int], list[dict]] = defaultdict(list)

    for r_old, dense in source_dense_by_r.items():
        for old_a, map_a in mapping_by_old.items():
            source_slice_a = _atom_slice(source_metadata, old_a)
            target_a = int(map_a["new_atom"])
            for old_b, map_b in mapping_by_old.items():
                source_slice_b = _atom_slice(source_metadata, old_b)
                block = np.asarray(dense[source_slice_a, source_slice_b], dtype=complex)
                if not np.any(np.abs(block) > zero_tol):
                    continue

                target_b = int(map_b["new_atom"])
                r_new = map_target_r_vector(
                    lattice_transform_fractional,
                    np.asarray(r_old, dtype=int),
                    np.asarray(map_a["shift"], dtype=int),
                    np.asarray(map_b["shift"], dtype=int),
                )
                key = (tuple(int(value) for value in r_new.tolist()), target_a, target_b)
                grouped[key].append(
                    {
                        "source_r": tuple(int(value) for value in r_old),
                        "source_atoms": (old_a, old_b),
                        "target_atoms": (target_a, target_b),
                        "block": block,
                    }
                )

    return grouped


def collect_no_rotation_block_samples_from_xr(
    source_xr,
    source_metadata,
    target_metadata,
    atom_mapping: list[dict],
    lattice_transform_fractional: np.ndarray,
    zero_tol: float = 1.0e-12,
) -> dict[tuple[tuple[int, int, int], int, int], list[dict]]:
    return collect_no_rotation_block_samples_from_dense_blocks(
        source_dense_by_r=_dense_blocks_by_r(source_xr),
        source_metadata=source_metadata,
        target_metadata=target_metadata,
        atom_mapping=atom_mapping,
        lattice_transform_fractional=lattice_transform_fractional,
        zero_tol=zero_tol,
    )


def _pair_vector_error_summary(
    sample_groups: dict[tuple[tuple[int, int, int], int, int], list[dict]],
    source_metadata,
    target_metadata,
) -> dict:
    max_abs_component = 0.0
    max_norm = 0.0
    worst = None

    for (r_new, target_a, target_b), samples in sample_groups.items():
        target_vec = (
            np.asarray(r_new, dtype=float)
            + np.asarray(target_metadata.positions_frac[target_b], dtype=float)
            - np.asarray(target_metadata.positions_frac[target_a], dtype=float)
        ) @ np.asarray(target_metadata.lattice_vector, dtype=float)

        for sample in samples:
            old_a, old_b = sample["source_atoms"]
            source_vec = (
                np.asarray(sample["source_r"], dtype=float)
                + np.asarray(source_metadata.positions_frac[old_b], dtype=float)
                - np.asarray(source_metadata.positions_frac[old_a], dtype=float)
            ) @ np.asarray(source_metadata.lattice_vector, dtype=float)
            diff = source_vec - target_vec
            cur_abs = float(np.max(np.abs(diff)))
            cur_norm = float(np.linalg.norm(diff))
            if cur_abs > max_abs_component:
                max_abs_component = cur_abs
                max_norm = cur_norm
                worst = {
                    "source_r": list(sample["source_r"]),
                    "target_r": list(r_new),
                    "source_atoms_1based": [old_a + 1, old_b + 1],
                    "target_atoms_1based": [target_a + 1, target_b + 1],
                    "diff_cart": [float(value) for value in diff.tolist()],
                    "diff_norm": cur_norm,
                }

    return {
        "max_abs_component": max_abs_component,
        "max_norm": max_norm,
        "worst": worst,
    }


def validate_no_rotation_block_mapping_from_dense_blocks(
    source_dense_by_r: dict[tuple[int, int, int], np.ndarray],
    target_dense_by_r: dict[tuple[int, int, int], np.ndarray],
    source_metadata,
    target_metadata,
    atom_mapping: list[dict],
    lattice_transform_fractional: np.ndarray,
    zero_tol: float = 1.0e-12,
) -> dict:
    sample_groups = collect_no_rotation_block_samples_from_dense_blocks(
        source_dense_by_r=source_dense_by_r,
        source_metadata=source_metadata,
        target_metadata=target_metadata,
        atom_mapping=atom_mapping,
        lattice_transform_fractional=lattice_transform_fractional,
        zero_tol=zero_tol,
    )

    mapped_block_count = 0
    target_missing_keys: set[tuple[int, int, int]] = set()
    max_abs_diff = 0.0
    sum_abs_diff = 0.0
    fro_diff = 0.0
    fro_ref = 0.0
    worst = None

    duplicate_consistency_max_abs = 0.0
    duplicate_consistency_worst = None

    for (r_new, target_a, target_b), samples in sample_groups.items():
        target_dense = target_dense_by_r.get(r_new)
        target_slice_a = _atom_slice(target_metadata, target_a)
        target_slice_b = _atom_slice(target_metadata, target_b)
        reference_block = None
        if target_dense is None:
            target_missing_keys.add(r_new)
            reference_block = np.zeros(
                (
                    target_slice_a.stop - target_slice_a.start,
                    target_slice_b.stop - target_slice_b.start,
                ),
                dtype=complex,
            )
        else:
            reference_block = np.asarray(target_dense[target_slice_a, target_slice_b], dtype=complex)

        sample_reference = samples[0]["block"]
        for sample_index, sample in enumerate(samples, start=1):
            block = np.asarray(sample["block"], dtype=complex)
            diff = block - reference_block
            cur_max = float(np.max(np.abs(diff)))
            cur_sum = float(np.sum(np.abs(diff)))
            mapped_block_count += 1
            sum_abs_diff += cur_sum
            fro_diff += float(np.vdot(diff.ravel(), diff.ravel()).real)
            fro_ref += float(np.vdot(reference_block.ravel(), reference_block.ravel()).real)

            if cur_max > max_abs_diff:
                max_abs_diff = cur_max
                worst = {
                    "target_r": list(r_new),
                    "source_r": list(sample["source_r"]),
                    "source_atoms_1based": [sample["source_atoms"][0] + 1, sample["source_atoms"][1] + 1],
                    "target_atoms_1based": [target_a + 1, target_b + 1],
                    "sample_index": sample_index,
                }

            dup_max = float(np.max(np.abs(block - sample_reference)))
            if dup_max > duplicate_consistency_max_abs:
                duplicate_consistency_max_abs = dup_max
                duplicate_consistency_worst = {
                    "target_r": list(r_new),
                    "target_atoms_1based": [target_a + 1, target_b + 1],
                    "sample_index": sample_index,
                }

    rel_fro_diff = 0.0
    if fro_ref > 0.0:
        rel_fro_diff = float(np.sqrt(fro_diff / fro_ref))

    return {
        "mapped_block_count": mapped_block_count,
        "unique_target_key_count": len(sample_groups),
        "target_missing_count": len(target_missing_keys),
        "target_missing_keys": [list(key) for key in sorted(target_missing_keys)],
        "max_abs_diff": max_abs_diff,
        "sum_abs_diff": sum_abs_diff,
        "rel_fro_diff": rel_fro_diff,
        "worst": worst,
        "duplicate_consistency_max_abs": duplicate_consistency_max_abs,
        "duplicate_consistency_worst": duplicate_consistency_worst,
        "pair_vector_error": _pair_vector_error_summary(sample_groups, source_metadata, target_metadata),
    }


def validate_no_rotation_block_mapping_from_xr(
    source_xr,
    target_xr,
    source_metadata,
    target_metadata,
    atom_mapping: list[dict],
    lattice_transform_fractional: np.ndarray,
    zero_tol: float = 1.0e-12,
    full_matrix_from_hermitian: bool = True,
) -> dict:
    return validate_no_rotation_block_mapping_from_dense_blocks(
        source_dense_by_r=_dense_blocks_by_r_with_optional_full_reconstruction(
            source_xr,
            full_matrix_from_hermitian=full_matrix_from_hermitian,
        ),
        target_dense_by_r=_dense_blocks_by_r_with_optional_full_reconstruction(
            target_xr,
            full_matrix_from_hermitian=full_matrix_from_hermitian,
        ),
        source_metadata=source_metadata,
        target_metadata=target_metadata,
        atom_mapping=atom_mapping,
        lattice_transform_fractional=lattice_transform_fractional,
        zero_tol=zero_tol,
    )


def validate_abacus_supercell_to_primitive_no_rotation(
    source_stru_path: str | Path,
    target_stru_path: str | Path,
    source_hr_path: str | Path,
    source_sr_path: str | Path,
    target_hr_path: str | Path,
    target_sr_path: str | Path,
    nspin: int,
    hr_unit: str,
    report_path: str | Path | None = None,
    zero_tol: float = 1.0e-12,
    full_matrix_from_hermitian: bool = True,
) -> dict:
    source_lattice = read_stru_lattice_vector(source_stru_path)
    target_lattice = read_stru_lattice_vector(target_stru_path)
    lattice_transform = infer_integer_lattice_transform(source_lattice, target_lattice)

    source_metadata = _build_metadata_from_stru(Path(source_stru_path), source_lattice, int(nspin))
    target_metadata = _build_metadata_from_stru(Path(target_stru_path), target_lattice, int(nspin))
    atom_mapping = build_standardization_atom_mapping(source_metadata, target_metadata)

    source_hr = abacus_readHR(int(nspin), str(source_hr_path), hr_unit)
    source_sr = abacus_readSR(int(nspin), str(source_sr_path))
    target_hr = abacus_readHR(int(nspin), str(target_hr_path), hr_unit)
    target_sr = abacus_readSR(int(nspin), str(target_sr_path))

    report = {
        "source_stru": str(Path(source_stru_path).resolve()),
        "target_stru": str(Path(target_stru_path).resolve()),
        "source_hr": str(Path(source_hr_path).resolve()),
        "target_hr": str(Path(target_hr_path).resolve()),
        "source_sr": str(Path(source_sr_path).resolve()),
        "target_sr": str(Path(target_sr_path).resolve()),
        "nspin": int(nspin),
        "hr_unit": str(hr_unit),
        "full_matrix_from_hermitian": bool(full_matrix_from_hermitian),
        "lattice_transform_fractional": [[int(value) for value in row] for row in lattice_transform.tolist()],
        "atom_mapping": [
            {
                "old_atom_1based": int(item["old_atom"]) + 1,
                "new_atom_1based": int(item["new_atom"]) + 1,
                "shift": [int(value) for value in np.asarray(item["shift"], dtype=int).tolist()],
                "species": str(item["species"]),
            }
            for item in atom_mapping
        ],
        "HR": validate_no_rotation_block_mapping_from_xr(
            source_xr=source_hr,
            target_xr=target_hr,
            source_metadata=source_metadata,
            target_metadata=target_metadata,
            atom_mapping=atom_mapping,
            lattice_transform_fractional=lattice_transform,
            zero_tol=zero_tol,
            full_matrix_from_hermitian=full_matrix_from_hermitian,
        ),
        "SR": validate_no_rotation_block_mapping_from_xr(
            source_xr=source_sr,
            target_xr=target_sr,
            source_metadata=source_metadata,
            target_metadata=target_metadata,
            atom_mapping=atom_mapping,
            lattice_transform_fractional=lattice_transform,
            zero_tol=zero_tol,
            full_matrix_from_hermitian=full_matrix_from_hermitian,
        ),
    }

    if report_path is not None:
        target_path = Path(report_path)
        target_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report
