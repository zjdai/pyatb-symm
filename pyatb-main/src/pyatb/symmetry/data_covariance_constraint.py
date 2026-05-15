from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import spglib
from ase.data import atomic_numbers
from scipy.sparse import coo_matrix

from pyatb.io.abacus_read_xr import abacus_readHR, abacus_readSR
from pyatb.symmetry.Dk_matrix import build_atom_local_rotation, find_atom_mapping
from pyatb.symmetry.hs_covariance import _dense_blocks_by_r_with_optional_full_reconstruction, read_stru_lattice_vector
from pyatb.symmetry.hs_standardize import (
    _atom_slice,
    _build_metadata_from_stru,
    _unit_scale_from_hr_unit,
    _write_abacus_sparse_xr,
)
from pyatb.tb.multixr import multiXR


@dataclass
class MatrixSummary:
    max_abs: float
    mean_abs: float
    rms_abs: float
    rel_fro: float
    element_count: int
    missing_predicted_R_count: int
    extra_predicted_R_count: int

    def to_dict(self) -> dict:
        return {
            "max_abs": float(self.max_abs),
            "mean_abs": float(self.mean_abs),
            "rms_abs": float(self.rms_abs),
            "rel_fro": float(self.rel_fro),
            "element_count": int(self.element_count),
            "missing_predicted_R_count": int(self.missing_predicted_R_count),
            "extra_predicted_R_count": int(self.extra_predicted_R_count),
        }


def _cart_rotation_from_fractional(rotation_frac: np.ndarray, lattice_row: np.ndarray) -> np.ndarray:
    a_cols = np.asarray(lattice_row, dtype=float).T
    return a_cols @ np.asarray(rotation_frac, dtype=float) @ np.linalg.inv(a_cols)


def _integral_r(vec: np.ndarray, tol: float = 1.0e-6) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    rounded = np.rint(arr).astype(int)
    err = float(np.max(np.abs(arr - rounded)))
    if err > tol:
        raise ValueError(f"Mapped R is non-integral: vec={arr.tolist()}, max_err={err}")
    return rounded


def _empty_like_block(template: dict[tuple[int, int, int], np.ndarray], basis_num: int) -> np.ndarray:
    if template:
        return np.zeros_like(next(iter(template.values())), dtype=complex)
    return np.zeros((int(basis_num), int(basis_num)), dtype=complex)


def _synchronize_block_keys(
    hr_blocks: dict[tuple[int, int, int], np.ndarray],
    sr_blocks: dict[tuple[int, int, int], np.ndarray],
    basis_num: int,
):
    all_keys = sorted(set(hr_blocks) | set(sr_blocks))
    synced_hr = {tuple(int(x) for x in key): np.asarray(value, dtype=complex) for key, value in hr_blocks.items()}
    synced_sr = {tuple(int(x) for x in key): np.asarray(value, dtype=complex) for key, value in sr_blocks.items()}
    for key in all_keys:
        if key not in synced_hr:
            synced_hr[key] = _empty_like_block(synced_hr, int(basis_num))
        if key not in synced_sr:
            synced_sr[key] = _empty_like_block(synced_sr, int(basis_num))
    return synced_hr, synced_sr


def get_symmetry_operations_from_metadata(metadata, symprec: float = 1.0e-5) -> list[dict]:
    numbers = np.asarray([atomic_numbers[s] for s in metadata.species_by_atom], dtype=int)
    cell = (
        np.asarray(metadata.lattice_vector, dtype=float),
        np.asarray(metadata.positions_frac, dtype=float),
        numbers,
    )
    dataset = spglib.get_symmetry_dataset(cell, symprec=float(symprec), _throw=True)
    if dataset is None:
        raise RuntimeError("spglib.get_symmetry failed while building covariance operations.")

    operations = []
    rotations = np.asarray(dataset.rotations, dtype=int)
    translations = np.asarray(dataset.translations, dtype=float)
    for index, (rot, tau) in enumerate(zip(rotations, translations, strict=True), start=1):
        rot = np.asarray(rot, dtype=int)
        tau = np.asarray(tau, dtype=float)
        cart_rot = _cart_rotation_from_fractional(rot, np.asarray(metadata.lattice_vector, dtype=float))
        operations.append(
            {
                "index": int(index),
                "rotation": rot,
                "translation": tau,
                "cart_rotation": cart_rot,
            }
        )
    return operations


def load_abacus_hs_blocks(
    stru_path: str | Path,
    hr_path: str | Path,
    sr_path: str | Path,
    nspin: int = 4,
    hr_unit: str = "Ry",
):
    lattice = np.asarray(read_stru_lattice_vector(stru_path), dtype=float)
    metadata = _build_metadata_from_stru(Path(stru_path), lattice, int(nspin))

    hr = abacus_readHR(int(nspin), str(hr_path), hr_unit)
    sr = abacus_readSR(int(nspin), str(sr_path))

    hr_blocks = _dense_blocks_by_r_with_optional_full_reconstruction(hr, full_matrix_from_hermitian=True)
    sr_blocks = _dense_blocks_by_r_with_optional_full_reconstruction(sr, full_matrix_from_hermitian=True)
    return metadata, hr_blocks, sr_blocks


def _dense_blocks_to_multixr(
    xr_tag: str,
    blocks: dict[tuple[int, int, int], np.ndarray],
    basis_num: int,
    nspin: int,
    unit_scale: float = 1.0,
):
    basis_num = int(basis_num)
    nspin = int(nspin)
    r_keys = sorted(blocks.keys())
    r_num = len(r_keys)
    triu_size = int((basis_num + 1) * basis_num / 2)

    r_direct_coor = np.zeros((r_num, 3), dtype=int)
    record_row: list[int] = []
    record_col: list[int] = []
    record_data: list[complex] = []
    num = np.zeros(basis_num, dtype=int)
    for i in range(basis_num - 1):
        num[i + 1] = num[i] + basis_num - i

    for i_r, r_key in enumerate(r_keys):
        r_direct_coor[i_r, :] = np.asarray(r_key, dtype=int)
        dense = np.asarray(blocks[r_key], dtype=complex) / float(unit_scale)
        upper = np.triu(dense)
        if nspin != 4:
            if float(np.max(np.abs(upper.imag))) > 1.0e-8:
                raise ValueError("Non-spinor XR conversion received complex values beyond tolerance.")
            upper = upper.real

        nz_row, nz_col = np.nonzero(upper)
        for row, col in zip(nz_row.tolist(), nz_col.tolist(), strict=False):
            tri_col = int(col) + int(num[int(row)]) - int(row)
            raw_value = upper[int(row), int(col)]
            if nspin == 4:
                quantized_value = complex(
                    float(f"{float(np.real(raw_value)):.16e}"),
                    float(f"{float(np.imag(raw_value)):.16e}"),
                )
            else:
                quantized_value = complex(float(f"{float(raw_value):.16e}"), 0.0)
            record_row.append(int(i_r))
            record_col.append(int(tri_col))
            record_data.append(quantized_value)

    xr_sparse = coo_matrix(
        (record_data, (record_row, record_col)),
        shape=(r_num, triu_size),
        dtype=complex,
    ).tocsc()
    xr_obj = multiXR(str(xr_tag))
    xr_obj.set_XR(int(r_num), r_direct_coor, int(basis_num), xr_sparse)
    return xr_obj


def build_multixr_from_dense_blocks(
    hr_blocks: dict[tuple[int, int, int], np.ndarray],
    sr_blocks: dict[tuple[int, int, int], np.ndarray],
    basis_num: int,
    nspin: int,
    hr_unit: str,
):
    # `hr_blocks`/`sr_blocks` are already internal XR values (eV for HR),
    # so no unit rescaling is needed here.
    hr_blocks, sr_blocks = _synchronize_block_keys(hr_blocks, sr_blocks, int(basis_num))
    hr_obj = _dense_blocks_to_multixr(
        "H",
        hr_blocks,
        int(basis_num),
        int(nspin),
        unit_scale=1.0,
    )
    sr_obj = _dense_blocks_to_multixr(
        "S",
        sr_blocks,
        int(basis_num),
        int(nspin),
        unit_scale=1.0,
    )
    return hr_obj, sr_obj


def transform_blocks_by_operation(
    source_blocks: dict[tuple[int, int, int], np.ndarray],
    metadata,
    operation: dict,
    map_tol: float = 1.0e-5,
    zero_tol: float = 1.0e-14,
    nonzero_block_tol: float = 1.0e-9,
) -> dict[tuple[int, int, int], np.ndarray]:
    context = _prepare_operation_context(metadata, operation, map_tol=float(map_tol))
    return transform_blocks_with_context(
        source_blocks=source_blocks,
        metadata=metadata,
        context=context,
        zero_tol=zero_tol,
        nonzero_block_tol=nonzero_block_tol,
    )


def _prepare_operation_context(
    metadata,
    operation: dict,
    map_tol: float = 1.0e-5,
) -> dict:
    rot = np.asarray(operation["rotation"], dtype=int)
    tau = np.asarray(operation["translation"], dtype=float)
    cart_rot = np.asarray(operation["cart_rotation"], dtype=float)

    map_rows = find_atom_mapping(
        metadata,
        {
            "rotation": rot,
            "translation": tau,
        },
        tol=float(map_tol),
    )
    map_by_src = {int(item["source_atom"]): item for item in map_rows}

    d_by_atom = {
        ia: np.asarray(
            build_atom_local_rotation(
                metadata,
                ia,
                cart_rot,
                passive_basis=False,
            ),
            dtype=complex,
        )
        for ia in range(len(metadata.species_by_atom))
    }

    atom_rows = []
    natom = len(metadata.species_by_atom)
    for ia in range(natom):
        row = map_by_src[ia]
        ia_prime = int(row["target_atom"])
        atom_rows.append(
            {
                "source_atom": int(ia),
                "target_atom": int(ia_prime),
                "cell_shift": np.asarray(row["cell_shift"], dtype=int),
                "source_slice": _atom_slice(metadata, ia),
                "target_slice": _atom_slice(metadata, ia_prime),
                "d": d_by_atom[ia],
                "d_dag": d_by_atom[ia].conj().T,
            }
        )

    pair_rows = []
    for row_a in atom_rows:
        for row_b in atom_rows:
            pair_rows.append(
                {
                    "source_a": int(row_a["source_atom"]),
                    "source_b": int(row_b["source_atom"]),
                    "target_a": int(row_a["target_atom"]),
                    "target_b": int(row_b["target_atom"]),
                    "source_slice_a": row_a["source_slice"],
                    "source_slice_b": row_b["source_slice"],
                    "target_slice_a": row_a["target_slice"],
                    "target_slice_b": row_b["target_slice"],
                    "shift_diff": np.asarray(row_b["cell_shift"] - row_a["cell_shift"], dtype=int),
                    "d_left": row_a["d"],
                    "d_right_dag": row_b["d_dag"],
                }
            )
    pair_row_by_source = {
        (int(item["source_a"]), int(item["source_b"])): item
        for item in pair_rows
    }

    return {
        "rotation": rot,
        "translation": tau,
        "cart_rotation": cart_rot,
        "map_by_src": map_by_src,
        "d_by_atom": d_by_atom,
        "pair_rows": pair_rows,
        "pair_row_by_source": pair_row_by_source,
    }


def prepare_operation_contexts(
    metadata,
    operations: list[dict],
    map_tol: float = 1.0e-5,
) -> list[tuple[dict, dict]]:
    return [
        (op, _prepare_operation_context(metadata, op, map_tol=float(map_tol)))
        for op in operations
    ]


def transform_blocks_with_context(
    source_blocks: dict[tuple[int, int, int], np.ndarray],
    metadata,
    context: dict,
    zero_tol: float = 1.0e-14,
    nonzero_block_tol: float = 1.0e-9,
    return_touched_pairs: bool = False,
    active_pair_index: dict[tuple[int, int, int], set[tuple[int, int]]] | None = None,
) -> dict[tuple[int, int, int], np.ndarray] | tuple[dict[tuple[int, int, int], np.ndarray], set[tuple[tuple[int, int, int], int, int]]]:
    rot = np.asarray(context["rotation"], dtype=int)
    pair_rows = context.get("pair_rows")
    pair_row_by_source = context.get("pair_row_by_source")
    if pair_rows is None:
        raise ValueError("Operation context missing pair_rows cache.")
    if pair_row_by_source is None:
        raise ValueError("Operation context missing pair_row_by_source cache.")
    block_tol = max(float(zero_tol), float(nonzero_block_tol))
    basis_num = int(metadata.basis_num)

    transformed: dict[tuple[int, int, int], np.ndarray] = {}
    touched_pairs: set[tuple[tuple[int, int, int], int, int]] = set()

    for r_old, dense in source_blocks.items():
        r_old_vec = np.asarray(r_old, dtype=int)
        dense = np.asarray(dense, dtype=complex)
        if active_pair_index is not None and r_old in active_pair_index:
            pair_iter = (
                pair_row_by_source[source_pair]
                for source_pair in active_pair_index.get(r_old, set())
                if source_pair in pair_row_by_source
            )
        else:
            pair_iter = pair_rows
        for pair in pair_iter:
            sl_a = pair["source_slice_a"]
            sl_b = pair["source_slice_b"]
            block = dense[sl_a, sl_b]
            # Only operate on atom-pair blocks that are numerically non-zero.
            if float(np.max(np.abs(block))) < block_tol:
                continue

            r_new = _integral_r(rot @ r_old_vec + pair["shift_diff"])
            r_new_key = tuple(int(x) for x in r_new.tolist())
            target = transformed.setdefault(
                r_new_key,
                np.zeros((basis_num, basis_num), dtype=complex),
            )
            sl_ap = pair["target_slice_a"]
            sl_bp = pair["target_slice_b"]
            target[sl_ap, sl_bp] += pair["d_left"] @ block @ pair["d_right_dag"]
            touched_pairs.add((r_new_key, int(pair["target_a"]), int(pair["target_b"])))
    if return_touched_pairs:
        return transformed, touched_pairs
    return transformed


def build_active_pair_index(
    source_blocks: dict[tuple[int, int, int], np.ndarray],
    context: dict,
    nonzero_block_tol: float = 1.0e-9,
) -> dict[tuple[int, int, int], set[tuple[int, int]]]:
    pair_rows = context.get("pair_rows")
    if pair_rows is None:
        raise ValueError("Operation context missing pair_rows cache.")

    block_tol = float(nonzero_block_tol)
    active_index: dict[tuple[int, int, int], set[tuple[int, int]]] = {}
    for r_key, dense in source_blocks.items():
        dense = np.asarray(dense, dtype=complex)
        pairs = set()
        for pair in pair_rows:
            sl_a = pair["source_slice_a"]
            sl_b = pair["source_slice_b"]
            block = dense[sl_a, sl_b]
            if float(np.max(np.abs(block))) < block_tol:
                continue
            pairs.add((int(pair["source_a"]), int(pair["source_b"])))
        if pairs:
            active_index[tuple(int(x) for x in r_key)] = pairs
    return active_index


def merge_active_pair_index(
    active_index: dict[tuple[int, int, int], set[tuple[int, int]]],
    touched_pairs: set[tuple[tuple[int, int, int], int, int]],
) -> dict[tuple[int, int, int], set[tuple[int, int]]]:
    for r_key, a, b in touched_pairs:
        key = tuple(int(x) for x in r_key)
        active_index.setdefault(key, set()).add((int(a), int(b)))
    return active_index


def compare_block_sets_on_candidate_pairs(
    reference_blocks: dict[tuple[int, int, int], np.ndarray],
    predicted_blocks: dict[tuple[int, int, int], np.ndarray],
    metadata,
    candidate_index: dict[tuple[int, int, int], set[tuple[int, int]]],
) -> MatrixSummary:
    basis_num = int(metadata.basis_num)
    ref_zero = _empty_like_block(reference_blocks, basis_num)
    pred_zero = _empty_like_block(predicted_blocks, basis_num)
    atom_slices = [_atom_slice(metadata, i) for i in range(len(metadata.species_by_atom))]

    total_count = 0
    sum_abs = 0.0
    sum_sq = 0.0
    sum_ref_sq = 0.0
    max_abs = 0.0

    for r_key, pairs in candidate_index.items():
        ref = np.asarray(reference_blocks.get(r_key, ref_zero), dtype=complex)
        pred = np.asarray(predicted_blocks.get(r_key, pred_zero), dtype=complex)
        for a, b in pairs:
            sl_a = atom_slices[int(a)]
            sl_b = atom_slices[int(b)]
            diff_block = pred[sl_a, sl_b] - ref[sl_a, sl_b]
            abs_block = np.abs(diff_block)
            total_count += int(diff_block.size)
            sum_abs += float(np.sum(abs_block))
            sum_sq += float(np.vdot(diff_block.ravel(), diff_block.ravel()).real)
            ref_block = ref[sl_a, sl_b]
            sum_ref_sq += float(np.vdot(ref_block.ravel(), ref_block.ravel()).real)
            block_max = float(np.max(np.abs(diff_block)))
            if block_max > max_abs:
                max_abs = block_max

    mean_abs = float(sum_abs / total_count) if total_count > 0 else 0.0
    rms_abs = float(np.sqrt(sum_sq / total_count)) if total_count > 0 else 0.0
    rel_fro = float(np.sqrt(sum_sq / sum_ref_sq)) if sum_ref_sq > 0.0 else 0.0
    return MatrixSummary(
        max_abs=max_abs,
        mean_abs=mean_abs,
        rms_abs=rms_abs,
        rel_fro=rel_fro,
        element_count=int(total_count),
        missing_predicted_R_count=0,
        extra_predicted_R_count=0,
    )


def compare_block_sets(
    reference_blocks: dict[tuple[int, int, int], np.ndarray],
    predicted_blocks: dict[tuple[int, int, int], np.ndarray],
    basis_num: int,
) -> MatrixSummary:
    all_keys = set(reference_blocks.keys()) | set(predicted_blocks.keys())
    ref_zero = _empty_like_block(reference_blocks, basis_num)
    pred_zero = _empty_like_block(predicted_blocks, basis_num)

    total_count = 0
    sum_abs = 0.0
    sum_sq = 0.0
    sum_ref_sq = 0.0
    max_abs = 0.0
    missing_pred = 0
    extra_pred = 0

    for r_key in all_keys:
        ref = np.asarray(reference_blocks.get(r_key, ref_zero), dtype=complex)
        pred = np.asarray(predicted_blocks.get(r_key, pred_zero), dtype=complex)
        if r_key not in predicted_blocks:
            missing_pred += 1
        if r_key not in reference_blocks:
            extra_pred += 1

        diff = pred - ref
        abs_diff = np.abs(diff)

        total_count += int(diff.size)
        sum_abs += float(np.sum(abs_diff))
        sum_sq += float(np.vdot(diff.ravel(), diff.ravel()).real)
        sum_ref_sq += float(np.vdot(ref.ravel(), ref.ravel()).real)

        cur_max = float(np.max(abs_diff))
        if cur_max > max_abs:
            max_abs = cur_max

    mean_abs = float(sum_abs / total_count) if total_count > 0 else 0.0
    rms_abs = float(np.sqrt(sum_sq / total_count)) if total_count > 0 else 0.0
    rel_fro = float(np.sqrt(sum_sq / sum_ref_sq)) if sum_ref_sq > 0.0 else 0.0

    return MatrixSummary(
        max_abs=max_abs,
        mean_abs=mean_abs,
        rms_abs=rms_abs,
        rel_fro=rel_fro,
        element_count=total_count,
        missing_predicted_R_count=missing_pred,
        extra_predicted_R_count=extra_pred,
    )


def _max_error_element_detail(
    reference_blocks: dict[tuple[int, int, int], np.ndarray],
    predicted_blocks: dict[tuple[int, int, int], np.ndarray],
    basis_num: int,
) -> dict:
    all_keys = set(reference_blocks.keys()) | set(predicted_blocks.keys())
    ref_zero = _empty_like_block(reference_blocks, basis_num)
    pred_zero = _empty_like_block(predicted_blocks, basis_num)

    best_abs = -1.0
    best = {
        "R": [0, 0, 0],
        "row": 1,
        "col": 1,
        "abs_diff": 0.0,
        "diff_real": 0.0,
        "diff_imag": 0.0,
        "reference_real": 0.0,
        "reference_imag": 0.0,
        "predicted_real": 0.0,
        "predicted_imag": 0.0,
    }
    for r_key in all_keys:
        ref = np.asarray(reference_blocks.get(r_key, ref_zero), dtype=complex)
        pred = np.asarray(predicted_blocks.get(r_key, pred_zero), dtype=complex)
        diff = pred - ref
        abs_diff = np.abs(diff)
        local_index = np.unravel_index(int(np.argmax(abs_diff)), abs_diff.shape)
        local_abs = float(abs_diff[local_index])
        if local_abs <= best_abs:
            continue
        best_abs = local_abs
        rr, cc = int(local_index[0]), int(local_index[1])
        best = {
            "R": [int(r_key[0]), int(r_key[1]), int(r_key[2])],
            "row": rr + 1,
            "col": cc + 1,
            "abs_diff": local_abs,
            "diff_real": float(np.real(diff[rr, cc])),
            "diff_imag": float(np.imag(diff[rr, cc])),
            "reference_real": float(np.real(ref[rr, cc])),
            "reference_imag": float(np.imag(ref[rr, cc])),
            "predicted_real": float(np.real(pred[rr, cc])),
            "predicted_imag": float(np.imag(pred[rr, cc])),
        }
    return best


def compare_block_sets_with_detail(
    reference_blocks: dict[tuple[int, int, int], np.ndarray],
    predicted_blocks: dict[tuple[int, int, int], np.ndarray],
    basis_num: int,
) -> tuple[MatrixSummary, dict]:
    all_keys = set(reference_blocks.keys()) | set(predicted_blocks.keys())
    ref_zero = _empty_like_block(reference_blocks, basis_num)
    pred_zero = _empty_like_block(predicted_blocks, basis_num)

    total_count = 0
    sum_abs = 0.0
    sum_sq = 0.0
    sum_ref_sq = 0.0
    max_abs = 0.0
    missing_pred = 0
    extra_pred = 0

    best_abs = -1.0
    best = {
        "R": [0, 0, 0],
        "row": 1,
        "col": 1,
        "abs_diff": 0.0,
        "diff_real": 0.0,
        "diff_imag": 0.0,
        "reference_real": 0.0,
        "reference_imag": 0.0,
        "predicted_real": 0.0,
        "predicted_imag": 0.0,
    }

    for r_key in all_keys:
        ref = np.asarray(reference_blocks.get(r_key, ref_zero), dtype=complex)
        pred = np.asarray(predicted_blocks.get(r_key, pred_zero), dtype=complex)
        if r_key not in predicted_blocks:
            missing_pred += 1
        if r_key not in reference_blocks:
            extra_pred += 1

        diff = pred - ref
        abs_diff = np.abs(diff)

        total_count += int(diff.size)
        sum_abs += float(np.sum(abs_diff))
        sum_sq += float(np.vdot(diff.ravel(), diff.ravel()).real)
        sum_ref_sq += float(np.vdot(ref.ravel(), ref.ravel()).real)

        local_index = np.unravel_index(int(np.argmax(abs_diff)), abs_diff.shape)
        local_abs = float(abs_diff[local_index])
        if local_abs > max_abs:
            max_abs = local_abs
        if local_abs > best_abs:
            best_abs = local_abs
            rr, cc = int(local_index[0]), int(local_index[1])
            best = {
                "R": [int(r_key[0]), int(r_key[1]), int(r_key[2])],
                "row": rr + 1,
                "col": cc + 1,
                "abs_diff": local_abs,
                "diff_real": float(np.real(diff[rr, cc])),
                "diff_imag": float(np.imag(diff[rr, cc])),
                "reference_real": float(np.real(ref[rr, cc])),
                "reference_imag": float(np.imag(ref[rr, cc])),
                "predicted_real": float(np.real(pred[rr, cc])),
                "predicted_imag": float(np.imag(pred[rr, cc])),
            }

    mean_abs = float(sum_abs / total_count) if total_count > 0 else 0.0
    rms_abs = float(np.sqrt(sum_sq / total_count)) if total_count > 0 else 0.0
    rel_fro = float(np.sqrt(sum_sq / sum_ref_sq)) if sum_ref_sq > 0.0 else 0.0
    summary = MatrixSummary(
        max_abs=max_abs,
        mean_abs=mean_abs,
        rms_abs=rms_abs,
        rel_fro=rel_fro,
        element_count=total_count,
        missing_predicted_R_count=missing_pred,
        extra_predicted_R_count=extra_pred,
    )
    return summary, best


def average_block_sets(
    lhs_blocks: dict[tuple[int, int, int], np.ndarray],
    rhs_blocks: dict[tuple[int, int, int], np.ndarray],
    basis_num: int,
) -> dict[tuple[int, int, int], np.ndarray]:
    all_keys = sorted(set(lhs_blocks.keys()) | set(rhs_blocks.keys()))
    lhs_zero = _empty_like_block(lhs_blocks, basis_num)
    rhs_zero = _empty_like_block(rhs_blocks, basis_num)

    averaged = {}
    for r_key in all_keys:
        lhs = np.asarray(lhs_blocks.get(r_key, lhs_zero), dtype=complex)
        rhs = np.asarray(rhs_blocks.get(r_key, rhs_zero), dtype=complex)
        averaged[r_key] = 0.5 * (lhs + rhs)
    return averaged


def average_block_sets_on_touched_pairs(
    lhs_blocks: dict[tuple[int, int, int], np.ndarray],
    rhs_blocks: dict[tuple[int, int, int], np.ndarray],
    metadata,
    touched_pairs: set[tuple[tuple[int, int, int], int, int]],
    in_place: bool = False,
) -> dict[tuple[int, int, int], np.ndarray]:
    basis_num = int(metadata.basis_num)
    averaged = lhs_blocks if bool(in_place) else {k: np.asarray(v, dtype=complex).copy() for k, v in lhs_blocks.items()}
    zero = _empty_like_block(lhs_blocks, basis_num)
    atom_slices = [_atom_slice(metadata, i) for i in range(len(metadata.species_by_atom))]
    touched_by_r: dict[tuple[int, int, int], list[tuple[int, int]]] = {}

    for r_key, a, b in touched_pairs:
        touched_by_r.setdefault(r_key, []).append((int(a), int(b)))

    for r_key, pairs in touched_by_r.items():
        rhs = np.asarray(rhs_blocks.get(r_key, zero), dtype=complex)
        if r_key in averaged:
            lhs = np.asarray(averaged[r_key], dtype=complex)
            if not bool(in_place):
                lhs = lhs.copy()
        else:
            lhs = np.asarray(zero, dtype=complex).copy()

        for a, b in pairs:
            sl_a = atom_slices[a]
            sl_b = atom_slices[b]
            lhs[sl_a, sl_b] = 0.5 * (lhs[sl_a, sl_b] + rhs[sl_a, sl_b])
        averaged[r_key] = lhs

    return averaged


def hermitize_blocks(
    blocks: dict[tuple[int, int, int], np.ndarray],
    basis_num: int,
) -> dict[tuple[int, int, int], np.ndarray]:
    zero = _empty_like_block(blocks, basis_num)
    all_keys = set(blocks.keys())
    all_keys.update({tuple(int(-x) for x in key) for key in blocks.keys()})

    hermitized: dict[tuple[int, int, int], np.ndarray] = {}
    for r_key in sorted(all_keys):
        nr_key = tuple(int(-x) for x in r_key)
        lhs = np.asarray(blocks.get(r_key, zero), dtype=complex)
        rhs = np.asarray(blocks.get(nr_key, zero), dtype=complex)
        hermitized[r_key] = 0.5 * (lhs + rhs.conj().T)
    return hermitized


def _self_covariance_statistics_with_contexts(
    blocks: dict[tuple[int, int, int], np.ndarray],
    metadata,
    operation_contexts: list[tuple[dict, dict]],
    nonzero_block_tol: float = 1.0e-9,
) -> dict:
    per_op = []
    max_abs = 0.0
    mean_abs_sum = 0.0
    rms_sq_sum = 0.0
    rel_fro_max = 0.0

    for op, context in operation_contexts:
        transformed = transform_blocks_with_context(
            blocks,
            metadata,
            context,
            nonzero_block_tol=float(nonzero_block_tol),
        )
        summary = compare_block_sets(blocks, transformed, int(metadata.basis_num))
        per_op.append(
            {
                "operation_index": int(op["index"]),
                "max_abs": float(summary.max_abs),
                "mean_abs": float(summary.mean_abs),
                "rms_abs": float(summary.rms_abs),
                "rel_fro": float(summary.rel_fro),
                "element_count": int(summary.element_count),
                "missing_predicted_R_count": int(summary.missing_predicted_R_count),
                "extra_predicted_R_count": int(summary.extra_predicted_R_count),
            }
        )
        max_abs = max(max_abs, float(summary.max_abs))
        mean_abs_sum += float(summary.mean_abs)
        rms_sq_sum += float(summary.rms_abs ** 2)
        rel_fro_max = max(rel_fro_max, float(summary.rel_fro))

    op_count = len(per_op)
    return {
        "operation_count": int(op_count),
        "global_max_abs": float(max_abs),
        "mean_abs_over_operations": float(mean_abs_sum / op_count) if op_count > 0 else 0.0,
        "rms_abs_over_operations": float(np.sqrt(rms_sq_sum / op_count)) if op_count > 0 else 0.0,
        "max_rel_fro_over_operations": float(rel_fro_max),
        "operations": per_op,
    }


def self_covariance_statistics(
    blocks: dict[tuple[int, int, int], np.ndarray],
    metadata,
    operations: list[dict],
    map_tol: float = 1.0e-5,
    nonzero_block_tol: float = 1.0e-9,
    operation_contexts: list[tuple[dict, dict]] | None = None,
) -> dict:
    if operation_contexts is None:
        operation_contexts = prepare_operation_contexts(
            metadata,
            operations,
            map_tol=float(map_tol),
        )
    return _self_covariance_statistics_with_contexts(
        blocks,
        metadata,
        operation_contexts,
        nonzero_block_tol=float(nonzero_block_tol),
    )


def sequential_symmetrize_hs(
    hr_blocks: dict[tuple[int, int, int], np.ndarray],
    sr_blocks: dict[tuple[int, int, int], np.ndarray],
    metadata,
    operations: list[dict],
    hr_max_abs_threshold_ry: float = 1.0e-3,
    operation_target_max_abs_ry: float = 1.0e-8,
    max_iter_per_operation: int | None = 5,
    map_tol: float = 1.0e-5,
    nonzero_block_tol: float = 1.0e-9,
    verbose: bool = False,
    operation_contexts: list[tuple[dict, dict]] | None = None,
) -> tuple[dict[tuple[int, int, int], np.ndarray], dict[tuple[int, int, int], np.ndarray], list[dict]]:
    max_iter_per_operation = int(5 if max_iter_per_operation is None else max_iter_per_operation)
    if max_iter_per_operation <= 0:
        raise ValueError(f"max_iter_per_operation must be positive, got {max_iter_per_operation}.")

    cur_hr = {k: np.asarray(v, dtype=complex).copy() for k, v in hr_blocks.items()}
    cur_sr = {k: np.asarray(v, dtype=complex).copy() for k, v in sr_blocks.items()}

    if operation_contexts is None:
        operation_contexts = prepare_operation_contexts(
            metadata,
            operations,
            map_tol=float(map_tol),
        )
    if not operation_contexts:
        return cur_hr, cur_sr, []

    # Track numerically active atom-pairs by R and reuse the index across iterations.
    active_context = operation_contexts[0][1]
    active_hr_index = build_active_pair_index(
        cur_hr,
        active_context,
        nonzero_block_tol=float(nonzero_block_tol),
    )
    active_sr_index = build_active_pair_index(
        cur_sr,
        active_context,
        nonzero_block_tol=float(nonzero_block_tol),
    )

    history = []
    op_count = len(operation_contexts)
    if verbose:
        print(
            f"[covsymm] start sequential symmetrization: operations={op_count}, "
            f"max_iter_per_operation={max_iter_per_operation}, "
            f"target={float(operation_target_max_abs_ry):.3e}, "
            f"nonzero_block_tol={float(nonzero_block_tol):.3e}",
            flush=True,
        )

    for op_pos, (op, context) in enumerate(operation_contexts, start=1):
        op_row = {
            "operation_index": int(op["index"]),
            "operation_position": int(op_pos),
            "target_max_abs_ry": float(operation_target_max_abs_ry),
            "max_iter_per_operation": int(max_iter_per_operation),
            "nonzero_block_tol": float(nonzero_block_tol),
            "iterations": [],
            "converged": False,
        }
        for op_iter in range(1, max_iter_per_operation + 1):
            tr_hr, touched_hr = transform_blocks_with_context(
                cur_hr,
                metadata,
                context,
                nonzero_block_tol=float(nonzero_block_tol),
                return_touched_pairs=True,
                active_pair_index=active_hr_index,
            )
            tr_sr, touched_sr = transform_blocks_with_context(
                cur_sr,
                metadata,
                context,
                nonzero_block_tol=float(nonzero_block_tol),
                return_touched_pairs=True,
                active_pair_index=active_sr_index,
            )
            candidate_hr_index = {k: set(v) for k, v in active_hr_index.items()}
            candidate_sr_index = {k: set(v) for k, v in active_sr_index.items()}
            merge_active_pair_index(candidate_hr_index, touched_hr)
            merge_active_pair_index(candidate_sr_index, touched_sr)

            stat_hr = compare_block_sets_on_candidate_pairs(
                cur_hr,
                tr_hr,
                metadata,
                candidate_hr_index,
            )
            stat_sr = compare_block_sets_on_candidate_pairs(
                cur_sr,
                tr_sr,
                metadata,
                candidate_sr_index,
            )
            op_row["iterations"].append(
                {
                    "iteration_index": int(op_iter),
                    "active_pairs_HR": int(len(touched_hr)),
                    "active_pairs_SR": int(len(touched_sr)),
                    "HR": stat_hr.to_dict(),
                    "SR": stat_sr.to_dict(),
                }
            )
            if verbose:
                print(
                    f"[covsymm] op {op_pos}/{op_count} (#{int(op['index'])}) iter {op_iter}/{max_iter_per_operation}: "
                    f"active_pairs(HR/SR)={len(touched_hr)}/{len(touched_sr)} | "
                    f"HR max={stat_hr.max_abs:.3e}, mean={stat_hr.mean_abs:.3e}, rms={stat_hr.rms_abs:.3e}, rel={stat_hr.rel_fro:.3e} | "
                    f"SR max={stat_sr.max_abs:.3e}, mean={stat_sr.mean_abs:.3e}, rms={stat_sr.rms_abs:.3e}, rel={stat_sr.rel_fro:.3e}",
                    flush=True,
                )

            # Update the full block set, not just the touched atom-pair subset.
            # The transformed matrix may contribute to pairs that were not marked
            # as touched in the current iteration, and dropping those terms can
            # destroy the positive definiteness of S(k) even if the symmetry
            # residuals look numerically small.
            cur_hr = average_block_sets(cur_hr, tr_hr, int(metadata.basis_num))
            cur_sr = average_block_sets(cur_sr, tr_sr, int(metadata.basis_num))
            merge_active_pair_index(active_hr_index, touched_hr)
            merge_active_pair_index(active_sr_index, touched_sr)
            if float(stat_hr.max_abs) <= float(operation_target_max_abs_ry):
                op_row["converged"] = True
                if verbose:
                    print(
                        f"[covsymm] op {op_pos}/{op_count} (#{int(op['index'])}) converged: "
                        f"HR max={stat_hr.max_abs:.3e} <= {float(operation_target_max_abs_ry):.3e}",
                        flush=True,
                    )
                break

        final_iter = op_row["iterations"][-1]
        op_row["final_hr_max_abs"] = float(final_iter["HR"]["max_abs"])
        op_row["final_sr_max_abs"] = float(final_iter["SR"]["max_abs"])
        history.append(op_row)

    final_hr = _self_covariance_statistics_with_contexts(
        cur_hr,
        metadata,
        operation_contexts,
        nonzero_block_tol=float(nonzero_block_tol),
    )
    final_sr = _self_covariance_statistics_with_contexts(
        cur_sr,
        metadata,
        operation_contexts,
        nonzero_block_tol=float(nonzero_block_tol),
    )
    worst = None
    for op, context in operation_contexts:
        transformed_hr = transform_blocks_with_context(
            cur_hr,
            metadata,
            context,
            nonzero_block_tol=float(nonzero_block_tol),
        )
        summary, detail = compare_block_sets_with_detail(cur_hr, transformed_hr, int(metadata.basis_num))
        row = {
            "operation_index": int(op["index"]),
            "max_abs": float(summary.max_abs),
            "detail": detail,
        }
        if verbose:
            print(
                f"[covsymm] post-symmetrization op #{int(op['index'])}: "
                f"HR max={float(summary.max_abs):.3e} at R={detail['R']} ({detail['row']},{detail['col']})",
                flush=True,
            )
        if worst is None or row["max_abs"] > float(worst["max_abs"]):
            worst = row

    history.append(
        {
            "final_symmetry_error": {
                "HR": final_hr,
                "SR": final_sr,
                "worst_error_element_in_symmetry_scan": worst,
            },
        }
    )

    return cur_hr, cur_sr, history


def write_symmetrized_hs(
    hr_blocks: dict[tuple[int, int, int], np.ndarray],
    sr_blocks: dict[tuple[int, int, int], np.ndarray],
    output_hr_path: str | Path,
    output_sr_path: str | Path,
    basis_num: int,
    nspin: int,
    hr_unit: str,
) -> None:
    hr_blocks, sr_blocks = _synchronize_block_keys(hr_blocks, sr_blocks, int(basis_num))
    _write_abacus_sparse_xr(
        Path(output_hr_path),
        hr_blocks,
        int(basis_num),
        int(nspin),
        unit_scale=_unit_scale_from_hr_unit(str(hr_unit)),
    )
    _write_abacus_sparse_xr(
        Path(output_sr_path),
        sr_blocks,
        int(basis_num),
        int(nspin),
        unit_scale=1.0,
    )
