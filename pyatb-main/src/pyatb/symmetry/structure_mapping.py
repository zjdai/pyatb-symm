from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms
from scipy.optimize import linear_sum_assignment


_ROW_VECTOR_RELATION = "source_lattice @ rotation_matrix = supercell_matrix @ target_lattice"


@dataclass
class StructureMappingResult:
    source_lattice: np.ndarray
    target_lattice: np.ndarray
    rotation_matrix: np.ndarray
    supercell_matrix: np.ndarray
    fractional_translation: np.ndarray
    global_shift: np.ndarray
    atom_mapping: list[dict]
    max_lattice_error: float
    max_atom_error: float
    mean_atom_error: float
    max_fractional_error: float
    matrix_relation: str = _ROW_VECTOR_RELATION


def _wrap_fractional(frac: np.ndarray, tol: float = 1.0e-9) -> np.ndarray:
    wrapped = np.asarray(frac, dtype=float) - np.floor(np.asarray(frac, dtype=float))
    wrapped[np.abs(wrapped) <= tol] = 0.0
    wrapped[np.abs(wrapped - 1.0) <= tol] = 0.0
    return wrapped


def _canonical_shift(shift: np.ndarray, tol: float = 1.0e-9) -> np.ndarray:
    value = _wrap_fractional(np.asarray(shift, dtype=float), tol=tol)
    value[np.abs(value) <= tol] = 0.0
    return value


def _integer_matrix(matrix: np.ndarray, tol: float) -> np.ndarray:
    raw = np.asarray(matrix, dtype=float).reshape(3, 3)
    rounded = np.rint(raw).astype(int)
    err = float(np.max(np.abs(raw - rounded)))
    if err > tol:
        raise ValueError(f"Supercell matrix is not integral within tolerance: max_err={err:.6e}.")
    return rounded


def validate_lattice_relation(
    source_lattice: np.ndarray,
    target_lattice: np.ndarray,
    rotation_matrix: np.ndarray,
    supercell_matrix: np.ndarray,
    tol: float = 1.0e-5,
) -> float:
    """Validate the row-vector structure relation used by pyatb.

    Fractional row coordinates are converted to Cartesian coordinates by
    ``cart = frac @ lattice``.  A Cartesian rotation is applied on the right,
    so a source lattice transformed to the target lattice obeys
    ``source_lattice @ rotation_matrix = supercell_matrix @ target_lattice``.
    """

    source = np.asarray(source_lattice, dtype=float).reshape(3, 3)
    target = np.asarray(target_lattice, dtype=float).reshape(3, 3)
    rotation = np.asarray(rotation_matrix, dtype=float).reshape(3, 3)
    supercell = np.asarray(supercell_matrix, dtype=float).reshape(3, 3)

    lhs = source @ rotation
    rhs = supercell @ target
    err = float(np.max(np.abs(lhs - rhs)))
    if err <= float(tol):
        return err

    inv_target = np.linalg.inv(target)
    inferred = lhs @ inv_target
    inferred_int = np.rint(inferred).astype(int)
    diagnostics = {
        _ROW_VECTOR_RELATION: err,
        "source_lattice = supercell_matrix @ target_lattice @ rotation_matrix.T": float(
            np.max(np.abs(source - supercell @ target @ rotation.T))
        ),
        "source_lattice = target_lattice @ supercell_matrix @ rotation_matrix": float(
            np.max(np.abs(source - target @ supercell @ rotation))
        ),
        "inferred_supercell_matrix": inferred.tolist(),
        "rounded_inferred_supercell_matrix": inferred_int.tolist(),
        "inferred_supercell_rounding_error": float(np.max(np.abs(inferred - inferred_int))),
    }
    raise ValueError(
        "Structure mapping lattice relation failed. "
        f"Expected row-vector relation '{_ROW_VECTOR_RELATION}', max_err={err:.6e}, "
        f"diagnostics={diagnostics}"
    )


def _candidate_global_shifts(
    source_frac_in_target: np.ndarray,
    target_frac: np.ndarray,
    source_numbers: np.ndarray,
    target_numbers: np.ndarray,
    fractional_translation: np.ndarray | None,
    tol: float,
) -> list[np.ndarray]:
    candidates: list[np.ndarray] = []

    def add(candidate: np.ndarray) -> None:
        shift = _canonical_shift(candidate, tol=tol)
        if not any(float(np.max(np.abs(shift - item - np.rint(shift - item)))) <= tol for item in candidates):
            candidates.append(shift)

    add(np.zeros(3, dtype=float))
    if fractional_translation is not None:
        add(np.asarray(fractional_translation, dtype=float).reshape(3))
        add(-np.asarray(fractional_translation, dtype=float).reshape(3))

    for old_idx, source_number in enumerate(source_numbers.tolist()):
        for new_idx, target_number in enumerate(target_numbers.tolist()):
            if int(source_number) != int(target_number):
                continue
            add(source_frac_in_target[int(old_idx)] - target_frac[int(new_idx)])

    return candidates


def _score_global_shift(
    global_shift: np.ndarray,
    source_frac_in_target: np.ndarray,
    target_frac: np.ndarray,
    target_lattice: np.ndarray,
    source_numbers: np.ndarray,
    target_numbers: np.ndarray,
    source_symbols: list[str],
) -> dict | None:
    n_source = len(source_numbers)
    n_target = len(target_numbers)
    big = 1.0e30

    def pair_data(old_idx: int, new_idx: int):
        if int(source_numbers[old_idx]) != int(target_numbers[new_idx]):
            return None
        raw = source_frac_in_target[old_idx] - target_frac[new_idx] - global_shift
        shift = np.rint(raw).astype(int)
        frac_residual = raw - shift
        cart_residual = frac_residual @ target_lattice
        cart_err = float(np.max(np.abs(cart_residual)))
        frac_err = float(np.max(np.abs(frac_residual)))
        return shift, cart_err, frac_err

    if n_source == n_target:
        cost = np.full((n_source, n_target), big, dtype=float)
        shift_table: dict[tuple[int, int], np.ndarray] = {}
        frac_error_table: dict[tuple[int, int], float] = {}
        for old_idx in range(n_source):
            for new_idx in range(n_target):
                data = pair_data(old_idx, new_idx)
                if data is None:
                    continue
                shift, cart_err, frac_err = data
                cost[old_idx, new_idx] = cart_err
                shift_table[(old_idx, new_idx)] = shift
                frac_error_table[(old_idx, new_idx)] = frac_err

        row_ind, col_ind = linear_sum_assignment(cost)
        if not np.array_equal(row_ind, np.arange(n_source)):
            return None
        if any(cost[int(i), int(j)] >= big * 0.5 for i, j in zip(row_ind, col_ind, strict=True)):
            return None

        mapping = []
        cart_errors = []
        frac_errors = []
        for old_idx, new_idx in zip(row_ind.tolist(), col_ind.tolist(), strict=True):
            shift = np.asarray(shift_table[(int(old_idx), int(new_idx))], dtype=int)
            mapping.append(
                {
                    "old_atom": int(old_idx),
                    "new_atom": int(new_idx),
                    "shift": shift,
                    "species": str(source_symbols[int(old_idx)]),
                }
            )
            cart_errors.append(float(cost[int(old_idx), int(new_idx)]))
            frac_errors.append(float(frac_error_table[(int(old_idx), int(new_idx))]))
    else:
        mapping = []
        cart_errors = []
        frac_errors = []
        used_images: set[tuple[int, tuple[int, int, int]]] = set()
        for old_idx in range(n_source):
            best = None
            for new_idx in range(n_target):
                data = pair_data(old_idx, new_idx)
                if data is None:
                    continue
                shift, cart_err, frac_err = data
                image_key = (int(new_idx), tuple(int(value) for value in shift.tolist()))
                if image_key in used_images:
                    continue
                if best is None or cart_err < best[0]:
                    best = (cart_err, frac_err, int(new_idx), shift, image_key)
            if best is None:
                return None
            used_images.add(best[4])
            mapping.append(
                {
                    "old_atom": int(old_idx),
                    "new_atom": int(best[2]),
                    "shift": np.asarray(best[3], dtype=int),
                    "species": str(source_symbols[int(old_idx)]),
                }
            )
            cart_errors.append(float(best[0]))
            frac_errors.append(float(best[1]))

    return {
        "global_shift": np.asarray(global_shift, dtype=float),
        "atom_mapping": mapping,
        "max_atom_error": float(max(cart_errors) if cart_errors else 0.0),
        "mean_atom_error": float(np.mean(cart_errors) if cart_errors else 0.0),
        "max_fractional_error": float(max(frac_errors) if frac_errors else 0.0),
    }


def build_structure_mapping(
    source_atoms: Atoms,
    target_atoms: Atoms,
    rotation_matrix: np.ndarray,
    supercell_matrix: np.ndarray,
    fractional_translation: np.ndarray | None = None,
    tol: float = 1.0e-5,
    lattice_tol: float | None = None,
) -> StructureMappingResult:
    """Build and validate source-atom to target-atom mapping for HS standardization."""

    source_lattice = np.asarray(source_atoms.cell.array, dtype=float)
    target_lattice = np.asarray(target_atoms.cell.array, dtype=float)
    rotation = np.asarray(rotation_matrix, dtype=float).reshape(3, 3)
    supercell = _integer_matrix(supercell_matrix, tol=max(float(tol), 1.0e-8))
    lattice_error = validate_lattice_relation(
        source_lattice,
        target_lattice,
        rotation,
        supercell,
        tol=float(lattice_tol if lattice_tol is not None else tol),
    )

    source_frac = np.asarray(source_atoms.get_scaled_positions(wrap=False), dtype=float)
    target_frac = _wrap_fractional(np.asarray(target_atoms.get_scaled_positions(wrap=False), dtype=float), tol=tol)
    source_numbers = np.asarray(source_atoms.get_atomic_numbers(), dtype=int)
    target_numbers = np.asarray(target_atoms.get_atomic_numbers(), dtype=int)
    source_symbols = list(source_atoms.get_chemical_symbols())

    if len(source_numbers) == 0:
        return StructureMappingResult(
            source_lattice=source_lattice,
            target_lattice=target_lattice,
            rotation_matrix=rotation,
            supercell_matrix=supercell,
            fractional_translation=np.zeros(3, dtype=float)
            if fractional_translation is None
            else np.asarray(fractional_translation, dtype=float).reshape(3),
            global_shift=np.zeros(3, dtype=float),
            atom_mapping=[],
            max_lattice_error=float(lattice_error),
            max_atom_error=0.0,
            mean_atom_error=0.0,
            max_fractional_error=0.0,
        )

    source_frac_in_target = source_frac @ supercell
    frac_shift = (
        np.zeros(3, dtype=float)
        if fractional_translation is None
        else np.asarray(fractional_translation, dtype=float).reshape(3)
    )
    candidates = _candidate_global_shifts(
        source_frac_in_target,
        target_frac,
        source_numbers,
        target_numbers,
        frac_shift,
        tol=max(float(tol), 1.0e-9),
    )

    best = None
    for candidate in candidates:
        score = _score_global_shift(
            candidate,
            source_frac_in_target,
            target_frac,
            target_lattice,
            source_numbers,
            target_numbers,
            source_symbols,
        )
        if score is None:
            continue
        key = (
            float(score["max_atom_error"]),
            float(score["mean_atom_error"]),
            float(score["max_fractional_error"]),
        )
        if best is None or key < best[0]:
            best = (key, score)

    if best is None:
        raise ValueError("Failed to build source-to-target atom mapping: no species-compatible mapping was found.")

    score = best[1]
    if float(score["max_atom_error"]) > float(tol):
        raise ValueError(
            "Failed to build source-to-target atom mapping within tolerance: "
            f"max_cart_error={float(score['max_atom_error']):.6e}, "
            f"mean_cart_error={float(score['mean_atom_error']):.6e}, "
            f"max_fractional_error={float(score['max_fractional_error']):.6e}."
        )

    return StructureMappingResult(
        source_lattice=source_lattice,
        target_lattice=target_lattice,
        rotation_matrix=rotation,
        supercell_matrix=supercell,
        fractional_translation=frac_shift,
        global_shift=np.asarray(score["global_shift"], dtype=float),
        atom_mapping=list(score["atom_mapping"]),
        max_lattice_error=float(lattice_error),
        max_atom_error=float(score["max_atom_error"]),
        mean_atom_error=float(score["mean_atom_error"]),
        max_fractional_error=float(score["max_fractional_error"]),
    )


def structure_mapping_to_tuples(
    mapping_result: StructureMappingResult,
) -> list[tuple[int, int, np.ndarray]]:
    return [
        (
            int(item["old_atom"]),
            int(item["new_atom"]),
            np.asarray(item["shift"], dtype=int),
        )
        for item in mapping_result.atom_mapping
    ]


def structure_mapping_to_dicts(mapping_result: StructureMappingResult) -> list[dict]:
    return [
        {
            "old_atom": int(item["old_atom"]),
            "new_atom": int(item["new_atom"]),
            "shift": np.asarray(item["shift"], dtype=int),
            "species": str(item.get("species", "")),
        }
        for item in mapping_result.atom_mapping
    ]


def structure_mapping_summary(mapping_result: StructureMappingResult) -> dict:
    return {
        "matrix_relation": str(mapping_result.matrix_relation),
        "max_lattice_error": float(mapping_result.max_lattice_error),
        "max_atom_error": float(mapping_result.max_atom_error),
        "mean_atom_error": float(mapping_result.mean_atom_error),
        "max_fractional_error": float(mapping_result.max_fractional_error),
        "global_shift": np.asarray(mapping_result.global_shift, dtype=float).tolist(),
        "fractional_translation": np.asarray(mapping_result.fractional_translation, dtype=float).tolist(),
        "supercell_matrix": np.asarray(mapping_result.supercell_matrix, dtype=int).tolist(),
    }
