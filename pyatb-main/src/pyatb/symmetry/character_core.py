from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np

_COMBO_CACHE: dict[tuple[int, int], tuple[tuple[int, ...], ...]] = {}


def _cached_combinations_with_replacement(num_items: int, term_count: int) -> tuple[tuple[int, ...], ...]:
    key = (int(num_items), int(term_count))
    cached = _COMBO_CACHE.get(key)
    if cached is None:
        cached = tuple(combinations_with_replacement(range(int(num_items)), int(term_count)))
        _COMBO_CACHE[key] = cached
    return cached


def _filter_irreps_by_spin(reolution_irreps, spinful: bool | None):
    irreps = list(reolution_irreps)
    if spinful is None:
        return irreps

    filtered = []
    for irrep in irreps:
        raw_name = getattr(irrep, "raw_name", getattr(irrep, "name", ""))
        reality = int(getattr(irrep, "reality", 1))
        is_double_valued = str(raw_name).startswith("-") and reality == -1
        if spinful and is_double_valued:
            filtered.append(irrep)
        if not spinful and not is_double_valued:
            filtered.append(irrep)
    return filtered or irreps


def _current_operation_phase(k_direct, operation) -> complex:
    k = np.asarray(k_direct, dtype=float).reshape(-1)
    tau = np.asarray(getattr(operation, "translation", np.zeros(3, dtype=float)), dtype=float).reshape(-1)
    if k.size < 3 or tau.size < 3:
        return 1.0 + 0.0j
    angle = -2.0 * np.pi * float(np.dot(k[:3], tau[:3]))
    return np.exp(1j * angle)


def _source_phase_override_enabled(resolution) -> bool:
    return bool(getattr(resolution, "phase_from_source_operations", False)) and bool(
        getattr(resolution, "cornwell_satisfied", True)
    )


def _phase_is_nontrivial(phase: complex, tol: float = 1.0e-8) -> bool:
    return abs(complex(phase) - (1.0 + 0.0j)) > tol


def _should_use_operation_phase(phase_kind: int, resolution, phase: complex) -> bool:
    if int(phase_kind) == 2:
        return _source_phase_override_enabled(resolution)
    if int(phase_kind) != 1:
        return False
    return _source_phase_override_enabled(resolution) and _phase_is_nontrivial(phase)


def _should_use_coeff_phase(phase_kind: int, resolution) -> bool:
    return int(phase_kind) == 2 and not _source_phase_override_enabled(resolution)


def _resolved_irrep_characters(irrep, resolution, phase_k_direct=None, phase_operations=None) -> np.ndarray:
    table = np.asarray(getattr(irrep, "characters", []), dtype=complex).reshape(-1).copy()
    if table.size == 0 or resolution is None:
        return table

    if not bool(getattr(resolution, "cornwell_satisfied", True)):
        table = np.conj(table)

    phase_kinds = np.asarray(getattr(irrep, "phase_kinds", []), dtype=int).reshape(-1)
    coeff_uvw = np.asarray(getattr(irrep, "coeff_uvw", []), dtype=float)
    if coeff_uvw.ndim != 2 or coeff_uvw.shape[1] < 3:
        return table

    count = min(table.size, phase_kinds.size, coeff_uvw.shape[0])
    if count <= 0:
        return table

    k_conv = np.asarray(getattr(resolution, "k_conv", np.zeros(3, dtype=float)), dtype=float).reshape(-1)
    for j in range(count):
        phase_kind = int(phase_kinds[j])
        if _should_use_coeff_phase(phase_kind, resolution) and k_conv.size >= 3:
            angle = -np.pi * float(np.dot(coeff_uvw[j, :3], k_conv[:3]))
            table[j] *= np.exp(1j * angle)
        elif phase_k_direct is not None and phase_operations is not None and j < len(phase_operations):
            phase = _current_operation_phase(phase_k_direct, phase_operations[j])
            if _should_use_operation_phase(phase_kind, resolution, phase):
                table[j] *= phase
    return table


def _resolved_irrep_character_slice(
    irrep,
    resolution,
    active_operation_indices,
    table_operation_indices,
    phase_k_direct=None,
    phase_operations=None,
) -> np.ndarray:
    raw_table = np.asarray(getattr(irrep, "characters", []), dtype=complex).reshape(-1).copy()
    if raw_table.size == 0 or resolution is None:
        return raw_table

    if not bool(getattr(resolution, "cornwell_satisfied", True)):
        raw_table = np.conj(raw_table)

    active = np.asarray(active_operation_indices, dtype=int).reshape(-1)
    table_active = np.asarray(table_operation_indices, dtype=int).reshape(-1)
    phase_kinds = np.asarray(getattr(irrep, "phase_kinds", []), dtype=int).reshape(-1)
    coeff_uvw = np.asarray(getattr(irrep, "coeff_uvw", []), dtype=float)
    k_conv = np.asarray(getattr(resolution, "k_conv", np.zeros(3, dtype=float)), dtype=float).reshape(-1)

    values: list[complex] = []
    for active_idx, table_idx in zip(active, table_active):
        if table_idx < 0 or table_idx >= raw_table.size:
            values.append(np.nan + 0.0j)
            continue
        value = raw_table[int(table_idx)]
        phase_kind = int(phase_kinds[int(table_idx)]) if table_idx < phase_kinds.size else 1
        if table_idx < phase_kinds.size:
            if (
                _should_use_coeff_phase(phase_kind, resolution)
                and coeff_uvw.ndim == 2
                and coeff_uvw.shape[1] >= 3
                and k_conv.size >= 3
            ):
                angle = -np.pi * float(np.dot(coeff_uvw[int(table_idx), :3], k_conv[:3]))
                value *= np.exp(1j * angle)
            elif (
                phase_k_direct is not None
                and phase_operations is not None
                and int(active_idx) < len(phase_operations)
            ):
                phase = _current_operation_phase(phase_k_direct, phase_operations[int(active_idx)])
                if _should_use_operation_phase(phase_kind, resolution, phase):
                    value *= phase
        values.append(value)
    return np.asarray(values, dtype=complex)


def group_degenerate_bands(energies: np.ndarray, tol: float = 5.0e-4) -> list[tuple[int, int]]:
    values = np.asarray(energies, dtype=float).reshape(-1)
    if values.size == 0:
        return []

    groups: list[tuple[int, int]] = []
    start = 0
    for idx in range(1, values.size):
        if abs(values[idx] - values[idx - 1]) > tol:
            groups.append((start, idx - 1))
            start = idx
    groups.append((start, values.size - 1))
    return groups


def assign_irrep_from_characters(
    characters: np.ndarray,
    resolution,
    active_operation_indices: list[int] | np.ndarray,
    tol: float = 1.0e-6,
    spinful: bool | None = None,
    table_operation_indices: list[int] | np.ndarray | None = None,
    phase_k_direct=None,
    phase_operations=None,
):
    target = np.asarray(characters, dtype=complex).reshape(-1)
    active = np.asarray(active_operation_indices, dtype=int).reshape(-1)
    table_active = active if table_operation_indices is None else np.asarray(table_operation_indices, dtype=int).reshape(-1)
    if table_active.size != active.size:
        raise ValueError("table_operation_indices must have the same length as active_operation_indices.")
    if target.size != table_active.size:
        raise ValueError("character vector length must match active_operation_indices.")

    best_name = None
    best_error = np.inf
    for irrep in _filter_irreps_by_spin(resolution.entry.irreps, spinful):
        table = _resolved_irrep_character_slice(
            irrep,
            resolution,
            active,
            table_active,
            phase_k_direct=phase_k_direct,
            phase_operations=phase_operations,
        )
        if table.size != target.size:
            continue
        diff = table - target
        err = float(np.max(np.abs(diff))) if diff.size else 0.0
        if err < best_error:
            best_error = err
            best_name = getattr(irrep, "name", getattr(irrep, "raw_name", None))

    if best_name is None or best_error > tol:
        raise ValueError("Failed to assign irrep from calculated characters.")

    return best_name


def calculate_subspace_characters(
    eigenvectors: np.ndarray,
    overlap: np.ndarray,
    operation_matrices: list[np.ndarray],
    band_range: tuple[int, int],
) -> np.ndarray:
    start, stop = band_range
    subspace = np.asarray(eigenvectors[:, start : stop + 1], dtype=complex)
    overlap = np.asarray(overlap, dtype=complex)

    characters = []
    for op_matrix in operation_matrices:
        reduced = subspace.conj().T @ overlap @ np.asarray(op_matrix, dtype=complex) @ subspace
        characters.append(complex(np.trace(reduced)))
    return np.asarray(characters, dtype=complex)


def assign_irrep_combination(
    characters: np.ndarray,
    resolution,
    active_operation_indices: list[int] | np.ndarray,
    max_terms: int = 4,
    tol: float = 5.0e-2,
    spinful: bool | None = None,
    table_operation_indices: list[int] | np.ndarray | None = None,
    phase_k_direct=None,
    phase_operations=None,
):
    active = np.asarray(active_operation_indices, dtype=int).reshape(-1)
    table_active = active if table_operation_indices is None else np.asarray(table_operation_indices, dtype=int).reshape(-1)
    target = np.asarray(characters, dtype=complex).reshape(-1)
    if table_active.size != active.size:
        raise ValueError("table_operation_indices must have the same length as active_operation_indices.")
    if target.size != table_active.size:
        raise ValueError("character vector length must match active_operation_indices.")
    irreps = _filter_irreps_by_spin(resolution.entry.irreps, spinful)
    sliced_tables = []
    irrep_labels = []
    for idx, irrep in enumerate(irreps):
        table = _resolved_irrep_character_slice(
            irrep,
            resolution,
            active,
            table_active,
            phase_k_direct=phase_k_direct,
            phase_operations=phase_operations,
        )
        if table.size != target.size:
            continue
        sliced_tables.append(table)
        irrep_labels.append(getattr(irrep, "name", getattr(irrep, "raw_name", f"irrep{idx + 1}")))

    best = None
    for term_count in range(1, max_terms + 1):
        for combo in _cached_combinations_with_replacement(len(sliced_tables), term_count):
            trial = np.zeros_like(target)
            labels = []
            for idx in combo:
                trial = trial + sliced_tables[idx]
                labels.append(irrep_labels[idx])
            err = float(np.max(np.abs(trial - target))) if target.size else 0.0
            if best is None or err < best[0]:
                best = (err, labels)
            if err <= tol:
                return " + ".join(labels)

    if best is None:
        raise ValueError("Failed to assign irreps from character combination.")
    if best[0] <= tol * max(1, len(best[1])):
        return " + ".join(best[1])
    return "??"
