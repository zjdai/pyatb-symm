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


def _cornwell_satisfied(resolution) -> bool:
    return bool(getattr(resolution, "cornwell_satisfied", True))


def _uses_nonsymmorphic_factor_system(resolution) -> bool:
    return not _cornwell_satisfied(resolution)


def _should_use_coeff_phase(phase_kind: int, resolution) -> bool:
    # IRVSP applies coeff_uvw only in its nonsymmorphic kLG-table branch
    # (FGT=.FALSE.).  Cornwell-satisfied k points are classified by the
    # ordinary point-group branch, so phase_kind=2 must not force an
    # additional table phase there.
    return int(phase_kind) == 2 and _uses_nonsymmorphic_factor_system(resolution)



def _operation_phase_factors(k_direct, operations, operation_indices) -> np.ndarray:
    indices = np.asarray(operation_indices, dtype=int).reshape(-1)
    factors = np.ones(indices.size, dtype=complex)
    if k_direct is None or operations is None:
        return factors
    for pos, op_idx in enumerate(indices):
        if 0 <= int(op_idx) < len(operations):
            factors[pos] = _current_operation_phase(k_direct, operations[int(op_idx)])
    return factors


def _translation_phase_factors(k_direct, translations) -> np.ndarray:
    arr = np.asarray(translations, dtype=float)
    if arr.size == 0:
        return np.ones(0, dtype=complex)
    arr = np.atleast_2d(arr)
    factors = np.ones(arr.shape[0], dtype=complex)
    if k_direct is None or arr.shape[1] < 3:
        return factors
    k = np.asarray(k_direct, dtype=float).reshape(-1)
    if k.size < 3:
        return factors
    for pos, tau in enumerate(arr):
        angle = -2.0 * np.pi * float(np.dot(k[:3], tau[:3]))
        factors[pos] = np.exp(1j * angle)
    return factors


def _table_phase_factors(
    resolution,
    table_operation_indices,
    phase_k_direct=None,
    table_operation_translations=None,
) -> np.ndarray:
    table_active = np.asarray(table_operation_indices, dtype=int).reshape(-1)
    factors = np.ones(table_active.size, dtype=complex)
    if resolution is None or not _uses_nonsymmorphic_factor_system(resolution):
        return factors

    if table_operation_translations is not None and phase_k_direct is not None:
        representative_phases = _translation_phase_factors(phase_k_direct, table_operation_translations)
        if representative_phases.size == table_active.size:
            return representative_phases

    irreps = list(getattr(getattr(resolution, "entry", None), "irreps", []))
    if not irreps:
        return factors
    ref_irrep = irreps[0]
    phase_kinds = np.asarray(getattr(ref_irrep, "phase_kinds", []), dtype=int).reshape(-1)
    coeff_uvw = np.asarray(getattr(ref_irrep, "coeff_uvw", []), dtype=float)
    k_conv = np.asarray(getattr(resolution, "k_conv", np.zeros(3, dtype=float)), dtype=float).reshape(-1)
    if coeff_uvw.ndim != 2 or coeff_uvw.shape[1] < 3 or k_conv.size < 3:
        return factors

    for pos, table_idx in enumerate(table_active):
        idx = int(table_idx)
        if idx < 0 or idx >= phase_kinds.size or idx >= coeff_uvw.shape[0]:
            continue
        if _should_use_coeff_phase(int(phase_kinds[idx]), resolution):
            angle = -np.pi * float(np.dot(coeff_uvw[idx, :3], k_conv[:3]))
            factors[pos] = np.exp(1j * angle)
    return factors


def _representative_corrected_characters(
    characters: np.ndarray,
    resolution,
    active_operation_indices,
    table_operation_indices,
    phase_k_direct=None,
    phase_operations=None,
    table_operation_translations=None,
) -> np.ndarray:
    target = np.asarray(characters, dtype=complex).reshape(-1).copy()
    if not _uses_nonsymmorphic_factor_system(resolution):
        return target

    active = np.asarray(active_operation_indices, dtype=int).reshape(-1)
    table_active = np.asarray(table_operation_indices, dtype=int).reshape(-1)
    if target.size != active.size or active.size != table_active.size:
        return target
    if phase_k_direct is None or phase_operations is None:
        return target

    active_phases = _operation_phase_factors(phase_k_direct, phase_operations, active)
    table_phases = _table_phase_factors(
        resolution,
        table_active,
        phase_k_direct=phase_k_direct,
        table_operation_translations=table_operation_translations,
    )
    correction = np.ones(target.size, dtype=complex)
    mask = np.abs(active_phases) > 1.0e-14
    correction[mask] = table_phases[mask] / active_phases[mask]
    return target * correction

def _comparison_characters(
    characters: np.ndarray,
    resolution,
    active_operation_indices,
    table_operation_indices=None,
    phase_k_direct=None,
    phase_operations=None,
) -> np.ndarray:
    target = np.asarray(characters, dtype=complex).reshape(-1).copy()
    active = np.asarray(active_operation_indices, dtype=int).reshape(-1)

    if _uses_nonsymmorphic_factor_system(resolution):
        return target

    # When Cornwell is satisfied, the space-group character is a point-group
    # character times KPH({R|tau}) = exp(-2*pi*i*k_prim.tau).  Remove KPH before
    # comparing with ordinary point-group characters.
    if phase_k_direct is not None and phase_operations is not None:
        for pos, active_idx in enumerate(active):
            if pos >= target.size or int(active_idx) < 0 or int(active_idx) >= len(phase_operations):
                continue
            target[pos] *= np.conj(_current_operation_phase(phase_k_direct, phase_operations[int(active_idx)]))
    return target


def _comparison_character_candidates(
    characters: np.ndarray,
    resolution,
    active_operation_indices,
    table_operation_indices=None,
    phase_k_direct=None,
    phase_operations=None,
    table_operation_translations=None,
) -> list[np.ndarray]:
    raw = np.asarray(characters, dtype=complex).reshape(-1).copy()
    table_active = active_operation_indices if table_operation_indices is None else table_operation_indices
    if _uses_nonsymmorphic_factor_system(resolution):
        corrected = _representative_corrected_characters(
            raw,
            resolution,
            active_operation_indices,
            table_active,
            phase_k_direct=phase_k_direct,
            phase_operations=phase_operations,
            table_operation_translations=table_operation_translations,
        )
        if np.allclose(corrected, raw, atol=1.0e-10):
            return [corrected]
        return [corrected, raw]

    normalized = _comparison_characters(
        raw,
        resolution,
        active_operation_indices,
        table_operation_indices=table_operation_indices,
        phase_k_direct=phase_k_direct,
        phase_operations=phase_operations,
    )
    if np.allclose(normalized, raw, atol=1.0e-10):
        return [normalized]
    # Keep the IRVSP-style KPH-normalized convention first, but fall back to
    # the raw Dk characters for source-origin conventions where the calculated
    # traces already match the kLittleGroups table.
    return [normalized, raw]


def _resolved_irrep_characters(irrep, resolution, phase_k_direct=None, phase_operations=None) -> np.ndarray:
    table = np.asarray(getattr(irrep, "characters", []), dtype=complex).reshape(-1).copy()
    if table.size == 0 or resolution is None:
        return table

    if _uses_nonsymmorphic_factor_system(resolution):
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

    if _uses_nonsymmorphic_factor_system(resolution):
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
    table_operation_translations=None,
):
    raw_target = np.asarray(characters, dtype=complex).reshape(-1)
    active = np.asarray(active_operation_indices, dtype=int).reshape(-1)
    table_active = active if table_operation_indices is None else np.asarray(table_operation_indices, dtype=int).reshape(-1)
    if table_active.size != active.size:
        raise ValueError("table_operation_indices must have the same length as active_operation_indices.")
    if raw_target.size != table_active.size:
        raise ValueError("character vector length must match active_operation_indices.")
    targets = _comparison_character_candidates(
        raw_target,
        resolution,
        active,
        table_operation_indices=table_active,
        phase_k_direct=phase_k_direct,
        phase_operations=phase_operations,
        table_operation_translations=table_operation_translations,
    )

    best_name = None
    best_error = np.inf
    for target in targets:
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
            if err <= tol:
                return getattr(irrep, "name", getattr(irrep, "raw_name", None))

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
    table_operation_translations=None,
):
    active = np.asarray(active_operation_indices, dtype=int).reshape(-1)
    table_active = active if table_operation_indices is None else np.asarray(table_operation_indices, dtype=int).reshape(-1)
    raw_target = np.asarray(characters, dtype=complex).reshape(-1)
    if table_active.size != active.size:
        raise ValueError("table_operation_indices must have the same length as active_operation_indices.")
    if raw_target.size != table_active.size:
        raise ValueError("character vector length must match active_operation_indices.")
    targets = _comparison_character_candidates(
        raw_target,
        resolution,
        active,
        table_operation_indices=table_active,
        phase_k_direct=phase_k_direct,
        phase_operations=phase_operations,
        table_operation_translations=table_operation_translations,
    )
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
        if not targets or table.size != targets[0].size:
            continue
        sliced_tables.append(table)
        irrep_labels.append(getattr(irrep, "name", getattr(irrep, "raw_name", f"irrep{idx + 1}")))

    best = None
    for target in targets:
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
