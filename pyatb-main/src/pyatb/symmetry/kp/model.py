from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Sequence

import numpy as np

from pyatb import OUTPUT_PATH, RANK, RUNNING_LOG
from pyatb.io.abacus_read_xr import abacus_readHR, abacus_readSR, abacus_readrR
from pyatb.symmetry.character import Character
from pyatb.symmetry.character_core import _resolved_irrep_character_slice
from pyatb.symmetry.Dk_matrix import (
    axis_angle_from_cartesian_rotation,
    build_dk_matrix,
    spin_half_matrix_from_axis_angle,
    spin_half_matrix_from_cartesian_rotation,
)
from pyatb.tb.tb import tb as TBModel

from pyatb.symmetry.kp.types import (
    KP_DIRECTIONS,
    KP_FREE_ELECTRON_COEFFICIENT_EV_A2,
    ZEEMAN_FIELD_DIRECTIONS,
    KPointBandSelection,
    _kp_zeeman_enabled,
    _normalize_selections,
)


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


def _is_float_token(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _moment_from_values(values: Sequence[str] | Sequence[float] | None) -> np.ndarray:
    if not values:
        return np.zeros(3, dtype=float)

    numeric = []
    for value in values:
        if not _is_float_token(str(value)):
            break
        numeric.append(float(value))

    if not numeric:
        return np.zeros(3, dtype=float)
    if len(numeric) == 1:
        return np.array([0.0, 0.0, numeric[0]], dtype=float)
    return np.asarray(numeric[:3], dtype=float)


def _strip_stru_comment(line: str) -> str:
    text = str(line)
    for marker in ("#", "//"):
        text = text.split(marker, 1)[0]
    return text.strip()


def _parse_abacus_stru_magnetic_moments(stru_path: str | Path, atom_count: int) -> np.ndarray:
    """Parse ABACUS STRU atom magnetic moments as Cartesian axial vectors."""

    raw_lines = [
        cleaned
        for line in Path(stru_path).read_text(encoding="utf-8").splitlines()
        if (cleaned := _strip_stru_comment(line))
    ]
    try:
        positions_index = next(
            idx for idx, line in enumerate(raw_lines) if line.strip().upper() == "ATOMIC_POSITIONS"
        )
    except StopIteration:
        return np.zeros((int(atom_count), 3), dtype=float)

    idx = positions_index + 1
    if idx >= len(raw_lines):
        return np.zeros((int(atom_count), 3), dtype=float)
    idx += 1  # coordinate mode

    moments: list[np.ndarray] = []
    while idx < len(raw_lines):
        idx += 1  # species label
        if idx >= len(raw_lines):
            break
        species_default = _moment_from_values(raw_lines[idx].split())
        idx += 1
        if idx >= len(raw_lines):
            break
        atom_num = int(raw_lines[idx].split()[0])
        idx += 1
        for _ in range(atom_num):
            if idx >= len(raw_lines):
                raise ValueError(f"Malformed STRU magnetic moment block in {stru_path}.")
            tokens = raw_lines[idx].split()
            atom_moment = species_default
            lowered = [token.lower() for token in tokens]
            if "mag" in lowered:
                mag_index = lowered.index("mag")
                atom_moment = _moment_from_values(tokens[mag_index + 1 :])
            moments.append(np.asarray(atom_moment, dtype=float))
            idx += 1

    if len(moments) != int(atom_count):
        raise ValueError(
            f"Parsed {len(moments)} magnetic moments from {stru_path}, expected {int(atom_count)}."
        )
    return np.vstack(moments).astype(float)


def _kp_magnetic_moments(
    *,
    stru_path: str | Path,
    atom_count: int,
    mag_tag: int,
    mag: str | Sequence[float],
) -> tuple[np.ndarray, str]:
    if int(mag_tag) == 1:
        values = np.asarray(mag, dtype=float).reshape(-1)
        required = 3 * int(atom_count)
        if values.size < required:
            raise ValueError(
                f"KP.mag_tag=1 requires at least {required} magnetic moment values; got {values.size}."
            )
        return values[:required].reshape(int(atom_count), 3), "kp_input_mag"

    return _parse_abacus_stru_magnetic_moments(stru_path, int(atom_count)), "stru"


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _complex_matrix_to_pairs(matrix: np.ndarray) -> list[list[list[float]]]:
    arr = np.asarray(matrix, dtype=complex)
    return [
        [[float(value.real), float(value.imag)] for value in row]
        for row in arr
    ]


def _complex_vector_to_pairs(vector: np.ndarray) -> list[list[float]]:
    arr = np.asarray(vector, dtype=complex).reshape(-1)
    return [[float(value.real), float(value.imag)] for value in arr]


def _complex_array_from_pairs(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.size == 0:
        return np.zeros((0,), dtype=complex)
    if arr.shape[-1] == 2:
        numeric = np.asarray(values, dtype=float)
        return numeric[..., 0] + 1j * numeric[..., 1]
    return np.asarray(values, dtype=complex)


def _complex_scalar_to_pair(value: complex) -> list[float]:
    number = complex(value)
    return [float(number.real), float(number.imag)]


def _reciprocal_lattice_vectors(lattice: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(lattice, dtype=float)).T


def _vector3(value: Any, fallback: Sequence[float]) -> list[float]:
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        arr = np.asarray(fallback, dtype=float).reshape(-1)
    if arr.size < 3:
        arr = np.asarray(fallback, dtype=float).reshape(-1)
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def _display_float(value: Any) -> float:
    number = float(value)
    return 0.0 if abs(number) < 5.0e-12 else number


def _magnetic_spacegroup_setting(number: Any, scheme: str) -> str:
    del scheme
    return str(number)


def _magnetic_spacegroup_type_label(type_number: Any) -> str:
    labels = {
        1: "type-I",
        2: "type-II",
        3: "type-III",
        4: "type-IV",
    }
    try:
        return labels[int(type_number)]
    except (KeyError, TypeError, ValueError):
        return "unknown"


def _kpoint_star_transform_text(record: Mapping[str, Any] | None, analysis_result: Mapping[str, Any]) -> str:
    if record is None:
        return "The k-point is transformed by Identity operation to k-star"

    resolution = record.get("resolution")
    star_op_index = int(getattr(resolution, "rotation_index", 1))
    if star_op_index == 0:
        return "The k-point is transformed by inversion-equivalent operation (-I) to k-star"

    operations = list(analysis_result.get("operations") or [])
    if 1 <= star_op_index <= len(operations):
        star_op = operations[star_op_index - 1]
        star_matrix = np.asarray(getattr(star_op, "inverse_rotation", np.eye(3)), dtype=float)
        if np.allclose(star_matrix, np.eye(3), atol=1.0e-8):
            return "The k-point is transformed by Identity operation to k-star"
        symbol = getattr(star_op, "symbol", "unknown")
        return f"The k-point is transformed by symmetry operation #{star_op_index} ({symbol}) to k-star"

    return f"The k-point is transformed by unknown star operation #{star_op_index} to k-star"


def _kp_model_kpoint_records(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for result in results:
        rows = list(result.get("rows") or [])
        if not rows:
            continue

        payload = result.get("character_payload")
        analysis_result = payload.get("analysis_result", {}) if isinstance(payload, Mapping) else {}
        character_records = list(analysis_result.get("kpoint_records") or [])
        character_record = character_records[0] if character_records else None
        resolution = character_record.get("resolution") if isinstance(character_record, Mapping) else None

        kpoint = _vector3(result.get("kpoint"), [0.0, 0.0, 0.0])
        primitive_basis = _vector3(
            getattr(resolution, "rotated_k_prim", getattr(resolution, "mapped_k_prim", None)),
            character_record.get("character_k_direct", kpoint) if isinstance(character_record, Mapping) else kpoint,
        )
        conventional_basis = _vector3(getattr(resolution, "k_conv", None), kpoint)
        band_indices = [int(row["band"]) for row in rows if "band" in row]
        irreps = [str(row.get("irrep", "")).strip() for row in rows if str(row.get("irrep", "")).strip()]
        target_band_range = [int(value) for value in result.get("band", [])]
        if len(target_band_range) != 2:
            target_band_range = [band_indices[0], band_indices[-1]] if band_indices else []

        records.append(
            {
                "k_index": int(rows[0].get("k_index", result.get("selection_index", 1))),
                "k_direct": kpoint,
                "k_name": str(rows[0].get("k_name") or result.get("label") or "").strip(),
                "star_transform": _kpoint_star_transform_text(character_record, analysis_result),
                "primitive_basis": primitive_basis,
                "conventional_basis": conventional_basis,
                "band_range": target_band_range,
                "band_rep": " + ".join(irreps),
            }
        )
    return records


def _spgrep_reference_kpoint_from_record(record: Mapping[str, Any]) -> list[float]:
    return _vector3(record.get("character_k_direct", record.get("k_direct")), [0.0, 0.0, 0.0])


def _symmetry_operation_keeps_kpoint(
    rotation: Sequence[Sequence[float]] | np.ndarray,
    kpoint: Sequence[float] | np.ndarray,
    *,
    antiunitary: bool,
    tol: float = 1.0e-6,
) -> bool:
    try:
        reciprocal_rotation = np.linalg.inv(np.asarray(rotation, dtype=float))
    except np.linalg.LinAlgError:
        return False
    kpoint_array = np.asarray(kpoint, dtype=float).reshape(3)
    mapped = reciprocal_rotation @ kpoint_array
    if antiunitary:
        mapped = -mapped
    diff = mapped - kpoint_array
    return bool(np.allclose(diff - np.rint(diff), 0.0, atol=tol, rtol=0.0))


def _lowdin_reference_kpoint_from_result(result: Mapping[str, Any]) -> list[float]:
    payload = result.get("character_payload")
    if isinstance(payload, Mapping):
        records = list(payload.get("analysis_result", {}).get("kpoint_records") or [])
        if records:
            return _spgrep_reference_kpoint_from_record(records[0])
    return _vector3(result.get("kpoint"), [0.0, 0.0, 0.0])


def _target_indices_from_band_range(band_range: Sequence[int]) -> np.ndarray:
    if len(band_range) != 2:
        raise ValueError("KP band range must contain two band numbers.")
    band_start = int(band_range[0])
    band_stop = int(band_range[1])
    if band_start <= 0 or band_stop < band_start:
        raise ValueError("KP band range must be positive and ordered.")
    return np.arange(band_start - 1, band_stop, dtype=int)


def _quadratic_monomial_pairs() -> list[tuple[int, int]]:
    return [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]


def _ordered_direction_tuples_for_exponent(exponent: Sequence[int]) -> list[tuple[int, int, int]]:
    counts = tuple(int(value) for value in exponent)
    tuples: list[tuple[int, int, int]] = []
    for first in range(3):
        for second in range(3):
            for third in range(3):
                if (
                    (first, second, third).count(0),
                    (first, second, third).count(1),
                    (first, second, third).count(2),
                ) == counts:
                    tuples.append((first, second, third))
    return tuples


def _levi_civita_tensor() -> np.ndarray:
    tensor = np.zeros((3, 3, 3), dtype=int)
    tensor[0, 1, 2] = tensor[1, 2, 0] = tensor[2, 0, 1] = 1
    tensor[1, 0, 2] = tensor[2, 1, 0] = tensor[0, 2, 1] = -1
    return tensor


def _spin_pauli_target_matrices_from_overlap(
    wavefunctions: np.ndarray,
    overlap: np.ndarray,
    *,
    target_indices: Sequence[int],
    nspin: int,
) -> np.ndarray | None:
    if int(nspin) != 4:
        return None

    coefficients = np.asarray(wavefunctions, dtype=complex)
    overlap_matrix = np.asarray(overlap, dtype=complex)
    target = np.asarray(target_indices, dtype=int).reshape(-1)
    basis_count = int(coefficients.shape[0])
    if basis_count <= 0 or basis_count % 2 != 0:
        return None
    if coefficients.ndim != 2 or overlap_matrix.shape != (basis_count, basis_count):
        return None
    if np.any(target < 0) or np.any(target >= coefficients.shape[1]):
        return None

    spinless_basis_count = basis_count // 2
    spatial_overlap = 0.5 * (
        overlap_matrix[0::2, 0::2] + overlap_matrix[1::2, 1::2]
    )
    pauli_matrices = np.asarray(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, -1.0j], [1.0j, 0.0]],
            [[1.0, 0.0], [0.0, -1.0]],
        ],
        dtype=complex,
    )
    target_coefficients = coefficients[:, target]
    projected = np.zeros((3, target.size, target.size), dtype=complex)
    for direction, pauli in enumerate(pauli_matrices):
        operator = np.kron(spatial_overlap.reshape(spinless_basis_count, spinless_basis_count), pauli)
        projected[direction] = target_coefficients.conj().T @ operator @ target_coefficients
    return projected


def _clean_complex_array(values: np.ndarray, tol: float = 1.0e-12) -> np.ndarray:
    out = np.asarray(values, dtype=complex).copy()
    out[np.abs(out.real) < tol] = 1j * out[np.abs(out.real) < tol].imag
    out[np.abs(out.imag) < tol] = out[np.abs(out.imag) < tol].real
    return out


def _build_numeric_lowdin_kp_analysis(
    eigenvalues: Sequence[float],
    velocity_matrices: Sequence[Sequence[Sequence[complex]]],
    *,
    target_indices: Sequence[int],
    reference_k_direct: Sequence[float],
    reference_k_cartesian: Sequence[float],
    band_range: Sequence[int],
    selection_index: int,
    basis_convention: str = "pyatb eigenvector basis at the reference k point",
    hamiltonian_data_convention: str = "current pyatb TB H/S data",
    hamiltonian_data_paths: Mapping[str, str] | None = None,
    free_electron_coefficient: float = KP_FREE_ELECTRON_COEFFICIENT_EV_A2,
    spin_target_matrices: Sequence[Sequence[Sequence[complex]]] | None = None,
    denominator_tol: float = 1.0e-10,
) -> dict[str, Any]:
    """Build the second-order numeric k.p model from pyatb velocity matrices.

    ``velocity_matrices`` is interpreted as the linear coefficient
    ``V_i = (hbar / m) pi_i`` in the pyatb eigenvector basis at ``k0``.
    The expansion variable is Cartesian ``q`` in Angstrom^{-1}.
    """

    energies = np.asarray(eigenvalues, dtype=float).reshape(-1)
    velocities = np.asarray(velocity_matrices, dtype=complex)
    if velocities.ndim != 3 or velocities.shape[0] != 3 or velocities.shape[1] != velocities.shape[2]:
        raise ValueError("velocity_matrices must have shape (3, band_count, band_count).")
    if velocities.shape[1] != energies.size:
        raise ValueError("eigenvalues and velocity_matrices use different band counts.")

    target = np.asarray(target_indices, dtype=int).reshape(-1)
    if target.size == 0:
        raise ValueError("target_indices must contain at least one band index.")
    if np.any(target < 0) or np.any(target >= energies.size):
        raise ValueError("target_indices contains an out-of-range band index.")
    if np.unique(target).size != target.size:
        raise ValueError("target_indices must not contain duplicates.")

    remote = np.asarray([index for index in range(energies.size) if index not in set(target)], dtype=int)
    dimension = int(target.size)
    target_energies = energies[target]
    linear = velocities[:, target[:, None], target[None, :]]

    lowdin_tensor = np.zeros((3, 3, dimension, dimension), dtype=complex)
    for remote_index in remote:
        delta = target_energies - energies[remote_index]
        if np.any(np.abs(delta) <= denominator_tol):
            remote_band = int(remote_index + 1)
            raise ValueError(
                "Cannot build second-order Lowdin k.p model: "
                f"remote band {remote_band} is degenerate with the target subspace."
            )
        denominator = (1.0 / delta[:, None]) + (1.0 / delta[None, :])
        for left_direction in range(3):
            left = velocities[left_direction, target, remote_index]
            for right_direction in range(3):
                right = np.conj(velocities[right_direction, target, remote_index])
                lowdin_tensor[left_direction, right_direction] += 0.5 * np.outer(left, right) * denominator

    free_tensor = np.zeros_like(lowdin_tensor)
    identity = np.eye(dimension, dtype=complex)
    for direction in range(3):
        free_tensor[direction, direction] = float(free_electron_coefficient) * identity
    total_tensor = lowdin_tensor + free_tensor

    zeeman_orbital = np.zeros((3, dimension, dimension), dtype=complex)
    if abs(float(free_electron_coefficient)) > denominator_tol:
        levi_civita = _levi_civita_tensor()
        zeeman_prefactor = -1.0j / (4.0 * float(free_electron_coefficient))
        for remote_index in remote:
            delta = target_energies - energies[remote_index]
            if np.any(np.abs(delta) <= denominator_tol):
                continue
            denominator = (1.0 / delta[:, None]) + (1.0 / delta[None, :])
            for field_direction in range(3):
                for left_direction in range(3):
                    for right_direction in range(3):
                        sign = int(levi_civita[left_direction, right_direction, field_direction])
                        if sign == 0:
                            continue
                        left = velocities[left_direction, target, remote_index]
                        right = velocities[right_direction, remote_index, target]
                        zeeman_orbital[field_direction] += (
                            zeeman_prefactor * float(sign) * np.outer(left, right) * denominator
                        )
    if spin_target_matrices is None:
        zeeman_spin = np.zeros_like(zeeman_orbital)
    else:
        zeeman_spin = np.asarray(spin_target_matrices, dtype=complex)
        if zeeman_spin.shape != zeeman_orbital.shape:
            raise ValueError("spin_target_matrices must have shape (3, target_dimension, target_dimension).")
    zeeman_total = zeeman_orbital + zeeman_spin

    remote_energies = energies[remote]
    target_remote_denominator = 1.0 / (target_energies[:, None] - remote_energies[None, :])
    target_block = np.ix_(target, target)
    remote_block = np.ix_(remote, remote)
    target_remote_block = np.ix_(target, remote)
    remote_target_block = np.ix_(remote, target)
    cubic_tensor = np.zeros((3, 3, 3, dimension, dimension), dtype=complex)
    if remote.size:
        cubic_raw = np.zeros_like(cubic_tensor)
        for left_direction in range(3):
            velocity_left_target_remote = velocities[left_direction][target_remote_block]
            velocity_left_target_target = velocities[left_direction][target_block]
            for middle_direction in range(3):
                velocity_middle_remote_remote = velocities[middle_direction][remote_block]
                velocity_middle_remote_target = velocities[middle_direction][remote_target_block]
                velocity_middle_target_remote = velocities[middle_direction][target_remote_block]
                for right_direction in range(3):
                    velocity_right_remote_target = velocities[right_direction][remote_target_block]
                    velocity_right_target_target = velocities[right_direction][target_block]

                    remote_remote = 0.5 * (
                        np.einsum(
                            "al,lr,rb,al,ar->ab",
                            velocity_left_target_remote,
                            velocity_middle_remote_remote,
                            velocity_right_remote_target,
                            target_remote_denominator,
                            target_remote_denominator,
                            optimize=True,
                        )
                        + np.einsum(
                            "al,lr,rb,bl,br->ab",
                            velocity_left_target_remote,
                            velocity_middle_remote_remote,
                            velocity_right_remote_target,
                            target_remote_denominator,
                            target_remote_denominator,
                            optimize=True,
                        )
                    )
                    target_remote_target = -0.5 * (
                        np.einsum(
                            "al,lm,mb,bl,ml->ab",
                            velocity_left_target_remote,
                            velocity_middle_remote_target,
                            velocity_right_target_target,
                            target_remote_denominator,
                            target_remote_denominator,
                            optimize=True,
                        )
                        + np.einsum(
                            "am,ml,lb,al,ml->ab",
                            velocity_left_target_target,
                            velocity_middle_target_remote,
                            velocity_right_remote_target,
                            target_remote_denominator,
                            target_remote_denominator,
                            optimize=True,
                        )
                    )
                    cubic_raw[left_direction, middle_direction, right_direction] = (
                        remote_remote + target_remote_target
                    )
        for left_direction in range(3):
            for middle_direction in range(3):
                for right_direction in range(3):
                    cubic_tensor[left_direction, middle_direction, right_direction] = (
                        cubic_raw[left_direction, middle_direction, right_direction]
                        + cubic_raw[left_direction, right_direction, middle_direction]
                        + cubic_raw[middle_direction, left_direction, right_direction]
                        + cubic_raw[right_direction, middle_direction, left_direction]
                        + cubic_raw[middle_direction, right_direction, left_direction]
                        + cubic_raw[right_direction, left_direction, middle_direction]
                    ) / 6.0

    quadratic_monomials = []
    lowdin_monomials = []
    for left_direction, right_direction in _quadratic_monomial_pairs():
        label = _monomial_label(
            tuple(
                2 if axis == left_direction == right_direction else
                1 if axis in (left_direction, right_direction) else
                0
                for axis in range(3)
            )
        )
        if left_direction == right_direction:
            total_matrix = total_tensor[left_direction, right_direction]
            lowdin_matrix = lowdin_tensor[left_direction, right_direction]
        else:
            total_matrix = (
                total_tensor[left_direction, right_direction]
                + total_tensor[right_direction, left_direction]
            )
            lowdin_matrix = (
                lowdin_tensor[left_direction, right_direction]
                + lowdin_tensor[right_direction, left_direction]
            )
        quadratic_monomials.append(
            {
                "label": label,
                "directions": [KP_DIRECTIONS[left_direction], KP_DIRECTIONS[right_direction]],
                "matrix": _complex_matrix_to_pairs(_clean_complex_array(total_matrix)),
            }
        )
        lowdin_monomials.append(
            {
                "label": label,
                "directions": [KP_DIRECTIONS[left_direction], KP_DIRECTIONS[right_direction]],
                "matrix": _complex_matrix_to_pairs(_clean_complex_array(lowdin_matrix)),
            }
        )

    cubic_monomials = []
    for exponent in _monomial_exponents(3, 3):
        label = _monomial_label(exponent)
        matrix = np.zeros((dimension, dimension), dtype=complex)
        for directions in _ordered_direction_tuples_for_exponent(exponent):
            matrix += cubic_tensor[directions]
        cubic_monomials.append(
            {
                "label": label,
                "directions": [KP_DIRECTIONS[index] for index in range(3) for _ in range(int(exponent[index]))],
                "matrix": _complex_matrix_to_pairs(_clean_complex_array(matrix)),
            }
        )

    return {
        "selection_index": int(selection_index),
        "reference_k_direct": [float(value) for value in reference_k_direct],
        "reference_k_cartesian": [float(value) for value in reference_k_cartesian],
        "expansion_variable": "q = k - k0 in Cartesian coordinates, Angstrom^-1",
        "basis_convention": str(basis_convention),
        "hamiltonian_data_convention": str(hamiltonian_data_convention),
        "hamiltonian_data_paths": dict(hamiltonian_data_paths or {}),
        "velocity_convention": "V_i = (hbar / m_e) pi_i = dH/dk_i-like linear coefficient from pyatb velocity_matrix",
        "formula": "H_ab(q)=E_a delta_ab + sum_i V^i_ab q_i + sum_Q C^Q_ab Q(q)",
        "zeeman_convention": "H_Z = (mu_B / 2) sum_i G_i B_i",
        "zeeman_formula": (
            "G^orb,k_ab=-i/(4C) sum_lij epsilon_ijk V^i_a l V^j_l b "
            "[1/(E_a-E_l)+1/(E_b-E_l)], C=hbar^2/(2m_e)"
        ),
        "second_order_formula": (
            "L^ij_ab=1/2 sum_l V^i_a l conj(V^j_b l) "
            "[1/(E_a-E_l)+1/(E_b-E_l)]"
        ),
        "third_order_formula": (
            "L^ijk_ab includes remote-remote and target-remote-target Lowdin chains, "
            "then is symmetrized over the three Cartesian k indices."
        ),
        "free_electron_coefficient_eV_A2": float(free_electron_coefficient),
        "band_range": [int(value) for value in band_range],
        "target_band_indices_one_based": [int(index + 1) for index in target],
        "target_array_indices_zero_based": [int(index) for index in target],
        "remote_band_count": int(remote.size),
        "constant_energies_eV": [float(value) for value in target_energies],
        "linear_matrices": [
            {
                "direction": KP_DIRECTIONS[direction],
                "matrix": _complex_matrix_to_pairs(_clean_complex_array(linear[direction])),
            }
            for direction in range(3)
        ],
        "zeeman_orbital_matrices": [
            {
                "field": ZEEMAN_FIELD_DIRECTIONS[direction],
                "matrix": _complex_matrix_to_pairs(_clean_complex_array(zeeman_orbital[direction])),
            }
            for direction in range(3)
        ],
        "zeeman_spin_matrices": [
            {
                "field": ZEEMAN_FIELD_DIRECTIONS[direction],
                "matrix": _complex_matrix_to_pairs(_clean_complex_array(zeeman_spin[direction])),
            }
            for direction in range(3)
        ],
        "zeeman_matrices": [
            {
                "field": ZEEMAN_FIELD_DIRECTIONS[direction],
                "matrix": _complex_matrix_to_pairs(_clean_complex_array(zeeman_total[direction])),
            }
            for direction in range(3)
        ],
        "lowdin_quadratic_tensor": [
            [
                _complex_matrix_to_pairs(_clean_complex_array(lowdin_tensor[left_direction, right_direction]))
                for right_direction in range(3)
            ]
            for left_direction in range(3)
        ],
        "lowdin_cubic_tensor": [
            [
                [
                    _complex_matrix_to_pairs(_clean_complex_array(cubic_tensor[left_direction, middle_direction, right_direction]))
                    for right_direction in range(3)
                ]
                for middle_direction in range(3)
            ]
            for left_direction in range(3)
        ],
        "lowdin_quadratic_monomial_matrices": lowdin_monomials,
        "quadratic_monomial_matrices": quadratic_monomials,
        "cubic_monomial_matrices": cubic_monomials,
    }


def _first_existing_route(route: Any) -> str | None:
    if route is None:
        return None
    if isinstance(route, (str, Path)):
        text = str(route)
        return text if text else None
    try:
        for item in route:
            text = str(item)
            if text:
                return text
    except TypeError:
        text = str(route)
        return text if text else None
    return None


def _lowdin_tb_from_character_payload(
    reference_tb,
    payload: Mapping[str, Any] | None,
    *,
    rR_route: Any = None,
    rR_unit: str = "Angstrom",
    HR_unit: str | None = None,
):
    if not isinstance(payload, Mapping):
        return reference_tb

    payload_rR_path = payload.get("active_rR_path")
    rR_path = _first_existing_route(payload_rR_path) or _first_existing_route(rR_route)
    if rR_path is None:
        return reference_tb
    active_rR_unit = str(payload.get("active_rR_unit", rR_unit))

    active_hr_path = payload.get("active_hr_path")
    active_sr_path = payload.get("active_sr_path")
    active_stru_path = payload.get("active_stru_path")
    if not active_hr_path or not active_sr_path or not active_stru_path:
        return reference_tb

    if int(getattr(reference_tb, "nspin", 0)) == 2:
        return reference_tb

    lattice_constant = float(payload.get("lattice_constant", getattr(reference_tb, "lattice_constant", 1.0)))
    lattice_vector = np.asarray(payload.get("lattice_vector", getattr(reference_tb, "lattice_vector", np.eye(3))), dtype=float)
    max_kpoint_num = getattr(reference_tb, "max_kpoint_num", None)
    active_tb = TBModel(
        int(reference_tb.nspin),
        lattice_constant,
        lattice_vector,
        max_kpoint_num,
    )
    active_hr = abacus_readHR(int(reference_tb.nspin), str(active_hr_path), str(payload.get("HR_unit", HR_unit or "Ry")))
    active_sr = abacus_readSR(int(reference_tb.nspin), str(active_sr_path))
    is_sparse = bool(getattr(reference_tb, "HSR_iSsparse", False))
    active_tb.set_solver_HSR(active_hr, active_sr, is_sparse)
    active_rR = abacus_readrR(str(rR_path), active_rR_unit)
    active_tb.set_solver_rR(active_rR[0], active_rR[1], active_rR[2], is_sparse)
    active_tb.read_stru(str(active_stru_path), need_orb=True)
    active_tb.kp_lowdin_data_convention = "CHARACTER active symmetrized H/S/rR data"
    active_tb.kp_lowdin_data_paths = {
        "stru": str(active_stru_path),
        "HR": str(active_hr_path),
        "SR": str(active_sr_path),
        "rR": str(rR_path),
    }
    return active_tb


def _active_hs_tb_from_character_payload(
    reference_tb,
    payload: Mapping[str, Any] | None,
    *,
    HR_unit: str | None = None,
):
    if not isinstance(payload, Mapping):
        return reference_tb

    active_hr_path = payload.get("active_hr_path")
    active_sr_path = payload.get("active_sr_path")
    active_stru_path = payload.get("active_stru_path")
    if not active_hr_path or not active_sr_path or not active_stru_path:
        return reference_tb
    if not Path(active_hr_path).exists() or not Path(active_sr_path).exists() or not Path(active_stru_path).exists():
        return reference_tb

    if int(getattr(reference_tb, "nspin", 0)) == 2:
        return reference_tb

    lattice_constant = float(payload.get("lattice_constant", getattr(reference_tb, "lattice_constant", 1.0)))
    lattice_vector = np.asarray(payload.get("lattice_vector", getattr(reference_tb, "lattice_vector", np.eye(3))), dtype=float)
    max_kpoint_num = getattr(reference_tb, "max_kpoint_num", None)
    active_tb = TBModel(
        int(reference_tb.nspin),
        lattice_constant,
        lattice_vector,
        max_kpoint_num,
    )
    active_hr = abacus_readHR(int(reference_tb.nspin), str(active_hr_path), str(payload.get("HR_unit", HR_unit or "Ry")))
    active_sr = abacus_readSR(int(reference_tb.nspin), str(active_sr_path))
    is_sparse = bool(getattr(reference_tb, "HSR_iSsparse", False))
    active_tb.set_solver_HSR(active_hr, active_sr, is_sparse)
    active_tb.read_stru(str(active_stru_path), need_orb=True)
    active_tb.kp_alignment_data_convention = "CHARACTER active symmetrized H/S data"
    active_tb.kp_alignment_data_paths = {
        "stru": str(active_stru_path),
        "HR": str(active_hr_path),
        "SR": str(active_sr_path),
    }
    return active_tb


def _numeric_lowdin_kp_analyses_for_results(
    tb,
    results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    tb_solver = getattr(tb, "tb_solver", None)
    if tb_solver is None or not hasattr(tb_solver, "diago_H"):
        return []
    if not bool(getattr(tb, "has_rR", False)):
        return []
    if not hasattr(tb_solver, "get_velocity_basis_k"):
        return []

    analyses: list[dict[str, Any]] = []
    for result in results:
        band_range = list(result.get("band") or [])
        if len(band_range) != 2:
            continue
        k_direct = np.asarray([_lowdin_reference_kpoint_from_result(result)], dtype=float)
        character_row = _character_target_representation_row(result)
        if (
            isinstance(character_row, Mapping)
            and character_row.get("target_eigenvectors_full") is not None
            and character_row.get("target_eigenvalues_full") is not None
        ):
            wavefunctions = np.asarray(character_row["target_eigenvectors_full"], dtype=complex)
            eigenvalues_for_analysis = np.asarray(character_row["target_eigenvalues_full"], dtype=float)
            basis_convention = "CHARACTER eigenvector basis at the reference k point"
        else:
            eigenvectors, eigenvalues = tb_solver.diago_H(k_direct)
            wavefunctions = np.asarray(eigenvectors[0], dtype=complex)
            eigenvalues_for_analysis = np.asarray(eigenvalues[0], dtype=float)
            basis_convention = "pyatb eigenvector basis at the reference k point"
        velocity_basis = tb_solver.get_velocity_basis_k(k_direct)
        velocity_matrices = np.asarray(
            [
                wavefunctions.conj().T @ np.asarray(velocity_basis[0, direction], dtype=complex) @ wavefunctions
                for direction in range(3)
            ],
            dtype=complex,
        )
        target_indices = _target_indices_from_band_range(band_range)
        spin_target_matrices = None
        if isinstance(character_row, Mapping) and character_row.get("target_overlap") is not None:
            spin_target_matrices = _spin_pauli_target_matrices_from_overlap(
                wavefunctions,
                np.asarray(character_row.get("target_overlap"), dtype=complex),
                target_indices=target_indices,
                nspin=int(getattr(tb, "nspin", 1)),
            )
        if hasattr(tb, "direct_to_cartesian_kspace"):
            k_cartesian = np.asarray(tb.direct_to_cartesian_kspace(k_direct), dtype=float)[0]
        else:
            k_cartesian = k_direct[0]
        analyses.append(
            _build_numeric_lowdin_kp_analysis(
                eigenvalues_for_analysis,
                velocity_matrices,
                target_indices=target_indices,
                reference_k_direct=k_direct[0],
                reference_k_cartesian=k_cartesian,
                band_range=band_range,
                selection_index=int(result.get("selection_index", len(analyses) + 1)),
                basis_convention=basis_convention,
                hamiltonian_data_convention=getattr(
                    tb,
                    "kp_lowdin_data_convention",
                    "current pyatb TB H/S data",
                ),
                hamiltonian_data_paths=getattr(tb, "kp_lowdin_data_paths", {}),
                spin_target_matrices=spin_target_matrices,
            )
        )
    return analyses


def _unitarize_by_polar(matrix: np.ndarray) -> np.ndarray:
    u_matrix, _singular_values, vh_matrix = np.linalg.svd(np.asarray(matrix, dtype=complex), full_matrices=False)
    return u_matrix @ vh_matrix


def _schur_project_intertwiner(
    numeric_matrices: Sequence[np.ndarray],
    standard_matrices: Sequence[np.ndarray],
    seed: np.ndarray,
    *,
    numeric_antiunitary_matrices: Sequence[np.ndarray] | None = None,
    standard_antiunitary_matrices: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    projected = np.zeros_like(np.asarray(seed, dtype=complex))
    for numeric, standard in zip(numeric_matrices, standard_matrices, strict=True):
        projected += np.asarray(numeric, dtype=complex) @ seed @ np.asarray(standard, dtype=complex).conj().T
    count = len(numeric_matrices)
    numeric_antiunitary = list(numeric_antiunitary_matrices or [])
    standard_antiunitary = list(standard_antiunitary_matrices or [])
    for numeric, standard in zip(numeric_antiunitary, standard_antiunitary, strict=True):
        projected += (
            np.asarray(numeric, dtype=complex)
            @ np.asarray(seed, dtype=complex).conj()
            @ np.asarray(standard, dtype=complex).conj().T
        )
        count += 1
    projected /= float(count)
    return projected


def _schur_intertwiner_between_representations(
    left_matrices: Sequence[np.ndarray],
    right_matrices: Sequence[np.ndarray],
    *,
    tol: float = 1.0e-9,
) -> np.ndarray:
    left = [np.asarray(matrix, dtype=complex) for matrix in left_matrices]
    right = [np.asarray(matrix, dtype=complex) for matrix in right_matrices]
    if not left or len(left) != len(right):
        raise ValueError("left_matrices and right_matrices must have the same nonzero length.")
    dimension = int(left[0].shape[0])
    if any(matrix.shape != (dimension, dimension) for matrix in left + right):
        raise ValueError("All representation matrices must have the same square shape.")

    seeds = [np.eye(dimension, dtype=complex)]
    for row in range(dimension):
        for col in range(dimension):
            seed = np.zeros((dimension, dimension), dtype=complex)
            seed[row, col] = 1.0
            seeds.extend([seed, 1.0j * seed])

    best_projected = None
    best_score = None
    for seed in seeds:
        projected = _schur_project_intertwiner(left, right, seed)
        singular_values = np.linalg.svd(projected, compute_uv=False)
        rank = int(np.count_nonzero(singular_values > tol))
        score = (rank, float(singular_values[-1]), float(np.linalg.norm(projected)))
        if best_score is None or score > best_score:
            best_score = score
            best_projected = projected

    if best_projected is None or int(best_score[0]) < dimension:
        raise ValueError("Failed to build a full-rank Schur intertwiner.")
    return _unitarize_by_polar(best_projected)


def _spinless_time_reversal_sewing_matrix(
    unitary_matrices: Sequence[np.ndarray],
    *,
    tol: float = 1.0e-9,
) -> np.ndarray:
    """Return the spinless time-reversal sewing matrix in the chosen irrep basis.

    For NSOC calculations time reversal is complex conjugation in a real-space
    orbital basis, but after spgrep chooses a complex irrep basis the anti-linear
    matrix is generally not identity.  It must satisfy D(g) A = A D(g)^*.
    """

    matrices = [np.asarray(matrix, dtype=complex) for matrix in unitary_matrices]
    if not matrices:
        return np.eye(0, dtype=complex)
    return _schur_intertwiner_between_representations(
        matrices,
        [matrix.conj() for matrix in matrices],
        tol=tol,
    )


def _canonical_antiunitary_sewing_matrix(
    unitary_matrices: Sequence[np.ndarray],
    *,
    spin_orbit: bool,
    tol: float = 1.0e-8,
) -> np.ndarray:
    matrices = [np.asarray(matrix, dtype=complex) for matrix in unitary_matrices]
    if not matrices:
        return np.eye(0, dtype=complex)
    dimension = int(matrices[0].shape[0])
    if dimension == 1:
        return np.eye(1, dtype=complex)
    if all(np.allclose(matrix, np.diag(np.diag(matrix)), atol=tol, rtol=0.0) for matrix in matrices):
        diagonals = [np.diag(matrix) for matrix in matrices]
        out = np.zeros((dimension, dimension), dtype=complex)
        used: set[int] = set()
        for left in range(dimension):
            if left in used:
                continue
            partner = None
            for right in range(left, dimension):
                if right in used:
                    continue
                if all(
                    abs(complex(diagonal[right]) - complex(diagonal[left]).conjugate()) <= tol
                    for diagonal in diagonals
                ):
                    partner = right
                    break
            if partner is None:
                partner = left
            if partner == left:
                out[left, left] = 1.0
                used.add(left)
            else:
                if bool(spin_orbit):
                    out[left, partner] = -1.0
                    out[partner, left] = 1.0
                else:
                    out[left, partner] = 1.0
                    out[partner, left] = 1.0
                used.update({left, partner})
        if len(used) == dimension:
            return out
    try:
        return _spinless_time_reversal_sewing_matrix(matrices, tol=tol)
    except ValueError:
        return np.eye(dimension, dtype=complex)


def _antiunitary_block_phase_refinement(
    transformed_antiunitary_matrices: Sequence[np.ndarray],
    standard_antiunitary_matrices: Sequence[np.ndarray],
    *,
    block_dimensions: Sequence[int],
    tol: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    dimensions = [int(value) for value in block_dimensions if int(value) > 0]
    total_dimension = int(sum(dimensions))
    if total_dimension <= 0:
        return np.eye(0, dtype=complex), {"status": "skipped", "reason": "empty block dimensions"}

    block_by_basis_index: list[int] = []
    for block_index, dimension in enumerate(dimensions):
        block_by_basis_index.extend([block_index] * int(dimension))
    block_by_basis_index_array = np.asarray(block_by_basis_index, dtype=int)

    equations: list[np.ndarray] = []
    right_hand_side: list[float] = []
    weights: list[float] = []
    for transformed, standard in zip(
        transformed_antiunitary_matrices,
        standard_antiunitary_matrices,
        strict=True,
    ):
        current = np.asarray(transformed, dtype=complex)
        target = np.asarray(standard, dtype=complex)
        if current.shape != (total_dimension, total_dimension) or target.shape != current.shape:
            continue
        for row in range(total_dimension):
            for col in range(total_dimension):
                if abs(current[row, col]) <= tol or abs(target[row, col]) <= tol:
                    continue
                equation = np.zeros(len(dimensions), dtype=float)
                equation[block_by_basis_index_array[row]] += 1.0
                equation[block_by_basis_index_array[col]] += 1.0
                equations.append(equation)
                right_hand_side.append(float(-np.angle(target[row, col] / current[row, col])))
                weights.append(float(abs(current[row, col])))

    if not equations:
        return (
            np.eye(total_dimension, dtype=complex),
            {"status": "skipped", "reason": "no overlapping nonzero antiunitary matrix elements"},
        )

    matrix = np.vstack(equations)
    rhs = np.asarray(right_hand_side, dtype=float)
    weight = np.sqrt(np.asarray(weights, dtype=float))
    weighted_matrix = weight[:, None] * matrix
    weighted_rhs = weight * rhs
    phases = np.linalg.lstsq(weighted_matrix, weighted_rhs, rcond=None)[0]
    diagonal = np.asarray(
        [np.exp(1.0j * phases[block_index]) for block_index in block_by_basis_index],
        dtype=complex,
    )
    phase_matrix = np.diag(diagonal)

    before_max = 0.0
    after_max = 0.0
    for transformed, standard in zip(
        transformed_antiunitary_matrices,
        standard_antiunitary_matrices,
        strict=True,
    ):
        current = np.asarray(transformed, dtype=complex)
        target = np.asarray(standard, dtype=complex)
        if current.shape != (total_dimension, total_dimension) or target.shape != current.shape:
            continue
        before_max = max(before_max, float(np.max(np.abs(current - target))))
        refined = phase_matrix.conj().T @ current @ phase_matrix.conj()
        after_max = max(after_max, float(np.max(np.abs(refined - target))))

    return (
        phase_matrix,
        {
            "status": "applied",
            "block_dimensions": dimensions,
            "block_phases_rad": [float(value) for value in phases],
            "equation_count": int(len(equations)),
            "max_abs_antiunitary_difference_before": before_max,
            "max_abs_antiunitary_difference_after": after_max,
            "phase_matrix": _complex_matrix_to_pairs(phase_matrix),
        },
    )


def _schur_representation_alignment(
    *,
    operation_indices: Sequence[int],
    numeric_matrices: Sequence[np.ndarray],
    standard_matrices: Sequence[np.ndarray],
    antiunitary_operation_indices: Sequence[int] | None = None,
    numeric_antiunitary_matrices: Sequence[np.ndarray] | None = None,
    standard_antiunitary_matrices: Sequence[np.ndarray] | None = None,
    standard_block_dimensions: Sequence[int] | None = None,
    project_antiunitary: bool = True,
    tol: float = 1.0e-9,
) -> dict[str, Any]:
    """Align two equivalent corepresentations using Schur projection.

    The returned ``U`` follows ``D_standard(g) ~= U^dagger D_numeric(g) U`` for
    unitary operations and ``T_standard(a) ~= U^dagger T_numeric(a) U^*`` for
    antiunitary operations.
    """

    numeric = [np.asarray(matrix, dtype=complex) for matrix in numeric_matrices]
    standard = [np.asarray(matrix, dtype=complex) for matrix in standard_matrices]
    numeric_antiunitary = [np.asarray(matrix, dtype=complex) for matrix in (numeric_antiunitary_matrices or [])]
    standard_antiunitary = [np.asarray(matrix, dtype=complex) for matrix in (standard_antiunitary_matrices or [])]
    if not numeric or len(numeric) != len(standard):
        raise ValueError("numeric_matrices and standard_matrices must have the same nonzero length.")
    if len(numeric_antiunitary) != len(standard_antiunitary):
        raise ValueError("numeric_antiunitary_matrices and standard_antiunitary_matrices must have the same length.")
    dimension = int(numeric[0].shape[0])
    if any(matrix.shape != (dimension, dimension) for matrix in numeric):
        raise ValueError("All numeric representation matrices must have the same square shape.")
    if any(matrix.shape != (dimension, dimension) for matrix in standard):
        raise ValueError("All standard representation matrices must have the same square shape.")
    if any(matrix.shape != (dimension, dimension) for matrix in numeric_antiunitary):
        raise ValueError("All numeric antiunitary matrices must have the same square shape.")
    if any(matrix.shape != (dimension, dimension) for matrix in standard_antiunitary):
        raise ValueError("All standard antiunitary matrices must have the same square shape.")
    projector_numeric_antiunitary = numeric_antiunitary if bool(project_antiunitary) else []
    projector_standard_antiunitary = standard_antiunitary if bool(project_antiunitary) else []

    seeds = [np.eye(dimension, dtype=complex)]
    for row in range(dimension):
        for col in range(dimension):
            seed = np.zeros((dimension, dimension), dtype=complex)
            seed[row, col] = 1.0
            seeds.append(seed)
            if projector_numeric_antiunitary:
                seeds.append(1.0j * seed)

    best_projected = None
    best_score = None
    best_singular_values: np.ndarray | None = None
    for seed in seeds:
        projected = _schur_project_intertwiner(
            numeric,
            standard,
            seed,
            numeric_antiunitary_matrices=projector_numeric_antiunitary,
            standard_antiunitary_matrices=projector_standard_antiunitary,
        )
        singular_values = np.linalg.svd(projected, compute_uv=False)
        rank = int(np.count_nonzero(singular_values > tol))
        score = (rank, float(singular_values[-1]), float(np.linalg.norm(projected)))
        if best_score is None or score > best_score:
            best_score = score
            best_projected = projected
            best_singular_values = singular_values

    if best_projected is None:
        raise ValueError("Failed to build a Schur-projected intertwiner.")

    if int(best_score[0]) < dimension:
        rng = np.random.default_rng(12345)
        for _ in range(64):
            seed = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension))
            projected = _schur_project_intertwiner(
                numeric,
                standard,
                seed,
                numeric_antiunitary_matrices=projector_numeric_antiunitary,
                standard_antiunitary_matrices=projector_standard_antiunitary,
            )
            singular_values = np.linalg.svd(projected, compute_uv=False)
            rank = int(np.count_nonzero(singular_values > tol))
            score = (rank, float(singular_values[-1]), float(np.linalg.norm(projected)))
            if score > best_score:
                best_score = score
                best_projected = projected
                best_singular_values = singular_values
            if rank == dimension:
                break

    unitary = _unitarize_by_polar(best_projected)
    phase_refinement: dict[str, Any] | None = None
    if numeric_antiunitary and standard_antiunitary and standard_block_dimensions:
        antiunitary_after_unitary = [
            unitary.conj().T @ matrix @ unitary.conj()
            for matrix in numeric_antiunitary
        ]
        phase_matrix, phase_refinement = _antiunitary_block_phase_refinement(
            antiunitary_after_unitary,
            standard_antiunitary,
            block_dimensions=standard_block_dimensions,
            tol=tol,
        )
        unitary = unitary @ phase_matrix
    transformed = [unitary.conj().T @ matrix @ unitary for matrix in numeric]
    differences = [left - right for left, right in zip(transformed, standard, strict=True)]
    transformed_antiunitary = [
        unitary.conj().T @ matrix @ unitary.conj()
        for matrix in numeric_antiunitary
    ]
    antiunitary_differences = [
        left - right
        for left, right in zip(transformed_antiunitary, standard_antiunitary, strict=True)
    ]
    all_differences = differences + antiunitary_differences
    max_abs = max(float(np.max(np.abs(diff))) for diff in all_differences)
    frobenius = float(np.sqrt(sum(float(np.linalg.norm(diff)) ** 2 for diff in all_differences)))
    unitary_max_abs = max(float(np.max(np.abs(diff))) for diff in differences)
    antiunitary_max_abs = (
        max(float(np.max(np.abs(diff))) for diff in antiunitary_differences)
        if antiunitary_differences
        else 0.0
    )

    return {
        "method": "Schur corepresentation projection/intertwiner followed by polar unitarization",
        "convention": "D_standard(g) ~= U^dagger D_numeric(g) U; T_standard(a) ~= U^dagger T_numeric(a) U^*",
        "operation_indices": [int(index) for index in operation_indices],
        "antiunitary_operation_indices": [int(index) for index in (antiunitary_operation_indices or [])],
        "antiunitary_projector_applied": bool(project_antiunitary),
        "dimension": dimension,
        "projected_intertwiner_singular_values": [float(value) for value in np.asarray(best_singular_values).reshape(-1)],
        "unitary_numeric_to_standard": _complex_matrix_to_pairs(unitary),
        "unitarity_error": float(np.linalg.norm(unitary.conj().T @ unitary - np.eye(dimension))),
        "max_abs_representation_difference": max_abs,
        "unitary_max_abs_representation_difference": unitary_max_abs,
        "antiunitary_max_abs_representation_difference": antiunitary_max_abs,
        "frobenius_representation_difference": frobenius,
        "antiunitary_phase_refinement": phase_refinement,
        "transformed_numeric_matrices": [_complex_matrix_to_pairs(matrix) for matrix in transformed],
        "difference_matrices": [_complex_matrix_to_pairs(diff) for diff in differences],
        "transformed_numeric_antiunitary_matrices": [
            _complex_matrix_to_pairs(matrix) for matrix in transformed_antiunitary
        ],
        "antiunitary_difference_matrices": [
            _complex_matrix_to_pairs(diff) for diff in antiunitary_differences
        ],
    }


def _target_labels_from_result(result: Mapping[str, Any]) -> list[str]:
    labels: list[str] = []
    for row in result.get("rows") or []:
        for label in _split_irrep_terms(str(row.get("irrep", ""))):
            labels.append(label)
    return labels


def _character_target_representation_row(result: Mapping[str, Any]) -> Mapping[str, Any] | None:
    payload = result.get("character_payload")
    if not isinstance(payload, Mapping):
        return None
    target_band = [int(value) for value in result.get("band", [])]
    for row in payload.get("character_rows", []) or []:
        if [int(value) for value in row.get("target_band_range", [])] == target_band:
            matrices = row.get("target_representation_matrices")
            if matrices:
                return row
    return None


def _representation_alignment_analyses_for_results(
    info: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    *,
    tb: Any = None,
    symm_prec: float = 1.0e-6,
) -> list[dict[str, Any]]:
    spgrep_info = info.get("spgrep_operations", {})
    character_table = info.get("character_table", {})
    if not spgrep_info:
        return []

    analyses = []
    for result in results:
        row = _character_target_representation_row(result)
        if row is None:
            continue
        target_labels = _target_labels_from_result(result)
        corep_groups = _spgrep_corep_groups_for_irrep_terms(
            target_labels,
            spgrep_info=spgrep_info,
            character_table=character_table,
        )
        if not corep_groups:
            continue
        corep_indices = [int(group["corep_index"]) for group in corep_groups]
        target_corep_labels = [str(group["label"]) for group in corep_groups]

        standard_operation_indices, standard_matrices = _target_matrices_from_corep_indices(
            spgrep_info,
            corep_indices,
        )
        if not standard_matrices:
            continue
        target_corep_dimensions = _target_corep_dimensions(spgrep_info, corep_indices)

        numeric_operation_indices = [int(index) for index in row.get("target_operation_indices", [])]
        numeric_matrices_all = [np.asarray(matrix, dtype=complex) for matrix in row.get("target_representation_matrices", [])]
        operation_aliases = _spgrep_unitary_operation_aliases(spgrep_info)
        standard_by_operation: dict[int, np.ndarray] = {}
        for index, matrix in zip(standard_operation_indices, standard_matrices, strict=True):
            for alias in operation_aliases.get(int(index), [int(index)]):
                standard_by_operation[int(alias)] = np.asarray(matrix, dtype=complex)

        operation_indices = []
        numeric_matrices = []
        matched_standard_matrices = []
        for index, matrix in zip(numeric_operation_indices, numeric_matrices_all, strict=True):
            standard = standard_by_operation.get(int(index))
            if standard is None:
                continue
            operation_indices.append(int(index))
            numeric_matrices.append(matrix)
            matched_standard_matrices.append(standard)
        if not operation_indices:
            continue

        antiunitary_operation_indices: list[int] = []
        antiunitary_display_operation_indices: list[int] = []
        numeric_antiunitary_matrices: list[np.ndarray] = []
        matched_standard_antiunitary_matrices: list[np.ndarray] = []
        numeric_antiunitary_by_spgrep = _antiunitary_numeric_matrices_from_row(
            info,
            result,
            row,
            tb=tb,
            symm_prec=symm_prec,
        )
        for antiunitary_operation in spgrep_info.get("antiunitary_operations", []) or []:
            antiunitary_operation_index = int(antiunitary_operation.get("antiunitary_operation_index", 0))
            spgrep_operation_index = int(antiunitary_operation.get("spgrep_operation_index", 0))
            numeric_antiunitary = numeric_antiunitary_by_spgrep.get(spgrep_operation_index)
            if numeric_antiunitary is None:
                continue
            standard_antiunitary = _target_antiunitary_matrix_for_operation(
                spgrep_info,
                corep_indices,
                antiunitary_operation,
                standard_by_operation,
            )
            if standard_antiunitary is None:
                continue
            antiunitary_operation_indices.append(antiunitary_operation_index)
            antiunitary_display_operation_indices.append(
                int(antiunitary_operation.get("display_operation_index", antiunitary_operation_index))
            )
            numeric_antiunitary_matrices.append(np.asarray(numeric_antiunitary, dtype=complex))
            matched_standard_antiunitary_matrices.append(np.asarray(standard_antiunitary, dtype=complex))

        if not antiunitary_operation_indices:
            time_reversal_operation = _pure_time_reversal_operation(spgrep_info)
            target_time_reversal = row.get("target_time_reversal_matrix")
            if time_reversal_operation is not None and target_time_reversal is not None:
                antiunitary_operation_index = int(time_reversal_operation.get("antiunitary_operation_index", 0))
                standard_time_reversal = _target_antiunitary_matrix_from_corep_indices(
                    spgrep_info,
                    corep_indices,
                    antiunitary_operation_index,
                )
                if standard_time_reversal is not None:
                    antiunitary_operation_indices.append(antiunitary_operation_index)
                    antiunitary_display_operation_indices.append(
                        int(time_reversal_operation.get("display_operation_index", antiunitary_operation_index))
                    )
                    numeric_antiunitary_matrices.append(np.asarray(target_time_reversal, dtype=complex))
                    matched_standard_antiunitary_matrices.append(np.asarray(standard_time_reversal, dtype=complex))

        if antiunitary_operation_indices:
            constraint_operation = _anti_linear_constraint_operation(spgrep_info)
            constraint_index = (
                int(constraint_operation.get("antiunitary_operation_index", 0))
                if constraint_operation is not None
                else 0
            )
            constraint_display_index = _anti_linear_display_operation_index(constraint_operation)
            selected_position = None
            for position, (operation_index, display_index) in enumerate(
                zip(antiunitary_operation_indices, antiunitary_display_operation_indices, strict=False)
            ):
                if constraint_index and int(operation_index) == constraint_index:
                    selected_position = position
                    break
                if constraint_display_index and int(display_index) == constraint_display_index:
                    selected_position = position
                    break
            if selected_position is None:
                selected_position = 0
            antiunitary_operation_indices = [antiunitary_operation_indices[selected_position]]
            antiunitary_display_operation_indices = [antiunitary_display_operation_indices[selected_position]]
            numeric_antiunitary_matrices = [numeric_antiunitary_matrices[selected_position]]
            matched_standard_antiunitary_matrices = [matched_standard_antiunitary_matrices[selected_position]]

        alignment = _schur_representation_alignment(
            operation_indices=operation_indices,
            numeric_matrices=numeric_matrices,
            standard_matrices=matched_standard_matrices,
            antiunitary_operation_indices=antiunitary_operation_indices,
            numeric_antiunitary_matrices=numeric_antiunitary_matrices,
            standard_antiunitary_matrices=matched_standard_antiunitary_matrices,
            standard_block_dimensions=target_corep_dimensions,
            project_antiunitary=False,
        )

        analyses.append(
            {
                "selection_index": int(result.get("selection_index", len(analyses) + 1)),
                "target_label": " + ".join(target_labels),
                "target_corep_labels": target_corep_labels,
                "target_corep_indices": [int(index) for index in corep_indices],
                "target_corep_dimensions": [int(value) for value in target_corep_dimensions],
                "band_range": [int(value) for value in result.get("band", [])],
                "matrix_convention": "D_num(g)=C^dagger S(k) D_g C; D_spgrep is block-diag target corep matrix",
                "alignment_convention": alignment["convention"],
                "operation_indices": operation_indices,
                "antiunitary_operation_indices": antiunitary_operation_indices,
                "antiunitary_display_operation_indices": antiunitary_display_operation_indices,
                "numeric_representation_matrices": [_complex_matrix_to_pairs(matrix) for matrix in numeric_matrices],
                "spgrep_representation_matrices": [_complex_matrix_to_pairs(matrix) for matrix in matched_standard_matrices],
                "numeric_antiunitary_representation_matrices": [
                    _complex_matrix_to_pairs(matrix) for matrix in numeric_antiunitary_matrices
                ],
                "spgrep_antiunitary_representation_matrices": [
                    _complex_matrix_to_pairs(matrix) for matrix in matched_standard_antiunitary_matrices
                ],
                "schur_alignment": alignment,
            }
        )
    return analyses


def _normalize_k_direction(
    k_direction: str | Sequence[str] | None = "xyz",
) -> tuple[str, tuple[int, ...], tuple[str, ...]]:
    if k_direction is None:
        text = "xyz"
    elif isinstance(k_direction, str):
        text = k_direction
    else:
        text = "".join(str(item) for item in k_direction)
    text = text.strip().lower().replace(",", "").replace(" ", "")
    if not text:
        raise ValueError("KP.k_direction must contain one or more of x, y, z.")
    allowed = {"x", "y", "z"}
    if any(char not in allowed for char in text):
        raise ValueError("KP.k_direction must contain only x, y, z.")
    if len(set(text)) != len(text):
        raise ValueError("KP.k_direction must not repeat x, y, or z.")
    canonical = "".join(char for char in "xyz" if char in set(text))
    indices = tuple("xyz".index(char) for char in canonical)
    labels = tuple(KP_DIRECTIONS[index] for index in indices)
    return canonical, indices, labels


def _subtransform_for_k_labels(transform: np.ndarray, variable_labels: Sequence[str]) -> np.ndarray:
    matrix = np.asarray(transform, dtype=complex)
    labels = tuple(str(label) for label in variable_labels)
    if matrix.shape == (3, 3) and labels and all(label in KP_DIRECTIONS for label in labels):
        indices = tuple(KP_DIRECTIONS.index(label) for label in labels)
        return matrix[np.ix_(indices, indices)]
    return matrix


def _kp_fit_monomial_labels(
    max_order: int = 3,
    *,
    variable_labels: Sequence[str] = KP_DIRECTIONS,
) -> list[str]:
    max_order = max(0, int(max_order))
    labels = tuple(str(label) for label in variable_labels)
    return ["1"] + [
        _monomial_label(exponent, labels)
        for order in range(1, max_order + 1)
        for exponent in _monomial_exponents(order, len(labels))
    ]


def _numeric_lowdin_tensor_from_analysis(
    analysis: Mapping[str, Any],
    *,
    monomial_labels: Sequence[str],
) -> np.ndarray:
    dimension = len(analysis.get("constant_energies_eV", []))
    tensor = np.zeros((len(monomial_labels), dimension, dimension), dtype=complex)
    monomial_index = {str(label): index for index, label in enumerate(monomial_labels)}
    if "1" in monomial_index:
        tensor[monomial_index["1"]] = np.diag(np.asarray(analysis.get("constant_energies_eV", []), dtype=float))
    for item in analysis.get("linear_matrices", []):
        label = str(item.get("direction", ""))
        if label in monomial_index:
            tensor[monomial_index[label]] = np.asarray(_complex_array_from_pairs(item.get("matrix", [])), dtype=complex)
    for item in analysis.get("quadratic_monomial_matrices", []):
        label = str(item.get("label", ""))
        if label in monomial_index:
            tensor[monomial_index[label]] = np.asarray(_complex_array_from_pairs(item.get("matrix", [])), dtype=complex)
    for item in analysis.get("cubic_monomial_matrices", []):
        label = str(item.get("label", ""))
        if label in monomial_index:
            tensor[monomial_index[label]] = np.asarray(_complex_array_from_pairs(item.get("matrix", [])), dtype=complex)
    return tensor


def _numeric_zeeman_tensor_from_analysis(
    analysis: Mapping[str, Any],
    *,
    field_labels: Sequence[str] = ZEEMAN_FIELD_DIRECTIONS,
) -> np.ndarray:
    dimension = len(analysis.get("constant_energies_eV", []))
    tensor = np.zeros((len(field_labels), dimension, dimension), dtype=complex)
    field_index = {str(label): index for index, label in enumerate(field_labels)}
    for item in analysis.get("zeeman_matrices", []):
        label = str(item.get("field", ""))
        if label in field_index:
            tensor[field_index[label]] = np.asarray(_complex_array_from_pairs(item.get("matrix", [])), dtype=complex)
    return tensor


def _formal_basis_tensors(
    info: Mapping[str, Any],
    *,
    dimension: int,
    monomial_labels: Sequence[str],
    max_order: int = 2,
    analysis_key: str = "final_kp_model_analyses",
) -> tuple[list[str], np.ndarray]:
    hermitian_basis_labels, hermitian_basis = _hermitian_matrix_basis(int(dimension))
    hermitian_by_label = {
        label: matrix for label, matrix in zip(hermitian_basis_labels, hermitian_basis, strict=True)
    }
    monomial_index = {str(label): index for index, label in enumerate(monomial_labels)}

    labels: list[str] = []
    tensors: list[np.ndarray] = []
    for analysis in info.get(analysis_key, []) or []:
        if int(analysis.get("order", 0)) > int(max_order):
            continue
        for term in analysis.get("terms", []) or []:
            tensor = np.zeros((len(monomial_labels), dimension, dimension), dtype=complex)
            for raw in term.get("linear_combination", []) or []:
                h_label = str(raw.get("hermitian_basis_label", ""))
                q_label = str(raw.get("k_basis_label", ""))
                if h_label not in hermitian_by_label or q_label not in monomial_index:
                    continue
                tensor[monomial_index[q_label]] += complex(*raw.get("coefficient", [0.0, 0.0])) * hermitian_by_label[h_label]
            labels.append(str(term.get("label", "")))
            tensors.append(tensor)
    if not tensors:
        return [], np.zeros((0, len(monomial_labels), dimension, dimension), dtype=complex)
    return labels, np.asarray(tensors, dtype=complex)


def _stack_real_tensor(tensor: np.ndarray) -> np.ndarray:
    array = np.asarray(tensor, dtype=complex)
    return np.concatenate([array.real.reshape(-1), array.imag.reshape(-1)])


def _fit_formal_parameters_to_tensor(
    formal_tensors: np.ndarray,
    target_tensor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    design = np.column_stack([_stack_real_tensor(tensor) for tensor in formal_tensors])
    target = _stack_real_tensor(target_tensor)
    parameters = np.linalg.pinv(design, rcond=1.0e-10) @ target
    fitted_vector = design @ parameters
    residual = fitted_vector - target
    fitted_tensor = np.tensordot(parameters, formal_tensors, axes=(0, 0))
    return (
        np.asarray(parameters, dtype=float),
        np.asarray(fitted_tensor, dtype=complex),
        float(np.linalg.norm(residual)),
        float(np.linalg.norm(residual) / max(np.linalg.norm(target), 1.0e-30)),
    )


def _schur_kp_parameter_fit_analyses(
    info: Mapping[str, Any],
    representation_analyses: Sequence[Mapping[str, Any]],
    *,
    max_order: int = 3,
    variable_labels: Sequence[str] = KP_DIRECTIONS,
) -> list[dict[str, Any]]:
    numeric_analyses = list(info.get("numeric_lowdin_kp_analyses", []) or [])
    if not numeric_analyses or not representation_analyses:
        return []

    max_order = max(0, int(max_order))
    monomial_labels = _kp_fit_monomial_labels(
        max_order=max_order,
        variable_labels=variable_labels,
    )
    fits = []
    for rep_analysis in representation_analyses:
        selection_index = int(rep_analysis.get("selection_index", 0))
        numeric = next(
            (
                item
                for item in numeric_analyses
                if int(item.get("selection_index", 0)) == selection_index
            ),
            None,
        )
        if numeric is None:
            continue
        numeric_tensor = _numeric_lowdin_tensor_from_analysis(
            numeric,
            monomial_labels=monomial_labels,
        )
        if numeric_tensor.size == 0:
            continue
        dimension = int(numeric_tensor.shape[1])
        unitary = np.asarray(
            _complex_array_from_pairs(
                rep_analysis.get("schur_alignment", {}).get("unitary_numeric_to_standard", [])
            ),
            dtype=complex,
        )
        if unitary.shape != (dimension, dimension):
            continue
        transformed_tensor = np.asarray(
            [unitary.conj().T @ matrix @ unitary for matrix in numeric_tensor],
            dtype=complex,
        )
        formal_labels, formal_tensors = _formal_basis_tensors(
            info,
            dimension=dimension,
            monomial_labels=monomial_labels,
            max_order=max_order,
        )
        if not formal_labels:
            continue
        parameters, fitted_tensor, residual_norm, relative_residual = _fit_formal_parameters_to_tensor(
            formal_tensors,
            transformed_tensor,
        )
        difference = transformed_tensor - fitted_tensor
        records = []
        for index, label in enumerate(monomial_labels):
            diff = difference[index]
            records.append(
                {
                    "monomial": str(label),
                    "raw_numeric_matrix": _complex_matrix_to_pairs(numeric_tensor[index]),
                    "transformed_numeric_matrix": _complex_matrix_to_pairs(transformed_tensor[index]),
                    "formal_fit_matrix": _complex_matrix_to_pairs(fitted_tensor[index]),
                    "difference_matrix": _complex_matrix_to_pairs(diff),
                    "max_abs_difference": float(np.max(np.abs(diff))),
                    "frobenius_difference": float(np.linalg.norm(diff)),
                }
            )
        fits.append(
            {
                "selection_index": selection_index,
                "convention": "H_form_fit(q) ~= U_schur^dagger H_numeric(q) U_schur",
                "max_order": int(max_order),
                "k_direction": "".join(label[-1] for label in variable_labels),
                "monomial_basis": list(monomial_labels),
                "formal_parameter_labels": formal_labels,
                "formal_parameters": [float(value) for value in parameters],
                "absolute_residual": residual_norm,
                "relative_residual": relative_residual,
                "max_abs_difference_all": float(max(record["max_abs_difference"] for record in records)),
                "records": records,
            }
        )
    return fits


def _kp_error_q_grid(
    radius: float,
    grid: int,
    *,
    direction_indices: Sequence[int] = (0, 1, 2),
) -> np.ndarray:
    grid_count = int(grid)
    if grid_count <= 0:
        raise ValueError("KP.kp_grid must be a positive integer.")
    radius_value = float(radius)
    if radius_value < 0.0:
        raise ValueError("KP.kp_radius must be zero or a positive float.")
    if grid_count == 1:
        axis = np.array([0.0], dtype=float)
    else:
        axis = np.linspace(-radius_value, radius_value, grid_count, dtype=float)
    active = {int(index) for index in direction_indices}
    axes = [axis if index in active else np.array([0.0], dtype=float) for index in range(3)]
    return np.asarray(
        [[kx, ky, kz] for kx in axes[0] for ky in axes[1] for kz in axes[2]],
        dtype=float,
    )


def _kp_monomial_value(label: str, q_cartesian: Sequence[float]) -> float:
    text = str(label).strip()
    if text == "1":
        return 1.0
    q = np.asarray(q_cartesian, dtype=float).reshape(3)
    value = 1.0
    axis_values = {"kx": q[0], "ky": q[1], "kz": q[2]}
    for factor in text.split("*"):
        if "^" in factor:
            axis, exponent_text = factor.split("^", 1)
            exponent = int(exponent_text)
        else:
            axis = factor
            exponent = 1
        if axis not in axis_values:
            raise ValueError(f"Unknown k monomial factor: {factor}")
        value *= float(axis_values[axis]) ** exponent
    return float(value)


def _kp_fitted_monomial_matrices(
    fit_analysis: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    matrices: dict[str, np.ndarray] = {}
    for record in fit_analysis.get("records", []) or []:
        label = str(record.get("monomial", ""))
        matrix = np.asarray(
            _complex_array_from_pairs(record.get("formal_fit_matrix", [])),
            dtype=complex,
        )
        if label and matrix.ndim == 2:
            matrices[label] = matrix
    return matrices


def _evaluate_kp_hamiltonian_from_fit(
    fit_analysis: Mapping[str, Any],
    q_cartesian: Sequence[float],
) -> np.ndarray:
    matrices = _kp_fitted_monomial_matrices(fit_analysis)
    if not matrices:
        return np.zeros((0, 0), dtype=complex)
    first_matrix = next(iter(matrices.values()))
    hamiltonian = np.zeros_like(first_matrix, dtype=complex)
    for label in fit_analysis.get("monomial_basis", []) or matrices.keys():
        label_text = str(label)
        matrix = matrices.get(label_text)
        if matrix is None:
            continue
        hamiltonian += _kp_monomial_value(label_text, q_cartesian) * matrix
    return _clean_complex_array(hamiltonian)


def _direct_band_energies_for_kpoints(
    tb,
    k_direct_points: np.ndarray,
    band_range: Sequence[int],
) -> np.ndarray:
    tb_solver = getattr(tb, "tb_solver", None)
    if tb_solver is None:
        raise ValueError("KP energy-error analysis requires a tb_solver.")
    if len(band_range) != 2:
        raise ValueError("KP.band must contain band_start band_stop for energy-error analysis.")
    lower = int(band_range[0])
    upper = int(band_range[1])
    chunk_size = max(1, int(getattr(tb, "max_kpoint_num", len(k_direct_points)) or len(k_direct_points)))
    chunks: list[np.ndarray] = []
    for start in range(0, len(k_direct_points), chunk_size):
        chunk = np.asarray(k_direct_points[start:start + chunk_size], dtype=float)
        if hasattr(tb_solver, "diago_H_eigenvaluesOnly_range"):
            values = tb_solver.diago_H_eigenvaluesOnly_range(chunk, lower, upper)
        elif hasattr(tb_solver, "diago_H_eigenvaluesOnly"):
            all_values = tb_solver.diago_H_eigenvaluesOnly(chunk)
            values = np.asarray(all_values, dtype=float)[:, lower - 1:upper]
        else:
            _eigenvectors, all_values = tb_solver.diago_H(chunk)
            values = np.asarray(all_values, dtype=float)[:, lower - 1:upper]
        chunks.append(np.asarray(values, dtype=float))
    if not chunks:
        return np.zeros((0, upper - lower + 1), dtype=float)
    return np.concatenate(chunks, axis=0)


def _kp_energy_error_analyses(
    tb,
    fit_analyses: Sequence[Mapping[str, Any]],
    numeric_analyses: Sequence[Mapping[str, Any]],
    *,
    radius: float,
    grid: int,
    direction_indices: Sequence[int] = (0, 1, 2),
) -> list[dict[str, Any]]:
    radius_value = float(radius)
    if radius_value <= 0.0:
        return []
    grid_count = int(grid)
    active_indices = tuple(int(index) for index in direction_indices)
    q_points = _kp_error_q_grid(
        radius_value,
        grid_count,
        direction_indices=active_indices,
    )
    grid_shape = [grid_count if index in set(active_indices) else 1 for index in range(3)]
    numeric_by_selection = {
        int(item.get("selection_index", 0)): item
        for item in numeric_analyses or []
    }
    analyses: list[dict[str, Any]] = []
    for fit in fit_analyses or []:
        selection_index = int(fit.get("selection_index", 0))
        numeric = numeric_by_selection.get(selection_index)
        if numeric is None:
            continue
        band_range = list(numeric.get("band_range", []))
        if len(band_range) != 2:
            continue
        k0_direct = np.asarray(numeric.get("reference_k_direct", []), dtype=float).reshape(-1)
        if k0_direct.size != 3:
            continue
        if hasattr(tb, "direct_to_cartesian_kspace"):
            k0_cartesian = np.asarray(tb.direct_to_cartesian_kspace(k0_direct.reshape(1, 3)), dtype=float)[0]
        else:
            k0_cartesian = np.asarray(numeric.get("reference_k_cartesian", k0_direct), dtype=float).reshape(3)
        k_cartesian_points = k0_cartesian.reshape(1, 3) + q_points
        if hasattr(tb, "cartesian_to_direct_kspace"):
            k_direct_points = np.asarray(tb.cartesian_to_direct_kspace(k_cartesian_points), dtype=float)
        else:
            k_direct_points = k_cartesian_points
        direct_energies = _direct_band_energies_for_kpoints(tb, k_direct_points, band_range)
        kp_energies = []
        for q_point in q_points:
            kp_hamiltonian = _evaluate_kp_hamiltonian_from_fit(fit, q_point)
            if kp_hamiltonian.size == 0:
                kp_energies = []
                break
            kp_energies.append(np.linalg.eigvalsh(kp_hamiltonian).real)
        if not kp_energies:
            continue
        kp_energy_array = np.asarray(kp_energies, dtype=float)
        if direct_energies.shape != kp_energy_array.shape:
            continue
        error = kp_energy_array - direct_energies
        abs_error = np.abs(error)
        worst_flat_index = int(np.argmax(abs_error))
        worst_point_index, worst_band_index = np.unravel_index(worst_flat_index, abs_error.shape)
        analyses.append(
            {
                "selection_index": selection_index,
                "radius_A^-1": radius_value,
                "grid_points_per_axis": grid_count,
                "grid_shape": [int(value) for value in grid_shape],
                "k_direction": "".join("xyz"[index] for index in active_indices),
                "point_count": int(q_points.shape[0]),
                "band_count": int(kp_energy_array.shape[1]),
                "direct_diagonalization_band_range": [int(band_range[0]), int(band_range[1])],
                "q_grid_convention": "q = k - k0 in Cartesian coordinates, Angstrom^-1",
                "kp_energy_convention": "eigenvalues of fitted k.p Hamiltonian",
                "direct_energy_convention": "direct diagonalization from the active symmetrized H/S data",
                "max_abs_error_eV": float(np.max(abs_error)),
                "mean_abs_error_eV": float(np.mean(abs_error)),
                "rms_error_eV": float(np.sqrt(np.mean(abs_error**2))),
                "per_band_max_abs_error_eV": [float(value) for value in np.max(abs_error, axis=0)],
                "per_band_mean_abs_error_eV": [float(value) for value in np.mean(abs_error, axis=0)],
                "worst_point_index": int(worst_point_index + 1),
                "worst_band_offset": int(worst_band_index + 1),
                "worst_band_index": int(band_range[0] + worst_band_index),
                "worst_q_cartesian": [float(value) for value in q_points[worst_point_index]],
                "worst_k_direct": [float(value) for value in k_direct_points[worst_point_index]],
                "worst_direct_energy_eV": float(direct_energies[worst_point_index, worst_band_index]),
                "worst_kp_energy_eV": float(kp_energy_array[worst_point_index, worst_band_index]),
                "worst_signed_error_eV": float(error[worst_point_index, worst_band_index]),
            }
        )
    return analyses


def _schur_zeeman_parameter_fit_analyses(
    info: Mapping[str, Any],
    representation_analyses: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    numeric_analyses = list(info.get("numeric_lowdin_kp_analyses", []) or [])
    if not numeric_analyses or not representation_analyses:
        return []

    field_labels = list(ZEEMAN_FIELD_DIRECTIONS)
    fits = []
    for rep_analysis in representation_analyses:
        selection_index = int(rep_analysis.get("selection_index", 0))
        numeric = next(
            (
                item
                for item in numeric_analyses
                if int(item.get("selection_index", 0)) == selection_index
            ),
            None,
        )
        if numeric is None:
            continue
        numeric_tensor = _numeric_zeeman_tensor_from_analysis(
            numeric,
            field_labels=field_labels,
        )
        if numeric_tensor.size == 0:
            continue
        dimension = int(numeric_tensor.shape[1])
        unitary = np.asarray(
            _complex_array_from_pairs(
                rep_analysis.get("schur_alignment", {}).get("unitary_numeric_to_standard", [])
            ),
            dtype=complex,
        )
        if unitary.shape != (dimension, dimension):
            continue
        transformed_tensor = np.asarray(
            [unitary.conj().T @ matrix @ unitary for matrix in numeric_tensor],
            dtype=complex,
        )
        formal_labels, formal_tensors = _formal_basis_tensors(
            info,
            dimension=dimension,
            monomial_labels=field_labels,
            max_order=1,
            analysis_key="final_zeeman_model_analyses",
        )
        if not formal_labels:
            continue
        parameters, fitted_tensor, residual_norm, relative_residual = _fit_formal_parameters_to_tensor(
            formal_tensors,
            transformed_tensor,
        )
        difference = transformed_tensor - fitted_tensor
        records = []
        for index, label in enumerate(field_labels):
            diff = difference[index]
            records.append(
                {
                    "field": str(label),
                    "raw_numeric_matrix": _complex_matrix_to_pairs(numeric_tensor[index]),
                    "transformed_numeric_matrix": _complex_matrix_to_pairs(transformed_tensor[index]),
                    "formal_fit_matrix": _complex_matrix_to_pairs(fitted_tensor[index]),
                    "difference_matrix": _complex_matrix_to_pairs(diff),
                    "max_abs_difference": float(np.max(np.abs(diff))),
                    "frobenius_difference": float(np.linalg.norm(diff)),
                }
            )
        fits.append(
            {
                "selection_index": selection_index,
                "convention": "G_form_fit(B) ~= U_schur^dagger G_numeric(B) U_schur; H_Z=(mu_B/2) sum_i G_i B_i",
                "field_basis": field_labels,
                "formal_parameter_labels": formal_labels,
                "formal_parameters": [float(value) for value in parameters],
                "absolute_residual": residual_norm,
                "relative_residual": relative_residual,
                "max_abs_difference_all": float(max(record["max_abs_difference"] for record in records)),
                "records": records,
            }
        )
    return fits


def _sort_spgrep_unitary_operations_by_character_order(
    unitary_source_indices: Sequence[int],
    operation_by_source_index: Mapping[int, Mapping[str, Any]],
    character_operations: Sequence[Any] | None,
) -> list[int]:
    if not character_operations:
        return list(unitary_source_indices)

    remaining = list(unitary_source_indices)
    sorted_indices: list[int] = []
    for character_operation in character_operations:
        character_rotation = np.asarray(getattr(character_operation, "rotation", None), dtype=int)
        matched_index = None
        for source_index in remaining:
            spgrep_rotation = np.asarray(operation_by_source_index[source_index]["rotation"], dtype=int)
            if np.array_equal(spgrep_rotation, character_rotation):
                matched_index = source_index
                break
        if matched_index is None:
            continue
        sorted_indices.append(matched_index)
        remaining.remove(matched_index)
    sorted_indices.extend(remaining)
    return sorted_indices


def _hermitian_matrix_basis(dimension: int) -> tuple[list[str], list[np.ndarray]]:
    labels: list[str] = []
    basis: list[np.ndarray] = []
    n = int(dimension)
    if n <= 0:
        return labels, basis

    for i in range(n):
        matrix = np.zeros((n, n), dtype=complex)
        matrix[i, i] = 1.0
        labels.append(f"diag({i + 1})")
        basis.append(matrix)

    scale = 1.0 / np.sqrt(2.0)
    for i in range(n):
        for j in range(i + 1, n):
            matrix = np.zeros((n, n), dtype=complex)
            matrix[i, j] = scale
            matrix[j, i] = scale
            labels.append(f"re({i + 1},{j + 1})")
            basis.append(matrix)

            matrix = np.zeros((n, n), dtype=complex)
            matrix[i, j] = -1j * scale
            matrix[j, i] = 1j * scale
            labels.append(f"im({i + 1},{j + 1})")
            basis.append(matrix)

    return labels, basis


def _corep_operation_matrices(corep: Mapping[str, Any]) -> tuple[list[int], list[np.ndarray]]:
    operation_matrices = sorted(
        list(corep.get("operation_matrices") or []),
        key=lambda item: int(item.get("operation_index", 0)),
    )
    operation_indices: list[int] = []
    matrices: list[np.ndarray] = []
    for item in operation_matrices:
        operation_indices.append(int(item.get("operation_index", 0)))
        matrices.append(np.asarray(_complex_array_from_pairs(item.get("matrix", [])), dtype=complex))
    return operation_indices, matrices


def _block_diag_matrices(matrices: Sequence[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(matrix, dtype=complex) for matrix in matrices]
    dimension = int(sum(array.shape[0] for array in arrays))
    out = np.zeros((dimension, dimension), dtype=complex)
    start = 0
    for array in arrays:
        stop = start + int(array.shape[0])
        out[start:stop, start:stop] = array
        start = stop
    return out


def _target_matrices_from_corep_indices(
    spgrep_info: Mapping[str, Any],
    corep_indices: Sequence[int],
) -> tuple[list[int], list[np.ndarray]]:
    coreps = list(spgrep_info.get("coreps") or [])
    matrices_by_corep: list[tuple[list[int], list[np.ndarray]]] = []
    for corep_index in corep_indices:
        corep = next(
            (
                item
                for item in coreps
                if int(item.get("corep_index", 0)) == int(corep_index)
            ),
            None,
        )
        if corep is None:
            return [], []
        operation_indices, matrices = _corep_operation_matrices(corep)
        if not matrices:
            return [], []
        matrices_by_corep.append((operation_indices, matrices))

    reference_indices = matrices_by_corep[0][0]
    if any(indices != reference_indices for indices, _matrices in matrices_by_corep[1:]):
        return [], []

    combined_matrices = []
    for op_pos in range(len(reference_indices)):
        combined_matrices.append(
            _block_diag_matrices([matrices[op_pos] for _indices, matrices in matrices_by_corep])
        )
    return reference_indices, combined_matrices


def _spgrep_unitary_operation_aliases(spgrep_info: Mapping[str, Any]) -> dict[int, list[int]]:
    aliases: dict[int, list[int]] = {}
    for operation in spgrep_info.get("operations", []) or []:
        local_index = int(operation.get("operation_index", 0))
        if local_index <= 0:
            continue
        values = [local_index]
        display_index = int(operation.get("display_operation_index", local_index))
        if display_index > 0 and display_index not in values:
            values.append(display_index)
        aliases[local_index] = values
    return aliases


def _target_corep_dimensions(
    spgrep_info: Mapping[str, Any],
    corep_indices: Sequence[int],
) -> list[int]:
    dimensions: list[int] = []
    coreps = list(spgrep_info.get("coreps") or [])
    for corep_index in corep_indices:
        corep = next(
            (
                item
                for item in coreps
                if int(item.get("corep_index", 0)) == int(corep_index)
            ),
            None,
        )
        if corep is None:
            return []
        operation_matrices = list(corep.get("operation_matrices") or [])
        if not operation_matrices:
            return []
        matrix = np.asarray(_complex_array_from_pairs(operation_matrices[0].get("matrix", [])), dtype=complex)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            return []
        dimensions.append(int(matrix.shape[0]))
    return dimensions


def _operator_representation_matrices(
    target_matrices: Sequence[np.ndarray],
    basis: Sequence[np.ndarray],
) -> list[np.ndarray]:
    representation_matrices: list[np.ndarray] = []
    basis_arrays = [np.asarray(matrix, dtype=complex) for matrix in basis]
    basis_count = len(basis_arrays)
    for target in target_matrices:
        matrix = np.asarray(target, dtype=complex)
        transformed_rep = np.zeros((basis_count, basis_count), dtype=complex)
        for column, basis_matrix in enumerate(basis_arrays):
            transformed = matrix.conj().T @ basis_matrix @ matrix
            for row, projector in enumerate(basis_arrays):
                transformed_rep[row, column] = np.trace(projector @ transformed)
        representation_matrices.append(transformed_rep)
    return representation_matrices


def _irrep_display_label(irrep: Any) -> str:
    raw_name = str(getattr(irrep, "raw_name", getattr(irrep, "name", ""))).strip()
    if raw_name.startswith("-"):
        return raw_name[1:]
    return str(getattr(irrep, "name", raw_name)).strip() or raw_name


def _is_double_valued_irrep(irrep: Any) -> bool:
    raw_name = str(getattr(irrep, "raw_name", getattr(irrep, "name", ""))).strip()
    return raw_name.startswith("-")


def _character_table_summary(analysis_result: Mapping[str, Any]) -> dict[str, Any]:
    kpoint_records = list(analysis_result.get("kpoint_records") or [])
    if not kpoint_records:
        return {}
    record = kpoint_records[0]
    resolution = record.get("resolution") if isinstance(record, Mapping) else None
    if resolution is None:
        return {}

    active_indices = list(
        record.get("character_operation_indices")
        or record.get("database_operation_indices")
        or record.get("table_operation_indices")
        or []
    )
    if not active_indices:
        return {}
    table_indices = list(record.get("table_operation_indices", active_indices))
    source_operations = list(analysis_result.get("operations") or [])
    phase_k_direct = record.get("phase_k_direct")
    table_operation_translations = record.get("table_operation_translations")

    irreps = []
    for irrep in getattr(getattr(resolution, "entry", None), "irreps", []):
        characters = _resolved_irrep_character_slice(
            irrep,
            resolution,
            active_indices,
            table_indices,
            phase_k_direct=phase_k_direct,
            phase_operations=source_operations,
            table_operation_translations=table_operation_translations,
        )
        label = _irrep_display_label(irrep)
        raw_name = str(getattr(irrep, "raw_name", getattr(irrep, "name", label)))
        irreps.append(
            {
                "label": label,
                "raw_name": raw_name,
                "dimension": int(getattr(irrep, "dimension", round(float(np.real(characters[0]))))),
                "reality": int(getattr(irrep, "reality", 0)),
                "double_valued": _is_double_valued_irrep(irrep),
                "characters": _complex_vector_to_pairs(characters),
            }
        )

    return {
        "k_name": str(getattr(resolution.entry, "name", record.get("k_name", ""))),
        "operation_indices": [int(index) + 1 for index in active_indices],
        "display_operation_indices": list(range(1, len(active_indices) + 1)),
        "irreps": irreps,
    }


def _character_table_candidates(character_table: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    irreps = list(character_table.get("irreps") or [])
    single_valued = [irrep for irrep in irreps if not bool(irrep.get("double_valued", False))]
    return single_valued or irreps


def _linear_combination_terms(
    coefficients: np.ndarray,
    basis_labels: Sequence[str],
    *,
    tol: float,
) -> list[dict[str, Any]]:
    terms = []
    for index, coefficient in enumerate(np.asarray(coefficients, dtype=complex).reshape(-1)):
        if abs(coefficient) <= tol:
            continue
        terms.append(
            {
                "basis_index": int(index + 1),
                "basis_label": str(basis_labels[index]),
                "coefficient": _complex_scalar_to_pair(coefficient),
            }
        )
    return terms


def _canonicalize_vector_phase(vector: np.ndarray, tol: float) -> np.ndarray:
    out = np.asarray(vector, dtype=complex).copy()
    if out.size == 0:
        return out
    pivot = int(np.argmax(np.abs(out)))
    if abs(out[pivot]) <= tol:
        return out
    out *= np.exp(-1j * np.angle(out[pivot]))
    if out[pivot].real < 0:
        out *= -1.0
    out[np.abs(out) < tol] = 0.0
    return out


def _operator_irrep_analysis(
    character_table: Mapping[str, Any],
    *,
    target_label: str,
    operation_indices: Sequence[int],
    target_matrices: Sequence[np.ndarray],
    target_corep_indices: Sequence[int],
    tol: float = 1.0e-8,
) -> dict[str, Any]:
    if not target_matrices:
        return {}
    dimension = int(np.asarray(target_matrices[0]).shape[0])
    target_characters = np.asarray([np.trace(matrix) for matrix in target_matrices], dtype=complex)
    operator_characters = np.conj(target_characters) * target_characters
    group_order = int(len(operator_characters))

    candidates = _character_table_candidates(character_table)
    decomposition = []
    projectable_irreps = []
    for irrep in candidates:
        irrep_characters = np.asarray(_complex_array_from_pairs(irrep.get("characters", [])), dtype=complex).reshape(-1)
        if irrep_characters.size != group_order:
            continue
        raw_multiplicity = np.vdot(irrep_characters, operator_characters) / float(group_order)
        rounded = int(round(float(raw_multiplicity.real)))
        is_integer = (
            abs(raw_multiplicity.imag) <= 1.0e-6
            and abs(raw_multiplicity.real - rounded) <= 1.0e-6
        )
        if abs(raw_multiplicity) <= 1.0e-6:
            continue
        entry = {
            "label": str(irrep.get("label", "")),
            "dimension": int(irrep.get("dimension", 0)),
            "multiplicity": int(rounded) if is_integer else float(raw_multiplicity.real),
            "raw_multiplicity": _complex_scalar_to_pair(raw_multiplicity),
            "is_integer": bool(is_integer),
            "characters": _complex_vector_to_pairs(irrep_characters),
        }
        decomposition.append(entry)
        if is_integer and rounded > 0:
            projectable_irreps.append((irrep, irrep_characters, rounded))

    basis_labels, basis = _hermitian_matrix_basis(dimension)
    operator_matrices = _operator_representation_matrices(target_matrices, basis)
    projected_basis = []
    for irrep, irrep_characters, _multiplicity in projectable_irreps:
        irrep_dimension = int(irrep.get("dimension", 0))
        projector = np.zeros((len(basis), len(basis)), dtype=complex)
        for character, operator_matrix in zip(irrep_characters, operator_matrices):
            projector += np.conj(character) * operator_matrix
        projector *= float(irrep_dimension) / float(group_order)
        u_matrix, singular_values, _vh = np.linalg.svd(projector)
        rank = int(np.count_nonzero(singular_values > 1.0e-7))
        matrices = []
        for basis_index in range(rank):
            coefficients = _canonicalize_vector_phase(u_matrix[:, basis_index], tol)
            matrix = np.zeros((dimension, dimension), dtype=complex)
            for coefficient, basis_matrix in zip(coefficients, basis):
                matrix += coefficient * basis_matrix
            matrices.append(
                {
                    "label": f"{irrep.get('label', '')}_basis_{basis_index + 1}",
                    "linear_combination": _linear_combination_terms(coefficients, basis_labels, tol=tol),
                    "matrix": _complex_matrix_to_pairs(matrix),
                }
            )
        projected_basis.append(
            {
                "irrep_label": str(irrep.get("label", "")),
                "irrep_dimension": irrep_dimension,
                "rank": rank,
                "singular_values": [float(value) for value in singular_values],
                "matrices": matrices,
            }
        )

    return {
        "target_label": str(target_label),
        "target_corep_indices": [int(index) for index in target_corep_indices],
        "operator_representation": (
            f"({target_label})^* x ({target_label})"
            if " + " in str(target_label)
            else f"{target_label}^* x {target_label}"
        ),
        "dimension": dimension,
        "operation_indices": [int(index) for index in operation_indices],
        "band_characters": _complex_vector_to_pairs(target_characters),
        "operator_characters": _complex_vector_to_pairs(operator_characters),
        "representation_matrices": [_complex_matrix_to_pairs(matrix) for matrix in operator_matrices],
        "character_table_source": character_table.get("source", "CHARACTER"),
        "decomposition": decomposition,
        "hermitian_basis": [
            {
                "basis_index": int(index + 1),
                "label": label,
                "matrix": _complex_matrix_to_pairs(matrix),
            }
            for index, (label, matrix) in enumerate(zip(basis_labels, basis))
        ],
        "projected_basis": projected_basis,
    }


def _split_irrep_terms(value: str) -> list[str]:
    terms = []
    for part in re.split(r"\s+\+\s+|⊕", str(value)):
        label = part.strip()
        if not label or label == "??":
            continue
        terms.append(label)
    return terms


def _operator_irrep_analyses_for_results(
    info: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    spgrep_info = info.get("spgrep_operations", {})
    character_table = info.get("character_table", {})
    if not spgrep_info or not character_table:
        return []

    analyses = []
    for result in results:
        target_labels: list[str] = []
        for row in result.get("rows") or []:
            for label in _split_irrep_terms(str(row.get("irrep", ""))):
                target_labels.append(label)
        if not target_labels:
            continue
        corep_groups = _spgrep_corep_groups_for_irrep_terms(
            target_labels,
            spgrep_info=spgrep_info,
            character_table=character_table,
        )
        if not corep_groups:
            continue
        corep_indices = [int(group["corep_index"]) for group in corep_groups]
        operation_indices, target_matrices = _target_matrices_from_corep_indices(spgrep_info, corep_indices)
        target_label = " + ".join(target_labels)
        analysis = _operator_irrep_analysis(
            character_table,
            target_label=target_label,
            operation_indices=operation_indices,
            target_matrices=target_matrices,
            target_corep_indices=corep_indices,
        )
        if analysis:
            analysis["target_corep_labels"] = [str(group["label"]) for group in corep_groups]
            analyses.append(analysis)
    return analyses


def _monomial_exponents(order: int, variable_count: int = 3) -> list[tuple[int, ...]]:
    n = int(order)
    if n < 0:
        return []
    if int(variable_count) == 1:
        return [(n,)]

    exponents: list[tuple[int, ...]] = []
    for value in range(n, -1, -1):
        for tail in _monomial_exponents(n - value, int(variable_count) - 1):
            exponents.append((value, *tail))
    return exponents


def _monomial_label(exponent: Sequence[int], variable_labels: Sequence[str] = ("kx", "ky", "kz")) -> str:
    pieces = []
    for power, label in zip(exponent, variable_labels):
        value = int(power)
        if value == 0:
            continue
        if value == 1:
            pieces.append(str(label))
        else:
            pieces.append(f"{label}^{value}")
    return "*".join(pieces) if pieces else "1"


def _multiply_polynomials(
    left: Mapping[tuple[int, ...], complex],
    right: Mapping[tuple[int, ...], complex],
) -> dict[tuple[int, ...], complex]:
    product: dict[tuple[int, ...], complex] = {}
    for left_exp, left_coeff in left.items():
        for right_exp, right_coeff in right.items():
            exponent = tuple(int(a) + int(b) for a, b in zip(left_exp, right_exp))
            product[exponent] = product.get(exponent, 0.0 + 0.0j) + left_coeff * right_coeff
    return product


def _linear_form_power(coefficients: Sequence[complex], power: int) -> dict[tuple[int, ...], complex]:
    variable_count = len(coefficients)
    polynomial: dict[tuple[int, ...], complex] = {tuple([0] * variable_count): 1.0 + 0.0j}
    linear_terms = {}
    for index, coefficient in enumerate(coefficients):
        if abs(coefficient) <= 1.0e-12:
            continue
        exponent = [0] * variable_count
        exponent[index] = 1
        linear_terms[tuple(exponent)] = complex(coefficient)
    if not linear_terms:
        linear_terms = {tuple([0] * variable_count): 0.0 + 0.0j}
    for _ in range(int(power)):
        polynomial = _multiply_polynomials(polynomial, linear_terms)
    return polynomial


def _polynomial_transform_matrix(transform: np.ndarray, exponents: Sequence[tuple[int, ...]]) -> np.ndarray:
    matrix = np.asarray(transform, dtype=complex)
    exponent_to_index = {tuple(exponent): index for index, exponent in enumerate(exponents)}
    out = np.zeros((len(exponents), len(exponents)), dtype=complex)
    for column, exponent in enumerate(exponents):
        polynomial: dict[tuple[int, ...], complex] = {tuple([0] * matrix.shape[0]): 1.0 + 0.0j}
        for row_index, power in enumerate(exponent):
            polynomial = _multiply_polynomials(
                polynomial,
                _linear_form_power(matrix[row_index, :], int(power)),
            )
        for transformed_exp, coefficient in polynomial.items():
            row = exponent_to_index.get(tuple(transformed_exp))
            if row is not None:
                out[row, column] += coefficient
    out[np.abs(out) < 1.0e-12] = 0.0
    return out


def _monomial_exponent_from_label(
    label: str,
    variable_labels: Sequence[str] = ("kx", "ky", "kz"),
) -> tuple[int, ...] | None:
    text = str(label).strip()
    if text == "1":
        return tuple([0] * len(variable_labels))
    variable_index = {str(name): index for index, name in enumerate(variable_labels)}
    exponents = [0] * len(variable_labels)
    for piece in text.split("*"):
        item = piece.strip()
        if not item:
            continue
        if "^" in item:
            name, power_text = item.split("^", 1)
            power = int(power_text)
        else:
            name = item
            power = 1
        index = variable_index.get(name)
        if index is None:
            return None
        exponents[index] += int(power)
    return tuple(exponents)


def _polynomial_transform_matrix_for_labels(
    transform: np.ndarray,
    basis_labels: Sequence[str],
    *,
    variable_labels: Sequence[str] = ("kx", "ky", "kz"),
) -> np.ndarray:
    exponents = [
        _monomial_exponent_from_label(label, variable_labels=variable_labels)
        for label in basis_labels
    ]
    if any(exponent is None for exponent in exponents):
        return np.eye(len(basis_labels), dtype=complex)
    return _polynomial_transform_matrix(np.asarray(transform, dtype=complex), [tuple(exp) for exp in exponents if exp is not None])


def _reciprocal_rotation_from_operation(
    operation: Mapping[str, Any],
    lattice: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> np.ndarray:
    rotation = np.asarray(operation.get("rotation", np.eye(3, dtype=int)), dtype=float)
    if lattice is None:
        return np.linalg.inv(rotation)
    lattice_matrix = np.asarray(lattice, dtype=float)
    if lattice_matrix.shape != (3, 3):
        return np.linalg.inv(rotation)
    col_lattice = lattice_matrix.T
    cart_rotation = col_lattice @ rotation @ np.linalg.inv(col_lattice)
    return np.linalg.inv(cart_rotation)


def _cartesian_rotation_from_operation(
    operation: Mapping[str, Any],
    lattice: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> np.ndarray:
    rotation = np.asarray(operation.get("rotation", np.eye(3, dtype=int)), dtype=float)
    if lattice is None:
        return rotation
    lattice_matrix = np.asarray(lattice, dtype=float)
    if lattice_matrix.shape != (3, 3):
        return rotation
    col_lattice = lattice_matrix.T
    return col_lattice @ rotation @ np.linalg.inv(col_lattice)


def _factor_spin_matrix_from_cartesian_rotation(cart_rotation: np.ndarray) -> np.ndarray:
    rot = np.asarray(cart_rotation, dtype=float)
    proper_rot = -rot if float(np.linalg.det(rot)) < 0.0 else rot
    axis, angle, _ = axis_angle_from_cartesian_rotation(proper_rot)
    canonical_axis = _canonicalize_axis(axis)
    if abs(float(angle) - float(np.pi)) < 1.0e-6:
        signed_angle = -float(angle)
    elif float(np.dot(axis, canonical_axis)) < 0.0:
        signed_angle = -float(angle)
    else:
        signed_angle = float(angle)
    return spin_half_matrix_from_axis_angle(canonical_axis, signed_angle)


def _dk_operation_from_summary(
    info: Mapping[str, Any],
    operation: Mapping[str, Any],
) -> dict[str, Any]:
    lattice = np.asarray(info.get("lattice_vectors", np.eye(3)), dtype=float)
    rotation = np.asarray(operation.get("rotation", np.eye(3, dtype=int)), dtype=int)
    translation = np.asarray(_vector3(operation.get("translation"), [0.0, 0.0, 0.0]), dtype=float)
    cart_rotation = _cartesian_rotation_from_operation({"rotation": rotation}, lattice=lattice)
    out: dict[str, Any] = {
        "rotation": rotation,
        "translation": translation,
        "cart_rotation": cart_rotation,
    }
    if int(info.get("nspin", 1)) == 4:
        out["spin_matrix"] = _factor_spin_matrix_from_cartesian_rotation(cart_rotation)
    return out


def _zeeman_field_transform_from_operation(
    operation: Mapping[str, Any],
    lattice: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> np.ndarray:
    cart_rotation = _cartesian_rotation_from_operation(operation, lattice=lattice)
    pseudovector_rotation = float(round(np.linalg.det(cart_rotation))) * cart_rotation
    return np.linalg.inv(pseudovector_rotation)


def _projected_polynomial_basis(
    *,
    irrep: Mapping[str, Any],
    irrep_characters: np.ndarray,
    representation_matrices: Sequence[np.ndarray],
    basis_labels: Sequence[str],
    group_order: int,
    tol: float = 1.0e-8,
) -> dict[str, Any]:
    irrep_dimension = int(irrep.get("dimension", 0))
    projector = np.zeros_like(np.asarray(representation_matrices[0], dtype=complex))
    for character, representation_matrix in zip(irrep_characters, representation_matrices):
        projector += np.conj(character) * representation_matrix
    projector *= float(irrep_dimension) / float(group_order)
    u_matrix, singular_values, _vh = np.linalg.svd(projector)
    rank = int(np.count_nonzero(singular_values > 1.0e-7))
    functions = []
    for basis_index in range(rank):
        coefficients = _canonicalize_vector_phase(u_matrix[:, basis_index], tol)
        functions.append(
            {
                "label": f"{irrep.get('label', '')}_k_basis_{basis_index + 1}",
                "linear_combination": _linear_combination_terms(coefficients, basis_labels, tol=tol),
            }
        )
    return {
        "irrep_label": str(irrep.get("label", "")),
        "irrep_dimension": irrep_dimension,
        "rank": rank,
        "singular_values": [float(value) for value in singular_values],
        "functions": functions,
    }


def _k_polynomial_irrep_analyses(
    spgrep_info: Mapping[str, Any],
    character_table: Mapping[str, Any],
    *,
    orders: Sequence[int] = (1, 2, 3),
    lattice: Sequence[Sequence[float]] | np.ndarray | None = None,
    k_direction: str | Sequence[str] | None = "xyz",
) -> list[dict[str, Any]]:
    operations = list(spgrep_info.get("operations") or [])
    if not operations or not character_table:
        return []

    k_direction_text, direction_indices, variable_labels = _normalize_k_direction(k_direction)
    candidates = _character_table_candidates(character_table)
    operation_indices = [int(operation.get("operation_index", index + 1)) for index, operation in enumerate(operations)]
    analyses = []
    for order in orders:
        exponents = _monomial_exponents(int(order), len(variable_labels))
        basis_labels = [_monomial_label(exponent, variable_labels) for exponent in exponents]
        representation_matrices = [
            _polynomial_transform_matrix(
                _reciprocal_rotation_from_operation(operation, lattice=lattice)[np.ix_(direction_indices, direction_indices)],
                exponents,
            )
            for operation in operations
        ]
        characters = np.asarray([np.trace(matrix) for matrix in representation_matrices], dtype=complex)
        group_order = int(len(characters))

        decomposition = []
        projectable_irreps = []
        for irrep in candidates:
            irrep_characters = np.asarray(_complex_array_from_pairs(irrep.get("characters", [])), dtype=complex).reshape(-1)
            if irrep_characters.size != group_order:
                continue
            raw_multiplicity = np.vdot(irrep_characters, characters) / float(group_order)
            rounded = int(round(float(raw_multiplicity.real)))
            is_integer = (
                abs(raw_multiplicity.imag) <= 1.0e-6
                and abs(raw_multiplicity.real - rounded) <= 1.0e-6
            )
            if abs(raw_multiplicity) <= 1.0e-6:
                continue
            decomposition.append(
                {
                    "label": str(irrep.get("label", "")),
                    "dimension": int(irrep.get("dimension", 0)),
                    "multiplicity": int(rounded) if is_integer else float(raw_multiplicity.real),
                    "raw_multiplicity": _complex_scalar_to_pair(raw_multiplicity),
                    "is_integer": bool(is_integer),
                    "characters": _complex_vector_to_pairs(irrep_characters),
                }
            )
            if is_integer and rounded > 0:
                projectable_irreps.append((irrep, irrep_characters))

        projected_basis = [
            _projected_polynomial_basis(
                irrep=irrep,
                irrep_characters=irrep_characters,
                representation_matrices=representation_matrices,
                basis_labels=basis_labels,
                group_order=group_order,
            )
            for irrep, irrep_characters in projectable_irreps
        ]
        analyses.append(
            {
                "order": int(order),
                "k_direction": k_direction_text,
                "variable_labels": list(variable_labels),
                "coordinate_basis": "cartesian",
                "operation_indices": operation_indices,
                "characters": _complex_vector_to_pairs(characters),
                "representation_matrices": [_complex_matrix_to_pairs(matrix) for matrix in representation_matrices],
                "decomposition": decomposition,
                "monomial_basis": [
                    {
                        "basis_index": int(index + 1),
                        "label": label,
                        "exponents": [int(value) for value in exponent],
                    }
                    for index, (label, exponent) in enumerate(zip(basis_labels, exponents))
                ],
                "projected_basis": projected_basis,
            }
        )
    return analyses


def _zeeman_field_irrep_analyses(
    spgrep_info: Mapping[str, Any],
    character_table: Mapping[str, Any],
    *,
    lattice: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> list[dict[str, Any]]:
    operations = list(spgrep_info.get("operations") or [])
    if not operations or not character_table:
        return []

    exponents = _monomial_exponents(1, 3)
    basis_labels = [
        _monomial_label(exponent, variable_labels=ZEEMAN_FIELD_DIRECTIONS)
        for exponent in exponents
    ]
    representation_matrices = [
        _polynomial_transform_matrix(_zeeman_field_transform_from_operation(operation, lattice=lattice), exponents)
        for operation in operations
    ]
    characters = np.asarray([np.trace(matrix) for matrix in representation_matrices], dtype=complex)
    group_order = int(len(characters))
    candidates = _character_table_candidates(character_table)
    decomposition = []
    projectable_irreps = []
    for irrep in candidates:
        irrep_characters = np.asarray(_complex_array_from_pairs(irrep.get("characters", [])), dtype=complex).reshape(-1)
        if irrep_characters.size != group_order:
            continue
        raw_multiplicity = np.vdot(irrep_characters, characters) / float(group_order)
        rounded = int(round(float(raw_multiplicity.real)))
        is_integer = (
            abs(raw_multiplicity.imag) <= 1.0e-6
            and abs(raw_multiplicity.real - rounded) <= 1.0e-6
        )
        if abs(raw_multiplicity) <= 1.0e-6:
            continue
        decomposition.append(
            {
                "label": str(irrep.get("label", "")),
                "dimension": int(irrep.get("dimension", 0)),
                "multiplicity": int(rounded) if is_integer else float(raw_multiplicity.real),
                "raw_multiplicity": _complex_scalar_to_pair(raw_multiplicity),
                "is_integer": bool(is_integer),
                "characters": _complex_vector_to_pairs(irrep_characters),
            }
        )
        if is_integer and rounded > 0:
            projectable_irreps.append((irrep, irrep_characters))

    projected_basis = [
        _projected_polynomial_basis(
            irrep=irrep,
            irrep_characters=irrep_characters,
            representation_matrices=representation_matrices,
            basis_labels=basis_labels,
            group_order=group_order,
        )
        for irrep, irrep_characters in projectable_irreps
    ]
    operation_indices = [int(operation.get("operation_index", index + 1)) for index, operation in enumerate(operations)]
    return [
        {
            "order": 1,
            "coordinate_basis": "cartesian_pseudovector",
            "operation_indices": operation_indices,
            "characters": _complex_vector_to_pairs(characters),
            "representation_matrices": [_complex_matrix_to_pairs(matrix) for matrix in representation_matrices],
            "decomposition": decomposition,
            "monomial_basis": [
                {
                    "basis_index": int(index + 1),
                    "label": label,
                    "exponents": [int(value) for value in exponent],
                }
                for index, (label, exponent) in enumerate(zip(basis_labels, exponents))
            ],
            "projected_basis": projected_basis,
        }
    ]


def _coefficient_vector_from_terms(
    terms: Sequence[Mapping[str, Any]],
    basis_count: int,
) -> np.ndarray:
    vector = np.zeros(int(basis_count), dtype=complex)
    for term in terms:
        index = int(term.get("basis_index", 0)) - 1
        if 0 <= index < vector.size:
            vector[index] += complex(*term.get("coefficient", [0.0, 0.0]))
    return vector


def _conjugate_irrep_label(character_table: Mapping[str, Any], label: str, tol: float = 1.0e-7) -> str:
    irreps = _character_table_candidates(character_table)
    target = next((item for item in irreps if str(item.get("label", "")) == str(label)), None)
    if target is None:
        return str(label)
    target_characters = np.asarray(_complex_array_from_pairs(target.get("characters", [])), dtype=complex).reshape(-1)
    for candidate in irreps:
        candidate_characters = np.asarray(_complex_array_from_pairs(candidate.get("characters", [])), dtype=complex).reshape(-1)
        if candidate_characters.shape == target_characters.shape and np.allclose(
            candidate_characters,
            np.conj(target_characters),
            atol=tol,
            rtol=0.0,
        ):
            return str(candidate.get("label", label))
    return str(label)


def _projected_vectors_from_basis(
    projected: Mapping[str, Any],
    *,
    basis_count: int,
    item_key: str,
) -> list[np.ndarray]:
    vectors = []
    for item in projected.get(item_key, []):
        vectors.append(_coefficient_vector_from_terms(item.get("linear_combination", []), basis_count))
    return vectors


def _kp_product_terms(
    coefficients: np.ndarray,
    *,
    hermitian_labels: Sequence[str],
    monomial_labels: Sequence[str],
    tol: float,
) -> list[dict[str, Any]]:
    n_q = len(monomial_labels)
    terms = []
    for flat_index, coefficient in enumerate(np.asarray(coefficients, dtype=complex).reshape(-1)):
        if abs(coefficient) <= tol:
            continue
        x_index = int(flat_index // n_q)
        q_index = int(flat_index % n_q)
        terms.append(
            {
                "hermitian_basis_index": int(x_index + 1),
                "hermitian_basis_label": str(hermitian_labels[x_index]),
                "k_basis_index": int(q_index + 1),
                "k_basis_label": str(monomial_labels[q_index]),
                "coefficient": _complex_scalar_to_pair(coefficient),
            }
        )
    return terms


def _real_basis_from_complex_columns(
    columns: np.ndarray,
    *,
    tol: float,
) -> tuple[np.ndarray, list[float]]:
    matrix = np.asarray(columns, dtype=complex)
    if matrix.size == 0:
        return np.zeros((matrix.shape[0] if matrix.ndim == 2 else 0, 0), dtype=float), []
    real_matrix = np.column_stack([matrix.real, matrix.imag])
    real_matrix[np.abs(real_matrix) < tol] = 0.0
    if real_matrix.size == 0:
        return np.zeros((matrix.shape[0], 0), dtype=float), []
    u_matrix, singular_values, _vh = np.linalg.svd(real_matrix, full_matrices=False)
    rank = int(np.count_nonzero(singular_values > 1.0e-7))
    basis = u_matrix[:, :rank].astype(float, copy=True)
    for column in range(basis.shape[1]):
        basis[:, column] = _canonicalize_real_vector(basis[:, column], tol)
    return basis, [float(value) for value in singular_values]


def _canonicalize_real_vector(vector: np.ndarray, tol: float) -> np.ndarray:
    out = np.asarray(vector, dtype=float).copy().reshape(-1)
    if out.size == 0:
        return out
    pivot = int(np.argmax(np.abs(out)))
    if abs(out[pivot]) <= tol:
        out[:] = 0.0
        return out
    if out[pivot] < 0:
        out *= -1.0
    out[np.abs(out) < tol] = 0.0
    return out


def _kp_form_solution_analyses(
    operator_analyses: Sequence[Mapping[str, Any]],
    k_polynomial_analyses: Sequence[Mapping[str, Any]],
    character_table: Mapping[str, Any],
    *,
    tol: float = 1.0e-8,
) -> list[dict[str, Any]]:
    if not operator_analyses or not k_polynomial_analyses:
        return []
    operator_analysis = operator_analyses[0]
    hermitian_labels = [str(item.get("label", "")) for item in operator_analysis.get("hermitian_basis", [])]
    operator_rep_matrices = [
        np.asarray(_complex_array_from_pairs(matrix), dtype=complex)
        for matrix in operator_analysis.get("representation_matrices", [])
    ]
    if not hermitian_labels or not operator_rep_matrices:
        return []

    operator_projected = {
        str(item.get("irrep_label", "")): item
        for item in operator_analysis.get("projected_basis", [])
    }
    solutions = []
    for k_analysis in k_polynomial_analyses:
        monomial_labels = [str(item.get("label", "")) for item in k_analysis.get("monomial_basis", [])]
        k_rep_matrices = [
            np.asarray(_complex_array_from_pairs(matrix), dtype=complex)
            for matrix in k_analysis.get("representation_matrices", [])
        ]
        if not monomial_labels or not k_rep_matrices or len(k_rep_matrices) != len(operator_rep_matrices):
            continue

        product_projector = np.zeros(
            (len(hermitian_labels) * len(monomial_labels), len(hermitian_labels) * len(monomial_labels)),
            dtype=complex,
        )
        for x_matrix, q_matrix in zip(operator_rep_matrices, k_rep_matrices):
            product_projector += np.kron(x_matrix, np.linalg.inv(q_matrix))
        product_projector /= float(len(k_rep_matrices))

        pair_solutions = []
        for k_projected in k_analysis.get("projected_basis", []):
            k_label = str(k_projected.get("irrep_label", ""))
            x_label = _conjugate_irrep_label(character_table, k_label)
            x_projected = operator_projected.get(x_label)
            if not x_projected:
                continue
            x_vectors = _projected_vectors_from_basis(
                x_projected,
                basis_count=len(hermitian_labels),
                item_key="matrices",
            )
            q_vectors = _projected_vectors_from_basis(
                k_projected,
                basis_count=len(monomial_labels),
                item_key="functions",
            )
            candidate_vectors = [
                np.kron(x_vector, q_vector)
                for x_vector in x_vectors
                for q_vector in q_vectors
                if np.linalg.norm(x_vector) > tol and np.linalg.norm(q_vector) > tol
            ]
            if not candidate_vectors:
                continue
            candidate_matrix = np.column_stack(candidate_vectors)
            projected_candidates = product_projector @ candidate_matrix
            real_basis, singular_values = _real_basis_from_complex_columns(projected_candidates, tol=tol)
            rank = int(real_basis.shape[1])
            terms = []
            for invariant_index in range(rank):
                coefficients = real_basis[:, invariant_index].astype(complex)
                terms.append(
                    {
                        "label": f"H{int(k_analysis.get('order', 0))}_{k_label}_{invariant_index + 1}",
                        "linear_combination": _kp_product_terms(
                            coefficients,
                            hermitian_labels=hermitian_labels,
                            monomial_labels=monomial_labels,
                            tol=tol,
                        ),
                    }
                )
            pair_solutions.append(
                {
                    "k_irrep_label": k_label,
                    "hermitian_irrep_label": x_label,
                    "rank": rank,
                    "candidate_count": int(candidate_matrix.shape[1]),
                    "singular_values": singular_values,
                    "terms": terms,
                }
            )
        solutions.append(
            {
                "order": int(k_analysis.get("order", 0)),
                "irrep_pair_solutions": pair_solutions,
            }
        )
    return solutions


def _pure_time_reversal_operation(spgrep_info: Mapping[str, Any], *, tol: float = 1.0e-8) -> Mapping[str, Any] | None:
    for operation in spgrep_info.get("antiunitary_operations", []):
        rotation = np.asarray(operation.get("rotation", np.eye(3)), dtype=float)
        translation = np.asarray(_vector3(operation.get("translation"), [0.0, 0.0, 0.0]), dtype=float)
        if not bool(operation.get("time_reversal", False)):
            continue
        if not np.allclose(rotation, np.eye(3), atol=tol, rtol=0.0):
            continue
        if not np.allclose(translation - np.rint(translation), 0.0, atol=tol, rtol=0.0):
            continue
        return operation
    return None


def _anti_linear_constraint_operation(spgrep_info: Mapping[str, Any]) -> Mapping[str, Any] | None:
    pure_time_reversal = _pure_time_reversal_operation(spgrep_info)
    if pure_time_reversal is not None:
        return pure_time_reversal
    operations = sorted(
        list(spgrep_info.get("antiunitary_operations", []) or []),
        key=lambda item: int(item.get("display_operation_index", item.get("antiunitary_operation_index", 0))),
    )
    return operations[0] if operations else None


def _anti_linear_display_operation_index(operation: Mapping[str, Any] | None) -> int:
    if not operation:
        return 0
    return int(
        operation.get(
            "display_operation_index",
            operation.get("spgrep_operation_index", operation.get("antiunitary_operation_index", 0)),
        )
    )


def _target_antiunitary_matrix_from_corep_indices(
    spgrep_info: Mapping[str, Any],
    corep_indices: Sequence[int],
    antiunitary_operation_index: int,
) -> np.ndarray | None:
    coreps = list(spgrep_info.get("coreps") or [])
    matrices = []
    for corep_index in corep_indices:
        corep = next(
            (
                item
                for item in coreps
                if int(item.get("corep_index", 0)) == int(corep_index)
            ),
            None,
        )
        if corep is None:
            return None
        operation = next(
            (
                item
                for item in corep.get("antiunitary_operation_matrices", [])
                if int(item.get("antiunitary_operation_index", 0)) == int(antiunitary_operation_index)
            ),
            None,
        )
        if operation is None:
            return None
        matrices.append(np.asarray(_complex_array_from_pairs(operation.get("matrix", [])), dtype=complex))
    if not matrices:
        return None
    return _block_diag_matrices(matrices)


def _target_antiunitary_reference_matrix_from_corep_indices(
    spgrep_info: Mapping[str, Any],
    corep_indices: Sequence[int],
) -> np.ndarray | None:
    coreps = list(spgrep_info.get("coreps") or [])
    matrices = []
    for corep_index in corep_indices:
        corep = next(
            (
                item
                for item in coreps
                if int(item.get("corep_index", 0)) == int(corep_index)
            ),
            None,
        )
        if corep is None:
            return None
        _operation_indices, unitary_matrices = _corep_operation_matrices(corep)
        if not unitary_matrices:
            return None
        matrices.append(
            _canonical_antiunitary_sewing_matrix(
                unitary_matrices,
                spin_orbit=bool(spgrep_info.get("spin_orbit", False)),
            )
        )
    if not matrices:
        return None
    return _block_diag_matrices(matrices)


def _target_antiunitary_matrix_for_operation(
    spgrep_info: Mapping[str, Any],
    corep_indices: Sequence[int],
    antiunitary_operation: Mapping[str, Any],
    standard_by_operation: Mapping[int, np.ndarray],
) -> np.ndarray | None:
    antiunitary_operation_index = int(antiunitary_operation.get("antiunitary_operation_index", 0))
    direct = _target_antiunitary_matrix_from_corep_indices(
        spgrep_info,
        corep_indices,
        antiunitary_operation_index,
    )
    uses_reference_relation = "reference_antiunitary_operation_index" in antiunitary_operation
    if direct is not None and not uses_reference_relation:
        return direct

    reference_index = int(
        antiunitary_operation.get(
            "reference_antiunitary_operation_index",
            antiunitary_operation_index,
        )
    )
    reference = _target_antiunitary_reference_matrix_from_corep_indices(
        spgrep_info,
        corep_indices,
    )
    if reference is None:
        reference = _target_antiunitary_matrix_from_corep_indices(
            spgrep_info,
            corep_indices,
            reference_index,
        )
    if reference is None:
        return None

    left_index = int(antiunitary_operation.get("left_unitary_operation_index", 1))
    left = standard_by_operation.get(left_index)
    if left is None:
        display_index = int(antiunitary_operation.get("left_unitary_display_operation_index", 0))
        if display_index:
            left = standard_by_operation.get(display_index)
    if left is None:
        return reference
    return np.asarray(left, dtype=complex) @ np.asarray(reference, dtype=complex)


def _antiunitary_numeric_matrices_from_row(
    info: Mapping[str, Any],
    result: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    tb: Any = None,
    symm_prec: float = 1.0e-6,
) -> dict[int, np.ndarray]:
    matrices_by_spgrep_index: dict[int, np.ndarray] = {}
    stored_indices = [int(index) for index in row.get("target_antiunitary_spgrep_operation_indices", []) or []]
    stored_matrices = [
        np.asarray(matrix, dtype=complex)
        for matrix in row.get("target_antiunitary_representation_matrices", []) or []
    ]
    for index, matrix in zip(stored_indices, stored_matrices, strict=False):
        matrices_by_spgrep_index[int(index)] = matrix
    if matrices_by_spgrep_index or tb is None:
        return matrices_by_spgrep_index

    time_reversal_matrix = Character._time_reversal_basis_matrix(tb)
    if time_reversal_matrix is None:
        return {}
    eigenvectors = row.get("target_eigenvectors_full")
    overlap = row.get("target_overlap")
    band_range = [int(value) for value in row.get("target_band_range", []) or []]
    if eigenvectors is None or overlap is None or len(band_range) != 2:
        return {}
    start = int(band_range[0]) - 1
    stop = int(band_range[1])
    target_subspace = np.asarray(eigenvectors, dtype=complex)[:, start:stop]
    overlap_matrix = np.asarray(overlap, dtype=complex)
    k_direct = np.asarray(_lowdin_reference_kpoint_from_result(result), dtype=float)
    for operation in info.get("spgrep_operations", {}).get("antiunitary_operations", []) or []:
        spgrep_operation_index = int(operation.get("spgrep_operation_index", 0))
        if spgrep_operation_index <= 0:
            continue
        try:
            dk_operation = _dk_operation_from_summary(info, operation)
            unitary_part = build_dk_matrix(
                tb,
                k_direct,
                dk_operation,
                map_tol=float(symm_prec),
            )
        except Exception:
            continue
        matrices_by_spgrep_index[spgrep_operation_index] = (
            target_subspace.conj().T
            @ overlap_matrix
            @ np.asarray(unitary_part, dtype=complex)
            @ np.asarray(time_reversal_matrix, dtype=complex)
            @ target_subspace.conj()
        )
    return matrices_by_spgrep_index


def _antiunitary_operator_representation_matrix(
    target_matrix: np.ndarray,
    basis: Sequence[np.ndarray],
) -> np.ndarray:
    target = np.asarray(target_matrix, dtype=complex)
    basis_arrays = [np.asarray(matrix, dtype=complex) for matrix in basis]
    basis_count = len(basis_arrays)
    representation = np.zeros((basis_count, basis_count), dtype=complex)
    for column, basis_matrix in enumerate(basis_arrays):
        transformed = target.conj().T @ basis_matrix.conj() @ target
        for row, projector in enumerate(basis_arrays):
            representation[row, column] = np.trace(projector @ transformed)
    return representation


def _kp_product_coefficient_matrix(
    terms: Sequence[Mapping[str, Any]],
    *,
    hermitian_basis_count: int,
    k_basis_count: int | None = None,
) -> np.ndarray:
    inferred_k_basis_count = 0
    for term in terms:
        inferred_k_basis_count = max(inferred_k_basis_count, int(term.get("k_basis_index", 0)))
    if k_basis_count is None:
        k_basis_count = inferred_k_basis_count
    coefficients = np.zeros((int(hermitian_basis_count), int(k_basis_count)), dtype=complex)
    for term in terms:
        x_index = int(term.get("hermitian_basis_index", 0)) - 1
        k_index = int(term.get("k_basis_index", 0)) - 1
        if 0 <= x_index < coefficients.shape[0] and 0 <= k_index < coefficients.shape[1]:
            coefficients[x_index, k_index] += complex(*term.get("coefficient", [0.0, 0.0]))
    return coefficients


def _time_reversal_constraint_analyses(
    info: Mapping[str, Any],
    operator_analyses: Sequence[Mapping[str, Any]],
    kp_form_solution_analyses: Sequence[Mapping[str, Any]],
    *,
    variable_labels: Sequence[str] = KP_DIRECTIONS,
    tol: float = 1.0e-7,
) -> list[dict[str, Any]]:
    spgrep_info = info.get("spgrep_operations", {})
    anti_linear_operation = _anti_linear_constraint_operation(spgrep_info)
    if anti_linear_operation is None or not operator_analyses or not kp_form_solution_analyses:
        return []

    operator_analysis = operator_analyses[0]
    corep_indices = [int(index) for index in operator_analysis.get("target_corep_indices", [])]
    antiunitary_operation_index = int(anti_linear_operation.get("antiunitary_operation_index", 0))
    operation_indices, target_matrices = _target_matrices_from_corep_indices(spgrep_info, corep_indices)
    standard_by_operation = {
        int(index): np.asarray(matrix, dtype=complex)
        for index, matrix in zip(operation_indices, target_matrices, strict=False)
    }
    target_matrix = _target_antiunitary_matrix_for_operation(
        spgrep_info,
        corep_indices,
        anti_linear_operation,
        standard_by_operation,
    )
    if target_matrix is None:
        return []

    dimension = int(operator_analysis.get("dimension", target_matrix.shape[0]))
    hermitian_labels, hermitian_basis = _hermitian_matrix_basis(dimension)
    if not hermitian_basis:
        return []
    tr_operator_matrix = _antiunitary_operator_representation_matrix(target_matrix, hermitian_basis)

    term_checks = []
    for analysis in kp_form_solution_analyses:
        order = int(analysis.get("order", 0))
        parity = -1 if order % 2 else 1
        monomial_labels = _monomial_labels_from_terms(
            [
                raw_term
                for pair in analysis.get("irrep_pair_solutions", [])
                for term in pair.get("terms", [])
                for raw_term in term.get("linear_combination", [])
            ]
        )
        if not monomial_labels:
            continue
        base_transform = _reciprocal_rotation_from_operation(
            anti_linear_operation,
            lattice=info.get("lattice_vectors"),
        )
        base_transform = _subtransform_for_k_labels(base_transform, variable_labels)
        function_transform = _polynomial_transform_matrix_for_labels(
            -base_transform,
            monomial_labels,
            variable_labels=variable_labels,
        )
        function_transform = np.linalg.inv(np.real_if_close(function_transform, tol=1000).real)
        for pair in analysis.get("irrep_pair_solutions", []):
            for term in pair.get("terms", []):
                coefficients = _kp_product_coefficient_matrix(
                    term.get("linear_combination", []),
                    hermitian_basis_count=len(hermitian_labels),
                    k_basis_count=len(monomial_labels),
                )
                if coefficients.size == 0:
                    continue
                transformed = tr_operator_matrix @ np.conj(coefficients) @ function_transform.T
                diff = transformed - coefficients
                max_abs_diff = float(np.max(np.abs(diff))) if diff.size else 0.0
                term_checks.append(
                    {
                        "label": str(term.get("label", "")),
                        "order": order,
                        "parity": int(parity),
                        "k_irrep_label": str(pair.get("k_irrep_label", "")),
                        "hermitian_irrep_label": str(pair.get("hermitian_irrep_label", "")),
                        "max_abs_diff": max_abs_diff,
                        "status": "ok" if max_abs_diff <= tol else "violation",
                    }
                )

    return [
        {
            "target_label": str(operator_analysis.get("target_label", "")),
            "target_corep_indices": corep_indices,
            "anti_linear_operator_index": _anti_linear_display_operation_index(anti_linear_operation),
            "antiunitary_operation_index": antiunitary_operation_index,
            "spgrep_operation_index": int(anti_linear_operation.get("spgrep_operation_index", 0)),
            "k_transform": "q -> -R_cart^{-1} q",
            "matrix_convention": "A(a)^dagger H(aq)^* A(a) = H(q)",
            "tolerance": float(tol),
            "term_checks": term_checks,
        }
    ]


def _monomial_labels_from_terms(terms: Sequence[Mapping[str, Any]]) -> list[str]:
    labels_by_index: dict[int, str] = {}
    for term in terms:
        index = int(term.get("k_basis_index", 0))
        if index > 0 and index not in labels_by_index:
            labels_by_index[index] = str(term.get("k_basis_label", ""))
    if not labels_by_index:
        return []
    return [labels_by_index.get(index, f"Q{index}") for index in range(1, max(labels_by_index) + 1)]


def _kp_product_vector_from_terms(
    terms: Sequence[Mapping[str, Any]],
    *,
    hermitian_basis_count: int,
    k_basis_count: int,
) -> np.ndarray:
    vector = np.zeros(int(hermitian_basis_count) * int(k_basis_count), dtype=complex)
    for term in terms:
        x_index = int(term.get("hermitian_basis_index", 0)) - 1
        k_index = int(term.get("k_basis_index", 0)) - 1
        if 0 <= x_index < int(hermitian_basis_count) and 0 <= k_index < int(k_basis_count):
            flat_index = x_index * int(k_basis_count) + k_index
            vector[flat_index] += complex(*term.get("coefficient", [0.0, 0.0]))
    return vector


def _real_tr_subspace(
    basis: np.ndarray,
    transform: np.ndarray,
    *,
    eigenvalue: float,
    tol: float,
) -> tuple[np.ndarray, list[float]]:
    real_basis = np.asarray(basis, dtype=float)
    if real_basis.size == 0 or real_basis.shape[1] == 0:
        return np.zeros((real_basis.shape[0] if real_basis.ndim == 2 else 0, 0), dtype=float), []
    real_transform = np.asarray(transform, dtype=float)
    representation = real_basis.T @ real_transform @ real_basis
    projector = 0.5 * (np.eye(representation.shape[0]) + float(eigenvalue) * representation)
    u_matrix, singular_values, _vh = np.linalg.svd(projector)
    rank = int(np.count_nonzero(singular_values > 1.0e-7))
    vectors = real_basis @ u_matrix[:, :rank]
    for column in range(vectors.shape[1]):
        vectors[:, column] = _canonicalize_real_vector(vectors[:, column], tol)
    return vectors, [float(value) for value in singular_values]


def _final_kp_model_analyses(
    info: Mapping[str, Any],
    operator_analyses: Sequence[Mapping[str, Any]],
    kp_form_solution_analyses: Sequence[Mapping[str, Any]],
    *,
    term_prefix: str = "K",
    variable_labels: Sequence[str] = ("kx", "ky", "kz"),
    function_transform_kind: str = "polar",
    tol: float = 1.0e-8,
) -> list[dict[str, Any]]:
    if not operator_analyses or not kp_form_solution_analyses:
        return []

    operator_analysis = operator_analyses[0]
    dimension = int(operator_analysis.get("dimension", 0))
    hermitian_labels = [str(item.get("label", "")) for item in operator_analysis.get("hermitian_basis", [])]
    if dimension <= 0:
        dimension = int(round(np.sqrt(max(len(hermitian_labels), 1))))
    canonical_labels, hermitian_basis = _hermitian_matrix_basis(dimension)
    if not hermitian_labels:
        hermitian_labels = canonical_labels
    hermitian_basis_count = len(hermitian_labels)
    if hermitian_basis_count == 0:
        return []

    spgrep_info = info.get("spgrep_operations", {})
    anti_linear_operation = _anti_linear_constraint_operation(spgrep_info)
    anti_operator_matrix = None
    antiunitary_operation_index = 0
    spgrep_operation_index = 0
    anti_linear_operator_index = 0
    if anti_linear_operation is not None:
        antiunitary_operation_index = int(anti_linear_operation.get("antiunitary_operation_index", 0))
        spgrep_operation_index = int(anti_linear_operation.get("spgrep_operation_index", 0))
        anti_linear_operator_index = _anti_linear_display_operation_index(anti_linear_operation)
        corep_indices = [int(index) for index in operator_analysis.get("target_corep_indices", [])]
        operation_indices, target_matrices = _target_matrices_from_corep_indices(spgrep_info, corep_indices)
        standard_by_operation = {
            int(index): np.asarray(matrix, dtype=complex)
            for index, matrix in zip(operation_indices, target_matrices, strict=False)
        }
        target_matrix = _target_antiunitary_matrix_for_operation(
            spgrep_info,
            corep_indices,
            anti_linear_operation,
            standard_by_operation,
        )
        if target_matrix is not None:
            anti_operator_matrix = _antiunitary_operator_representation_matrix(target_matrix, hermitian_basis)
            anti_operator_matrix = np.real_if_close(anti_operator_matrix, tol=1000).real

    analyses = []
    for analysis in kp_form_solution_analyses:
        order = int(analysis.get("order", 0))
        source_terms = [
            term
            for pair in analysis.get("irrep_pair_solutions", [])
            for term in pair.get("terms", [])
        ]
        if not source_terms:
            continue
        monomial_labels = _monomial_labels_from_terms(
            [
                raw_term
                for term in source_terms
                for raw_term in term.get("linear_combination", [])
            ]
        )
        if not monomial_labels:
            continue
        source_vectors = [
            _kp_product_vector_from_terms(
                term.get("linear_combination", []),
                hermitian_basis_count=hermitian_basis_count,
                k_basis_count=len(monomial_labels),
            )
            for term in source_terms
        ]
        real_basis, singular_values = _real_basis_from_complex_columns(np.column_stack(source_vectors), tol=tol)
        real_candidate_rank = int(real_basis.shape[1])
        if real_candidate_rank == 0:
            continue

        anti_linear_operation_applied = anti_operator_matrix is not None
        tr_even_rank = real_candidate_rank
        tr_odd_rank = 0
        final_basis = real_basis
        tr_even_singular_values: list[float] = []
        tr_odd_singular_values: list[float] = []
        if anti_linear_operation_applied:
            lattice = info.get("lattice_vectors")
            if str(function_transform_kind) == "pseudovector":
                base_transform = _zeeman_field_transform_from_operation(anti_linear_operation or {}, lattice=lattice)
            else:
                base_transform = _reciprocal_rotation_from_operation(anti_linear_operation or {}, lattice=lattice)
                base_transform = _subtransform_for_k_labels(base_transform, variable_labels)
            function_transform = _polynomial_transform_matrix_for_labels(
                -base_transform,
                monomial_labels,
                variable_labels=variable_labels,
            )
            tr_full = np.kron(
                anti_operator_matrix,
                np.linalg.inv(np.real_if_close(function_transform, tol=1000).real),
            )
            even_basis, tr_even_singular_values = _real_tr_subspace(
                real_basis,
                tr_full,
                eigenvalue=1.0,
                tol=tol,
            )
            odd_basis, tr_odd_singular_values = _real_tr_subspace(
                real_basis,
                tr_full,
                eigenvalue=-1.0,
                tol=tol,
            )
            final_basis = even_basis
            tr_even_rank = int(even_basis.shape[1])
            tr_odd_rank = int(odd_basis.shape[1])

        terms = []
        for index in range(final_basis.shape[1]):
            vector = _canonicalize_real_vector(final_basis[:, index], tol)
            terms.append(
                {
                    "label": f"{term_prefix}{order}_{index + 1}",
                    "linear_combination": _kp_product_terms(
                        vector.astype(complex),
                        hermitian_labels=hermitian_labels,
                        monomial_labels=monomial_labels,
                        tol=tol,
                    ),
                }
            )

        analyses.append(
            {
                "order": order,
                "target_label": str(operator_analysis.get("target_label", "")),
                "source_term_count": int(len(source_terms)),
                "real_candidate_rank": real_candidate_rank,
                "tr_even_rank": tr_even_rank,
                "tr_odd_rank": tr_odd_rank,
                "anti_linear_operation_applied": bool(anti_linear_operation_applied),
                "anti_linear_operator_index": anti_linear_operator_index,
                "time_reversal_applied": bool(anti_linear_operation_applied),
                "antiunitary_operation_index": antiunitary_operation_index,
                "spgrep_operation_index": spgrep_operation_index,
                "singular_values": singular_values,
                "tr_even_singular_values": tr_even_singular_values,
                "tr_odd_singular_values": tr_odd_singular_values,
                "terms": terms,
            }
        )
    return analyses


def _spgrep_operations_summary(
    *,
    lattice: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    time_reversals: np.ndarray | None,
    kpoint: Sequence[float],
    spin_orbit: bool,
    character_operations: Sequence[Any] | None = None,
) -> dict[str, Any]:
    try:
        import spgrep
    except ModuleNotFoundError as exc:
        raise ImportError("spgrep is required for KP spgrep operation output.") from exc

    def _operation_keeps_kpoint(rotation: np.ndarray, *, antiunitary: bool, tol: float = 1.0e-6) -> bool:
        return _symmetry_operation_keeps_kpoint(
            rotation,
            kpoint_array,
            antiunitary=antiunitary,
            tol=tol,
        )

    def _is_pure_time_reversal_source(source_index: int, tol: float = 1.0e-8) -> bool:
        operation = operation_by_source_index[source_index]
        rotation = np.asarray(operation["rotation"], dtype=float)
        translation = np.asarray(operation["translation"], dtype=float)
        return (
            bool(operation["time_reversal"])
            and np.allclose(rotation, np.eye(3), atol=tol, rtol=0.0)
            and np.allclose(translation - np.rint(translation), 0.0, atol=tol, rtol=0.0)
        )

    lattice = np.asarray(lattice, dtype=float)
    rotations = np.asarray(rotations, dtype=int)
    translations = np.asarray(translations, dtype=float)
    time_reversal_array = None if time_reversals is None else np.asarray(time_reversals, dtype=int)
    kpoint_array = np.asarray(kpoint, dtype=float)
    if time_reversal_array is not None and bool(spin_orbit):
        call = "get_spacegroup_spinor_irreps_from_primitive_symmetry(time_reversals=magnetic)"
        output = spgrep.get_spacegroup_spinor_irreps_from_primitive_symmetry(
            lattice,
            rotations,
            translations,
            time_reversals=time_reversal_array,
            kpoint=kpoint_array,
        )
        coreps = output[0]
        anti_linear = np.asarray(output[-2], dtype=bool)
        mapping_little_group = np.asarray(output[-1], dtype=int)
    elif spin_orbit:
        call = "get_spacegroup_spinor_irreps_from_primitive_symmetry"
        output = spgrep.get_spacegroup_spinor_irreps_from_primitive_symmetry(
            lattice,
            rotations,
            translations,
            kpoint=kpoint_array,
        )
        coreps = output[0]
        anti_linear = np.zeros(len(output[-1]), dtype=bool)
        mapping_little_group = np.asarray(output[-1], dtype=int)
    else:
        call = "get_spacegroup_irreps_from_primitive_symmetry"
        source_indices_for_irreps = np.arange(len(rotations), dtype=int)
        if time_reversal_array is not None:
            source_indices_for_irreps = np.asarray(
                [idx for idx, flag in enumerate(time_reversal_array) if not bool(flag)],
                dtype=int,
            )
            call = "get_spacegroup_irreps_from_primitive_symmetry(unitary subgroup)"
        irrep_rotations = rotations[source_indices_for_irreps]
        irrep_translations = translations[source_indices_for_irreps]
        coreps, mapping_little_group = spgrep.get_spacegroup_irreps_from_primitive_symmetry(
            irrep_rotations,
            irrep_translations,
            kpoint_array,
        )
        mapping_little_group = source_indices_for_irreps[np.asarray(mapping_little_group, dtype=int)]
        anti_linear = np.zeros(len(mapping_little_group), dtype=bool)

    little_group_indices = {int(index) for index in mapping_little_group}
    anti_linear_by_operation = {
        int(operation_index): bool(anti_linear[position])
        for position, operation_index in enumerate(mapping_little_group)
        if position < len(anti_linear)
    }
    operation_by_source_index = {}
    for index, (rotation, translation) in enumerate(zip(rotations, translations), start=1):
        zero_based_index = index - 1
        time_reversal_flag = bool(time_reversal_array[zero_based_index]) if time_reversal_array is not None else False
        operation_by_source_index[zero_based_index] = {
            "spgrep_operation_index": int(index),
            "rotation": np.asarray(rotation, dtype=int),
            "translation": np.asarray(translation, dtype=float),
            "time_reversal": time_reversal_flag,
            "anti_linear": bool(anti_linear_by_operation.get(zero_based_index, False)) or time_reversal_flag,
            "in_little_group": bool(zero_based_index in little_group_indices),
        }
    if time_reversal_array is not None:
        for source_index, operation in operation_by_source_index.items():
            if bool(operation["time_reversal"]):
                operation["in_little_group"] = _operation_keeps_kpoint(
                    np.asarray(operation["rotation"], dtype=float),
                    antiunitary=True,
                )
    unitary_source_indices = [
        source_index
        for source_index, operation in operation_by_source_index.items()
        if not bool(operation["time_reversal"]) and bool(operation["in_little_group"])
    ]
    all_unitary_source_indices = [
        source_index
        for source_index, operation in operation_by_source_index.items()
        if not bool(operation["time_reversal"])
    ]
    antiunitary_source_indices = [
        int(source_index)
        for source_index, operation in operation_by_source_index.items()
        if bool(operation["time_reversal"])
        and bool(operation["in_little_group"])
        and _operation_keeps_kpoint(np.asarray(operation["rotation"], dtype=float), antiunitary=True)
    ]
    all_antiunitary_source_indices = [
        source_index
        for source_index, operation in operation_by_source_index.items()
        if bool(operation["time_reversal"])
    ]
    sorted_source_indices = _sort_spgrep_unitary_operations_by_character_order(
        unitary_source_indices,
        operation_by_source_index,
        character_operations,
    )
    sorted_all_unitary_source_indices = _sort_spgrep_unitary_operations_by_character_order(
        all_unitary_source_indices,
        operation_by_source_index,
        character_operations,
    )
    source_to_display_index = {
        source_index: display_index
        for display_index, source_index in enumerate(sorted_source_indices, start=1)
    }
    all_source_to_display_index = {
        source_index: display_index
        for display_index, source_index in enumerate(sorted_all_unitary_source_indices, start=1)
    }
    antiunitary_source_to_display_index = {
        source_index: display_index
        for display_index, source_index in enumerate(antiunitary_source_indices, start=1)
    }
    all_antiunitary_source_to_display_index = {
        source_index: display_index
        for display_index, source_index in enumerate(all_antiunitary_source_indices, start=1)
    }
    reference_antiunitary_source_index = antiunitary_source_indices[0] if antiunitary_source_indices else None

    def _translations_match_mod_lattice(left: np.ndarray, right: np.ndarray, tol: float = 1.0e-6) -> bool:
        diff = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
        return bool(np.allclose(diff - np.rint(diff), 0.0, atol=tol, rtol=0.0))

    def _left_unitary_source_for_antiunitary(source_index: int) -> int | None:
        if reference_antiunitary_source_index is None:
            return None
        target = operation_by_source_index[int(source_index)]
        reference = operation_by_source_index[int(reference_antiunitary_source_index)]
        target_rotation = np.asarray(target["rotation"], dtype=int)
        target_translation = np.asarray(target["translation"], dtype=float)
        reference_rotation = np.asarray(reference["rotation"], dtype=int)
        reference_translation = np.asarray(reference["translation"], dtype=float)
        for left_source_index in sorted_source_indices:
            left = operation_by_source_index[int(left_source_index)]
            left_rotation = np.asarray(left["rotation"], dtype=int)
            left_translation = np.asarray(left["translation"], dtype=float)
            composed_rotation = left_rotation @ reference_rotation
            composed_translation = left_rotation @ reference_translation + left_translation
            if not np.array_equal(composed_rotation, target_rotation):
                continue
            if _translations_match_mod_lattice(composed_translation, target_translation):
                return int(left_source_index)
        return None

    all_operations = []
    for source_index in sorted_all_unitary_source_indices:
        operation = operation_by_source_index[source_index]
        all_operations.append(
            {
                "operation_index": int(all_source_to_display_index[source_index]),
                "spgrep_operation_index": int(operation["spgrep_operation_index"]),
                "rotation": np.asarray(operation["rotation"], dtype=int).tolist(),
                "translation": np.asarray(operation["translation"], dtype=float).tolist(),
                "time_reversal": bool(operation["time_reversal"]),
                "anti_linear": bool(operation["anti_linear"]),
                "in_little_group": bool(operation["in_little_group"]),
            }
        )
    all_antiunitary_operations = []
    all_unitary_count = len(all_operations)
    for source_index in all_antiunitary_source_indices:
        operation = operation_by_source_index[source_index]
        anti_index = int(all_antiunitary_source_to_display_index[source_index])
        all_antiunitary_operations.append(
            {
                "antiunitary_operation_index": anti_index,
                "operation_index": int(all_unitary_count + anti_index),
                "spgrep_operation_index": int(operation["spgrep_operation_index"]),
                "rotation": np.asarray(operation["rotation"], dtype=int).tolist(),
                "translation": np.asarray(operation["translation"], dtype=float).tolist(),
                "time_reversal": bool(operation["time_reversal"]),
                "anti_linear": bool(operation["anti_linear"]),
                "in_little_group": bool(operation["in_little_group"]),
            }
        )
    operations = []
    for source_index in sorted_source_indices:
        operation = operation_by_source_index[source_index]
        operations.append(
            {
                "operation_index": int(source_to_display_index[source_index]),
                "display_operation_index": int(all_source_to_display_index.get(source_index, source_to_display_index[source_index])),
                "spgrep_operation_index": int(operation["spgrep_operation_index"]),
                "rotation": np.asarray(operation["rotation"], dtype=int).tolist(),
                "translation": np.asarray(operation["translation"], dtype=float).tolist(),
                "time_reversal": bool(operation["time_reversal"]),
                "anti_linear": bool(operation["anti_linear"]),
                "in_little_group": bool(operation["in_little_group"]),
            }
        )
    antiunitary_operations = []
    for source_index in antiunitary_source_indices:
        operation = operation_by_source_index[source_index]
        left_unitary_source_index = _left_unitary_source_for_antiunitary(source_index)
        relation: dict[str, Any] = {}
        if reference_antiunitary_source_index is not None:
            relation["reference_antiunitary_operation_index"] = int(
                antiunitary_source_to_display_index[reference_antiunitary_source_index]
            )
        if left_unitary_source_index is not None:
            relation.update(
                {
                    "left_unitary_operation_index": int(source_to_display_index[left_unitary_source_index]),
                    "left_unitary_display_operation_index": int(
                        all_source_to_display_index.get(
                            left_unitary_source_index,
                            source_to_display_index[left_unitary_source_index],
                        )
                    ),
                    "left_unitary_spgrep_operation_index": int(
                        operation_by_source_index[left_unitary_source_index]["spgrep_operation_index"]
                    ),
                }
            )
        antiunitary_operations.append(
            {
                "antiunitary_operation_index": int(antiunitary_source_to_display_index[source_index]),
                "display_operation_index": int(
                    all_unitary_count + all_antiunitary_source_to_display_index.get(
                        source_index,
                        antiunitary_source_to_display_index[source_index],
                    )
                ),
                "spgrep_operation_index": int(operation["spgrep_operation_index"]),
                "rotation": np.asarray(operation["rotation"], dtype=int).tolist(),
                "translation": np.asarray(operation["translation"], dtype=float).tolist(),
                "time_reversal": bool(operation["time_reversal"]),
                "anti_linear": bool(operation["anti_linear"]),
                "in_little_group": bool(operation["in_little_group"]),
                **relation,
            }
        )
    corep_summaries = []
    for corep_index, corep in enumerate(coreps, start=1):
        corep_array = np.asarray(corep, dtype=complex)
        operation_matrices = []
        antiunitary_operation_matrices = []
        for position, operation_index in enumerate(mapping_little_group):
            source_index = int(operation_index)
            if source_index in source_to_display_index:
                operation_matrices.append(
                    {
                        "operation_index": int(source_to_display_index[source_index]),
                        "spgrep_operation_index": int(source_index) + 1,
                        "matrix": _complex_matrix_to_pairs(corep_array[position]),
                    }
                )
            if source_index in antiunitary_source_to_display_index:
                antiunitary_operation_matrices.append(
                    {
                        "antiunitary_operation_index": int(antiunitary_source_to_display_index[source_index]),
                        "spgrep_operation_index": int(source_index) + 1,
                        "matrix": _complex_matrix_to_pairs(corep_array[position]),
                    }
                )
        if time_reversal_array is not None and not bool(spin_orbit) and corep_array.size:
            dimension = int(corep_array.shape[-1])
            unitary_corep_matrices = [
                np.asarray(corep_array[position], dtype=complex)
                for position, source_index in enumerate(mapping_little_group)
                if int(source_index) in source_to_display_index
            ]
            try:
                time_reversal_matrix = _spinless_time_reversal_sewing_matrix(unitary_corep_matrices)
            except ValueError:
                time_reversal_matrix = np.eye(dimension, dtype=complex)
            for source_index in antiunitary_source_indices:
                if not _is_pure_time_reversal_source(source_index):
                    continue
                antiunitary_operation_matrices.append(
                    {
                        "antiunitary_operation_index": int(antiunitary_source_to_display_index[source_index]),
                        "spgrep_operation_index": int(source_index) + 1,
                        "matrix": _complex_matrix_to_pairs(time_reversal_matrix),
                    }
                )
        corep_summaries.append(
            {
                "corep_index": int(corep_index),
                "shape": list(corep_array.shape),
                "operation_matrices": sorted(operation_matrices, key=lambda item: int(item["operation_index"])),
                "antiunitary_operation_matrices": sorted(
                    antiunitary_operation_matrices,
                    key=lambda item: int(item["antiunitary_operation_index"]),
                ),
            }
        )

    return {
        "call": call,
        "spin_orbit": bool(spin_orbit),
        "coordinate_convention": "x' = R x + tau",
        "kpoint": kpoint_array.tolist(),
        "little_group_mapping_zero_based": sorted_source_indices,
        "little_group_mapping_one_based": [source_to_display_index[index] for index in sorted_source_indices],
        "spgrep_little_group_mapping_zero_based": mapping_little_group.tolist(),
        "spgrep_little_group_mapping_one_based": [int(index) + 1 for index in mapping_little_group],
        "corep_shapes": [list(np.asarray(corep).shape) for corep in coreps],
        "all_operations": all_operations,
        "all_antiunitary_operations": all_antiunitary_operations,
        "little_group_linear_operation_indices": [
            int(all_source_to_display_index[index])
            for index in sorted_source_indices
            if index in all_source_to_display_index
        ],
        "little_group_anti_linear_operation_indices": [
            int(all_unitary_count + all_antiunitary_source_to_display_index[index])
            for index in antiunitary_source_indices
            if index in all_antiunitary_source_to_display_index
        ],
        "operations": operations,
        "antiunitary_operations": antiunitary_operations,
        "coreps": corep_summaries,
    }


def _spglib_magnetic_dataset_summary(dataset, spacegroup_type) -> dict[str, Any]:
    time_reversals = np.asarray(dataset.time_reversals, dtype=bool)
    summary = {
        "uni_number": int(dataset.uni_number),
        "msg_type": int(dataset.msg_type),
        "hall_number": int(dataset.hall_number),
        "tensor_rank": int(dataset.tensor_rank),
        "n_operations": int(dataset.n_operations),
        "unitary_operation_count": int(np.count_nonzero(~time_reversals)),
        "antiunitary_operation_count": int(np.count_nonzero(time_reversals)),
        "equivalent_atoms": np.asarray(dataset.equivalent_atoms, dtype=int).tolist(),
    }
    if spacegroup_type is not None:
        summary.update(
            {
                "bns_number": str(spacegroup_type.bns_number),
                "og_number": str(spacegroup_type.og_number),
                "number": int(spacegroup_type.number),
                "type": int(spacegroup_type.type),
                "litvin_number": int(spacegroup_type.litvin_number),
            }
        )
    return summary


def _spglib_spacegroup_dataset_summary(dataset) -> dict[str, Any]:
    return {
        "number": int(dataset.number),
        "international": str(dataset.international),
        "hall_symbol": str(dataset.hall),
    }


def _magnetic_spacegroup_type_summary(spacegroup_type) -> dict[str, Any]:
    if spacegroup_type is None:
        return {}
    bns_number = str(spacegroup_type.bns_number)
    og_number = str(spacegroup_type.og_number)
    type_number = int(spacegroup_type.type)
    return {
        "uni_number": int(spacegroup_type.uni_number),
        "litvin_number": int(spacegroup_type.litvin_number),
        "type": type_number,
        "type_label": _magnetic_spacegroup_type_label(type_number),
        "bns_number": bns_number,
        "og_number": og_number,
        "bns_setting": _magnetic_spacegroup_setting(bns_number, "bns"),
        "og_setting": _magnetic_spacegroup_setting(og_number, "og"),
    }


def _spin_group_summary(lattice: np.ndarray, rotations: np.ndarray, translations: np.ndarray, time_reversals: np.ndarray) -> dict[str, Any]:
    lattice = np.asarray(lattice, dtype=float)
    col_lattice = lattice.T
    inv_col_lattice = np.linalg.inv(col_lattice)
    unitary_indices = [idx for idx, flag in enumerate(np.asarray(time_reversals, dtype=bool)) if not bool(flag)]
    operations = []
    for idx in unitary_indices:
        rotation = np.asarray(rotations[idx], dtype=int)
        cart_rotation = col_lattice @ rotation @ inv_col_lattice
        proper_cart = -cart_rotation if np.linalg.det(cart_rotation) < 0 else cart_rotation
        spin_matrix = spin_half_matrix_from_cartesian_rotation(proper_cart)
        operations.append(
            {
                "operation_index": int(idx + 1),
                "rotation": rotation.tolist(),
                "translation": np.asarray(translations[idx], dtype=float).tolist(),
                "spin_matrix": _complex_matrix_to_pairs(spin_matrix),
            }
        )
    return {
        "enabled": True,
        "operation_count": len(operations),
        "operations": operations,
    }


def _collect_kp_symmetry_info(
    character_payload: Mapping[str, Any],
    *,
    tb,
    symm_prec: float,
    mag_tag: int,
    mag: str | Sequence[float],
) -> dict[str, Any]:
    try:
        import spglib
        from ase.io import read as ase_read
    except ModuleNotFoundError as exc:
        raise ImportError("spglib and ase are required for KP symmetry information.") from exc

    active_stru_path = Path(character_payload["active_stru_path"]).resolve()
    active_hr_path = Path(character_payload["active_hr_path"]).resolve()
    active_sr_path = Path(character_payload["active_sr_path"]).resolve()
    atoms = ase_read(str(active_stru_path), format="abacus")
    lattice = np.asarray(atoms.cell.array, dtype=float)
    positions = np.asarray(atoms.get_scaled_positions(wrap=False), dtype=float)
    numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
    analysis_result = character_payload.get("analysis_result", {})
    magnetic_moments, moment_source = _kp_magnetic_moments(
        stru_path=active_stru_path,
        atom_count=len(atoms),
        mag_tag=int(mag_tag),
        mag=mag,
    )

    spacegroup_dataset = spglib.get_symmetry_dataset(
        (lattice, positions, numbers),
        symprec=float(symm_prec),
        _throw=True,
    )
    if spacegroup_dataset is None:
        raise ValueError("spglib failed to get KP space-group dataset.")

    magnetic_cell = (lattice, positions, numbers, magnetic_moments)
    magnetic_dataset = spglib.get_magnetic_symmetry_dataset(
        magnetic_cell,
        is_axial=True,
        symprec=float(symm_prec),
        mag_symprec=float(symm_prec),
    )
    if magnetic_dataset is None:
        raise ValueError("spglib failed to get KP magnetic symmetry dataset.")
    magnetic_type = spglib.get_magnetic_spacegroup_type(int(magnetic_dataset.uni_number))

    nspin = int(getattr(tb, "nspin", 1))
    time_reversals = np.asarray(magnetic_dataset.time_reversals, dtype=bool)
    info: dict[str, Any] = {
        "nspin": nspin,
        "spin_mode": "spinor" if nspin == 4 else "spinless",
        "active_stru_path": str(active_stru_path),
        "active_hr_path": str(active_hr_path),
        "active_sr_path": str(active_sr_path),
        "lattice_vectors": lattice.tolist(),
        "reciprocal_lattice_vectors": _reciprocal_lattice_vectors(lattice).tolist(),
        "magnetic_moment_source": moment_source,
        "magnetic_moments": magnetic_moments.tolist(),
        "analysis": {
            "detected_group": analysis_result.get("detected_group"),
            "resolved_group": analysis_result.get("resolved_group"),
            "operation_count": analysis_result.get("operation_count"),
            "magnetic": bool(analysis_result.get("magnetic", False)),
        },
        "spglib_spacegroup_dataset": _spglib_spacegroup_dataset_summary(spacegroup_dataset),
        "spglib_magnetic_dataset": _spglib_magnetic_dataset_summary(magnetic_dataset, magnetic_type),
        "magnetic_spacegroup_type": _magnetic_spacegroup_type_summary(magnetic_type),
        "character_table": _character_table_summary(analysis_result),
    }
    if nspin == 4:
        info["spin_group"] = _spin_group_summary(
            lattice,
            np.asarray(magnetic_dataset.rotations, dtype=int),
            np.asarray(magnetic_dataset.translations, dtype=float),
            time_reversals,
        )
    else:
        info["spin_group"] = {"enabled": False, "operation_count": 0, "operations": []}
    kpoint_records = list(analysis_result.get("kpoint_records") or [])
    if kpoint_records:
        spgrep_kpoint = _spgrep_reference_kpoint_from_record(kpoint_records[0])
    else:
        spgrep_kpoint = [0.0, 0.0, 0.0]
    info["spgrep_operations"] = _spgrep_operations_summary(
        lattice=lattice,
        rotations=np.asarray(magnetic_dataset.rotations, dtype=int),
        translations=np.asarray(magnetic_dataset.translations, dtype=float),
        time_reversals=np.asarray(magnetic_dataset.time_reversals, dtype=int),
        kpoint=spgrep_kpoint,
        spin_orbit=nspin == 4,
        character_operations=analysis_result.get("operations"),
    )
    return info


def _write_kp_symmetry_info(info: Mapping[str, Any], output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "kp_symmetry_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    lines = [
        "KP symmetry information",
        f"nspin = {info.get('nspin')}",
        f"spin_mode = {info.get('spin_mode')}",
        f"active_stru_path = {info.get('active_stru_path')}",
        f"active_hr_path = {info.get('active_hr_path')}",
        f"active_sr_path = {info.get('active_sr_path')}",
        f"magnetic_moment_source = {info.get('magnetic_moment_source')}",
    ]
    dataset = info.get("spglib_magnetic_dataset", {})
    for key in ("uni_number", "msg_type", "bns_number", "og_number", "number", "n_operations", "unitary_operation_count", "antiunitary_operation_count"):
        if key in dataset:
            lines.append(f"spglib_magnetic_dataset.{key} = {dataset[key]}")
    spin_group = info.get("spin_group", {})
    lines.append(f"spin_group.enabled = {int(bool(spin_group.get('enabled', False)))}")
    lines.append(f"spin_group.operation_count = {int(spin_group.get('operation_count', 0))}")
    lines.append(f"kpoint_count = {len(info.get('kpoint_records') or [])}")
    (output_path / "kp_symmetry_info.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    spgrep_info = info.get("spgrep_operations", {})
    spgrep_lines = _format_spgrep_character_style_operations(
        spgrep_info,
        lattice=info.get("lattice_vectors"),
        star_line=None,
    )
    spgrep_lines.extend(
        _format_spgrep_all_operations(
            spgrep_info,
            star_line="*" * 93 if spgrep_lines else None,
        )
    )
    spgrep_lines.extend(
        _format_spgrep_operations(
            spgrep_info,
            star_line="*" * 93 if spgrep_lines else None,
        )
    )
    spgrep_lines.extend(
        _format_spgrep_trace_tables(
            spgrep_info,
            character_table=info.get("character_table", {}),
            star_line="*" * 93,
        )
    )
    for target_label, corep_index in _selected_spgrep_irrep_targets(info):
        spgrep_lines.extend(
            _format_spgrep_irrep_matrices(
                spgrep_info,
                target_label=target_label,
                star_line="*" * 93,
                corep_index=corep_index,
                character_table=info.get("character_table", {}),
            )
        )
        spgrep_lines.extend(
            _format_spgrep_antiunitary_irrep_matrices(
                spgrep_info,
                target_label=target_label,
                star_line="*" * 93,
                corep_index=corep_index,
                character_table=info.get("character_table", {}),
            )
        )
    if spgrep_lines:
        (output_path / "spgrep_operations.txt").write_text("\n".join(spgrep_lines) + "\n", encoding="utf-8")
    _write_kp_model_header(info, output_path / "kp_model.txt")


def _selected_spgrep_irrep_targets(info: Mapping[str, Any]) -> list[tuple[str, int | None]]:
    targets: list[tuple[str, int | None]] = []
    seen: set[tuple[str, int | None]] = set()
    for key in ("representation_alignment_analyses", "operator_irrep_analyses"):
        for analysis in info.get(key, []) or []:
            labels = _split_irrep_terms(str(analysis.get("target_label", "")))
            corep_indices = [int(value) for value in analysis.get("target_corep_indices", []) or []]
            for index, label in enumerate(labels):
                corep_index = corep_indices[index] if index < len(corep_indices) else None
                item = (label, corep_index)
                if item in seen:
                    continue
                seen.add(item)
                targets.append(item)
    if targets:
        return targets

    spgrep_info = info.get("spgrep_operations", {})
    character_table = info.get("character_table", {})
    for irrep in character_table.get("irreps", []) or []:
        label = str(irrep.get("label", "")).strip()
        if not label:
            continue
        corep_index = _spgrep_corep_index_for_irrep(
            label,
            spgrep_info=spgrep_info,
            character_table=character_table,
        )
        item = (label, corep_index)
        if corep_index is None or item in seen:
            continue
        seen.add(item)
        targets.append(item)
    return targets


def _spgrep_all_operation_records(spgrep_info: Mapping[str, Any]) -> list[dict[str, Any]]:
    unitary_by_spgrep_index = {
        int(operation.get("spgrep_operation_index", operation.get("operation_index", 0))): operation
        for operation in spgrep_info.get("operations", [])
    }
    antiunitary_by_spgrep_index = {
        int(operation.get("spgrep_operation_index", 0)): operation
        for operation in spgrep_info.get("antiunitary_operations", [])
    }
    mapping = [
        int(index)
        for index in spgrep_info.get("spgrep_little_group_mapping_one_based", [])
    ]
    if not mapping:
        mapping = sorted(set(unitary_by_spgrep_index) | set(antiunitary_by_spgrep_index))

    records = []
    for all_operation_index, spgrep_operation_index in enumerate(mapping, start=1):
        if spgrep_operation_index in unitary_by_spgrep_index:
            operation = dict(unitary_by_spgrep_index[spgrep_operation_index])
            operation["kind"] = "unitary"
            operation["all_operation_index"] = all_operation_index
            records.append(operation)
        elif spgrep_operation_index in antiunitary_by_spgrep_index:
            operation = dict(antiunitary_by_spgrep_index[spgrep_operation_index])
            operation["kind"] = "antiunitary"
            operation["all_operation_index"] = all_operation_index
            records.append(operation)
    return records


def _format_spgrep_all_operations(spgrep_info: Mapping[str, Any], *, star_line: str | None) -> list[str]:
    if not spgrep_info:
        return []

    operation_records = _spgrep_all_operation_records(spgrep_info)
    if not operation_records:
        return []

    kpoint = _vector3(spgrep_info.get("kpoint"), [0.0, 0.0, 0.0])
    coordinate_convention = spgrep_info.get("coordinate_convention", "x' = R x + tau")
    mapping = [str(int(index)) for index in spgrep_info.get("spgrep_little_group_mapping_one_based", [])]
    if not mapping:
        mapping = [str(int(operation.get("spgrep_operation_index", 0))) for operation in operation_records]

    lines: list[str] = []
    if star_line is not None:
        lines.append(star_line)
    lines.extend(
        [
            "Spgrep all little-group symmetry operations",
            f"coordinate convention: {coordinate_convention}",
            f"spgrep call: {spgrep_info.get('call', 'unknown')}",
            f"kpoint: {_display_float(kpoint[0]):.6f} {_display_float(kpoint[1]):.6f} {_display_float(kpoint[2]):.6f}",
            f"spgrep little group mapping (1-based): {' '.join(mapping)}",
            "operation convention: unitary uses D; antiunitary uses D K",
        ]
    )
    for operation in operation_records:
        rotation = np.asarray(operation.get("rotation", np.eye(3, dtype=int)), dtype=int)
        translation = _vector3(operation.get("translation"), [0.0, 0.0, 0.0])
        kind = str(operation.get("kind", "unknown"))
        lines.extend(
            [
                f"operation {int(operation.get('all_operation_index', 0))}",
                f"kind: {kind}",
            ]
        )
        if kind == "antiunitary":
            lines.append(f"antiunitary operation index: {int(operation.get('antiunitary_operation_index', 0))}")
        else:
            lines.append(f"unitary operation index: {int(operation.get('operation_index', 0))}")
        lines.extend(
            [
                f"spgrep operation index: {int(operation.get('spgrep_operation_index', 0))}",
                "R =",
            ]
        )
        for row in rotation:
            lines.append(f"  {int(row[0]):4d}{int(row[1]):3d}{int(row[2]):3d}")
        lines.extend(
            [
                f"tau = {_display_float(translation[0]):.6f} {_display_float(translation[1]):.6f} {_display_float(translation[2]):.6f}",
                f"time reversal: {'yes' if bool(operation.get('time_reversal', False)) else 'no'}",
                f"anti-linear: {'yes' if bool(operation.get('anti_linear', False)) else 'no'}",
                f"in little group: {'yes' if bool(operation.get('in_little_group', False)) else 'no'}",
            ]
        )
    return lines


def _canonicalize_axis(axis: np.ndarray, tol: float = 1.0e-8) -> np.ndarray:
    out = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(out))
    if norm <= tol:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    out = out / norm
    for value in out:
        if abs(value) > tol:
            if value < 0.0:
                out = -out
            break
    out[np.abs(out) < tol] = 0.0
    return out


def _signed_rotation_angle(cart_rotation: np.ndarray, axis: np.ndarray) -> float:
    canonical_axis = _canonicalize_axis(axis)
    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(ref, canonical_axis))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    u_vec = ref - canonical_axis * float(np.dot(ref, canonical_axis))
    u_norm = float(np.linalg.norm(u_vec))
    if u_norm <= 1.0e-12:
        return 0.0
    u_vec /= u_norm
    v_vec = np.cross(canonical_axis, u_vec)
    rotated = np.asarray(cart_rotation, dtype=float) @ u_vec
    return float(np.arctan2(np.dot(rotated, v_vec), np.dot(rotated, u_vec)))


def _spgrep_operation_symbol_description(
    cart_rotation: np.ndarray,
    *,
    time_reversal: bool,
    tol: float = 1.0e-7,
) -> tuple[str, str, np.ndarray, float, str]:
    cart = np.asarray(cart_rotation, dtype=float)
    det = float(np.linalg.det(cart))
    proper_cart = -cart if det < 0.0 else cart
    axis, angle_rad, improper = axis_angle_from_cartesian_rotation(cart)
    axis = _canonicalize_axis(axis)
    angle_deg = float(np.degrees(angle_rad))
    if abs(angle_deg) <= tol:
        if improper:
            symbol = "I"
            description = "inverse op."
        else:
            symbol = "E"
            description = "unity op."
        angle_deg = 0.0
        sense = "clockwise"
    else:
        order = max(1, int(round(360.0 / angle_deg)))
        angle_deg = 360.0 / float(order)
        symbol = f"IC{order}" if improper else f"C{order}"
        description = f"{angle_deg:.0f}-degree rotation"
        if improper:
            description += " times inversion"
        signed_angle = _signed_rotation_angle(proper_cart, axis)
        sense = "clockwise" if signed_angle > 0.0 else "counterclockwise"
    if time_reversal:
        symbol = f"{symbol}*T"
        description = f"{description} times time reversal"
    return symbol, description, axis, angle_deg, sense


def _format_spgrep_character_style_operations(
    spgrep_info: Mapping[str, Any],
    *,
    lattice: Sequence[Sequence[float]] | np.ndarray | None = None,
    star_line: str | None,
) -> list[str]:
    if not spgrep_info:
        return []

    unitary_records = list(spgrep_info.get("all_operations") or spgrep_info.get("operations") or [])
    antiunitary_records = list(
        spgrep_info.get("all_antiunitary_operations") or spgrep_info.get("antiunitary_operations") or []
    )
    operation_records = [
        dict(operation, kind="unitary")
        for operation in sorted(
            unitary_records,
            key=lambda item: int(item.get("operation_index", 0)),
        )
    ]
    operation_records.extend(
        dict(operation, kind="antiunitary")
        for operation in sorted(
            antiunitary_records,
            key=lambda item: int(item.get("antiunitary_operation_index", 0)),
        )
    )
    if not operation_records:
        return []

    separator = "---------------------------------------------------------------------------------------"
    lines: list[str] = []
    if star_line is not None:
        lines.append(star_line)
    lines.append("Symmetry operations Pi={Ri|taui+tm}   note: defined in Symmetrized Stru")
    lines.append(f"Number of linear operations: {len(unitary_records)}")
    lines.append(f"Number of  anti-linear operations: {len(antiunitary_records)}")
    lines.append(separator)
    for display_index, operation in enumerate(operation_records, start=1):
        rotation = np.asarray(operation.get("rotation", np.eye(3, dtype=int)), dtype=int)
        translation = np.asarray(_vector3(operation.get("translation"), [0.0, 0.0, 0.0]), dtype=float)
        inverse_rotation = np.rint(np.linalg.inv(rotation)).astype(int)
        cart_rotation = _cartesian_rotation_from_operation(operation, lattice=lattice)
        cart_rotation[np.abs(cart_rotation) < 5.0e-12] = 0.0
        symbol, description, axis, angle_deg, sense = _spgrep_operation_symbol_description(
            cart_rotation,
            time_reversal=bool(operation.get("time_reversal", False)),
        )
        lines.append(f"{display_index} ({symbol}): {description}")
        lines.append(f"main axes: ({axis[0]:6.3f}, {axis[1]:6.3f}, {axis[2]:6.3f})")
        lines.append(f"time reversal: {'yes' if bool(operation.get('time_reversal', False)) else 'no'}")
        lines.append(f"anti-linear: {'yes' if bool(operation.get('anti_linear', False)) else 'no'}")
        lines.append(f"{sense} rotation degree: {angle_deg:.0f}-degree")
        lines.append("    Ri       taui   inv(Ri)   Ri(Cartesian coord)")
        lines.append(
            f" {int(rotation[0, 0]):2d} {int(rotation[0, 1]):2d} {int(rotation[0, 2]):2d}"
            f" {translation[0]:7.3f}"
            f" {int(inverse_rotation[0, 0]):3d} {int(inverse_rotation[0, 1]):2d} {int(inverse_rotation[0, 2]):2d}"
            f" {cart_rotation[0, 0]:7.3f}{cart_rotation[0, 1]:7.3f}{cart_rotation[0, 2]:7.3f}"
        )
        lines.append(
            f" {int(rotation[1, 0]):2d} {int(rotation[1, 1]):2d} {int(rotation[1, 2]):2d}"
            f" {translation[1]:7.3f}"
            f" {int(inverse_rotation[1, 0]):3d} {int(inverse_rotation[1, 1]):2d} {int(inverse_rotation[1, 2]):2d}"
            f" {cart_rotation[1, 0]:7.3f}{cart_rotation[1, 1]:7.3f}{cart_rotation[1, 2]:7.3f}"
        )
        lines.append(
            f" {int(rotation[2, 0]):2d} {int(rotation[2, 1]):2d} {int(rotation[2, 2]):2d}"
            f" {translation[2]:7.3f}"
            f" {int(inverse_rotation[2, 0]):3d} {int(inverse_rotation[2, 1]):2d} {int(inverse_rotation[2, 2]):2d}"
            f" {cart_rotation[2, 0]:7.3f}{cart_rotation[2, 1]:7.3f}{cart_rotation[2, 2]:7.3f}"
        )
        lines.append(separator)
    return lines


def _format_spgrep_operations(spgrep_info: Mapping[str, Any], *, star_line: str | None) -> list[str]:
    if not spgrep_info:
        return []

    kpoint = _vector3(spgrep_info.get("kpoint"), [0.0, 0.0, 0.0])
    coordinate_convention = spgrep_info.get("coordinate_convention", "x' = R x + tau")
    mapping = [str(int(index)) for index in spgrep_info.get("little_group_mapping_one_based", [])]
    shape_text = ", ".join(
        "(" + ", ".join(str(int(dim)) for dim in shape) + ")"
        for shape in spgrep_info.get("corep_shapes", spgrep_info.get("irrep_shapes", []))
    )
    is_magnetic = "time_reversals" in str(spgrep_info.get("call", ""))
    lines: list[str] = []
    if star_line is not None:
        lines.append(star_line)
    lines.extend(
        [
            "Spgrep magnetic unitary symmetry operations" if is_magnetic else "Spgrep symmetry operations",
            f"coordinate convention: {coordinate_convention}",
            f"spgrep call: {spgrep_info.get('call', 'unknown')}",
            f"kpoint: {_display_float(kpoint[0]):.6f} {_display_float(kpoint[1]):.6f} {_display_float(kpoint[2]):.6f}",
            f"little group mapping (1-based): {' '.join(mapping)}",
            f"{'corep' if is_magnetic else 'irrep'} shapes: {shape_text}",
        ]
    )
    for operation in spgrep_info.get("operations", []):
        rotation = np.asarray(operation.get("rotation", np.eye(3, dtype=int)), dtype=int)
        translation = _vector3(operation.get("translation"), [0.0, 0.0, 0.0])
        lines.extend(
            [
                f"operation {int(operation.get('operation_index', 0))}",
                "R =",
            ]
        )
        for row in rotation:
            lines.append(f"  {int(row[0]):4d}{int(row[1]):3d}{int(row[2]):3d}")
        lines.extend(
            [
                f"tau = {_display_float(translation[0]):.6f} {_display_float(translation[1]):.6f} {_display_float(translation[2]):.6f}",
                f"time reversal: {'yes' if bool(operation.get('time_reversal', False)) else 'no'}",
                f"anti-linear: {'yes' if bool(operation.get('anti_linear', False)) else 'no'}",
                f"in little group: {'yes' if bool(operation.get('in_little_group', False)) else 'no'}",
            ]
        )
    antiunitary_operations = list(spgrep_info.get("antiunitary_operations") or [])
    if antiunitary_operations:
        if star_line is not None:
            lines.append(star_line)
        lines.append("Spgrep antiunitary time-reversal symmetry operations")
        for operation in antiunitary_operations:
            rotation = np.asarray(operation.get("rotation", np.eye(3, dtype=int)), dtype=int)
            translation = _vector3(operation.get("translation"), [0.0, 0.0, 0.0])
            lines.extend(
                [
                    f"antiunitary operation {int(operation.get('antiunitary_operation_index', 0))}",
                    f"spgrep operation index: {int(operation.get('spgrep_operation_index', 0))}",
                    "R =",
                ]
            )
            for row in rotation:
                lines.append(f"  {int(row[0]):4d}{int(row[1]):3d}{int(row[2]):3d}")
            lines.extend(
                [
                    f"tau = {_display_float(translation[0]):.6f} {_display_float(translation[1]):.6f} {_display_float(translation[2]):.6f}",
                    f"time reversal: {'yes' if bool(operation.get('time_reversal', False)) else 'no'}",
                    f"anti-linear: {'yes' if bool(operation.get('anti_linear', False)) else 'no'}",
                    f"in little group: {'yes' if bool(operation.get('in_little_group', False)) else 'no'}",
                ]
            )
    return lines


def _spgrep_corep_trace_rows(spgrep_info: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for corep in spgrep_info.get("coreps", []):
        operation_matrices = sorted(
            list(corep.get("operation_matrices") or []),
            key=lambda item: int(item.get("operation_index", 0)),
        )
        traces = []
        dimension = 0
        for item in operation_matrices:
            matrix = np.asarray(_complex_array_from_pairs(item.get("matrix", [])), dtype=complex)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                continue
            dimension = int(matrix.shape[0])
            traces.append(
                {
                    "operation_index": int(item.get("operation_index", 0)),
                    "trace": _complex_scalar_to_pair(np.trace(matrix)),
                }
            )
        if traces:
            rows.append(
                {
                    "corep_index": int(corep.get("corep_index", 0)),
                    "dimension": dimension,
                    "traces": traces,
                }
            )
    return rows


def _spgrep_trace_character_checks(
    spgrep_info: Mapping[str, Any],
    character_table: Mapping[str, Any],
    *,
    tol: float = 1.0e-6,
) -> list[dict[str, Any]]:
    trace_rows = {
        int(row.get("corep_index", 0)): row
        for row in _spgrep_corep_trace_rows(spgrep_info)
    }
    checks = []
    for irrep in character_table.get("irreps", []):
        label = str(irrep.get("label", "")).strip()
        corep_index = _spgrep_corep_index_for_irrep(
            label,
            spgrep_info=spgrep_info,
            character_table=character_table,
        )
        if corep_index is None or corep_index not in trace_rows:
            continue
        spgrep_traces = list(trace_rows[corep_index].get("traces", []))
        character_values = np.asarray(_complex_array_from_pairs(irrep.get("characters", [])), dtype=complex).reshape(-1)
        pair_count = min(len(spgrep_traces), int(character_values.size))
        entries = []
        max_abs_diff = 0.0
        for index in range(pair_count):
            spgrep_trace = complex(*spgrep_traces[index].get("trace", [0.0, 0.0]))
            character_value = complex(character_values[index])
            diff = spgrep_trace - character_value
            max_abs_diff = max(max_abs_diff, float(abs(diff)))
            entries.append(
                {
                    "operation_index": int(spgrep_traces[index].get("operation_index", index + 1)),
                    "spgrep_trace": _complex_scalar_to_pair(spgrep_trace),
                    "character": _complex_scalar_to_pair(character_value),
                    "diff": _complex_scalar_to_pair(diff),
                }
            )
        if len(spgrep_traces) != int(character_values.size):
            max_abs_diff = float("inf")
        checks.append(
            {
                "label": label,
                "corep_index": int(corep_index),
                "status": "ok" if max_abs_diff <= tol else "mismatch",
                "max_abs_diff": max_abs_diff,
                "entries": entries,
            }
        )
    return checks


def _format_spgrep_trace_tables(
    spgrep_info: Mapping[str, Any],
    *,
    character_table: Mapping[str, Any],
    star_line: str | None,
) -> list[str]:
    trace_rows = _spgrep_corep_trace_rows(spgrep_info)
    if not trace_rows:
        return []

    operation_indices = [
        int(operation.get("operation_index", 0))
        for operation in spgrep_info.get("operations", [])
    ]
    if not operation_indices:
        operation_indices = [
            int(item.get("operation_index", 0))
            for item in trace_rows[0].get("traces", [])
        ]

    lines: list[str] = []
    if star_line is not None:
        lines.append(star_line)
    lines.extend(
        [
            "Spgrep little-group representation trace table",
            "note: spgrep provides representation matrices; this table is generated from trace(D_g)",
            "operations: " + " ".join(str(index) for index in operation_indices),
        ]
    )
    for row in trace_rows:
        trace_text = " ".join(
            _format_character(item.get("trace", [0.0, 0.0]))
            for item in row.get("traces", [])
        )
        lines.append(
            f"corep {int(row.get('corep_index', 0))} dim={int(row.get('dimension', 0))} traces: {trace_text}"
        )

    checks = _spgrep_trace_character_checks(spgrep_info, character_table)
    if checks:
        if star_line is not None:
            lines.append(star_line)
        lines.append("Spgrep trace check against CHARACTER table")
        for check in checks:
            max_abs_diff = float(check.get("max_abs_diff", float("inf")))
            max_abs_text = f"{max_abs_diff:.6e}" if np.isfinite(max_abs_diff) else "inf"
            lines.append(
                f"{check.get('label', '')} -> corep {int(check.get('corep_index', 0))} "
                f"max_abs_diff = {max_abs_text} status = {check.get('status', 'unknown')}"
            )
            for entry in check.get("entries", []):
                lines.append(
                    f"op {int(entry.get('operation_index', 0))} "
                    f"spgrep = {_format_character(entry.get('spgrep_trace', [0.0, 0.0]))} "
                    f"character = {_format_character(entry.get('character', [0.0, 0.0]))} "
                    f"diff = {_format_character(entry.get('diff', [0.0, 0.0]))}"
                )
    return lines


def _format_complex_pair(value: Sequence[float]) -> str:
    real = _display_float(value[0])
    imag = _display_float(value[1])
    return f"({real: .6f}{imag:+.6f}i)"


def _format_complex_pair_3(value: Sequence[float]) -> str:
    real = float(value[0])
    imag = float(value[1])
    if abs(real) < 5.0e-4:
        real = 0.0
    if abs(imag) < 5.0e-4:
        imag = 0.0
    return f"({real:8.3f}{imag:+8.3f}i)"


def _format_character(value: Sequence[float]) -> str:
    real = _display_float(value[0])
    imag = _display_float(value[1])
    if abs(imag) < 5.0e-12:
        return f"{real: .6f}"
    return f"{real: .6f}{imag:+.6f}i"


def _format_real_symbolic(value: float, tol: float = 2.0e-6) -> str:
    number = 0.0 if abs(float(value)) < tol else float(value)
    sign = "-" if number < 0 else ""
    magnitude = abs(number)
    candidates = [
        (0.0, "0"),
        (1.0, "1"),
        (2.0, "2"),
        (3.0, "3"),
        (0.5, "1/2"),
        (1.0 / 3.0, "1/3"),
        (1.0 / np.sqrt(2.0), "1/sqrt(2)"),
        (1.0 / np.sqrt(3.0), "1/sqrt(3)"),
        (1.0 / np.sqrt(6.0), "1/sqrt(6)"),
        (np.sqrt(2.0) / 4.0, "sqrt(2)/4"),
        (np.sqrt(3.0) / 6.0, "sqrt(3)/6"),
        (1.0 / np.sqrt(20.0), "1/sqrt(20)"),
        (3.0 / np.sqrt(20.0), "3/sqrt(20)"),
        (np.sqrt(2.0 / 3.0), "sqrt(2/3)"),
        (np.sqrt(3.0) / 2.0, "sqrt(3)/2"),
    ]
    for candidate, text in candidates:
        if abs(magnitude - candidate) <= tol:
            if text == "0":
                return "0"
            return f"{sign}{text}"
    return f"{number:.6f}"


def _format_complex_symbolic(value: Sequence[float]) -> str:
    real = _display_float(value[0])
    imag = _display_float(value[1])
    if abs(imag) < 2.0e-6:
        return _format_real_symbolic(real)
    if abs(real) < 2.0e-6:
        if abs(imag - 1.0) < 2.0e-6:
            return "i"
        if abs(imag + 1.0) < 2.0e-6:
            return "-i"
        return f"{_format_real_symbolic(imag)}i"
    sign = "+" if imag >= 0 else "-"
    return f"{_format_real_symbolic(real)}{sign}{_format_real_symbolic(abs(imag))}i"


def _format_character_table(character_table: Mapping[str, Any], *, star_line: str | None) -> list[str]:
    irreps = list(character_table.get("irreps") or [])
    if not irreps:
        return []
    lines: list[str] = []
    if star_line is not None:
        lines.append(star_line)
    display_ops = [str(int(index)) for index in character_table.get("display_operation_indices", [])]
    lines.extend(
        [
            "Character table used for Hermitian matrix decomposition",
            f"k-point name: {character_table.get('k_name', 'unknown')}",
            f"operations: {' '.join(display_ops)}",
        ]
    )
    for irrep in irreps:
        kind = "double-valued" if bool(irrep.get("double_valued", False)) else "single-valued"
        characters = np.asarray(_complex_array_from_pairs(irrep.get("characters", [])), dtype=complex).reshape(-1)
        char_text = " ".join(_format_character(_complex_scalar_to_pair(value)) for value in characters)
        lines.append(
            f"{str(irrep.get('label', '')):<8s} dim={int(irrep.get('dimension', 0)):<2d} {kind:<13s} {char_text}"
        )
    return lines


def _format_decomposition(decomposition: Sequence[Mapping[str, Any]]) -> str:
    terms = []
    for item in decomposition:
        label = str(item.get("label", ""))
        multiplicity = item.get("multiplicity", 0)
        if isinstance(multiplicity, int) or (
            isinstance(multiplicity, float) and abs(multiplicity - round(multiplicity)) < 1.0e-8
        ):
            count = int(round(float(multiplicity)))
            if count == 1:
                terms.append(label)
            else:
                terms.append(f"{count}{label}")
        else:
            terms.append(f"({float(multiplicity):.6f}){label}")
    return " + ".join(terms) if terms else "unresolved"


def _format_linear_combination(terms: Sequence[Mapping[str, Any]]) -> str:
    if not terms:
        return "0"
    pieces = []
    for term in terms:
        coefficient = _format_complex_symbolic(term.get("coefficient", [0.0, 0.0]))
        pieces.append(f"{coefficient}*{term.get('basis_label', '')}")
    return " + ".join(pieces)


def _format_hermitian_basis_definition(label: str) -> str:
    text = str(label)
    if text.startswith("diag(") and text.endswith(")"):
        return text

    for prefix, expression in (
        ("re(", "1/sqrt(2) * (E_{i}{j} + E_{j}{i})"),
        ("im(", "1/sqrt(2) * (-iE_{i}{j} + iE_{j}{i})"),
    ):
        if not text.startswith(prefix) or not text.endswith(")"):
            continue
        parts = [part.strip() for part in text[len(prefix) : -1].split(",")]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            return text
        return f"{text} = {expression.format(i=parts[0], j=parts[1])}"

    return text


def _format_matrix_pairs(matrix_pairs: Sequence[Sequence[Sequence[float]]]) -> list[str]:
    lines = []
    for row in matrix_pairs:
        entries = " ".join(_format_complex_pair(entry) for entry in row)
        lines.append(f"  {entries}")
    return lines


def _format_matrix_pairs_3(matrix_pairs: Sequence[Sequence[Sequence[float]]]) -> list[str]:
    lines = []
    for row in matrix_pairs:
        entries = " ".join(_format_complex_pair_3(entry) for entry in row)
        lines.append(f"  {entries}")
    return lines


def _format_matrix_pair_comparison_3(
    numeric_matrix: Sequence[Sequence[Sequence[float]]],
    fitted_matrix: Sequence[Sequence[Sequence[float]]],
    *,
    title: str,
) -> list[str]:
    numeric_lines = [line.strip() for line in _format_matrix_pairs_3(numeric_matrix)]
    fitted_lines = [line.strip() for line in _format_matrix_pairs_3(fitted_matrix)]
    if not numeric_lines and not fitted_lines:
        return []

    row_count = max(len(numeric_lines), len(fitted_lines))
    numeric_lines.extend([""] * (row_count - len(numeric_lines)))
    fitted_lines.extend([""] * (row_count - len(fitted_lines)))
    left_title = "Numerical result:"
    right_title = "Fitted result:"
    left_width = max([len(left_title), *(len(line) for line in numeric_lines)]) + 4
    right_width = max([len(right_title), *(len(line) for line in fitted_lines)])
    total_width = left_width + 5 + right_width

    lines = [title, f"{left_title:^{left_width}}     {right_title:^{right_width}}"]
    lines.extend(
        f"{left:<{left_width}}     {right}"
        for left, right in zip(numeric_lines, fitted_lines, strict=False)
    )
    return lines


def _format_kpoint_record_lines(
    records: Sequence[Mapping[str, Any]],
    *,
    star_line: str | None,
    spgrep_info: Mapping[str, Any] | None = None,
) -> list[str]:
    spgrep = spgrep_info or {}
    little_linear_indices = [int(index) for index in spgrep.get("little_group_linear_operation_indices", []) or []]
    little_anti_linear_indices = [
        int(index) for index in spgrep.get("little_group_anti_linear_operation_indices", []) or []
    ]
    if little_linear_indices:
        linear_indices = " ".join(str(index) for index in little_linear_indices)
    else:
        linear_count = len(spgrep.get("operations") or [])
        linear_indices = " ".join(str(index) for index in range(1, linear_count + 1))
    if little_anti_linear_indices:
        anti_linear_indices = " ".join(str(index) for index in little_anti_linear_indices)
    else:
        linear_count = len(spgrep.get("operations") or [])
        anti_linear_count = len(spgrep.get("antiunitary_operations") or [])
        anti_linear_indices = " ".join(str(linear_count + index) for index in range(1, anti_linear_count + 1))
    lines: list[str] = []
    for record in records:
        kpoint = _vector3(record.get("k_direct"), [0.0, 0.0, 0.0])
        primitive_basis = _vector3(record.get("primitive_basis"), kpoint)
        conventional_basis = _vector3(record.get("conventional_basis"), kpoint)
        band_range = list(record.get("band_range") or [])
        if len(band_range) < 2:
            band_range = ["unknown", "unknown"]
        if star_line is not None:
            lines.append(star_line)
        lines.extend(
            [
                f"knum = {int(record.get('k_index', 1)):2d}    k = {_display_float(kpoint[0]):.6f} {_display_float(kpoint[1]):.6f} {_display_float(kpoint[2]):.6f}",
                f"The k-point name is {str(record.get('k_name', '')).strip():<3s}",
                str(record.get("star_transform", "The k-point is transformed by Identity operation to k-star")),
                f"Primitive    basis  {_display_float(primitive_basis[0]): .6f} {_display_float(primitive_basis[1]): .6f} {_display_float(primitive_basis[2]): .6f}",
                f"Conventional basis  {_display_float(conventional_basis[0]): .6f} {_display_float(conventional_basis[1]): .6f} {_display_float(conventional_basis[2]): .6f}",
            ]
        )
        if spgrep_info is not None:
            lines.append(f"Existence of linear operations: {linear_indices} ")
            lines.append(f"Existence of non-linear operations: {anti_linear_indices} ")
        lines.extend(
            [
                f"Band range : {band_range[0]}-{band_range[1]}",
                f"Band rep:    {record.get('band_rep', '')}",
            ]
        )
    return lines


def _representation_block_labels(analysis: Mapping[str, Any]) -> list[str]:
    labels = [str(label) for label in analysis.get("target_corep_labels", []) or []]
    dimensions = list(analysis.get("target_corep_dimensions", []) or [])
    if len(labels) == len(dimensions):
        return labels
    labels = _split_irrep_terms(str(analysis.get("target_label", "")))
    if len(labels) == len(dimensions):
        return labels
    return [f"block{index + 1}" for index in range(len(dimensions))]


def _representation_matrix_block_width(dimension: int) -> int:
    entry_width = len(_format_complex_pair_3([0.0, 0.0]))
    return int(dimension) * entry_width + max(0, int(dimension) - 1)


def _format_representation_block_header(labels: Sequence[str], dimensions: Sequence[int]) -> str:
    widths = [_representation_matrix_block_width(int(value)) for value in dimensions]
    if len(widths) != len(labels):
        widths = [max(len(str(label)), 31) for label in labels]
    return "  " + (" " * 12).join(
        f"{str(label):^{width}}"
        for label, width in zip(labels, widths, strict=True)
    ).rstrip()


def _format_block_matrix_rows(
    matrix_pairs: Sequence[Sequence[Sequence[float]]],
    dimensions: Sequence[int],
) -> list[str]:
    matrix = np.asarray(_complex_array_from_pairs(matrix_pairs), dtype=complex)
    dims = [int(value) for value in dimensions]
    if not dims or matrix.ndim != 2 or sum(dims) != matrix.shape[0] or matrix.shape[0] != matrix.shape[1]:
        return _format_matrix_pairs(matrix_pairs)

    starts = np.cumsum([0] + dims[:-1]).astype(int).tolist()
    widths = [_representation_matrix_block_width(dimension) for dimension in dims]
    max_dimension = max(dims)
    lines: list[str] = []
    for row in range(max_dimension):
        blocks = []
        for start, dimension, width in zip(starts, dims, widths, strict=True):
            if row >= dimension:
                blocks.append(" " * width)
                continue
            entries = " ".join(
                _format_complex_pair_3(_complex_scalar_to_pair(matrix[start + row, start + column]))
                for column in range(dimension)
            )
            blocks.append(f"{entries:<{width}s}")
        lines.append("  " + (" " * 12).join(blocks).rstrip())
    return lines


def _phase_matrix_from_refinement(
    phase_refinement: Mapping[str, Any] | None,
    dimensions: Sequence[int],
) -> np.ndarray | None:
    if not isinstance(phase_refinement, Mapping):
        return None
    phase_matrix_raw = phase_refinement.get("phase_matrix")
    if phase_matrix_raw is not None:
        phase_matrix = np.asarray(_complex_array_from_pairs(phase_matrix_raw), dtype=complex)
        if phase_matrix.ndim == 2 and phase_matrix.shape[0] == phase_matrix.shape[1]:
            return phase_matrix

    phases = list(phase_refinement.get("block_phases_rad", []) or [])
    dims = [int(value) for value in dimensions]
    if len(phases) != len(dims):
        return None
    diagonal = [
        np.exp(1.0j * float(phase))
        for phase, dimension in zip(phases, dims, strict=True)
        for _ in range(dimension)
    ]
    return np.diag(np.asarray(diagonal, dtype=complex))


def _format_schur_basis_transformation(
    analysis: Mapping[str, Any],
    labels: Sequence[str],
    dimensions: Sequence[int],
    *,
    separator: str,
) -> list[str]:
    alignment = analysis.get("schur_alignment", {})
    unitary_pairs = alignment.get("unitary_numeric_to_standard", [])
    if not unitary_pairs:
        return []

    unitary = np.asarray(_complex_array_from_pairs(unitary_pairs), dtype=complex)
    if unitary.ndim != 2 or unitary.shape[0] != unitary.shape[1]:
        return []

    phase_refinement = alignment.get("antiunitary_phase_refinement")
    phase_matrix = _phase_matrix_from_refinement(phase_refinement, dimensions)
    if phase_matrix is not None and phase_matrix.shape == unitary.shape:
        u0 = unitary @ phase_matrix.conj().T
    else:
        u0 = unitary

    lines = [
        "Basis transformation from Numerical basis to Symmetrized basis",
        "Through Schur Lemma : D_symm(g) ~= U^dagger D_num(g) U",
    ]
    if phase_matrix is not None and phase_matrix.shape == unitary.shape:
        phase_text = ", ".join(
            f"e^{{i alpha_{label}}} I_{int(dimension)}"
            for label, dimension in zip(labels, dimensions, strict=True)
        )
        lines.extend(
            [
                "U = U0 P_alpha",
                f"P_alpha = diag({phase_text})",
            ]
        )
    else:
        lines.append("U = U0")
    antiunitary_indices = list(analysis.get("antiunitary_operation_indices", []) or [])
    numeric_antiunitary = list(analysis.get("numeric_antiunitary_representation_matrices", []) or [])
    standard_antiunitary = list(analysis.get("spgrep_antiunitary_representation_matrices", []) or [])
    transformed_antiunitary = list(alignment.get("transformed_numeric_antiunitary_matrices", []) or [])
    if antiunitary_indices and numeric_antiunitary and standard_antiunitary:
        display_indices = [int(index) for index in analysis.get("antiunitary_display_operation_indices", []) or []]
        if display_indices:
            display_index = display_indices[0]
        else:
            operation_indices = [int(index) for index in analysis.get("operation_indices", []) or []]
            display_index = (max(operation_indices) if operation_indices else 0) + int(antiunitary_indices[0])
        lines.extend(
            [
                "Anti-linear symmetry representation matrices",
                f"  Operator indices: {display_index} ",
                _format_representation_block_header(labels, dimensions),
                "  Symmetrized basis:",
            ]
        )
        lines.extend(_format_block_matrix_rows(standard_antiunitary[0], dimensions))
        lines.append("  Numerical basis:")
        lines.extend(_format_block_matrix_rows(numeric_antiunitary[0], dimensions))
        if transformed_antiunitary:
            lines.append("  Numerical basis after transform:")
            lines.extend(_format_block_matrix_rows(transformed_antiunitary[0], dimensions))
    if phase_matrix is not None and phase_matrix.shape == unitary.shape:
        phases = list(phase_refinement.get("block_phases_rad", []) or []) if isinstance(phase_refinement, Mapping) else []
        for label, dimension, phase in zip(labels, dimensions, phases, strict=False):
            del dimension
            factor = np.exp(1.0j * float(phase))
            lines.append(f"e^{{i alpha_{label}}} = {_format_complex_pair(_complex_scalar_to_pair(factor))}")
    lines.append("U0 =")
    lines.extend(_format_matrix_pairs_3(_complex_matrix_to_pairs(u0)))
    lines.append("U =")
    lines.extend(_format_matrix_pairs_3(_complex_matrix_to_pairs(unitary)))
    lines.extend(
        [
            "Transformed representation matrix errors:",
            f"linear max_abs_difference = {float(alignment.get('unitary_max_abs_representation_difference', 0.0)):.12e}",
            f"anti-linear max_abs_difference = {float(alignment.get('antiunitary_max_abs_representation_difference', 0.0)):.12e}",
        ]
    )
    return lines


def _format_linear_representation_matrix_comparison_analyses(
    analyses: Sequence[Mapping[str, Any]],
    *,
    star_line: str | None,
) -> list[str]:
    if not analyses:
        return []

    separator = "---------------------------------------------------------------------------------------"
    lines: list[str] = []
    lines.extend([separator, "Symmetry representation matrices"])
    for analysis in analyses:
        dimensions = [int(value) for value in analysis.get("target_corep_dimensions", []) or []]
        labels = _representation_block_labels(analysis)
        alignment = analysis.get("schur_alignment", {})
        numeric_matrices = list(analysis.get("numeric_representation_matrices", []) or [])
        standard_matrices = list(analysis.get("spgrep_representation_matrices", []) or [])
        transformed_matrices = list(alignment.get("transformed_numeric_matrices", []) or [])
        operation_indices = list(analysis.get("operation_indices", []) or [])
        if not dimensions or not labels or not operation_indices:
            continue
        for pos, op_index in enumerate(operation_indices):
            if pos >= len(standard_matrices) or pos >= len(numeric_matrices):
                continue
            lines.append(f"  Operator indices: {int(op_index)} ")
            lines.append(_format_representation_block_header(labels, dimensions))
            lines.append("  Symmetrized basis:")
            lines.extend(_format_block_matrix_rows(standard_matrices[pos], dimensions))
            lines.append("  Numerical basis:")
            lines.extend(_format_block_matrix_rows(numeric_matrices[pos], dimensions))
            if pos < len(transformed_matrices):
                lines.append("  Numerical basis after transform:")
                lines.extend(_format_block_matrix_rows(transformed_matrices[pos], dimensions))
            lines.append(separator)
        lines.extend(
            _format_schur_basis_transformation(
                analysis,
                labels,
                dimensions,
                separator=separator,
            )
        )
        lines.append(separator)
    return lines


def _format_operator_irrep_analyses(
    analyses: Sequence[Mapping[str, Any]],
    *,
    character_table: Mapping[str, Any],
    star_line: str | None,
) -> list[str]:
    if not analyses:
        return []

    lines: list[str] = []
    for analysis in analyses:
        if star_line is not None:
            lines.append(star_line)
        lines.append("Hermitian basis Xi:")
        for item in analysis.get("hermitian_basis", []):
            label = _format_hermitian_basis_definition(str(item.get("label", "")))
            lines.append(f"X{int(item.get('basis_index', 0)):<2d} = {label}")
        lines.extend(
            [
                "Hermitian matrix irreducible decomposition",
                f"Operator representation: {analysis.get('operator_representation', 'unknown')}",
                f"Decomposition: {_format_decomposition(list(analysis.get('decomposition') or []))}",
            ]
        )
        separator = "---------------------------------------------------------------------------------------"
        lines.append(separator)
        for projected in analysis.get("projected_basis", []):
            lines.append(f"Projected Hermitian matrix basis for {projected.get('irrep_label', '')}")
            for matrix in projected.get("matrices", []):
                lines.append(f"{matrix.get('label', '')} = {_format_linear_combination(matrix.get('linear_combination', []))}")
                lines.append("matrix =")
                lines.extend(_format_matrix_pairs_3(matrix.get("matrix", [])))
            lines.append(separator)
    return lines


def _format_k_polynomial_irrep_analyses(
    analyses: Sequence[Mapping[str, Any]],
    *,
    star_line: str | None,
) -> list[str]:
    if not analyses:
        return []

    lines: list[str] = []
    if star_line is not None:
        lines.append(star_line)
    lines.extend(
        [
            "k polynomial irreducible decomposition",
            "k-transform convention: q' = R_cart^{-1} q in Cartesian coordinates",
            f"k_direction: {analyses[0].get('k_direction', 'xyz')}",
        ]
    )
    separator = "---------------------------------------------------------------------------------------"
    for analysis in analyses:
        if star_line is not None:
            lines.append(star_line)
        lines.extend(
            [
                f"Order = {int(analysis.get('order', 0))}",
                f"Decomposition: {_format_decomposition(list(analysis.get('decomposition') or []))}",
                "Monomial basis:",
            ]
        )
        for item in analysis.get("monomial_basis", []):
            exponents = " ".join(str(int(value)) for value in item.get("exponents", []))
            lines.append(f"Q{int(item.get('basis_index', 0)):<2d} = {item.get('label', ''):<16s} exponents=({exponents})")
        projected_basis = list(analysis.get("projected_basis", []))
        if projected_basis:
            lines.append(separator)
        for projected_index, projected in enumerate(projected_basis):
            lines.append(f"Projected k-polynomial basis for {projected.get('irrep_label', '')}")
            for function in projected.get("functions", []):
                lines.append(
                    f"{function.get('label', '')} = "
                    f"{_format_linear_combination(function.get('linear_combination', []))}"
                )
            if projected_index != len(projected_basis) - 1:
                lines.append(separator)
    return lines


def _format_zeeman_field_irrep_analyses(
    analyses: Sequence[Mapping[str, Any]],
    *,
    star_line: str | None,
) -> list[str]:
    if not analyses:
        return []

    lines: list[str] = []
    if star_line is not None:
        lines.append(star_line)
    lines.extend(
        [
            "Zeeman magnetic-field irreducible decomposition",
            "B-transform convention: B' = det(R_cart) R_cart B; functions use inverse transform",
            "Field basis: Cartesian pseudovector Bx, By, Bz",
        ]
    )
    separator = "---------------------------------------------------------------------------------------"
    for analysis in analyses:
        if star_line is not None:
            lines.append(star_line)
        lines.extend(
            [
                f"Decomposition: {_format_decomposition(list(analysis.get('decomposition') or []))}",
                "field basis:",
            ]
        )
        for item in analysis.get("monomial_basis", []):
            lines.append(f"B{int(item.get('basis_index', 0)):<2d} = {item.get('label', ''):<16s}")
        projected_basis = list(analysis.get("projected_basis", []))
        if projected_basis:
            lines.append(separator)
        for projected_index, projected in enumerate(projected_basis):
            lines.append(f"Projected Zeeman-field basis for {projected.get('irrep_label', '')}")
            for function in projected.get("functions", []):
                lines.append(
                    f"{function.get('label', '')} = "
                    f"{_format_linear_combination(function.get('linear_combination', []))}"
                )
            if projected_index != len(projected_basis) - 1:
                lines.append(separator)
    return lines


def _format_kp_product_combination(terms: Sequence[Mapping[str, Any]]) -> str:
    if not terms:
        return "0"
    pieces = []
    for term in terms:
        coefficient = _format_complex_symbolic(term.get("coefficient", [0.0, 0.0]))
        pieces.append(
            f"{coefficient} * {term.get('hermitian_basis_label', '')} * {term.get('k_basis_label', '')}"
        )
    return " + ".join(pieces)


def _invariant_solution_count(pair: Mapping[str, Any]) -> int:
    terms = list(pair.get("terms", []))
    if terms:
        return len(terms)
    if "rank" in pair:
        return int(pair.get("rank", 0))
    return int(pair.get("candidate_count", 0))


def _kp_product_support(terms: Sequence[Mapping[str, Any]], *, tol: float = 1.0e-10) -> set[tuple[str, str]]:
    support: set[tuple[str, str]] = set()
    for term in terms:
        coefficient = complex(*term.get("coefficient", [0.0, 0.0]))
        if abs(coefficient) <= tol:
            continue
        support.add((str(term.get("hermitian_basis_label", "")), str(term.get("k_basis_label", ""))))
    return support


def _kp_pair_product_support(pair: Mapping[str, Any]) -> set[tuple[str, str]]:
    support: set[tuple[str, str]] = set()
    for term in pair.get("terms", []):
        support.update(_kp_product_support(term.get("linear_combination", [])))
    return support


def _format_kp_form_solution_analyses(
    analyses: Sequence[Mapping[str, Any]],
    *,
    star_line: str | None,
    time_reversal_analyses: Sequence[Mapping[str, Any]] | None = None,
    final_kp_model_analyses: Sequence[Mapping[str, Any]] | None = None,
) -> list[str]:
    if not analyses:
        return []
    final_terms_by_order: dict[int, list[Mapping[str, Any]]] = {}
    final_anti_applied_by_order: dict[int, bool] = {}
    final_anti_operator_by_order: dict[int, int] = {}
    final_antiunitary_by_order: dict[int, int] = {}
    final_spgrep_by_order: dict[int, int] = {}
    for analysis in final_kp_model_analyses or []:
        order = int(analysis.get("order", 0))
        final_terms_by_order[order] = list(analysis.get("terms", []))
        final_anti_applied_by_order[order] = bool(
            analysis.get("anti_linear_operation_applied", analysis.get("time_reversal_applied", False))
        )
        final_anti_operator_by_order[order] = int(analysis.get("anti_linear_operator_index", 0))
        final_antiunitary_by_order[order] = int(analysis.get("antiunitary_operation_index", 0))
        final_spgrep_by_order[order] = int(analysis.get("spgrep_operation_index", 0))
    tr_allowed_labels = {
        str(check.get("label", ""))
        for analysis in (time_reversal_analyses or [])
        for check in analysis.get("term_checks", [])
        if str(check.get("status", "")) == "ok"
    }
    next_k_index_by_order: dict[int, int] = {}
    used_final_indices_by_order: dict[int, set[int]] = {}
    separator = "---------------------------------------------------------------------------------------"
    lines: list[str] = []
    if star_line is not None:
        lines.append(star_line)
    lines.append("Invariant k.p form solutions")
    for analysis in analyses:
        order = int(analysis.get("order", 0))
        first_pair_in_order = True
        for pair in analysis.get("irrep_pair_solutions", []):
            if first_pair_in_order:
                lines.append(f"k polynomial order = {order}")
                first_pair_in_order = False
            lines.append(
                f"k irrep {pair.get('k_irrep_label', '')} ------> "
                f"Hermitian irrep {pair.get('hermitian_irrep_label', '')}"
            )
            lines.append(f"Candidate_count = {_invariant_solution_count(pair)}")
            for term in pair.get("terms", []):
                lines.append(
                    f"{term.get('label', '')} = "
                    f"{_format_kp_product_combination(term.get('linear_combination', []))}"
                )
            order_anti_linear_applied = bool(time_reversal_analyses) or final_anti_applied_by_order.get(order, False)
            lines.append(f"Anti-linear operation applied: {'yes' if order_anti_linear_applied else 'no'}")
            if order_anti_linear_applied and final_anti_operator_by_order.get(order, 0):
                lines.append(f"anti-linear operator index: {final_anti_operator_by_order[order]}")
            if order_anti_linear_applied or final_terms_by_order:
                printed_count = 0
                if final_terms_by_order:
                    pair_support = _kp_pair_product_support(pair)
                    used_final_indices = used_final_indices_by_order.setdefault(order, set())
                    for final_index, final_term in enumerate(final_terms_by_order.get(order, [])):
                        if final_index in used_final_indices:
                            continue
                        final_support = _kp_product_support(final_term.get("linear_combination", []))
                        if not final_support or not final_support.issubset(pair_support):
                            continue
                        used_final_indices.add(final_index)
                        next_index = next_k_index_by_order.get(order, 1)
                        next_k_index_by_order[order] = next_index + 1
                        printed_count += 1
                        lines.append(
                            f"K{order}_{next_index} = "
                            f"{_format_kp_product_combination(final_term.get('linear_combination', []))}"
                        )
                else:
                    for term in pair.get("terms", []):
                        if str(term.get("label", "")) not in tr_allowed_labels:
                            continue
                        next_index = next_k_index_by_order.get(order, 1)
                        next_k_index_by_order[order] = next_index + 1
                        printed_count += 1
                        lines.append(
                            f"K{order}_{next_index} = "
                            f"{_format_kp_product_combination(term.get('linear_combination', []))}"
                        )
                if printed_count == 0:
                    lines.append(f"K{order}: none")
            lines.append(separator)
    return lines


def _format_zeeman_form_solution_analyses(
    analyses: Sequence[Mapping[str, Any]],
    *,
    star_line: str | None,
    final_zeeman_analyses: Sequence[Mapping[str, Any]] | None = None,
) -> list[str]:
    if not analyses:
        return []
    final_terms_by_order: dict[int, list[Mapping[str, Any]]] = {}
    final_anti_applied_by_order: dict[int, bool] = {}
    final_anti_operator_by_order: dict[int, int] = {}
    final_antiunitary_by_order: dict[int, int] = {}
    final_spgrep_by_order: dict[int, int] = {}
    for analysis in final_zeeman_analyses or []:
        order = int(analysis.get("order", 0))
        final_terms_by_order[order] = list(analysis.get("terms", []))
        final_anti_applied_by_order[order] = bool(
            analysis.get("anti_linear_operation_applied", analysis.get("time_reversal_applied", False))
        )
        final_anti_operator_by_order[order] = int(analysis.get("anti_linear_operator_index", 0))
        final_antiunitary_by_order[order] = int(analysis.get("antiunitary_operation_index", 0))
        final_spgrep_by_order[order] = int(analysis.get("spgrep_operation_index", 0))
    separator = "---------------------------------------------------------------------------------------"
    lines: list[str] = []
    if star_line is not None:
        lines.append(star_line)
    lines.append("Invariant Zeeman form solutions")
    for analysis in analyses:
        order = int(analysis.get("order", 0))
        used_final_indices: set[int] = set()
        first_pair_in_order = True
        for pair in analysis.get("irrep_pair_solutions", []):
            if first_pair_in_order:
                lines.append(f"Zeeman field order = {order}")
                first_pair_in_order = False
            lines.append(
                f"B irrep {pair.get('k_irrep_label', '')} ------> "
                f"Hermitian irrep {pair.get('hermitian_irrep_label', '')}"
            )
            lines.append(f"Candidate_count = {_invariant_solution_count(pair)}")
            pair_term_texts = set()
            for term in pair.get("terms", []):
                term_text = _format_kp_product_combination(term.get("linear_combination", []))
                pair_term_texts.add(term_text)
                lines.append(
                    f"{term.get('label', '')} = "
                    f"{term_text}"
                )
            order_anti_linear_applied = final_anti_applied_by_order.get(order, False)
            lines.append(f"Anti-linear operation applied: {'yes' if order_anti_linear_applied else 'no'}")
            if order_anti_linear_applied and final_anti_operator_by_order.get(order, 0):
                lines.append(f"anti-linear operator index: {final_anti_operator_by_order[order]}")
            if order_anti_linear_applied or final_terms_by_order:
                for final_index, final_term in enumerate(final_terms_by_order.get(order, [])):
                    if final_index in used_final_indices:
                        continue
                    term_text = _format_kp_product_combination(final_term.get("linear_combination", []))
                    if term_text not in pair_term_texts:
                        continue
                    used_final_indices.add(final_index)
                    lines.append(f"{final_term.get('label', '')} = {term_text}")
            lines.append(separator)
    return lines


def _format_time_reversal_constraint_analyses(
    analyses: Sequence[Mapping[str, Any]],
    *,
    star_line: str | None,
) -> list[str]:
    if not analyses:
        return []
    lines: list[str] = []
    if star_line is not None:
        lines.append(star_line)
    lines.append("Anti-linear constraint check for k.p invariant terms")
    for analysis in analyses:
        lines.extend(
            [
                f"target band representation: {analysis.get('target_label', '')}",
                "target corep indices: "
                + " ".join(str(int(index)) for index in analysis.get("target_corep_indices", [])),
                f"anti-linear operator index: {int(analysis.get('anti_linear_operator_index', 0))}",
                f"antiunitary operation index: {int(analysis.get('antiunitary_operation_index', 0))}",
                f"spgrep operation index: {int(analysis.get('spgrep_operation_index', 0))}",
                f"k transform: {analysis.get('k_transform', 'k -> -k')}",
                f"matrix convention: {analysis.get('matrix_convention', '')}",
                f"tolerance = {float(analysis.get('tolerance', 0.0)):.6e}",
            ]
        )
        for check in analysis.get("term_checks", []):
            lines.append(
                f"{check.get('label', '')}   "
                f"order = {int(check.get('order', 0))}   "
                f"parity = {int(check.get('parity', 1))}   "
                f"status = {check.get('status', 'unknown')}   "
                f"max_abs_diff = {float(check.get('max_abs_diff', 0.0)):.6e}"
            )
    return lines


def _format_final_kp_model_analyses(
    analyses: Sequence[Mapping[str, Any]],
    *,
    star_line: str | None,
) -> list[str]:
    if not analyses:
        return []
    lines: list[str] = []
    if star_line is not None:
        lines.append(star_line)
    lines.extend(
        [
            "Final real Hermitian anti-linear-even k.p basis",
            "basis convention: real coefficients multiplying Hermitian Xi and real k-polynomials",
            "anti-linear projection: keep +1 eigenspace within each order",
        ]
    )
    for analysis in analyses:
        if star_line is not None:
            lines.append(star_line)
        lines.extend(
            [
                f"order = {int(analysis.get('order', 0))}",
                f"source_term_count = {int(analysis.get('source_term_count', 0))}",
                f"real_candidate_rank = {int(analysis.get('real_candidate_rank', 0))}",
                f"tr_even_rank = {int(analysis.get('tr_even_rank', 0))}",
                f"tr_odd_rank = {int(analysis.get('tr_odd_rank', 0))}",
                "anti-linear operation applied: "
                f"{'yes' if bool(analysis.get('anti_linear_operation_applied', analysis.get('time_reversal_applied', False))) else 'no'}",
            ]
        )
        if bool(analysis.get("anti_linear_operation_applied", analysis.get("time_reversal_applied", False))):
            lines.extend(
                [
                    f"anti-linear operator index: {int(analysis.get('anti_linear_operator_index', 0))}",
                    f"antiunitary operation index: {int(analysis.get('antiunitary_operation_index', 0))}",
                    f"spgrep operation index: {int(analysis.get('spgrep_operation_index', 0))}",
                ]
            )
        for term in analysis.get("terms", []):
            lines.append(
                f"{term.get('label', '')} = "
                f"{_format_kp_product_combination(term.get('linear_combination', []))}"
            )
    return lines


def _format_final_zeeman_model_analyses(
    analyses: Sequence[Mapping[str, Any]],
    *,
    star_line: str | None,
) -> list[str]:
    if not analyses:
        return []
    lines: list[str] = []
    if star_line is not None:
        lines.append(star_line)
    lines.extend(
        [
            "Final real Hermitian anti-linear-even Zeeman basis",
            "basis convention: real coefficients multiplying Hermitian Xi and real B-field components",
            "anti-linear projection: B is transformed by the selected anti-linear operation; keep +1 eigenspace",
        ]
    )
    for analysis in analyses:
        if star_line is not None:
            lines.append(star_line)
        lines.extend(
            [
                f"order = {int(analysis.get('order', 0))}",
                f"source_term_count = {int(analysis.get('source_term_count', 0))}",
                f"real_candidate_rank = {int(analysis.get('real_candidate_rank', 0))}",
                f"tr_even_rank = {int(analysis.get('tr_even_rank', 0))}",
                f"tr_odd_rank = {int(analysis.get('tr_odd_rank', 0))}",
                "anti-linear operation applied: "
                f"{'yes' if bool(analysis.get('anti_linear_operation_applied', analysis.get('time_reversal_applied', False))) else 'no'}",
            ]
        )
        if bool(analysis.get("anti_linear_operation_applied", analysis.get("time_reversal_applied", False))):
            lines.extend(
                [
                    f"anti-linear operator index: {int(analysis.get('anti_linear_operator_index', 0))}",
                    f"antiunitary operation index: {int(analysis.get('antiunitary_operation_index', 0))}",
                    f"spgrep operation index: {int(analysis.get('spgrep_operation_index', 0))}",
                ]
            )
        for term in analysis.get("terms", []):
            lines.append(
                f"{term.get('label', '')} = "
                f"{_format_kp_product_combination(term.get('linear_combination', []))}"
            )
    return lines


def _format_numeric_lowdin_kp_analyses(
    analyses: Sequence[Mapping[str, Any]],
    *,
    star_line: str | None,
    max_order: int = 3,
) -> list[str]:
    if not analyses:
        return []

    max_order = int(max_order)
    lines: list[str] = []
    if star_line is not None:
        lines.append(star_line)
    lines.extend(
        [
            "Numerical Lowdin k.p model coefficients",
            "basis convention: listed per selection",
            "q convention: q = k - k0 in Cartesian coordinates, Angstrom^-1",
            f"k_direction: {analyses[0].get('k_direction', 'xyz')}",
            "velocity convention: V_i = (hbar / m_e) pi_i from pyatb velocity_matrix",
        ]
    )
    if max_order >= 3:
        lines.extend(
            [
                "H(q) = E0 + V_kx*qx + V_ky*qy + V_kz*qz + sum_Q C_Q Q(q) + sum_R D_R R(q)",
                "C_Q includes the free-electron hbar^2/(2m_e) term and the second-order Lowdin remote-band term",
                "D_R is the third-order Lowdin remote-band term",
            ]
        )
    else:
        lines.extend(
            [
                "H(q) = E0 + V_kx*qx + V_ky*qy + V_kz*qz + sum_Q C_Q Q(q)",
                "C_Q includes the free-electron hbar^2/(2m_e) term and the second-order Lowdin remote-band term",
            ]
        )
    for analysis in analyses:
        lines.extend(
            [
                f"hbar^2/(2m_e) = {float(analysis.get('free_electron_coefficient_eV_A2', 0.0)):.8f} eV Angstrom^2",
                "constant energies E0 (eV):",
            ]
        )
        for index, energy in enumerate(analysis.get("constant_energies_eV", []), start=1):
            lines.append(f"E{index:<2d} = {float(energy): .10f}")

        lines.append("Linear coefficient matrices V_i (eV Angstrom)")
        for item in analysis.get("linear_matrices", []):
            lines.append(f"V_{item.get('direction', '')} =")
            lines.extend(_format_matrix_pairs_3(item.get("matrix", [])))

        lines.append("Second-order total coefficient matrices C_Q (eV Angstrom^2)")
        for item in analysis.get("quadratic_monomial_matrices", []):
            lines.append(f"C_{item.get('label', '')} =")
            lines.extend(_format_matrix_pairs_3(item.get("matrix", [])))

        if max_order >= 3:
            lines.append("Third-order Lowdin coefficient matrices D_R (eV Angstrom^3)")
            for item in analysis.get("cubic_monomial_matrices", []):
                lines.append(f"D_{item.get('label', '')} =")
                lines.extend(_format_matrix_pairs_3(item.get("matrix", [])))
    return lines


def _format_numeric_zeeman_analyses(
    analyses: Sequence[Mapping[str, Any]],
    *,
    star_line: str | None,
) -> list[str]:
    if not analyses:
        return []

    lines: list[str] = []
    if star_line is not None:
        lines.append(star_line)
    lines.extend(
        [
            "Numerical Zeeman matrices from Lowdin perturbation",
            "convention: H_Z = (mu_B / 2) sum_i G_i B_i",
            "G_i = G_i^orb + sigma_i in the target subspace",
        ]
    )
    for analysis in analyses:
        if star_line is not None:
            lines.append(star_line)
        formula = str(analysis.get("zeeman_formula", "")).strip()
        if formula:
            lines.append(f"Formula: {formula}")
        lines.append("Orbital Zeeman G^orb_i matrices")
        for item in analysis.get("zeeman_orbital_matrices", []):
            lines.append(f"G_orb_{item.get('field', '')} =")
            lines.extend(_format_matrix_pairs_3(item.get("matrix", [])))
        lines.append("Spin Zeeman sigma_i matrices")
        for item in analysis.get("zeeman_spin_matrices", []):
            lines.append(f"sigma_{item.get('field', '')} =")
            lines.extend(_format_matrix_pairs_3(item.get("matrix", [])))
        lines.append("Total Zeeman G_i matrices")
        for item in analysis.get("zeeman_matrices", []):
            lines.append(f"G_{item.get('field', '')} =")
            lines.extend(_format_matrix_pairs_3(item.get("matrix", [])))
    return lines


def _format_numeric_lowdin_kp_status(
    status: Mapping[str, Any] | None,
    *,
    star_line: str | None,
) -> list[str]:
    if not status or bool(status.get("enabled", False)):
        return []
    lines: list[str] = []
    if star_line is not None:
        lines.append(star_line)
    lines.extend(
        [
            "Numerical Lowdin k.p model coefficients",
            f"status: {status.get('status', 'skipped')}",
            f"reason: {status.get('reason', 'unknown')}",
        ]
    )
    hint = str(status.get("hint", "")).strip()
    if hint:
        lines.append(f"hint: {hint}")
    return lines


def _format_representation_alignment_analyses(
    analyses: Sequence[Mapping[str, Any]],
    *,
    star_line: str | None,
) -> list[str]:
    if not analyses:
        return []

    lines: list[str] = []
    if star_line is not None:
        lines.append(star_line)
    lines.extend(
            [
                "Target-subspace symmetry representation alignment",
                "numeric representation: D_num(g)=C^dagger S(k) D_g C from CHARACTER",
                "symmetrized representation: spgrep target corep matrices in CHARACTER operation order",
                "Schur convention: D_symm(g) ~= U^dagger D_num(g) U; T_symm(a) ~= U^dagger T_num(a) U^*",
            ]
        )
    for analysis in analyses:
        alignment = analysis.get("schur_alignment", {})
        if star_line is not None:
            lines.append(star_line)
        lines.extend(
            [
                f"selection = {int(analysis.get('selection_index', 0))}",
                f"target band representation: {analysis.get('target_label', '')}",
                "target corep indices: "
                + " ".join(str(int(index)) for index in analysis.get("target_corep_indices", [])),
                "target corep dimensions: "
                + " ".join(str(int(value)) for value in analysis.get("target_corep_dimensions", [])),
                "band range : "
                + "-".join(str(int(index)) for index in analysis.get("band_range", [])),
                "operations: " + " ".join(str(int(index)) for index in analysis.get("operation_indices", [])),
                "antiunitary operations: "
                + " ".join(str(int(index)) for index in analysis.get("antiunitary_operation_indices", [])),
                f"max_abs_representation_difference = {float(alignment.get('max_abs_representation_difference', 0.0)):.12e}",
                f"unitary_max_abs_representation_difference = {float(alignment.get('unitary_max_abs_representation_difference', 0.0)):.12e}",
                f"antiunitary_max_abs_representation_difference = {float(alignment.get('antiunitary_max_abs_representation_difference', 0.0)):.12e}",
                f"frobenius_representation_difference = {float(alignment.get('frobenius_representation_difference', 0.0)):.12e}",
                f"unitarity_error = {float(alignment.get('unitarity_error', 0.0)):.12e}",
                "Schur unitary U_numeric_to_spgrep:",
            ]
        )
        phase_refinement = alignment.get("antiunitary_phase_refinement")
        if isinstance(phase_refinement, Mapping):
            lines.extend(
                [
                    f"antiunitary phase refinement: {phase_refinement.get('status', 'unknown')}",
                    "antiunitary block phases (rad): "
                    + " ".join(
                        f"{float(value): .12f}"
                        for value in phase_refinement.get("block_phases_rad", []) or []
                    ),
                    "antiunitary max difference before phase refinement = "
                    f"{float(phase_refinement.get('max_abs_antiunitary_difference_before', 0.0)):.12e}",
                    "antiunitary max difference after phase refinement = "
                    f"{float(phase_refinement.get('max_abs_antiunitary_difference_after', 0.0)):.12e}",
                ]
            )
        lines.extend(_format_matrix_pairs(alignment.get("unitary_numeric_to_standard", [])))
    return lines


def _selection_indexed_analysis(
    analyses: Sequence[Mapping[str, Any]] | None,
    selection_index: int,
) -> Mapping[str, Any]:
    for analysis in analyses or []:
        if int(analysis.get("selection_index", 0)) == int(selection_index):
            return analysis
    return {}


def _final_model_terms_by_label(
    analyses: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Mapping[str, Any]]:
    terms: dict[str, Mapping[str, Any]] = {}
    for analysis in analyses or []:
        for term in analysis.get("terms", []) or []:
            label = str(term.get("label", ""))
            if label:
                terms[label] = term
    return terms


def _format_symmetrized_model_formula_lines(
    analyses: Sequence[Mapping[str, Any]] | None,
    *,
    title: str,
) -> list[str]:
    terms = _final_model_terms_by_label(analyses)
    lines = [title]
    for label, term in terms.items():
        lines.append(
            f"{label} = "
            f"{_format_kp_product_combination(term.get('linear_combination', []))}"
        )
    return lines


def _format_fit_parameter_lines(
    analysis: Mapping[str, Any],
    *,
    separator: str,
) -> list[str]:
    lines = ["Parameters fitted:"]
    for label, value in zip(
        analysis.get("formal_parameter_labels", []),
        analysis.get("formal_parameters", []),
        strict=False,
    ):
        lines.append(f"{str(label):<8s} {float(value): .12f}")
    lines.extend(
        [
            f"absolute_residual = {float(analysis.get('absolute_residual', 0.0)):.12e}",
            f"relative_residual = {float(analysis.get('relative_residual', 0.0)):.12e}",
            f"max_abs_difference_all = {float(analysis.get('max_abs_difference_all', 0.0)):.12e}",
            "Detailed information:",
            separator,
        ]
    )
    return lines


def _kp_parameter_aliases(labels: Sequence[str]) -> dict[str, str]:
    prefixes = {0: "a", 1: "b", 2: "c", 3: "d"}
    counts: dict[int, int] = {}
    aliases: dict[str, str] = {}
    for raw_label in labels:
        label = str(raw_label)
        match = re.match(r"K(\d+)_", label)
        if match is None:
            aliases[label] = f"p{len(aliases) + 1}"
            continue
        order = int(match.group(1))
        counts[order] = counts.get(order, 0) + 1
        aliases[label] = f"{prefixes.get(order, 'p')}{counts[order]}"
    return aliases


def _zeeman_parameter_aliases(labels: Sequence[str]) -> dict[str, str]:
    return {str(label): f"g{index}" for index, label in enumerate(labels, start=1)}


def _infer_model_dimension(analyses: Sequence[Mapping[str, Any]] | None) -> int:
    dimension = 0
    for term in _final_model_terms_by_label(analyses).values():
        for component in term.get("linear_combination", []) or []:
            label = str(component.get("hermitian_basis_label", ""))
            indices = [int(value) for value in re.findall(r"\d+", label)]
            if indices:
                dimension = max(dimension, max(indices))
    return max(4, dimension)


def _symbolic_factor_product(factors: Sequence[str]) -> str:
    cleaned = [str(item) for item in factors if str(item) and str(item) != "1"]
    return "*".join(cleaned) if cleaned else "1"


def _symbolic_parameter_term(
    coefficient: complex,
    *,
    parameter: str,
    variable: str,
) -> str:
    coefficient = complex(coefficient)
    if abs(coefficient) < 1.0e-10:
        return ""
    coefficient_text = _format_complex_symbolic(_complex_scalar_to_pair(coefficient))
    if coefficient_text == "1":
        factors = [parameter]
    elif coefficient_text == "-1":
        factors = [f"-{parameter}"]
    elif coefficient_text.startswith("-"):
        factors = [f"-{coefficient_text[1:]}", parameter]
    else:
        factors = [coefficient_text, parameter]
    if variable and variable != "1":
        factors.append(variable)
    return _symbolic_factor_product(factors)


def _summary_parameter_scales(
    final_model_analyses: Sequence[Mapping[str, Any]] | None,
) -> dict[str, float]:
    dimension = _infer_model_dimension(final_model_analyses)
    hermitian_labels, hermitian_basis = _hermitian_matrix_basis(dimension)
    hermitian_by_label = {
        label: matrix
        for label, matrix in zip(hermitian_labels, hermitian_basis, strict=True)
    }
    scales: dict[str, float] = {}
    for label, term in _final_model_terms_by_label(final_model_analyses).items():
        for component in term.get("linear_combination", []) or []:
            matrix = hermitian_by_label.get(str(component.get("hermitian_basis_label", "")))
            if matrix is None:
                continue
            coefficient = complex(*component.get("coefficient", [0.0, 0.0]))
            for row in range(dimension):
                for col in range(row, dimension):
                    value = coefficient * matrix[row, col]
                    magnitude = abs(value)
                    if magnitude > 1.0e-10:
                        scales[str(label)] = float(magnitude)
                        break
                if str(label) in scales:
                    break
            if str(label) in scales:
                break
        scales.setdefault(str(label), 1.0)
    return scales


def _join_symbolic_terms(terms: Sequence[str]) -> str:
    cleaned = [term for term in terms if term and term != "0"]
    if not cleaned:
        return "0"
    expression = cleaned[0]
    for term in cleaned[1:]:
        if term.startswith("-"):
            expression += " - " + term[1:]
        else:
            expression += " + " + term
    return expression


def _expanded_hamiltonian_lines(
    final_model_analyses: Sequence[Mapping[str, Any]] | None,
    aliases: Mapping[str, str],
    *,
    matrix_symbol: str,
    parameter_scales: Mapping[str, float] | None = None,
) -> list[str]:
    dimension = _infer_model_dimension(final_model_analyses)
    hermitian_labels, hermitian_basis = _hermitian_matrix_basis(dimension)
    hermitian_by_label = {
        label: matrix
        for label, matrix in zip(hermitian_labels, hermitian_basis, strict=True)
    }
    entry_terms: dict[tuple[int, int], list[str]] = {
        (row, col): [] for row in range(dimension) for col in range(row, dimension)
    }

    for label, term in _final_model_terms_by_label(final_model_analyses).items():
        parameter = aliases.get(label)
        if parameter is None:
            continue
        parameter_scale = float((parameter_scales or {}).get(label, 1.0))
        if abs(parameter_scale) <= 1.0e-12:
            parameter_scale = 1.0
        for component in term.get("linear_combination", []) or []:
            hermitian_label = str(component.get("hermitian_basis_label", ""))
            matrix = hermitian_by_label.get(hermitian_label)
            if matrix is None:
                continue
            variable = str(component.get("k_basis_label", ""))
            coefficient = complex(*component.get("coefficient", [0.0, 0.0]))
            for row in range(dimension):
                for col in range(row, dimension):
                    matrix_value = matrix[row, col]
                    if abs(matrix_value) <= 1.0e-10:
                        continue
                    text = _symbolic_parameter_term(
                        coefficient * matrix_value / parameter_scale,
                        parameter=parameter,
                        variable=variable,
                    )
                    if text:
                        entry_terms[(row, col)].append(text)

    lines = []
    for row in range(dimension):
        for col in range(row, dimension):
            lines.append(
                f"{matrix_symbol}{row + 1}{col + 1} = "
                f"{_join_symbolic_terms(entry_terms[(row, col)])}"
            )
    return lines


def _format_summary_parameter_values(
    analysis: Mapping[str, Any] | None,
    aliases: Mapping[str, str],
    *,
    parameter_scales: Mapping[str, float] | None = None,
) -> list[str]:
    lines = ["Parameters:"]
    if not analysis:
        return lines
    for label, value in zip(
        analysis.get("formal_parameter_labels", []),
        analysis.get("formal_parameters", []),
        strict=False,
    ):
        alias = aliases.get(str(label))
        if alias is not None:
            scale = float((parameter_scales or {}).get(str(label), 1.0))
            lines.append(f"{alias}: {float(value) * scale: .12f}")
    return lines


def _format_final_hamiltonian_summary(
    kp_fit_analyses: Sequence[Mapping[str, Any]] | None,
    final_kp_model_analyses: Sequence[Mapping[str, Any]] | None,
    zeeman_fit_analyses: Sequence[Mapping[str, Any]] | None,
    final_zeeman_model_analyses: Sequence[Mapping[str, Any]] | None,
    *,
    star_line: str | None,
) -> list[str]:
    if not kp_fit_analyses and not zeeman_fit_analyses:
        return []
    separator = "---------------------------------------------------------------------------------------"
    lines: list[str] = []
    if star_line is not None:
        lines.append(star_line)
    lines.append("Summary")
    if kp_fit_analyses:
        kp_fit = kp_fit_analyses[0]
        kp_aliases = _kp_parameter_aliases([str(label) for label in kp_fit.get("formal_parameter_labels", [])])
        kp_scales = _summary_parameter_scales(final_kp_model_analyses)
        lines.append("Symmetrized H(k) Hamiltonian:")
        lines.extend(
            _expanded_hamiltonian_lines(
                final_kp_model_analyses,
                kp_aliases,
                matrix_symbol="H(k)",
                parameter_scales=kp_scales,
            )
        )
        lines.extend(_format_summary_parameter_values(kp_fit, kp_aliases, parameter_scales=kp_scales))
    if zeeman_fit_analyses:
        if kp_fit_analyses:
            lines.append(separator)
        zeeman_fit = zeeman_fit_analyses[0]
        zeeman_aliases = _zeeman_parameter_aliases(
            [str(label) for label in zeeman_fit.get("formal_parameter_labels", [])]
        )
        zeeman_scales = _summary_parameter_scales(final_zeeman_model_analyses)
        lines.append("Symmetrized Zeeman Hamiltonian:")
        lines.extend(
            _expanded_hamiltonian_lines(
                final_zeeman_model_analyses,
                zeeman_aliases,
                matrix_symbol="H(z)",
                parameter_scales=zeeman_scales,
            )
        )
        lines.extend(
            _format_summary_parameter_values(
                zeeman_fit,
                zeeman_aliases,
                parameter_scales=zeeman_scales,
            )
        )
    if star_line is not None:
        lines.append(star_line)
    lines.append("Finished")
    return lines


def _format_schur_kp_parameter_fit_analyses(
    analyses: Sequence[Mapping[str, Any]],
    *,
    star_line: str | None,
    final_model_analyses: Sequence[Mapping[str, Any]] | None = None,
    numeric_analyses: Sequence[Mapping[str, Any]] | None = None,
    max_order: int = 3,
) -> list[str]:
    if not analyses:
        return []

    max_order = int(max_order)
    lines: list[str] = []
    if star_line is not None:
        lines.append(star_line)
    separator = "---------------------------------------------------------------------------------------"
    lines.extend(
        [
            "Numerical Lowdin k.p model coefficients",
            "basis convention: listed per selection",
            "q convention: q = k - k0 in Cartesian coordinates, Angstrom^-1",
            f"k_direction: {analyses[0].get('k_direction', 'xyz')}",
            "velocity convention: V_i = (hbar / m_e) pi_i from pyatb velocity_matrix",
        ]
    )
    if max_order >= 3:
        lines.extend(
            [
                "H(q) = E0 + V_kx*qx + V_ky*qy + V_kz*qz + sum_Q C_Q Q(q) + sum_R D_R R(q)",
                "C_Q includes the free-electron hbar^2/(2m_e) term and the second-order Lowdin remote-band term",
                "D_R is the third-order Lowdin remote-band term",
            ]
        )
    else:
        lines.extend(
            [
                "H(q) = E0 + V_kx*qx + V_ky*qy + V_kz*qz + sum_Q C_Q Q(q)",
                "C_Q includes the free-electron hbar^2/(2m_e) term and the second-order Lowdin remote-band term",
            ]
        )
    for analysis in analyses:
        numeric = _selection_indexed_analysis(
            numeric_analyses,
            int(analysis.get("selection_index", 0)),
        )
        lines.append(
            "hbar^2/(2m_e) = "
            f"{float(numeric.get('free_electron_coefficient_eV_A2', 0.0)):.8f} eV Angstrom^2"
        )
        lines.append(separator)
        lines.extend(
            _format_symmetrized_model_formula_lines(
                final_model_analyses,
                title="Symmetrized formual for k.p Hamiltonian:",
            )
        )
        lines.append(separator)
        lines.extend(_format_fit_parameter_lines(analysis, separator=separator))
        records_by_monomial = {
            str(record.get("monomial", "")): record
            for record in analysis.get("records", []) or []
        }
        sections = [
            ("Constant coefficient matrix E0 (eV)", [("1", "E0")]),
            (
                "Linear coefficient matrices V_i (eV Angstrom)",
                [(label, f"V_{label}") for label in KP_DIRECTIONS],
            ),
            (
                "Second-order total coefficient matrices C_Q (eV Angstrom^2)",
                [
                    ("kx^2", "C_kx^2"),
                    ("kx*ky", "C_kx*ky"),
                    ("kx*kz", "C_kx*kz"),
                    ("ky^2", "C_ky^2"),
                    ("ky*kz", "C_ky*kz"),
                    ("kz^2", "C_kz^2"),
                ],
            ),
        ]
        if max_order >= 3:
            sections.append(
                (
                    "Third-order Lowdin coefficient matrices D_R (eV Angstrom^3)",
                    [
                        ("kx^3", "D_kx^3"),
                        ("kx^2*ky", "D_kx^2*ky"),
                        ("kx^2*kz", "D_kx^2*kz"),
                        ("kx*ky^2", "D_kx*ky^2"),
                        ("kx*ky*kz", "D_kx*ky*kz"),
                        ("kx*kz^2", "D_kx*kz^2"),
                        ("ky^3", "D_ky^3"),
                        ("ky^2*kz", "D_ky^2*kz"),
                        ("ky*kz^2", "D_ky*kz^2"),
                        ("kz^3", "D_kz^3"),
                    ],
                )
            )
        for section_title, section_records in sections:
            section_lines: list[str] = []
            for monomial, title in section_records:
                record = records_by_monomial.get(monomial)
                if not record:
                    continue
                section_lines.extend(
                    _format_matrix_pair_comparison_3(
                        record.get("raw_numeric_matrix", record.get("transformed_numeric_matrix", [])),
                        record.get("formal_fit_matrix", []),
                        title=title,
                    )
                )
            if section_lines:
                lines.append(section_title)
                lines.extend(section_lines)
                lines.append(separator)
        if lines and lines[-1] == separator:
                lines.pop()
    return lines


def _format_vector3(values: Sequence[float]) -> str:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size < 3:
        array = np.pad(array, (0, 3 - array.size), constant_values=0.0)
    return f"{array[0]: .6f} {array[1]: .6f} {array[2]: .6f}"


def _format_kp_energy_error_analyses(
    analyses: Sequence[Mapping[str, Any]],
    *,
    star_line: str | None,
) -> list[str]:
    if not analyses:
        return []
    lines: list[str] = []
    if star_line is not None:
        lines.append(star_line)
    lines.append("kp band test:")
    separator = "---------------------------------------------------------------------------------------"
    for index, analysis in enumerate(analyses):
        if index > 0:
            lines.append(separator)
            lines.append(f"selection = {int(analysis.get('selection_index', index + 1))}")
        grid = int(analysis.get("grid_points_per_axis", 0))
        grid_shape = list(analysis.get("grid_shape", []) or [])
        if len(grid_shape) != 3:
            grid_shape = [grid, grid, grid]
        lines.extend(
            [
                f"MP grid around k0: {int(grid_shape[0])} {int(grid_shape[1])} {int(grid_shape[2])}",
                f"Radius: {float(analysis.get('radius_A^-1', 0.0)):.6f} A^-1",
                f"Band max error:  {float(analysis.get('max_abs_error_eV', 0.0)):.12e} eV",
                f"Band mean error: {float(analysis.get('mean_abs_error_eV', 0.0)):.12e} eV",
            ]
        )
    return lines


def _format_schur_zeeman_parameter_fit_analyses(
    analyses: Sequence[Mapping[str, Any]],
    *,
    star_line: str | None,
    final_zeeman_model_analyses: Sequence[Mapping[str, Any]] | None = None,
    numeric_analyses: Sequence[Mapping[str, Any]] | None = None,
) -> list[str]:
    if not analyses:
        return []

    lines: list[str] = []
    if star_line is not None:
        lines.append(star_line)
    separator = "---------------------------------------------------------------------------------------"
    lines.extend(
        [
            "Numerical Zeeman matrices from Lowdin perturbation",
            "Hamiltonian convention: H_Z = (mu_B / 2) sum_i G_i B_i",
            "G_i = G_i^orb + sigma_i in the target subspace",
        ]
    )
    for analysis in analyses:
        numeric = _selection_indexed_analysis(
            numeric_analyses,
            int(analysis.get("selection_index", 0)),
        )
        formula = str(numeric.get("zeeman_formula", "")).strip()
        if formula:
            lines.append(formula)
        lines.extend(
            [
                "Schur-aligned numerical Zeeman parameter fit",
                "convention: G_form_fit(B) ~= U_schur^dagger G_numeric(B) U_schur",
            ]
        )
        if star_line is not None:
            lines.append(star_line)
        lines.extend(
            _format_symmetrized_model_formula_lines(
                final_zeeman_model_analyses,
                title="Symmetrized formual for Zeeman Hamiltonian:",
            )
        )
        lines.extend(_format_fit_parameter_lines(analysis, separator=separator))
        records_by_field = {
            str(record.get("field", "")): record
            for record in analysis.get("records", []) or []
        }
        section_lines: list[str] = []
        for field in ZEEMAN_FIELD_DIRECTIONS:
            record = records_by_field.get(field)
            if not record:
                continue
            section_lines.extend(
                _format_matrix_pair_comparison_3(
                    record.get("raw_numeric_matrix", record.get("transformed_numeric_matrix", [])),
                    record.get("formal_fit_matrix", []),
                    title=f"G_{field}",
                )
            )
        if section_lines:
            lines.append("Zeeman coefficient matrices G_i")
            lines.extend(section_lines)
            lines.append(separator)
        if lines and lines[-1] == separator:
            lines.pop()
    return lines


def _irrep_character_values(
    character_table: Mapping[str, Any],
    label: str,
) -> np.ndarray | None:
    target = str(label).strip()
    for irrep in character_table.get("irreps", []) or []:
        if str(irrep.get("label", "")).strip() != target:
            continue
        values = np.asarray(_complex_array_from_pairs(irrep.get("characters", [])), dtype=complex).reshape(-1)
        if values.size:
            return values
    return None


def _corep_trace_values(
    spgrep_info: Mapping[str, Any],
    corep_index: int,
) -> np.ndarray | None:
    for row in _spgrep_corep_trace_rows(spgrep_info):
        if int(row.get("corep_index", 0)) != int(corep_index):
            continue
        values = [
            complex(*trace.get("trace", [0.0, 0.0]))
            for trace in row.get("traces", []) or []
        ]
        if values:
            return np.asarray(values, dtype=complex).reshape(-1)
    return None


def _spgrep_corep_index_for_irrep(
    label: str,
    *,
    spgrep_info: Mapping[str, Any] | None = None,
    character_table: Mapping[str, Any] | None = None,
    tol: float = 1.0e-6,
) -> int | None:
    label_text = str(label).strip()
    spgrep = spgrep_info or {}
    table = character_table or {}
    character_values = _irrep_character_values(table, label_text)
    if character_values is not None and spgrep.get("coreps"):
        best_index: int | None = None
        best_diff = float("inf")
        for corep in spgrep.get("coreps", []) or []:
            corep_index = int(corep.get("corep_index", 0))
            trace_values = _corep_trace_values(spgrep, corep_index)
            if trace_values is None:
                continue
            count = min(int(character_values.size), int(trace_values.size))
            if count <= 0:
                continue
            diff = float(np.max(np.abs(trace_values[:count] - character_values[:count])))
            if diff < best_diff:
                best_diff = diff
                best_index = corep_index
        if best_index is not None and best_diff <= tol:
            return best_index

    # Backward-compatible fallback for the historical Bi2Se3 Gamma output.
    fallback = {
        "GM8": 1,
        "GM9": 2,
    }
    return fallback.get(label_text)


def _spgrep_corep_index_for_character_values(
    character_values: np.ndarray,
    *,
    spgrep_info: Mapping[str, Any],
    tol: float = 1.0e-6,
) -> int | None:
    values = np.asarray(character_values, dtype=complex).reshape(-1)
    if values.size == 0:
        return None

    best_index: int | None = None
    best_diff = float("inf")
    for row in _spgrep_corep_trace_rows(spgrep_info):
        corep_index = int(row.get("corep_index", 0))
        trace_values = _corep_trace_values(spgrep_info, corep_index)
        if trace_values is None:
            continue
        count = min(int(values.size), int(trace_values.size))
        if count <= 0:
            continue
        diff = float(np.max(np.abs(trace_values[:count] - values[:count])))
        if diff < best_diff:
            best_diff = diff
            best_index = corep_index
    if best_index is not None and best_diff <= tol:
        return best_index
    return None


def _spgrep_corep_groups_for_irrep_terms(
    labels: Sequence[str],
    *,
    spgrep_info: Mapping[str, Any],
    character_table: Mapping[str, Any],
    tol: float = 1.0e-6,
) -> list[dict[str, Any]]:
    terms = [str(label).strip() for label in labels if str(label).strip()]
    groups: list[dict[str, Any]] = []
    index = 0
    while index < len(terms):
        single_corep_index = _spgrep_corep_index_for_irrep(
            terms[index],
            spgrep_info=spgrep_info,
            character_table=character_table,
            tol=tol,
        )
        if single_corep_index is not None:
            groups.append(
                {
                    "label": terms[index],
                    "labels": [terms[index]],
                    "corep_index": int(single_corep_index),
                }
            )
            index += 1
            continue

        combined_values: np.ndarray | None = None
        matched: dict[str, Any] | None = None
        for stop in range(index + 1, len(terms) + 1):
            next_values = _irrep_character_values(character_table, terms[stop - 1])
            if next_values is None:
                break
            if combined_values is None:
                combined_values = np.asarray(next_values, dtype=complex).copy()
            else:
                if combined_values.size != np.asarray(next_values).size:
                    break
                combined_values = combined_values + np.asarray(next_values, dtype=complex)
            corep_index = _spgrep_corep_index_for_character_values(
                combined_values,
                spgrep_info=spgrep_info,
                tol=tol,
            )
            if corep_index is None:
                continue
            group_labels = terms[index:stop]
            matched = {
                "label": " + ".join(group_labels),
                "labels": list(group_labels),
                "corep_index": int(corep_index),
            }
            break
        if matched is None:
            return []
        groups.append(matched)
        index += len(matched["labels"])
    return groups


def _spgrep_corep_indices_for_irrep_terms(
    labels: Sequence[str],
    *,
    spgrep_info: Mapping[str, Any],
    character_table: Mapping[str, Any],
    tol: float = 1.0e-6,
) -> list[int]:
    return [
        int(group["corep_index"])
        for group in _spgrep_corep_groups_for_irrep_terms(
            labels,
            spgrep_info=spgrep_info,
            character_table=character_table,
            tol=tol,
        )
    ]


def _format_spgrep_irrep_matrices(
    spgrep_info: Mapping[str, Any],
    *,
    target_label: str,
    star_line: str | None,
    corep_index: int | None = None,
    character_table: Mapping[str, Any] | None = None,
) -> list[str]:
    if corep_index is None:
        corep_index = _spgrep_corep_index_for_irrep(
            target_label,
            spgrep_info=spgrep_info,
            character_table=character_table,
        )
    if corep_index is None:
        return []

    coreps = list(spgrep_info.get("coreps") or [])
    corep = next((item for item in coreps if int(item.get("corep_index", 0)) == corep_index), None)
    if corep is None:
        return []

    operations_by_index = {
        int(operation.get("operation_index", 0)): operation
        for operation in spgrep_info.get("operations", [])
    }
    lines: list[str] = []
    if star_line is not None:
        lines.append(star_line)
    lines.extend(
        [
            f"Spgrep representation matrices for {target_label}",
            f"corep index: {corep_index}",
            "matrix convention: unitary operations only; anti-linear operations are omitted",
        ]
    )
    for item in corep.get("operation_matrices", []):
        operation_index = int(item.get("operation_index", 0))
        operation = operations_by_index.get(operation_index, {})
        lines.extend(
            [
                f"operation {operation_index}",
                f"time reversal: {'yes' if bool(operation.get('time_reversal', False)) else 'no'}",
                f"anti-linear: {'yes' if bool(operation.get('anti_linear', False)) else 'no'}",
                "D =",
            ]
        )
        matrix = item.get("matrix", [])
        for row in matrix:
            entries = " ".join(_format_complex_pair(entry) for entry in row)
            lines.append(f"  {entries}")
    return lines


def _format_spgrep_antiunitary_irrep_matrices(
    spgrep_info: Mapping[str, Any],
    *,
    target_label: str,
    star_line: str | None,
    corep_index: int | None = None,
    character_table: Mapping[str, Any] | None = None,
) -> list[str]:
    if corep_index is None:
        corep_index = _spgrep_corep_index_for_irrep(
            target_label,
            spgrep_info=spgrep_info,
            character_table=character_table,
        )
    if corep_index is None:
        return []

    coreps = list(spgrep_info.get("coreps") or [])
    corep = next((item for item in coreps if int(item.get("corep_index", 0)) == corep_index), None)
    if corep is None:
        return []

    antiunitary_matrices = list(corep.get("antiunitary_operation_matrices") or [])
    if not antiunitary_matrices:
        return []

    antiunitary_operations_by_index = {
        int(operation.get("antiunitary_operation_index", 0)): operation
        for operation in spgrep_info.get("antiunitary_operations", [])
    }
    lines: list[str] = []
    if star_line is not None:
        lines.append(star_line)
    lines.extend(
        [
            f"Spgrep antiunitary representation matrices for {target_label}",
            f"corep index: {corep_index}",
            "matrix convention: anti-linear operations are written as D K",
        ]
    )
    for item in sorted(antiunitary_matrices, key=lambda value: int(value.get("antiunitary_operation_index", 0))):
        operation_index = int(item.get("antiunitary_operation_index", 0))
        operation = antiunitary_operations_by_index.get(operation_index, {})
        lines.extend(
            [
                f"antiunitary operation {operation_index}",
                f"spgrep operation index: {int(item.get('spgrep_operation_index', 0))}",
                f"time reversal: {'yes' if bool(operation.get('time_reversal', False)) else 'no'}",
                f"anti-linear: {'yes' if bool(operation.get('anti_linear', False)) else 'no'}",
                "D =",
            ]
        )
        matrix = item.get("matrix", [])
        for row in matrix:
            entries = " ".join(_format_complex_pair(entry) for entry in row)
            lines.append(f"  {entries}")
    return lines


def _write_kp_model_header(info: Mapping[str, Any], path: Path) -> None:
    star_line = "*" * 93
    spacegroup = info.get("spglib_spacegroup_dataset", {})
    sg_number = spacegroup.get("number", "unknown")
    sg_symbol = spacegroup.get("international", "unknown")
    hall_symbol = spacegroup.get("hall_symbol", "unknown")
    spin_orbit = bool(info.get("spin_group", {}).get("enabled", False))
    magnetic_type = info.get("magnetic_spacegroup_type", {})

    lines = [
        star_line,
        star_line,
        "                                   KP      MODEL ",
        star_line,
        star_line,
        f"Spglib determined space group No. {sg_number} ({sg_symbol}), Hall symbol {hall_symbol}.",
        f"Spin orbital interaction : {'yes' if spin_orbit else 'no'}",
    ]
    if spin_orbit:
        lines.extend(
            [
                f"Magnetic space group type : {magnetic_type.get('type_label', 'unknown')}",
                f"UNI number : {magnetic_type.get('uni_number', 'unknown')}",
                f"Magnetic space group setting (BNS): {magnetic_type.get('bns_setting', 'unknown')}",
                f"Magnetic space group setting (OG): {magnetic_type.get('og_setting', 'unknown')}",
            ]
        )
    lattice = np.asarray(info.get("lattice_vectors", []), dtype=float)
    reciprocal_lattice = np.asarray(info.get("reciprocal_lattice_vectors", []), dtype=float)
    if lattice.shape == (3, 3) and reciprocal_lattice.shape == (3, 3):
        lines.extend(
            [
                star_line,
                star_line,
                f"calculated stru: {info.get('active_stru_path', 'unknown')}",
                f"calculated hr file: {info.get('active_hr_path', 'unknown')}",
                f"calculated sr file: {info.get('active_sr_path', 'unknown')}",
                "Lattice vectors",
            ]
        )
        for idx, vector in enumerate(lattice, start=1):
            lines.append(
                f"a{idx:<1}"
                f"{_display_float(vector[0]):17.8f}"
                f"{_display_float(vector[1]):16.8f}"
                f"{_display_float(vector[2]):16.8f}"
            )
        lines.append("Reciprocal lattice vectors ")
        for idx, vector in enumerate(reciprocal_lattice, start=1):
            lines.append(
                f"b{idx:<1}"
                f"{_display_float(vector[0]):17.8f}"
                f"{_display_float(vector[1]):16.8f}"
                f"{_display_float(vector[2]):16.8f}"
            )
    spgrep_info = info.get("spgrep_operations", {})
    lines.extend(
        _format_spgrep_character_style_operations(
            spgrep_info,
            lattice=info.get("lattice_vectors"),
            star_line=star_line,
        )
    )
    lines.extend(
        _format_kpoint_record_lines(
            info.get("kpoint_records") or [],
            star_line=star_line,
            spgrep_info=spgrep_info,
        )
    )
    lines.extend(
        _format_linear_representation_matrix_comparison_analyses(
            info.get("representation_alignment_analyses", []),
            star_line=star_line,
        )
    )
    lines.extend(
        _format_operator_irrep_analyses(
            info.get("operator_irrep_analyses", []),
            character_table=info.get("character_table", {}),
            star_line=star_line,
        )
    )
    lines.extend(
        _format_k_polynomial_irrep_analyses(
            info.get("k_polynomial_irrep_analyses", []),
            star_line=star_line,
        )
    )
    lines.extend(
        _format_zeeman_field_irrep_analyses(
            info.get("zeeman_field_irrep_analyses", []),
            star_line=star_line,
        )
    )
    lines.extend(
        _format_kp_form_solution_analyses(
            info.get("kp_form_solution_analyses", []),
            star_line=star_line,
            time_reversal_analyses=info.get("time_reversal_constraint_analyses", []),
            final_kp_model_analyses=info.get("final_kp_model_analyses", []),
        )
    )
    lines.extend(
        _format_zeeman_form_solution_analyses(
            info.get("zeeman_form_solution_analyses", []),
            star_line=star_line,
            final_zeeman_analyses=info.get("final_zeeman_model_analyses", []),
        )
    )
    lines.extend(
        _format_numeric_lowdin_kp_status(
            info.get("numeric_lowdin_kp_status"),
            star_line=star_line,
        )
    )
    max_order = int(info.get("korder", 3))
    schur_kp_fits = info.get("schur_kp_parameter_fit_analyses", [])
    schur_zeeman_fits = info.get("schur_zeeman_parameter_fit_analyses", [])
    if schur_kp_fits:
        lines.extend(
            _format_schur_kp_parameter_fit_analyses(
                schur_kp_fits,
                star_line=star_line,
                final_model_analyses=info.get("final_kp_model_analyses", []),
                numeric_analyses=info.get("numeric_lowdin_kp_analyses", []),
                max_order=max_order,
            )
        )
    else:
        lines.extend(
            _format_numeric_lowdin_kp_analyses(
                info.get("numeric_lowdin_kp_analyses", []),
                star_line=star_line,
                max_order=max_order,
            )
        )
    if schur_zeeman_fits:
        lines.extend(
            _format_schur_zeeman_parameter_fit_analyses(
                schur_zeeman_fits,
                star_line=star_line,
                final_zeeman_model_analyses=info.get("final_zeeman_model_analyses", []),
                numeric_analyses=info.get("numeric_lowdin_kp_analyses", []),
            )
        )
    else:
        lines.extend(
            _format_numeric_zeeman_analyses(
                info.get("numeric_lowdin_kp_analyses", []),
                star_line=star_line,
            )
        )
    lines.extend(
        _format_final_hamiltonian_summary(
            schur_kp_fits,
            info.get("final_kp_model_analyses", []),
            schur_zeeman_fits,
            info.get("final_zeeman_model_analyses", []),
            star_line=star_line,
        )
    )
    energy_error_lines = _format_kp_energy_error_analyses(
        info.get("kp_energy_error_analyses", []),
        star_line=star_line,
    )
    if energy_error_lines:
        if lines and lines[-1] == "Finished":
            lines.pop()
            if lines and star_line is not None and lines[-1] == star_line:
                lines.pop()
        lines.extend(energy_error_lines)
        if star_line is not None:
            lines.append(star_line)
        lines.append("Finished")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        character_payload = calculator.calculate_character(
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
                "character_payload": character_payload if isinstance(character_payload, Mapping) else None,
            }
        )

    return results


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
    korder: int = 2,
    k_direction: str | Sequence[str] = "xyz",
    zeeman_term: str | bool | int = "yes",
    kp_radius: float = 0.0,
    kp_grid: int = 4,
    rR_route: str | Sequence[str] | None = None,
    rR_unit: str = "Angstrom",
    **character_kwargs,
) -> list[dict[str, Any]]:
    """Main-flow KP entry point that delegates first-stage irreps to CHARACTER."""

    if kpoint_mode != "direct":
        raise ValueError("KP currently supports only direct k-point mode.")
    rR_unit = rR_unit or "Angstrom"

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

    max_k_order = int(korder)
    if max_k_order < 0:
        raise ValueError("KP.korder must be zero or a positive integer.")
    k_orders = tuple(range(max_k_order + 1))
    k_direction_text, k_direction_indices, k_variable_labels = _normalize_k_direction(k_direction)
    include_zeeman = _kp_zeeman_enabled(zeeman_term)
    kp_radius_value = float(kp_radius)
    if kp_radius_value < 0.0:
        raise ValueError("KP.kp_radius must be zero or a positive float.")
    kp_grid_value = int(kp_grid)
    if kp_grid_value <= 0:
        raise ValueError("KP.kp_grid must be a positive integer.")

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
        rR_route=rR_route,
        rR_unit=rR_unit,
        **character_kwargs,
    )

    if RANK == 0:
        symmetry_info = None
        if results:
            payload = results[-1].get("character_payload")
            if isinstance(payload, Mapping):
                symmetry_info = _collect_kp_symmetry_info(
                    payload,
                    tb=tb,
                    symm_prec=float(symm_prec),
                    mag_tag=int(mag_tag),
                    mag=mag,
                )
                symmetry_info["korder"] = int(max_k_order)
                symmetry_info["k_direction"] = k_direction_text
                symmetry_info["k_direction_labels"] = list(k_variable_labels)
                symmetry_info["zeeman_term"] = "yes" if include_zeeman else "no"
                symmetry_info["kp_radius"] = float(kp_radius_value)
                symmetry_info["kp_grid"] = int(kp_grid_value)
                symmetry_info["kpoint_records"] = _kp_model_kpoint_records(results)
                operator_analyses = _operator_irrep_analyses_for_results(
                    symmetry_info,
                    results,
                )
                if operator_analyses:
                    symmetry_info["operator_irrep_analyses"] = operator_analyses
                k_polynomial_analyses = _k_polynomial_irrep_analyses(
                    symmetry_info.get("spgrep_operations", {}),
                    symmetry_info.get("character_table", {}),
                    orders=k_orders,
                    lattice=symmetry_info.get("lattice_vectors"),
                    k_direction=k_direction_text,
                )
                if k_polynomial_analyses:
                    symmetry_info["k_polynomial_irrep_analyses"] = k_polynomial_analyses
                zeeman_field_analyses = []
                if include_zeeman:
                    zeeman_field_analyses = _zeeman_field_irrep_analyses(
                        symmetry_info.get("spgrep_operations", {}),
                        symmetry_info.get("character_table", {}),
                        lattice=symmetry_info.get("lattice_vectors"),
                    )
                    if zeeman_field_analyses:
                        symmetry_info["zeeman_field_irrep_analyses"] = zeeman_field_analyses
                if operator_analyses and k_polynomial_analyses:
                    kp_form_solutions = _kp_form_solution_analyses(
                        operator_analyses,
                        k_polynomial_analyses,
                        symmetry_info.get("character_table", {}),
                    )
                    if kp_form_solutions:
                        symmetry_info["kp_form_solution_analyses"] = kp_form_solutions
                        time_reversal_checks = _time_reversal_constraint_analyses(
                            symmetry_info,
                            operator_analyses,
                            kp_form_solutions,
                            variable_labels=k_variable_labels,
                        )
                        if time_reversal_checks:
                            symmetry_info["time_reversal_constraint_analyses"] = time_reversal_checks
                        final_kp_model = _final_kp_model_analyses(
                            symmetry_info,
                            operator_analyses,
                            kp_form_solutions,
                            variable_labels=k_variable_labels,
                        )
                        if final_kp_model:
                            symmetry_info["final_kp_model_analyses"] = final_kp_model
                if operator_analyses and zeeman_field_analyses:
                    zeeman_form_solutions = _kp_form_solution_analyses(
                        operator_analyses,
                        zeeman_field_analyses,
                        symmetry_info.get("character_table", {}),
                    )
                    if zeeman_form_solutions:
                        symmetry_info["zeeman_form_solution_analyses"] = zeeman_form_solutions
                        final_zeeman_model = _final_kp_model_analyses(
                            symmetry_info,
                            operator_analyses,
                            zeeman_form_solutions,
                            term_prefix="Z",
                            variable_labels=ZEEMAN_FIELD_DIRECTIONS,
                            function_transform_kind="pseudovector",
                        )
                        if final_zeeman_model:
                            symmetry_info["final_zeeman_model_analyses"] = final_zeeman_model
                alignment_tb = _active_hs_tb_from_character_payload(
                    tb,
                    payload,
                    HR_unit=character_kwargs.get("HR_unit"),
                )
                representation_alignments = _representation_alignment_analyses_for_results(
                    symmetry_info,
                    results,
                    tb=alignment_tb,
                    symm_prec=float(symm_prec),
                )
                if representation_alignments:
                    symmetry_info["representation_alignment_analyses"] = representation_alignments
                lowdin_tb = _lowdin_tb_from_character_payload(
                    tb,
                    payload,
                    rR_route=rR_route,
                    rR_unit=rR_unit,
                    HR_unit=character_kwargs.get("HR_unit"),
                )
                if hasattr(lowdin_tb, "has_rR") and not bool(getattr(lowdin_tb, "has_rR", False)):
                    symmetry_info["numeric_lowdin_kp_status"] = {
                        "enabled": False,
                        "status": "skipped",
                        "reason": "rR matrix was not loaded, and pyatb velocity_matrix requires rR.",
                        "hint": "Set INPUT_PARAMETERS.rR_route and rR_unit, then rerun KP to compute numerical Lowdin coefficients.",
                    }
                else:
                    numeric_lowdin_kp = _numeric_lowdin_kp_analyses_for_results(lowdin_tb, results)
                    if numeric_lowdin_kp:
                        for analysis in numeric_lowdin_kp:
                            analysis["k_direction"] = k_direction_text
                        symmetry_info["numeric_lowdin_kp_analyses"] = numeric_lowdin_kp
                if representation_alignments and symmetry_info.get("numeric_lowdin_kp_analyses"):
                    schur_fits = _schur_kp_parameter_fit_analyses(
                        symmetry_info,
                        representation_alignments,
                        max_order=max_k_order,
                        variable_labels=k_variable_labels,
                    )
                    if schur_fits:
                        symmetry_info["schur_kp_parameter_fit_analyses"] = schur_fits
                        if kp_radius_value > 0.0:
                            energy_errors = _kp_energy_error_analyses(
                                lowdin_tb,
                                schur_fits,
                                symmetry_info.get("numeric_lowdin_kp_analyses", []),
                                radius=kp_radius_value,
                                grid=kp_grid_value,
                                direction_indices=k_direction_indices,
                            )
                            if energy_errors:
                                symmetry_info["kp_energy_error_analyses"] = energy_errors
                    if include_zeeman:
                        zeeman_fits = _schur_zeeman_parameter_fit_analyses(
                            symmetry_info,
                            representation_alignments,
                        )
                        if zeeman_fits:
                            symmetry_info["schur_zeeman_parameter_fit_analyses"] = zeeman_fits
                _write_kp_symmetry_info(symmetry_info, kp_output_path)

        with open(RUNNING_LOG, "a", encoding="utf-8") as handle:
            handle.write("\nKP Irreducible Representation Summary\n")
            handle.write(f"output_path = {kp_output_path.resolve()}\n")
            if results:
                handle.write(f"character_output_path = {Path(results[-1]['output_path']).resolve()}\n")
            if symmetry_info is not None:
                handle.write(f"kp_symmetry_info = {(kp_output_path / 'kp_symmetry_info.json').resolve()}\n")
            handle.write(f"selection_count = {len(results)}\n")

    return results


__all__ = [
    "KPointBandSelection",
    "calculate_kpoint_irreps",
    "calculate_kp_irreps",
]
