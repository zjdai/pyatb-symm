from __future__ import annotations

from typing import Sequence

import numpy as np


def _as_square_matrix(matrix: np.ndarray | Sequence[Sequence[complex]], name: str) -> np.ndarray:
    value = np.asarray(matrix, dtype=complex)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {value.shape}.")
    return value


def _as_velocity_matrix(velocity: np.ndarray | Sequence[np.ndarray], name: str) -> np.ndarray:
    value = np.asarray(velocity, dtype=complex)
    if value.ndim != 3 or value.shape[0] != 3 or value.shape[1] != value.shape[2]:
        raise ValueError(f"{name} must have shape (3, n, n), got shape {value.shape}.")
    return value


def _as_cartesian_rotation(rotation: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    value = np.asarray(rotation, dtype=float)
    if value.shape != (3, 3):
        raise ValueError(f"cartesian_rotation must have shape (3, 3), got shape {value.shape}.")
    return value


def time_reversal_basis_matrix(
    nspin: int,
    *,
    basis_dim: int | None = None,
    spinless_basis_num: int | None = None,
) -> np.ndarray:
    """Return the orbital-basis unitary part of time reversal.

    Current pyatb velocity checks support ``nspin=1`` and ``nspin=4``:

    - ``nspin=1``: ``U_T = I``.
    - ``nspin=4``: ``U_T = I_Norb kron sigma_y`` with
      ``sigma_y = [[0, -i], [i, 0]]``.
    """

    spin = int(nspin)
    if spin not in (1, 4):
        raise ValueError(f"nspin must be 1 or 4, got {nspin}.")

    if spin == 1:
        if basis_dim is None:
            if spinless_basis_num is None:
                raise ValueError("basis_dim or spinless_basis_num is required for nspin=1.")
            basis_dim = int(spinless_basis_num)
        return np.eye(int(basis_dim), dtype=complex)

    if spinless_basis_num is None:
        if basis_dim is None:
            raise ValueError("basis_dim or spinless_basis_num is required for nspin=4.")
        if int(basis_dim) % 2 != 0:
            raise ValueError(f"nspin=4 basis_dim must be even, got {basis_dim}.")
        spinless_basis_num = int(basis_dim) // 2

    expected_dim = 2 * int(spinless_basis_num)
    if basis_dim is not None and int(basis_dim) != expected_dim:
        raise ValueError(
            f"basis_dim={basis_dim} is inconsistent with spinless_basis_num={spinless_basis_num} for nspin=4."
        )

    spin_time_reversal = np.array(
        [
            [0.0, -1.0j],
            [1.0j, 0.0],
        ],
        dtype=complex,
    )
    return np.kron(np.eye(int(spinless_basis_num), dtype=complex), spin_time_reversal)


def effective_symmetry_representation(
    representation_matrix: np.ndarray | Sequence[Sequence[complex]],
    *,
    nspin: int,
    antiunitary: bool = False,
    time_reversal_matrix: np.ndarray | Sequence[Sequence[complex]] | None = None,
    representation_includes_time_reversal: bool = False,
) -> np.ndarray:
    """Return the linear matrix used around covariant matrices.

    ``representation_matrix`` is normally the spatial representation ``D_g``.
    For antiunitary operations this function returns ``D_g U_T``.  If the caller
    has already supplied ``D_g U_T``, set ``representation_includes_time_reversal``
    to ``True``.
    """

    representation = _as_square_matrix(representation_matrix, "representation_matrix")
    if not bool(antiunitary) or bool(representation_includes_time_reversal):
        return representation

    if time_reversal_matrix is None:
        time_reversal = time_reversal_basis_matrix(int(nspin), basis_dim=representation.shape[1])
    else:
        time_reversal = _as_square_matrix(time_reversal_matrix, "time_reversal_matrix")
        if time_reversal.shape != representation.shape:
            raise ValueError(
                "time_reversal_matrix shape must match representation_matrix shape: "
                f"{time_reversal.shape} vs {representation.shape}."
            )

    return representation @ time_reversal


def rotate_velocity_components(
    velocity_matrix: np.ndarray | Sequence[np.ndarray],
    cartesian_rotation: np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray:
    """Return ``sum_beta O[alpha, beta] V_beta`` for all Cartesian components."""

    velocity = _as_velocity_matrix(velocity_matrix, "velocity_matrix")
    rotation = _as_cartesian_rotation(cartesian_rotation)
    mixed = np.zeros_like(velocity, dtype=complex)
    for alpha in range(3):
        for beta in range(3):
            coefficient = float(rotation[alpha, beta])
            if abs(coefficient) > 1.0e-14:
                mixed[alpha] += coefficient * velocity[beta]
    return mixed


def transform_wavefunction_coefficients(
    representation_matrix: np.ndarray | Sequence[Sequence[complex]],
    coefficients: np.ndarray | Sequence[Sequence[complex]],
    *,
    nspin: int,
    antiunitary: bool = False,
    time_reversal_matrix: np.ndarray | Sequence[Sequence[complex]] | None = None,
    representation_includes_time_reversal: bool = False,
) -> np.ndarray:
    """Transform eigenvector coefficients from ``k`` to the symmetry-related ``k'``.

    Unitary:

    ``C_tr(k') = D_g(k') C(k)``

    Antiunitary:

    ``C_tr(k') = D_g(k') U_T C(k)^*``
    """

    representation = effective_symmetry_representation(
        representation_matrix,
        nspin=int(nspin),
        antiunitary=bool(antiunitary),
        time_reversal_matrix=time_reversal_matrix,
        representation_includes_time_reversal=bool(representation_includes_time_reversal),
    )
    coeff = np.asarray(coefficients, dtype=complex)
    if coeff.ndim not in (1, 2):
        raise ValueError(f"coefficients must be a vector or a matrix, got shape {coeff.shape}.")
    if coeff.shape[0] != representation.shape[1]:
        raise ValueError(
            f"coefficients first dimension {coeff.shape[0]} does not match representation dimension "
            f"{representation.shape[1]}."
        )
    if bool(antiunitary):
        coeff = np.conj(coeff)
    return representation @ coeff


def transform_orbital_velocity_matrix(
    representation_matrix: np.ndarray | Sequence[Sequence[complex]],
    cartesian_rotation: np.ndarray | Sequence[Sequence[float]],
    orbital_velocity_matrix: np.ndarray | Sequence[np.ndarray],
    *,
    nspin: int,
    antiunitary: bool = False,
    time_reversal_matrix: np.ndarray | Sequence[Sequence[complex]] | None = None,
    representation_includes_time_reversal: bool = False,
) -> np.ndarray:
    """Transform an orbital-basis velocity matrix to the symmetry-related k point.

    Unitary:

    ``V'_alpha = D_g [sum_beta O_alpha,beta V_beta] D_g^dag``

    Antiunitary:

    ``V'_alpha = D_g U_T [-conj(sum_beta O_alpha,beta V_beta)] U_T^dag D_g^dag``
    """

    representation = effective_symmetry_representation(
        representation_matrix,
        nspin=int(nspin),
        antiunitary=bool(antiunitary),
        time_reversal_matrix=time_reversal_matrix,
        representation_includes_time_reversal=bool(representation_includes_time_reversal),
    )
    mixed = rotate_velocity_components(orbital_velocity_matrix, cartesian_rotation)
    if mixed.shape[1] != representation.shape[1]:
        raise ValueError(
            f"orbital_velocity_matrix dimension {mixed.shape[1]} does not match representation dimension "
            f"{representation.shape[1]}."
        )
    if bool(antiunitary):
        mixed = -np.conj(mixed)
    return np.asarray([representation @ mixed[alpha] @ representation.conj().T for alpha in range(3)], dtype=complex)


def transform_band_velocity_matrix(
    cartesian_rotation: np.ndarray | Sequence[Sequence[float]],
    band_velocity_matrix: np.ndarray | Sequence[np.ndarray],
    *,
    antiunitary: bool = False,
) -> np.ndarray:
    """Transform a band-basis velocity matrix in transported gauge.

    Unitary transported gauge:

    ``V_tr,alpha(k') = sum_beta O_alpha,beta V_beta(k)``

    Antiunitary transported gauge:

    ``V_tr,alpha(k') = -conj(sum_beta O_alpha,beta V_beta(k))``
    """

    mixed = rotate_velocity_components(band_velocity_matrix, cartesian_rotation)
    if bool(antiunitary):
        mixed = -np.conj(mixed)
    return mixed


def switch_band_velocity_gauge(
    transported_band_velocity_matrix: np.ndarray | Sequence[np.ndarray],
    switch_matrix: np.ndarray | Sequence[Sequence[complex]],
) -> np.ndarray:
    """Switch band velocity from transported gauge to target gauge.

    The convention is ``C_tr = C_target B``.  Therefore
    ``V_target = B V_tr B^dag``.
    """

    velocity = _as_velocity_matrix(transported_band_velocity_matrix, "transported_band_velocity_matrix")
    switch = _as_square_matrix(switch_matrix, "switch_matrix")
    if velocity.shape[1] != switch.shape[0]:
        raise ValueError(
            f"velocity band dimension {velocity.shape[1]} does not match switch_matrix dimension {switch.shape[0]}."
        )
    return np.asarray([switch @ velocity[alpha] @ switch.conj().T for alpha in range(3)], dtype=complex)


def transform_band_velocity_to_target_gauge(
    cartesian_rotation: np.ndarray | Sequence[Sequence[float]],
    band_velocity_matrix: np.ndarray | Sequence[np.ndarray],
    switch_matrix: np.ndarray | Sequence[Sequence[complex]],
    *,
    antiunitary: bool = False,
) -> np.ndarray:
    """Transform band velocity and then apply the ``B`` gauge switch."""

    transported = transform_band_velocity_matrix(
        cartesian_rotation,
        band_velocity_matrix,
        antiunitary=bool(antiunitary),
    )
    return switch_band_velocity_gauge(transported, switch_matrix)


def band_gauge_switch_matrix(
    target_coefficients: np.ndarray | Sequence[Sequence[complex]],
    target_overlap_matrix: np.ndarray | Sequence[Sequence[complex]],
    representation_matrix: np.ndarray | Sequence[Sequence[complex]],
    source_coefficients: np.ndarray | Sequence[Sequence[complex]],
    *,
    nspin: int,
    antiunitary: bool = False,
    time_reversal_matrix: np.ndarray | Sequence[Sequence[complex]] | None = None,
    representation_includes_time_reversal: bool = False,
) -> np.ndarray:
    """Return ``B = C_target^dag S(k') C_tr(k')``.

    The returned matrix follows ``C_tr = C_target B``.
    """

    target_coeff = np.asarray(target_coefficients, dtype=complex)
    target_overlap = _as_square_matrix(target_overlap_matrix, "target_overlap_matrix")
    transported = transform_wavefunction_coefficients(
        representation_matrix,
        source_coefficients,
        nspin=int(nspin),
        antiunitary=bool(antiunitary),
        time_reversal_matrix=time_reversal_matrix,
        representation_includes_time_reversal=bool(representation_includes_time_reversal),
    )
    if target_coeff.ndim != 2:
        raise ValueError(f"target_coefficients must be a matrix, got shape {target_coeff.shape}.")
    if target_coeff.shape[0] != target_overlap.shape[0]:
        raise ValueError(
            f"target_coefficients first dimension {target_coeff.shape[0]} does not match overlap dimension "
            f"{target_overlap.shape[0]}."
        )
    if transported.shape[0] != target_overlap.shape[1]:
        raise ValueError(
            f"transported coefficient dimension {transported.shape[0]} does not match overlap dimension "
            f"{target_overlap.shape[1]}."
        )
    return target_coeff.conj().T @ target_overlap @ transported
