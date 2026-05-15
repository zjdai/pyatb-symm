from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OrbitalShell:
    atom_index: int
    species: str
    l: int
    zeta: int
    global_offset: int
    local_offset: int
    dim: int


@dataclass
class BasisMetadata:
    basis_num: int
    spinless_basis_num: int
    spin_factor: int
    lattice_vector: np.ndarray
    positions_frac: np.ndarray
    species_by_atom: list[str]
    shells: list[OrbitalShell]
    atom_ranges: dict[int, tuple[int, int]]


def _canonicalize_fractional_coordinates(frac: np.ndarray, tol: float = 1.0e-9) -> np.ndarray:
    wrapped = np.asarray(frac, dtype=float) - np.floor(np.asarray(frac, dtype=float))
    wrapped[np.abs(wrapped) <= tol] = 0.0
    wrapped[np.abs(wrapped - 1.0) <= tol] = 0.0
    return wrapped


def _normalize(vec: np.ndarray, tol: float = 1.0e-12) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < tol:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return vec / norm


def axis_angle_from_cartesian_rotation(rot_cart: np.ndarray, tol: float = 1.0e-8):
    """
    Convert a 3x3 Cartesian rotation matrix to axis-angle form.

    Returns
    -------
    axis : np.ndarray
        Unit rotation axis.
    angle : float
        Rotation angle in radians, constrained to [0, pi].
    improper : bool
        Whether the original operation is improper (determinant < 0).
    """
    rot = np.asarray(rot_cart, dtype=float)
    det = float(np.linalg.det(rot))
    improper = det < 0.0

    proper_rot = -rot if improper else rot

    trace = float(np.trace(proper_rot))
    cos_theta = max(-1.0, min(1.0, 0.5 * (trace - 1.0)))
    angle = float(np.arccos(cos_theta))

    if abs(angle) < tol:
        return np.array([1.0, 0.0, 0.0], dtype=float), 0.0, improper

    if abs(np.pi - angle) < 1.0e-6:
        eigvals, eigvecs = np.linalg.eig(proper_rot)
        best = 0
        best_err = 1.0e9
        for i, eig in enumerate(eigvals):
            err = abs(eig.real - 1.0) + abs(eig.imag)
            if err < best_err:
                best_err = err
                best = i
        axis = np.real(eigvecs[:, best])
        axis = _normalize(axis)
        return axis, angle, improper

    axis = np.array(
        [
            proper_rot[2, 1] - proper_rot[1, 2],
            proper_rot[0, 2] - proper_rot[2, 0],
            proper_rot[1, 0] - proper_rot[0, 1],
        ],
        dtype=float,
    )
    axis = _normalize(axis)
    return axis, angle, improper


def spin_half_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """SU(2) representation for spin-1/2 under a spatial rotation."""
    n = _normalize(np.asarray(axis, dtype=float))
    half = 0.5 * float(angle)
    c = np.cos(half)
    s = np.sin(half)

    return np.array(
        [
            [c - 1j * s * n[2], -1j * s * (n[0] - 1j * n[1])],
            [-1j * s * (n[0] + 1j * n[1]), c + 1j * s * n[2]],
        ],
        dtype=complex,
    )


def spin_half_matrix_from_cartesian_rotation(rot_cart: np.ndarray) -> np.ndarray:
    """
    Build the spin-1/2 rotation matrix from a 3x3 Cartesian operation.

    For improper operations, only the proper-rotation part contributes to spin.
    """
    axis, angle, _ = axis_angle_from_cartesian_rotation(rot_cart)
    return spin_half_matrix_from_axis_angle(axis, angle)


def _complex_basis_operators(l: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dim = 2 * l + 1
    lp = np.zeros((dim, dim), dtype=complex)
    lm = np.zeros((dim, dim), dtype=complex)
    lz = np.diag(np.arange(-l, l + 1, dtype=float)).astype(complex)

    for col, m in enumerate(range(-l, l + 1)):
        if m < l:
            lp[col + 1, col] = np.sqrt(l * (l + 1) - m * (m + 1))
        if m > -l:
            lm[col - 1, col] = np.sqrt(l * (l + 1) - m * (m - 1))

    lx = (lp + lm) / 2.0
    ly = (lp - lm) / (2.0j)
    return lx, ly, lz


def _abacus_basis_transform(l: int) -> np.ndarray:
    dim = 2 * l + 1
    transform = np.zeros((dim, dim), dtype=complex)
    transform[l, 0] = 1.0

    next_column = 1
    for m in range(1, l + 1):
        pos_index = l + m
        neg_index = l - m
        phase = (-1) ** m

        transform[pos_index, next_column] = 1.0 / np.sqrt(2.0)
        transform[neg_index, next_column] = phase / np.sqrt(2.0)
        next_column += 1

        transform[pos_index, next_column] = -1.0j / np.sqrt(2.0)
        transform[neg_index, next_column] = 1.0j * phase / np.sqrt(2.0)
        next_column += 1

    return transform


def angular_momentum_matrices_abacus(l: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if l < 0:
        raise ValueError("l must be non-negative")

    lx_complex, ly_complex, lz_complex = _complex_basis_operators(l)
    transform = _abacus_basis_transform(l)
    lx = transform.conj().T @ lx_complex @ transform
    ly = transform.conj().T @ ly_complex @ transform
    lz = transform.conj().T @ lz_complex @ transform
    return lx, ly, lz


def extract_abacus_basis_metadata(tb) -> BasisMetadata:
    cached = getattr(tb, "_symmetry_basis_metadata", None)
    if cached is not None:
        return cached

    lattice_vector = np.asarray(tb.lattice_vector, dtype=float)
    inv_lattice = np.linalg.inv(lattice_vector)
    basis_num = int(tb.basis_num)
    nspin = int(getattr(tb, "nspin", 1))

    positions_frac: list[np.ndarray] = []
    species_by_atom: list[str] = []
    shells: list[OrbitalShell] = []
    atom_ranges: dict[int, tuple[int, int]] = {}

    global_offset = 0
    atom_index = 0
    for atom_type in tb.stru_atom:
        species = str(atom_type.species)
        orbital_num = list(atom_type.orbital_num)
        cart_coords = np.asarray(atom_type.cartesian_coor, dtype=float)
        atom_local_dim = int(sum((2 * l + 1) * nz for l, nz in enumerate(orbital_num)))

        for ia in range(int(atom_type.atom_num)):
            frac = np.asarray(cart_coords[ia], dtype=float) @ inv_lattice
            frac = _canonicalize_fractional_coordinates(frac)
            positions_frac.append(frac)
            species_by_atom.append(species)
            atom_ranges[atom_index] = (global_offset, atom_local_dim)

            local_offset = 0
            for l, nzeta in enumerate(orbital_num):
                for zeta in range(int(nzeta)):
                    dim = 2 * l + 1
                    shells.append(
                        OrbitalShell(
                            atom_index=atom_index,
                            species=species,
                            l=l,
                            zeta=zeta + 1,
                            global_offset=global_offset + local_offset,
                            local_offset=local_offset,
                            dim=dim,
                        )
                    )
                    local_offset += dim

            global_offset += atom_local_dim
            atom_index += 1

    spin_factor = 1
    expected_basis = global_offset
    if nspin == 4:
        spin_factor = 2
        expected_basis = 2 * global_offset

    if expected_basis != basis_num:
        raise ValueError(
            f"ABACUS basis metadata mismatch: inferred {expected_basis} orbitals, solver reports {basis_num}."
        )

    metadata = BasisMetadata(
        basis_num=basis_num,
        spinless_basis_num=global_offset,
        spin_factor=spin_factor,
        lattice_vector=lattice_vector,
        positions_frac=np.asarray(positions_frac, dtype=float),
        species_by_atom=species_by_atom,
        shells=shells,
        atom_ranges=atom_ranges,
    )
    setattr(tb, "_symmetry_basis_metadata", metadata)
    return metadata


def _operation_field(operation, name: str, default=None):
    if isinstance(operation, dict):
        return operation.get(name, default)
    return getattr(operation, name, default)


def find_atom_mapping(metadata: BasisMetadata, operation, tol: float = 1.0e-6) -> list[dict[str, np.ndarray | int | str]]:
    rotation = np.asarray(_operation_field(operation, "rotation"), dtype=int)
    translation = np.asarray(_operation_field(operation, "translation"), dtype=float)
    positions_frac = _canonicalize_fractional_coordinates(metadata.positions_frac, tol=tol)

    mapping = []
    for atom_index, position in enumerate(positions_frac):
        transformed = rotation @ position + translation
        found = None
        for target_index, target_position in enumerate(positions_frac):
            if metadata.species_by_atom[target_index] != metadata.species_by_atom[atom_index]:
                continue
            residual = transformed - target_position
            shift = np.rint(residual).astype(int)
            wrapped = residual - shift
            wrapped -= np.rint(wrapped)
            if float(np.linalg.norm(wrapped)) <= tol:
                found = {
                    "source_atom": atom_index,
                    "target_atom": target_index,
                    "species": metadata.species_by_atom[atom_index],
                    "cell_shift": shift,
                }
                break
        if found is None:
            raise ValueError(f"Failed to map atom {atom_index + 1} under symmetry operation.")
        mapping.append(found)
    return mapping


def shell_rotation(
    l_value: int,
    proper_rotation_cart: np.ndarray,
    det: float,
    include_inversion_parity: bool = True,
) -> np.ndarray:
    if l_value == 0:
        block = np.eye(1, dtype=complex)
    else:
        axis, angle, _ = axis_angle_from_cartesian_rotation(proper_rotation_cart)
        lx, ly, lz = angular_momentum_matrices_abacus(l_value)
        generator = axis[0] * lx + axis[1] * ly + axis[2] * lz
        generator = 0.5 * (generator + generator.conj().T)
        eigenvalues, eigenvectors = np.linalg.eigh(generator)
        block = eigenvectors @ np.diag(np.exp(-1.0j * angle * eigenvalues)) @ eigenvectors.conj().T
    if det < 0.0 and include_inversion_parity:
        block = ((-1) ** l_value) * block
    return block


def atom_orbital_rotation(metadata: BasisMetadata, atom_index: int, operation) -> np.ndarray:
    atom_offset, atom_dim = metadata.atom_ranges[atom_index]
    del atom_offset

    cart_rotation = np.asarray(_operation_field(operation, "cart_rotation"), dtype=float)
    det = float(np.linalg.det(cart_rotation))
    proper_rotation = -cart_rotation if det < 0.0 else cart_rotation

    matrix = np.zeros((atom_dim, atom_dim), dtype=complex)
    block_cache: dict[int, np.ndarray] = {}
    for shell in metadata.shells:
        if shell.atom_index != atom_index:
            continue
        block = block_cache.get(shell.l)
        if block is None:
            block = shell_rotation(shell.l, proper_rotation, det, include_inversion_parity=True)
            block_cache[shell.l] = block
        start = shell.local_offset
        stop = start + shell.dim
        matrix[start:stop, start:stop] = block
    return matrix


def build_atom_local_rotation(
    metadata: BasisMetadata,
    atom_index: int,
    cart_rotation: np.ndarray,
    passive_basis: bool = False,
    spin_matrix: np.ndarray | None = None,
) -> np.ndarray:
    rotation_cart = np.asarray(cart_rotation, dtype=float)
    if passive_basis:
        rotation_cart = np.linalg.inv(rotation_cart)

    orbital = atom_orbital_rotation(metadata, atom_index, {"cart_rotation": rotation_cart})
    if metadata.spin_factor != 2:
        return orbital

    if spin_matrix is not None and not passive_basis:
        spin = np.asarray(spin_matrix, dtype=complex)
    else:
        spin = spin_half_matrix_from_cartesian_rotation(rotation_cart)
    return np.kron(orbital, spin)


def build_dk_matrix(
    tb,
    k_direct: np.ndarray,
    operation,
    include_cell_shift_phase: bool = True,
    map_tol: float = 1.0e-6,
) -> np.ndarray:
    metadata = extract_abacus_basis_metadata(tb)
    kpoint = np.asarray(k_direct, dtype=float).reshape(3)
    atom_mapping = find_atom_mapping(metadata, operation, tol=float(map_tol))

    matrix = np.zeros((metadata.basis_num, metadata.basis_num), dtype=complex)
    local_cache: dict[int, np.ndarray] = {}

    for mapping in atom_mapping:
        source_atom = int(mapping["source_atom"])
        target_atom = int(mapping["target_atom"])
        shift = np.asarray(mapping["cell_shift"], dtype=int)

        local = local_cache.get(source_atom)
        if local is None:
            spin_matrix = None
            if metadata.spin_factor == 2:
                op_spin = _operation_field(operation, "spin_matrix", None)
                if op_spin is not None:
                    spin_matrix = np.asarray(op_spin, dtype=complex)
            local = build_atom_local_rotation(
                metadata,
                source_atom,
                _operation_field(operation, "cart_rotation"),
                passive_basis=False,
                spin_matrix=spin_matrix,
            )
            local_cache[source_atom] = local

        phase = 1.0 + 0.0j
        if include_cell_shift_phase:
            phase = np.exp(-2.0j * np.pi * float(np.dot(kpoint, shift)))

        source_offset, source_dim = metadata.atom_ranges[source_atom]
        target_offset, target_dim = metadata.atom_ranges[target_atom]
        if source_dim != target_dim:
            raise ValueError("Mapped atoms have inconsistent local basis dimensions.")

        source_offset *= metadata.spin_factor
        target_offset *= metadata.spin_factor
        source_dim *= metadata.spin_factor
        target_dim *= metadata.spin_factor
        matrix[
            target_offset : target_offset + target_dim,
            source_offset : source_offset + source_dim,
        ] = phase * local

    return matrix
