from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


def test_canonicalize_fractional_coordinates_wraps_boundary_points(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.hs_standardize")

    frac = np.array(
        [
            [1.0, -1.0e-12, 0.5],
            [0.999999999999, 1.000000000001, -0.25],
        ],
        dtype=float,
    )

    wrapped = module.canonicalize_fractional_coordinates(frac, tol=1.0e-9)

    np.testing.assert_allclose(
        wrapped,
        np.array(
            [
                [0.0, 0.0, 0.5],
                [0.0, 0.0, 0.75],
            ],
            dtype=float,
        ),
        atol=1.0e-12,
    )


def test_map_target_r_uses_lattice_transform_and_atom_shifts(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.hs_standardize")
    b_matrix = np.eye(3, dtype=int)
    r_old = np.array([1, 0, -1], dtype=int)
    shift_a = np.array([1, 0, 0], dtype=int)
    shift_b = np.array([0, 1, 0], dtype=int)

    r_new = module.map_target_r_vector(b_matrix, r_old, shift_a, shift_b)

    np.testing.assert_array_equal(r_new, np.array([0, 1, -1], dtype=int))


def test_accumulate_block_merges_duplicate_targets(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.hs_standardize")
    blocks = {}

    module.accumulate_block(blocks, (0, 1, (0, 0, 0)), np.array([[1.0 + 0.0j]], dtype=complex))
    module.accumulate_block(blocks, (0, 1, (0, 0, 0)), np.array([[2.0 + 0.0j]], dtype=complex))

    assert blocks[(0, 1, (0, 0, 0))][0, 0] == 3.0 + 0.0j


def test_standardize_sparse_blocks_rewrites_target_keys(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.hs_standardize")

    result = module.standardize_sparse_blocks(
        source_blocks=[((0, 1, (1, 0, 0)), np.array([[1.0 + 0.0j]], dtype=complex))],
        atom_mapping=[
            {"old_atom": 0, "new_atom": 0, "shift": np.array([1, 0, 0], dtype=int)},
            {"old_atom": 1, "new_atom": 1, "shift": np.array([0, 0, 0], dtype=int)},
        ],
        lattice_transform=np.eye(3, dtype=int),
        orbital_rotations={0: np.eye(1, dtype=complex), 1: np.eye(1, dtype=complex)},
    )

    assert (0, 1, (1, 0, 0)) not in result
    assert result[(0, 1, (0, 0, 0))][0, 0] == 1.0 + 0.0j


def test_standardize_sparse_blocks_uses_passive_basis_rotation_order(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.hs_standardize")

    theta_a = np.pi / 4.0
    theta_b = np.pi / 6.0
    d_a = np.diag([np.exp(1.0j * theta_a), np.exp(-1.0j * theta_a)])
    d_b = np.diag([np.exp(1.0j * theta_b), np.exp(-1.0j * theta_b)])
    block = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=complex)

    result = module.standardize_sparse_blocks(
        source_blocks=[((0, 1, (0, 0, 0)), block)],
        atom_mapping=[
            {"old_atom": 0, "new_atom": 0, "shift": np.array([0, 0, 0], dtype=int)},
            {"old_atom": 1, "new_atom": 1, "shift": np.array([0, 0, 0], dtype=int)},
        ],
        lattice_transform=np.eye(3, dtype=int),
        orbital_rotations={0: d_a, 1: d_b},
    )

    expected = d_a.conj().T @ block @ d_b

    np.testing.assert_allclose(result[(0, 1, (0, 0, 0))], expected)


def test_full_dense_blocks_are_completed_from_minus_r_partner(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.hs_standardize")

    class _FakeXR:
        def __init__(self):
            self.basis_num = 2
            self.R_direct_coor = np.array([[1, 0, 0], [-1, 0, 0]], dtype=int)
            self.XR = csr_matrix(
                np.array(
                    [
                        [1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j],  # R=(1,0,0)
                        [4.0 + 0.0j, 5.0 + 0.0j, 6.0 + 0.0j],  # R=(-1,0,0)
                    ],
                    dtype=complex,
                )
            )

    dense_full = module._dense_blocks_by_r_with_optional_full_reconstruction(
        _FakeXR(),
        full_matrix_from_hermitian=True,
    )

    r_pos = (1, 0, 0)
    r_neg = (-1, 0, 0)
    np.testing.assert_allclose(dense_full[r_pos][1, 0], np.conj(dense_full[r_neg][0, 1]))
    np.testing.assert_allclose(dense_full[r_neg][1, 0], np.conj(dense_full[r_pos][0, 1]))
