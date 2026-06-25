from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix


def _toy_metadata(load_pyatb, lattice_vector, positions_frac, species_by_atom):
    module = load_pyatb("pyatb.symmetry.Dk_matrix")
    return module.BasisMetadata(
        basis_num=len(positions_frac),
        spinless_basis_num=len(positions_frac),
        spin_factor=1,
        lattice_vector=np.asarray(lattice_vector, dtype=float),
        positions_frac=np.asarray(positions_frac, dtype=float),
        species_by_atom=list(species_by_atom),
        shells=[],
        atom_ranges={index: (index, 1) for index in range(len(positions_frac))},
    )


def test_build_standardization_atom_mapping_supports_supercell_to_primitive(load_pyatb) -> None:
    hs_standardize = load_pyatb("pyatb.symmetry.hs_standardize")

    source = _toy_metadata(
        load_pyatb,
        lattice_vector=[[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        positions_frac=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        species_by_atom=["X", "X"],
    )
    target = _toy_metadata(
        load_pyatb,
        lattice_vector=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        positions_frac=[[0.0, 0.0, 0.0]],
        species_by_atom=["X"],
    )

    mapping = hs_standardize.build_standardization_atom_mapping(source, target)

    assert [item["new_atom"] for item in mapping] == [0, 0]
    np.testing.assert_array_equal(mapping[0]["shift"], np.array([0, 0, 0], dtype=int))
    np.testing.assert_array_equal(mapping[1]["shift"], np.array([1, 0, 0], dtype=int))


def test_collect_no_rotation_samples_keeps_duplicate_supercell_blocks_separate(load_pyatb) -> None:
    covariance = load_pyatb("pyatb.symmetry.hs_covariance")

    source = _toy_metadata(
        load_pyatb,
        lattice_vector=[[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        positions_frac=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        species_by_atom=["X", "X"],
    )
    target = _toy_metadata(
        load_pyatb,
        lattice_vector=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        positions_frac=[[0.0, 0.0, 0.0]],
        species_by_atom=["X"],
    )
    atom_mapping = [
        {"old_atom": 0, "new_atom": 0, "shift": np.array([0, 0, 0], dtype=int), "species": "X"},
        {"old_atom": 1, "new_atom": 0, "shift": np.array([1, 0, 0], dtype=int), "species": "X"},
    ]
    lattice_transform = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int)
    source_dense_by_r = {
        (0, 0, 0): np.array([[5.0, 0.0], [0.0, 5.0]], dtype=complex),
    }
    target_dense_by_r = {
        (0, 0, 0): np.array([[5.0]], dtype=complex),
    }

    grouped = covariance.collect_no_rotation_block_samples_from_dense_blocks(
        source_dense_by_r=source_dense_by_r,
        source_metadata=source,
        target_metadata=target,
        atom_mapping=atom_mapping,
        lattice_transform_fractional=lattice_transform,
    )

    key = ((0, 0, 0), 0, 0)
    assert key in grouped
    assert len(grouped[key]) == 2
    for sample in grouped[key]:
        np.testing.assert_allclose(sample["block"], np.array([[5.0 + 0.0j]], dtype=complex))

    report = covariance.validate_no_rotation_block_mapping_from_dense_blocks(
        source_dense_by_r=source_dense_by_r,
        target_dense_by_r=target_dense_by_r,
        source_metadata=source,
        target_metadata=target,
        atom_mapping=atom_mapping,
        lattice_transform_fractional=lattice_transform,
    )

    assert report["mapped_block_count"] == 2
    assert report["target_missing_count"] == 0
    assert report["max_abs_diff"] == 0.0
    assert report["duplicate_consistency_max_abs"] == 0.0


def test_full_dense_blocks_are_completed_from_minus_r_partner(load_pyatb) -> None:
    covariance = load_pyatb("pyatb.symmetry.hs_covariance")

    class _FakeXR:
        def __init__(self):
            self.basis_num = 2
            self.R_direct_coor = np.array([[1, 0, 0], [-1, 0, 0]], dtype=int)
            # Triangular vectors for basis_num=2: [m00, m01, m11]
            self.XR = csr_matrix(
                np.array(
                    [
                        [1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j],  # R=(1,0,0)
                        [4.0 + 0.0j, 5.0 + 0.0j, 6.0 + 0.0j],  # R=(-1,0,0)
                    ],
                    dtype=complex,
                )
            )

    dense_full = covariance._dense_blocks_by_r_with_optional_full_reconstruction(
        _FakeXR(),
        full_matrix_from_hermitian=True,
    )

    r_pos = (1, 0, 0)
    r_neg = (-1, 0, 0)
    assert r_pos in dense_full
    assert r_neg in dense_full
    np.testing.assert_allclose(dense_full[r_pos][1, 0], np.conj(dense_full[r_neg][0, 1]))
    np.testing.assert_allclose(dense_full[r_neg][1, 0], np.conj(dense_full[r_pos][0, 1]))


def test_vector_rR_covariance_mixes_cartesian_components(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.data_covariance_constraint")

    metadata = type("Meta", (), {"basis_num": 1})()
    rot_z_90 = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    context = {
        "rotation": np.eye(3, dtype=int),
        "cart_rotation": rot_z_90,
        "pair_rows": [
            {
                "source_slice_a": slice(0, 1),
                "source_slice_b": slice(0, 1),
                "target_slice_a": slice(0, 1),
                "target_slice_b": slice(0, 1),
                "target_a": 0,
                "target_b": 0,
                "shift_diff": np.zeros(3, dtype=int),
                "d_left": np.eye(1, dtype=complex),
                "d_right_dag": np.eye(1, dtype=complex),
            }
        ],
        "pair_row_by_source": {
            (0, 0): {
                "source_slice_a": slice(0, 1),
                "source_slice_b": slice(0, 1),
                "target_slice_a": slice(0, 1),
                "target_slice_b": slice(0, 1),
                "target_a": 0,
                "target_b": 0,
                "shift_diff": np.zeros(3, dtype=int),
                "d_left": np.eye(1, dtype=complex),
                "d_right_dag": np.eye(1, dtype=complex),
            }
        },
    }
    vector_blocks = [
        {(0, 0, 0): np.array([[1.0 + 0.0j]], dtype=complex)},
        {(0, 0, 0): np.array([[2.0 + 0.0j]], dtype=complex)},
        {(0, 0, 0): np.array([[3.0 + 0.0j]], dtype=complex)},
    ]

    transformed = module.transform_vector_blocks_with_context(
        vector_blocks,
        metadata,
        context,
        nonzero_block_tol=1.0e-12,
    )

    np.testing.assert_allclose(transformed[0][(0, 0, 0)], np.array([[-2.0 + 0.0j]]))
    np.testing.assert_allclose(transformed[1][(0, 0, 0)], np.array([[1.0 + 0.0j]]))
    np.testing.assert_allclose(transformed[2][(0, 0, 0)], np.array([[3.0 + 0.0j]]))


def test_write_symmetrized_rR_roundtrips_with_abacus_reader(load_pyatb, tmp_path) -> None:
    covariance = load_pyatb("pyatb.symmetry.data_covariance_constraint")
    hs_covariance = load_pyatb("pyatb.symmetry.hs_covariance")
    reader = load_pyatb("pyatb.io.abacus_read_xr")

    path = tmp_path / "data-rR-sparse-covsymm.csr"
    vector_blocks = [
        {(0, 0, 0): np.array([[1.25 + 0.0j]], dtype=complex)},
        {(0, 0, 0): np.array([[2.50 + 0.0j]], dtype=complex)},
        {(0, 0, 0): np.array([[3.75 + 0.0j]], dtype=complex)},
    ]

    covariance.write_symmetrized_rR(
        vector_blocks,
        path,
        basis_num=1,
        nspin=1,
        rR_unit="Angstrom",
    )

    roundtrip = reader.abacus_readrR(str(path), "Angstrom")
    dense = [
        hs_covariance._dense_blocks_by_r_with_optional_full_reconstruction(component, full_matrix_from_hermitian=True)
        for component in roundtrip
    ]
    np.testing.assert_allclose(dense[0][(0, 0, 0)], np.array([[1.25 + 0.0j]]))
    np.testing.assert_allclose(dense[1][(0, 0, 0)], np.array([[2.50 + 0.0j]]))
    np.testing.assert_allclose(dense[2][(0, 0, 0)], np.array([[3.75 + 0.0j]]))
