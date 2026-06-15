from __future__ import annotations

import numpy as np
from ase import Atoms


def _shifts(mapping_result):
    return [tuple(int(value) for value in item["shift"].tolist()) for item in mapping_result.atom_mapping]


def test_build_structure_mapping_preserves_abacus_upper_boundary_image_shift(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.structure_mapping")
    source = Atoms(
        "Sn",
        cell=np.eye(3, dtype=float),
        scaled_positions=[[0.5, 0.5, 0.99999995]],
        pbc=True,
    )
    target = Atoms(
        "Sn",
        cell=np.eye(3, dtype=float),
        scaled_positions=[[0.5, 0.5, 0.0]],
        pbc=True,
    )

    mapping = module.build_structure_mapping(
        source,
        target,
        rotation_matrix=np.eye(3, dtype=float),
        supercell_matrix=np.eye(3, dtype=int),
        boundary_tol=1.0e-6,
    )

    assert _shifts(mapping) == [(0, 0, 1)]


def test_build_structure_mapping_wraps_negative_boundary_without_noise_zeroing(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.structure_mapping")
    source = Atoms(
        "Sn",
        cell=np.eye(3, dtype=float),
        scaled_positions=[[0.5, 0.5, -5.0e-8]],
        pbc=True,
    )
    target = Atoms(
        "Sn",
        cell=np.eye(3, dtype=float),
        scaled_positions=[[0.5, 0.5, 0.0]],
        pbc=True,
    )

    mapping = module.build_structure_mapping(
        source,
        target,
        rotation_matrix=np.eye(3, dtype=float),
        supercell_matrix=np.eye(3, dtype=int),
        boundary_tol=1.0e-6,
    )

    assert _shifts(mapping) == [(0, 0, 1)]


def test_build_structure_mapping_keeps_real_supercell_images(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.structure_mapping")
    source = Atoms(
        "HH",
        cell=np.diag([2.0, 1.0, 1.0]),
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        pbc=True,
    )
    target = Atoms(
        "H",
        cell=np.eye(3, dtype=float),
        scaled_positions=[[0.0, 0.0, 0.0]],
        pbc=True,
    )

    mapping = module.build_structure_mapping(
        source,
        target,
        rotation_matrix=np.eye(3, dtype=float),
        supercell_matrix=np.diag([2, 1, 1]),
        boundary_tol=1.0e-6,
    )

    assert _shifts(mapping) == [(0, 0, 0), (1, 0, 0)]
