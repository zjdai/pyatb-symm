from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms


def _op(rotation, translation):
    return SimpleNamespace(
        rotation=np.array(rotation, dtype=int),
        translation=np.array(translation, dtype=float),
    )


def test_load_input_stru_wraps_fractional_positions_without_boundary_noise(
    load_pyatb,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_pyatb("pyatb.symmetry.symm_stru")
    (tmp_path / "STRU").write_text("ATOMIC_POSITIONS\n", encoding="utf-8")
    atoms = Atoms(
        "H",
        cell=np.eye(3, dtype=float),
        scaled_positions=[[-5.00000006e-8, 0.99999995, 0.25]],
        pbc=True,
    )
    monkeypatch.setattr(module, "INPUT_PATH", str(tmp_path))
    monkeypatch.setattr(module, "ase_read", lambda *args, **kwargs: atoms.copy())

    analyzer = module.SymmStructureAnalyzer(tb=None, output_path=str(tmp_path))
    loaded, source_path = analyzer._load_input_stru()

    assert source_path == tmp_path / "STRU"
    np.testing.assert_allclose(
        loaded.get_scaled_positions(wrap=False),
        np.array([[0.9999999499999994, 0.99999995, 0.25]], dtype=float),
        atol=1.0e-14,
        rtol=0.0,
    )


def test_load_input_stru_prefers_raw_direct_coordinates_over_ase_roundoff(
    load_pyatb,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_pyatb("pyatb.symmetry.symm_stru")
    (tmp_path / "STRU").write_text(
        """ATOMIC_POSITIONS
Direct

H
0.0
1
0.00000000000000 1.00000000000000 0.9999999499999994 1 1 1
""",
        encoding="utf-8",
    )
    atoms = Atoms(
        "H",
        cell=np.eye(3, dtype=float),
        scaled_positions=[[-1.0e-16, -1.0e-16, -1.0e-16]],
        pbc=True,
    )
    monkeypatch.setattr(module, "INPUT_PATH", str(tmp_path))
    monkeypatch.setattr(module, "ase_read", lambda *args, **kwargs: atoms.copy())

    analyzer = module.SymmStructureAnalyzer(tb=None, output_path=str(tmp_path))
    loaded, _ = analyzer._load_input_stru()

    np.testing.assert_allclose(
        loaded.get_scaled_positions(wrap=False),
        np.array([[0.0, 0.0, 0.9999999499999994]], dtype=float),
        atol=1.0e-14,
        rtol=0.0,
    )


def test_write_standardized_stru_preserves_abacus_upper_boundary_positions(
    load_pyatb,
    tmp_path: Path,
) -> None:
    module = load_pyatb("pyatb.symmetry.symm_stru")
    source_stru = tmp_path / "STRU"
    source_stru.write_text(
        """ATOMIC_SPECIES
H 1.0 H.upf

NUMERICAL_ORBITAL
H.orb

LATTICE_CONSTANT
1.8897261258369282

LATTICE_VECTORS
1 0 0
0 1 0
0 0 1

ATOMIC_POSITIONS
Direct

H
0.0
1
0.25 0.75 0.99999999996 1 1 1
""",
        encoding="utf-8",
    )
    target_stru = tmp_path / "STRU-symm"
    atoms = Atoms(
        "H",
        cell=np.eye(3, dtype=float),
        scaled_positions=[[0.25, 0.75, 0.99999999996]],
        pbc=True,
    )

    module.SymmStructureAnalyzer._write_standardized_stru(source_stru, atoms, target_stru)

    coord_line = next(
        line
        for line in target_stru.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("0.25")
    )
    coords = [float(value) for value in coord_line.split()[:3]]
    np.testing.assert_allclose(coords, [0.25, 0.75, 0.99999999996], atol=1.0e-15, rtol=0.0)
    assert coord_line.split()[2] != "1.0000000000"


def test_integer_image_atom_shifts_require_hs_rebuild(load_pyatb, tmp_path: Path) -> None:
    module = load_pyatb("pyatb.symmetry.symm_stru")
    analyzer = module.SymmStructureAnalyzer(tb=None, output_path=str(tmp_path))

    mapping = [
        (0, 1, np.array([0, 0, 1], dtype=int)),
        (1, 0, np.array([0, 0, -1], dtype=int)),
    ]

    assert not analyzer._mapping_is_permutation_only(mapping, tol=1.0e-6)


def test_database_reorder_rejects_rotation_only_translation_mismatch(load_pyatb, tmp_path: Path) -> None:
    module = load_pyatb("pyatb.symmetry.symm_stru")
    analyzer = module.SymmStructureAnalyzer(tb=None, output_path=str(tmp_path))
    identity = np.eye(3, dtype=int)
    operations = [_op(identity, [0.0, 0.0, 0.0])]
    db = SimpleNamespace(
        doubnum=2,
        symops=[_op(identity, [0.5, 0.0, 0.0])],
    )

    with pytest.raises(ValueError, match="translation mismatch"):
        analyzer._reorder_operations_with_database(operations, db)


def test_database_reorder_aligns_r_centered_conventional_ops_to_primitive(load_pyatb, tmp_path: Path) -> None:
    module = load_pyatb("pyatb.symmetry.symm_stru")
    analyzer = module.SymmStructureAnalyzer(tb=None, output_path=str(tmp_path))

    identity = np.eye(3, dtype=int)
    primitive_op = np.array([[-1, -1, 0], [0, 1, 0], [0, 0, -1]], dtype=int)
    conventional_op = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=int)
    operations = [
        _op(identity, [0.0, 0.0, 0.0]),
        _op(primitive_op, [0.0, 0.0, 0.0]),
    ]
    db = SimpleNamespace(
        doubnum=4,
        spacegroup_symbol="R-3m",
        kc2p=np.array(
            [
                [2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                [-1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                [-1.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0],
            ],
            dtype=float,
        ),
        symops=[_op(identity, [0.0, 0.0, 0.0]), _op(conventional_op, [0.0, 0.0, 0.0])],
    )

    reordered, warnings = analyzer._reorder_operations_with_database(operations, db)

    assert reordered == operations
    assert warnings == ["op #2 matched after database-conventional-to-primitive basis conversion"]


def test_origin_shift_solver_uses_row_vector_change_of_origin_formula(load_pyatb, tmp_path: Path) -> None:
    module = load_pyatb("pyatb.symmetry.symm_stru")
    analyzer = module.SymmStructureAnalyzer(tb=None, output_path=str(tmp_path))

    c4 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=int)
    op = module.SymmetryOperation(
        rotation=c4,
        translation=np.array([0.5, 0.5, 0.0], dtype=float),
        inverse_rotation=np.linalg.inv(c4).astype(int),
        cart_rotation=np.array(c4, dtype=float),
        euler_zyz=np.zeros(3, dtype=float),
        spin_matrix=np.eye(2, dtype=complex),
        factor_spin_matrix=None,
        symbol="C4",
        description="",
        axis=np.array([0.0, 0.0, 1.0], dtype=float),
    )
    db = SimpleNamespace(
        doubnum=2,
        symops=[_op(c4, [0.5, 0.0, 0.0])],
    )

    shift = analyzer._solve_origin_shift_from_operations([op], db, tol=1.0e-6)

    assert shift is not None
    shifted_tau = np.asarray(op.translation, dtype=float) + np.asarray(shift, dtype=float) @ (
        np.eye(3, dtype=float) - np.asarray(c4, dtype=float).T
    )
    diff = shifted_tau - np.array([0.5, 0.0, 0.0], dtype=float)
    diff -= np.rint(diff)
    np.testing.assert_allclose(diff, np.zeros(3, dtype=float), atol=1.0e-8)


def test_align_operations_to_reference_uses_origin_shifted_translation_matching(load_pyatb, tmp_path: Path) -> None:
    module = load_pyatb("pyatb.symmetry.symm_stru")
    analyzer = module.SymmStructureAnalyzer(tb=None, output_path=str(tmp_path))

    c4 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=int)
    identity = np.eye(3, dtype=int)
    shift = np.array([0.1, 0.2, 0.0], dtype=float)

    ref_ops = [
        module.SymmetryOperation(
            rotation=identity,
            translation=np.zeros(3, dtype=float),
            inverse_rotation=identity,
            cart_rotation=np.eye(3, dtype=float),
            euler_zyz=np.zeros(3, dtype=float),
            spin_matrix=np.eye(2, dtype=complex),
            factor_spin_matrix=None,
            symbol="E",
            description="unity op.",
            axis=np.array([1.0, 0.0, 0.0], dtype=float),
        ),
        module.SymmetryOperation(
            rotation=c4,
            translation=np.array([0.5, 0.0, 0.0], dtype=float),
            inverse_rotation=np.linalg.inv(c4).astype(int),
            cart_rotation=np.array(c4, dtype=float),
            euler_zyz=np.zeros(3, dtype=float),
            spin_matrix=np.eye(2, dtype=complex),
            factor_spin_matrix=None,
            symbol="C4",
            description="rotation",
            axis=np.array([0.0, 0.0, 1.0], dtype=float),
        ),
    ]
    candidate_ops = [
        module.SymmetryOperation(
            rotation=c4,
            translation=np.array([0.5, 0.0, 0.0], dtype=float) - shift @ (np.eye(3, dtype=float) - c4.T),
            inverse_rotation=np.linalg.inv(c4).astype(int),
            cart_rotation=np.array(c4, dtype=float),
            euler_zyz=np.zeros(3, dtype=float),
            spin_matrix=np.eye(2, dtype=complex),
            factor_spin_matrix=None,
            symbol="C4",
            description="rotation",
            axis=np.array([0.0, 0.0, 1.0], dtype=float),
        ),
        module.SymmetryOperation(
            rotation=identity,
            translation=np.zeros(3, dtype=float),
            inverse_rotation=identity,
            cart_rotation=np.eye(3, dtype=float),
            euler_zyz=np.zeros(3, dtype=float),
            spin_matrix=np.eye(2, dtype=complex),
            factor_spin_matrix=None,
            symbol="E",
            description="unity op.",
            axis=np.array([1.0, 0.0, 0.0], dtype=float),
        ),
    ]

    aligned = analyzer._align_operations_to_reference(ref_ops, candidate_ops, origin_shift=shift)

    assert [op.symbol for op in aligned] == ["E", "C4"]
    np.testing.assert_allclose(aligned[1].translation, candidate_ops[0].translation, atol=1.0e-12)


def test_database_alignment_summary_reports_translation_mismatch(load_pyatb, tmp_path: Path) -> None:
    module = load_pyatb("pyatb.symmetry.symm_stru")
    analyzer = module.SymmStructureAnalyzer(tb=None, output_path=str(tmp_path))

    c2 = np.diag([-1, -1, 1]).astype(int)
    operations = [
        module.SymmetryOperation(
            rotation=c2,
            translation=np.array([0.0, 0.0, 0.0], dtype=float),
            inverse_rotation=c2,
            cart_rotation=np.array(c2, dtype=float),
            euler_zyz=np.zeros(3, dtype=float),
            spin_matrix=np.eye(2, dtype=complex),
            factor_spin_matrix=None,
            symbol="C2",
            description="rotation",
            axis=np.array([0.0, 0.0, 1.0], dtype=float),
        )
    ]
    db = SimpleNamespace(
        doubnum=2,
        symops=[_op(c2, [0.5, 0.5, 0.0])],
    )

    matched, details = analyzer._database_alignment_summary(operations, db)

    assert matched is False
    assert len(details) == 1
    assert "translation mismatch" in details[0]
