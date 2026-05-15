from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _op(rotation, translation):
    return SimpleNamespace(
        rotation=np.array(rotation, dtype=int),
        translation=np.array(translation, dtype=float),
    )


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
    primitive_c3 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=int)
    conventional_c3 = np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]], dtype=int)
    operations = [
        _op(identity, [0.0, 0.0, 0.0]),
        _op(primitive_c3, [0.0, 0.0, 0.0]),
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
        symops=[_op(identity, [0.0, 0.0, 0.0]), _op(conventional_c3, [0.0, 0.0, 0.0])],
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
