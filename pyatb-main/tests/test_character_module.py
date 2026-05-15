from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
import warnings

import numpy as np
import pytest


class _FakeTB:
    nspin = 1
    max_kpoint_num = 10

    def read_stru(self, stru_file, need_orb=False):
        del stru_file, need_orb


class _FakeEntry:
    def __init__(self, name, k_conv, irreps, antisym=1):
        self.name = name
        self.k_conv = np.array(k_conv, dtype=float)
        self.irreps = irreps
        self.antisym = antisym

    @property
    def little_group_ops(self):
        if not self.irreps:
            return np.zeros((0,), dtype=int)
        return np.where(self.irreps[0].active_ops)[0]


class _FakeIrrep:
    def __init__(self, raw_name, name, reality, characters, active_ops, phase_kinds=None, factor_strings=None):
        self.raw_name = raw_name
        self.name = name
        self.reality = reality
        self.characters = np.array(characters, dtype=complex)
        self.active_ops = np.array(active_ops, dtype=bool)
        self.phase_kinds = np.array(
            phase_kinds if phase_kinds is not None else [1] * len(characters), dtype=int
        )
        self.factor_strings = factor_strings if factor_strings is not None else ["            "] * len(characters)


class _FakeResolution:
    def __init__(self, entry, rotated_k_prim, k_conv, entry_index=1, rotation_index=1, variable_count=0):
        self.entry = entry
        self.entry_index = entry_index
        self.rotated_k_prim = np.array(rotated_k_prim, dtype=float)
        self.k_conv = np.array(k_conv, dtype=float)
        self.rotation_index = rotation_index
        self.variable_count = variable_count


class _FakeDB:
    def __init__(self, resolution, traces):
        self.doubnum = len(traces)
        self._resolution = resolution
        self._traces = np.array(traces, dtype=complex)
        self.calls = []
        self.path = Path("/tmp/kLG_166.data")

    def resolve_kpoint_from_star(self, k_prim, inverse_rotations, has_inversion, little_group_size=None, detected_ops=None, tol=1e-5):
        self.calls.append(
            {
                "k_prim": np.array(k_prim, dtype=float),
                "inverse_rotations": inverse_rotations,
                "has_inversion": has_inversion,
                "little_group_size": little_group_size,
                "detected_ops": detected_ops,
            }
        )
        return self._resolution

    def irrep_table_characters(self, resolution, irrep):
        return self._traces.copy()



def test_character_creates_output_dir_and_runs_symmetry_stage(
    load_pyatb, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_pyatb("pyatb.symmetry.character")
    symm_mod = load_pyatb("pyatb.symmetry.symm_stru")
    monkeypatch.setattr(
        symm_mod.SymmStructureAnalyzer,
        "analyze_nonmagnetic",
        lambda *args, **kwargs: {"resolved_group": 1},
    )
    class _FakeMetadata:
        basis_num = 1
        species_by_atom = ["H"]
        positions_frac = np.array([[0.0, 0.0, 0.0]], dtype=float)
        lattice_vector = np.eye(3, dtype=float)

    monkeypatch.setattr(
        module,
        "load_abacus_hs_blocks",
        lambda *args, **kwargs: (_FakeMetadata(), {(0, 0, 0): np.eye(1, dtype=complex)}, {(0, 0, 0): np.eye(1, dtype=complex)}),
    )
    monkeypatch.setattr(module, "prepare_operation_contexts", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        module,
        "self_covariance_statistics",
        lambda *args, **kwargs: {"global_max_abs": 0.0, "mean_abs_over_operations": 0.0},
    )
    cal = module.Character(_FakeTB())

    cal.calculate_character(
        kpoint_mode="direct",
        kpoint_num=2,
        kpoint_direct_coor=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        group="auto",
        symm_prec=1e-5,
        occ_band=8,
        band=[7, 10],
        mag_tag=0,
        mag="auto",
        package="ABACUS",
    )

    assert (tmp_path / "Out" / "CHARACTER").exists()


def test_character_reads_original_stru_and_writes_trace_outputs(
    load_pyatb, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_pyatb("pyatb.symmetry.character")

    class _FakeSolver:
        def diago_H(self, k_direct_coor):
            del k_direct_coor
            return (
                np.eye(1, dtype=complex).reshape(1, 1, 1),
                np.array([[0.25]], dtype=float),
            )

        def get_Sk(self, k_direct_coor):
            del k_direct_coor
            return np.eye(1, dtype=complex).reshape(1, 1, 1)

    class _FlowTB:
        nspin = 1
        max_kpoint_num = 10
        basis_num = 1
        lattice_vector = np.eye(3)
        lattice_constant = 2.0
        stru_atom = []

        def __init__(self):
            self.read_calls = []
            self.tb_solver = _FakeSolver()

        def read_stru(self, stru_file, need_orb=False):
            self.read_calls.append((stru_file, need_orb))

    class _FlowIrrep:
        raw_name = "A"
        name = "A"
        characters = np.array([1.0 + 0.0j], dtype=complex)
        active_ops = np.array([True], dtype=bool)

    class _FlowEntry:
        name = "GM"
        irreps = [_FlowIrrep()]

    class _FlowResolution:
        entry = _FlowEntry()
        k_conv = np.array([0.0, 0.0, 0.0], dtype=float)

    tb = _FlowTB()
    cal = module.Character(tb)

    def _fake_analyze_nonmagnetic(*args, **kwargs):
        del args, kwargs
        report_path = tmp_path / "Out" / "CHARACTER" / "symmetry_character_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            (
                "Transformations:\n\n"
                "********************************************************************************\n\n"
                "knum =  1    kname= \n"
                "k = 0.000000 0.000000 0.000000\n\n"
                "       The k-point name is GM \n"
                "elemt ,symmetry ops, main axes\n"
                "   E     1  ( 1.000,  0.000,  0.000)\n\n"
                "********************************************************************************\n"
            ),
            encoding="utf-8",
        )
        return {
            "resolved_group": 1,
            "operations": [
                {
                    "rotation": np.eye(3, dtype=int),
                    "translation": np.zeros(3, dtype=float),
                    "cart_rotation": np.eye(3, dtype=float),
                    "spin_matrix": np.eye(2, dtype=complex),
                }
            ],
            "source_operations": [
                {
                    "rotation": np.eye(3, dtype=int),
                    "translation": np.zeros(3, dtype=float),
                    "cart_rotation": np.eye(3, dtype=float),
                    "spin_matrix": np.eye(2, dtype=complex),
                }
            ],
            "kpoint_records": [
                {
                    "k_index": 1,
                    "k_direct": np.array([0.0, 0.0, 0.0], dtype=float),
                    "active_operation_indices": [0],
                    "resolution": _FlowResolution(),
                    "k_name": "GM",
                }
            ],
        }

    monkeypatch.setattr(module.SymmStructureAnalyzer, "analyze_nonmagnetic", _fake_analyze_nonmagnetic)
    class _FakeMetadata:
        basis_num = 1
        species_by_atom = ["H"]
        positions_frac = np.array([[0.0, 0.0, 0.0]], dtype=float)
        lattice_vector = np.eye(3, dtype=float)

    monkeypatch.setattr(
        module,
        "load_abacus_hs_blocks",
        lambda *args, **kwargs: (_FakeMetadata(), {(0, 0, 0): np.eye(1, dtype=complex)}, {(0, 0, 0): np.eye(1, dtype=complex)}),
    )
    monkeypatch.setattr(module, "prepare_operation_contexts", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        module,
        "self_covariance_statistics",
        lambda *args, **kwargs: {"global_max_abs": 0.0, "mean_abs_over_operations": 0.0},
    )
    dk_calls = []

    def _fake_build_dk_matrix(tb_obj, k_direct, operation, map_tol=1.0e-6):
        dk_calls.append(float(map_tol))
        return np.eye(tb_obj.basis_num, dtype=complex)

    monkeypatch.setattr(module, "build_dk_matrix", _fake_build_dk_matrix, raising=False)

    cal.calculate_character(
        stru_file="STRU",
        kpoint_mode="direct",
        kpoint_num=1,
        kpoint_direct_coor=[[0.0, 0.0, 0.0]],
        group="auto",
        symm_prec=1e-5,
        occ_band=1,
        band=[1, 1],
        mag_tag=0,
        mag="auto",
    )

    assert tb.read_calls == [("STRU", True)]
    assert dk_calls == [1.0e-5]
    assert (tmp_path / "Out" / "CHARACTER" / "trace.txt").exists()
    assert (tmp_path / "Out" / "CHARACTER" / "band_irrep.txt").exists()
    assert "A" in (tmp_path / "Out" / "CHARACTER" / "band_irrep.txt").read_text(encoding="utf-8")

    trace_text = (tmp_path / "Out" / "CHARACTER" / "trace.txt").read_text(encoding="utf-8")
    trace_lines = trace_text.splitlines()
    assert trace_lines[0].strip() == "1"
    assert trace_lines[1].strip() == "1"
    assert trace_lines[2].strip() == "1"
    assert "knum =" not in trace_text
    assert "  1  1" in trace_text
    assert "0.250000" in trace_text

    report_text = (tmp_path / "Out" / "CHARACTER" / "symmetry_character_report.txt").read_text(encoding="utf-8")
    assert report_text.count("band") == 1
    assert report_text.count("degeneracy") == 1
    assert report_text.count("eigval") == 1
    assert "CALCULATED BAND CHARACTERS" not in report_text
    assert report_text.index("band") > report_text.index("knum =  1")
    assert report_text.index("band") < report_text.rindex("********************************************************************************")
    assert "=A" in report_text


def test_group_mismatch_raises_value_error(load_pyatb, tmp_path: Path) -> None:
    module = load_pyatb("pyatb.symmetry.symm_stru")

    with pytest.raises(ValueError):
        module.SymmStructureAnalyzer._resolve_spacegroup(167, 166)



def test_magnetic_branch_requires_explicit_spacegroup(load_pyatb, tmp_path: Path) -> None:
    module = load_pyatb("pyatb.symmetry.character")
    cal = module.Character(_FakeTB())

    with pytest.raises(ValueError, match="requires an explicit CHARACTER.group"):
        cal.calculate_character(
            kpoint_mode="direct",
            kpoint_num=1,
            kpoint_direct_coor=[[0.0, 0.0, 0.0]],
            group="auto",
            symm_prec=1e-5,
            occ_band=1,
            band=[1, 1],
            mag_tag=1,
            mag="auto",
        )


def test_magnetic_branch_truncates_extra_magnetic_moment_values(
    load_pyatb, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_pyatb("pyatb.symmetry.character")
    symm_mod = load_pyatb("pyatb.symmetry.symm_stru")
    captured = {}

    def _fake_analyze_magnetic(self, requested_group, symm_prec, magnetic_moments, kpoints_direct=None):
        del self, symm_prec, kpoints_direct
        captured["requested_group"] = requested_group
        captured["magnetic_moments"] = np.asarray(magnetic_moments, dtype=float)
        return {"resolved_group": int(requested_group)}

    monkeypatch.setattr(symm_mod.SymmStructureAnalyzer, "analyze_magnetic", _fake_analyze_magnetic)
    class _FakeMetadata:
        basis_num = 1
        species_by_atom = ["H"]
        positions_frac = np.array([[0.0, 0.0, 0.0]], dtype=float)
        lattice_vector = np.eye(3, dtype=float)

    monkeypatch.setattr(
        module,
        "load_abacus_hs_blocks",
        lambda *args, **kwargs: (_FakeMetadata(), {(0, 0, 0): np.eye(1, dtype=complex)}, {(0, 0, 0): np.eye(1, dtype=complex)}),
    )
    monkeypatch.setattr(module, "prepare_operation_contexts", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        module,
        "self_covariance_statistics",
        lambda *args, **kwargs: {"global_max_abs": 0.0, "mean_abs_over_operations": 0.0},
    )

    class _TwoAtomTB(_FakeTB):
        stru_atom = [type("AtomType", (), {"atom_num": 2})()]

    cal = module.Character(_TwoAtomTB())
    cal.calculate_character(
        kpoint_mode="direct",
        kpoint_num=1,
        kpoint_direct_coor=[[0.0, 0.0, 0.0]],
        group=1,
        symm_prec=1e-5,
        occ_band=1,
        band=[1, 1],
        mag_tag=1,
        mag=[0, 0, 1, 0, 0, -1, 9, 9, 9],
    )

    assert captured["requested_group"] == 1
    np.testing.assert_allclose(
        captured["magnetic_moments"],
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=float),
    )


def test_magnetic_requested_group_must_be_unitary_subgroup(load_pyatb, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_pyatb("pyatb.symmetry.symm_stru")
    analyzer = module.SymmStructureAnalyzer(_FakeTB(), output_path=str(tmp_path))

    unitary_ops = [
        module.SymmetryOperation(
            rotation=np.eye(3, dtype=int),
            translation=np.zeros(3, dtype=float),
            inverse_rotation=np.eye(3, dtype=int),
            cart_rotation=np.eye(3, dtype=float),
            euler_zyz=np.zeros(3, dtype=float),
            spin_matrix=np.eye(2, dtype=complex),
            symbol="E",
            description="unity op.",
            axis=np.array([1.0, 0.0, 0.0], dtype=float),
        )
    ]

    requested_ops = [
        unitary_ops[0],
        module.SymmetryOperation(
            rotation=-np.eye(3, dtype=int),
            translation=np.zeros(3, dtype=float),
            inverse_rotation=-np.eye(3, dtype=int),
            cart_rotation=-np.eye(3, dtype=float),
            euler_zyz=np.zeros(3, dtype=float),
            spin_matrix=np.eye(2, dtype=complex),
            symbol="I",
            description="inverse op.",
            axis=np.array([1.0, 0.0, 0.0], dtype=float),
        ),
    ]

    assert analyzer._requested_group_is_unitary_subgroup(requested_ops[:1], unitary_ops)
    assert not analyzer._requested_group_is_unitary_subgroup(requested_ops, unitary_ops)


def test_magnetic_group_mismatch_writes_character_report(
    load_pyatb, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_pyatb("pyatb.symmetry.symm_stru")
    analyzer = module.SymmStructureAnalyzer(_FakeTB(), output_path=str(tmp_path / "Out" / "CHARACTER"))
    analyzer._output_path.mkdir(parents=True, exist_ok=True)

    def _fake_load_input_stru():
        from ase import Atoms

        return (
            Atoms(numbers=[1], scaled_positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=True),
            tmp_path / "STRU",
        )

    monkeypatch.setattr(analyzer, "_load_input_stru", _fake_load_input_stru)
    monkeypatch.setattr(
        module.SymmStructureAnalyzer,
        "_get_magnetic_unitary_symmetry_data",
        staticmethod(
            lambda atoms, magnetic_moments, symm_prec: (
                {"rotations": np.array([np.eye(3, dtype=int)]), "translations": np.zeros((1, 3), dtype=float)},
                1,
            )
        ),
    )
    monkeypatch.setattr(module.KLittleGroupsDB, "load", staticmethod(lambda path: _FakeDB(_FakeResolution(_FakeEntry("GM", [0, 0, 0], []), [0, 0, 0], [0, 0, 0]), [1 + 0j])))
    monkeypatch.setattr(
        module.SymmStructureAnalyzer,
        "_reorder_operations_with_database",
        lambda self, operations, db: (_ for _ in ()).throw(ValueError("not subgroup")),
    )

    with pytest.raises(ValueError, match="alignment with kLittleGroups database failed"):
        analyzer.analyze_magnetic(2, 1e-5, np.array([[0.0, 0.0, 1.0]]), kpoints_direct=np.zeros((1, 3)))

    text = (tmp_path / "Out" / "CHARACTER" / "symmetry_character_report.txt").read_text(encoding="utf-8")
    assert "Unitary operations form space group 1" in text
    assert "User requested space group 2" in text
    assert "alignment with kLittleGroups database failed" in text


def test_magnetic_identical_group_strict_alignment_failure_raises(
    load_pyatb, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_pyatb("pyatb.symmetry.symm_stru")
    analyzer = module.SymmStructureAnalyzer(_FakeTB(), output_path=str(tmp_path / "Out" / "CHARACTER"))
    analyzer._output_path.mkdir(parents=True, exist_ok=True)

    def _fake_load_input_stru():
        from ase import Atoms

        return (
            Atoms(numbers=[1], scaled_positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=True),
            tmp_path / "STRU",
        )

    op = module.SymmetryOperation(
        rotation=np.eye(3, dtype=int),
        translation=np.array([0.0, 0.0, 0.32236646], dtype=float),
        inverse_rotation=np.eye(3, dtype=int),
        cart_rotation=np.eye(3, dtype=float),
        euler_zyz=np.zeros(3, dtype=float),
        spin_matrix=np.eye(2, dtype=complex),
        symbol="E",
        description="unity op.",
        axis=np.array([1.0, 0.0, 0.0], dtype=float),
    )

    class _OneOperationDB:
        doubnum = 2
        path = Path("/tmp/kLG_1.data")
        symops = [
            type("Op", (), {"rotation": np.eye(3, dtype=int), "translation": np.zeros(3, dtype=float)})(),
        ]

    monkeypatch.setattr(analyzer, "_load_input_stru", _fake_load_input_stru)
    monkeypatch.setattr(
        module.SymmStructureAnalyzer,
        "_get_magnetic_unitary_symmetry_data",
        staticmethod(
            lambda atoms, magnetic_moments, symm_prec: (
                {"rotations": np.array([np.eye(3, dtype=int)]), "translations": np.array([[0.0, 0.0, 0.32236646]])},
                1,
            )
        ),
    )
    monkeypatch.setattr(module.KLittleGroupsDB, "load", staticmethod(lambda path: _OneOperationDB()))
    monkeypatch.setattr(analyzer, "_build_symmetry_operations", lambda atoms, sym_data: [op])
    monkeypatch.setattr(analyzer, "_sort_operations_irvsp_like", lambda operations: operations)
    monkeypatch.setattr(
        analyzer,
        "_reorder_operations_with_database",
        lambda operations, db: (_ for _ in ()).throw(ValueError("translation mismatch")),
    )

    with pytest.raises(ValueError, match="alignment with kLittleGroups database failed"):
        analyzer.analyze_magnetic(1, 1e-5, np.array([[0.0, 0.0, 1.0]]), kpoints_direct=np.zeros((0, 3)))


def test_magnetic_analyzer_standardizes_structure_before_database_alignment(
    load_pyatb, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from ase import Atoms

    module = load_pyatb("pyatb.symmetry.symm_stru")
    analyzer = module.SymmStructureAnalyzer(_FakeTB(), output_path=str(tmp_path / "Out" / "CHARACTER"))
    analyzer._output_path.mkdir(parents=True, exist_ok=True)

    source_atoms = Atoms(
        numbers=[1],
        scaled_positions=[[0.32236646, 0.0, 0.0]],
        cell=np.eye(3),
        pbc=True,
    )
    std_atoms = Atoms(
        numbers=[1],
        scaled_positions=[[0.0, 0.0, 0.0]],
        cell=np.eye(3),
        pbc=True,
    )
    source_op = module.SymmetryOperation(
        rotation=np.eye(3, dtype=int),
        translation=np.array([0.0, 0.0, 0.32236646], dtype=float),
        inverse_rotation=np.eye(3, dtype=int),
        cart_rotation=np.eye(3, dtype=float),
        euler_zyz=np.zeros(3, dtype=float),
        spin_matrix=np.eye(2, dtype=complex),
        symbol="E",
        description="unity op.",
        axis=np.array([1.0, 0.0, 0.0], dtype=float),
    )
    std_op = module.SymmetryOperation(
        rotation=np.eye(3, dtype=int),
        translation=np.zeros(3, dtype=float),
        inverse_rotation=np.eye(3, dtype=int),
        cart_rotation=np.eye(3, dtype=float),
        euler_zyz=np.zeros(3, dtype=float),
        spin_matrix=np.eye(2, dtype=complex),
        symbol="E",
        description="unity op.",
        axis=np.array([1.0, 0.0, 0.0], dtype=float),
    )

    class _OneOperationDB:
        doubnum = 2
        path = Path("/tmp/kLG_1.data")
        symops = [
            type("Op", (), {"rotation": np.eye(3, dtype=int), "translation": np.zeros(3, dtype=float)})(),
        ]

    monkeypatch.setattr(analyzer, "_load_input_stru", lambda: (source_atoms, tmp_path / "STRU"))
    monkeypatch.setattr(
        analyzer,
        "_standardize_magnetic_cell",
        lambda atoms, magnetic_moments, symm_prec: (
            1,
            std_atoms,
            source_atoms,
            {"rotations": np.array([np.eye(3, dtype=int)]), "translations": np.array([[0.0, 0.0, 0.32236646]])},
            {"rotations": np.array([np.eye(3, dtype=int)]), "translations": np.zeros((1, 3), dtype=float)},
            np.array([[0.0, 0.0, 1.0]], dtype=float),
        ),
    )
    monkeypatch.setattr(module.KLittleGroupsDB, "load", staticmethod(lambda path: _OneOperationDB()))

    def _fake_build_ops(atoms, sym_data):
        return [std_op] if atoms is std_atoms else [source_op]

    monkeypatch.setattr(analyzer, "_build_symmetry_operations", _fake_build_ops)
    monkeypatch.setattr(analyzer, "_sort_operations_irvsp_like", lambda operations: operations)

    result = analyzer.analyze_magnetic(1, 1e-5, np.array([[0.0, 0.0, 1.0]]), kpoints_direct=np.zeros((0, 3)))

    assert result["need_rebuild_hs"] is False
    np.testing.assert_allclose(result["operations"][0].translation, np.zeros(3))
    np.testing.assert_allclose(result["source_operations"][0].translation, np.array([0.0, 0.0, 0.32236646]))


def test_magnetic_report_writes_full_match_summary_and_omits_old_table_notes(
    load_pyatb, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from ase import Atoms

    module = load_pyatb("pyatb.symmetry.symm_stru")
    analyzer = module.SymmStructureAnalyzer(_FakeTB(), output_path=str(tmp_path / "Out" / "CHARACTER"))
    analyzer._output_path.mkdir(parents=True, exist_ok=True)

    atoms = Atoms(numbers=[1], scaled_positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=True)
    op = module.SymmetryOperation(
        rotation=np.eye(3, dtype=int),
        translation=np.zeros(3, dtype=float),
        inverse_rotation=np.eye(3, dtype=int),
        cart_rotation=np.eye(3, dtype=float),
        euler_zyz=np.zeros(3, dtype=float),
        spin_matrix=np.eye(2, dtype=complex),
        symbol="E",
        description="unity op.",
        axis=np.array([1.0, 0.0, 0.0], dtype=float),
    )

    class _OneOperationDB:
        doubnum = 2
        path = Path("/tmp/kLG_1.data")
        antisym = 0
        symops = [
            type("Op", (), {"rotation": np.eye(3, dtype=int), "translation": np.zeros(3, dtype=float)})(),
        ]

        def resolve_kpoint_from_star(self, *args, **kwargs):
            return _FakeResolution(_FakeEntry("GM", [0.0, 0.0, 0.0], []), [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    monkeypatch.setattr(analyzer, "_load_input_stru", lambda: (atoms, tmp_path / "STRU"))
    monkeypatch.setattr(
        analyzer,
        "_standardize_magnetic_cell",
        lambda atoms, magnetic_moments, symm_prec: (
            1,
            atoms,
            atoms,
            {"rotations": np.array([np.eye(3, dtype=int)]), "translations": np.zeros((1, 3), dtype=float)},
            {"rotations": np.array([np.eye(3, dtype=int)]), "translations": np.zeros((1, 3), dtype=float)},
            np.array([[0.0, 0.0, 1.0]], dtype=float),
        ),
    )
    monkeypatch.setattr(module.KLittleGroupsDB, "load", staticmethod(lambda path: _OneOperationDB()))
    monkeypatch.setattr(analyzer, "_build_symmetry_operations", lambda atoms, sym_data: [op])
    monkeypatch.setattr(analyzer, "_sort_operations_irvsp_like", lambda operations: operations)

    analyzer.analyze_magnetic(1, 1e-5, np.array([[0.0, 0.0, 1.0]]), kpoints_direct=np.zeros((1, 3)))

    text = (tmp_path / "Out" / "CHARACTER" / "symmetry_character_report.txt").read_text(encoding="utf-8")
    assert "All spglib symmetry operations fully match the kLittleGroups table." in text
    assert "We do NOT classify the elements into classes." not in text
    assert "Tables can be found on website" not in text


def test_character_data_symmetrization_uses_analysis_source_operations_for_magnetic_case(
    load_pyatb, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_pyatb("pyatb.symmetry.character")

    class _FakeSolver:
        def diago_H(self, k_direct_coor):
            del k_direct_coor
            return np.eye(1, dtype=complex).reshape(1, 1, 1), np.array([[0.0]], dtype=float)

        def get_Sk(self, k_direct_coor):
            del k_direct_coor
            return np.eye(1, dtype=complex).reshape(1, 1, 1)

    class _MagTB:
        nspin = 4
        max_kpoint_num = 10
        basis_num = 1
        lattice_vector = np.eye(3)
        lattice_constant = 1.0
        stru_atom = [type("AtomType", (), {"atom_num": 1})()]
        HSR_iSsparse = False

        def __init__(self):
            self.tb_solver = _FakeSolver()

        def read_stru(self, stru_file, need_orb=False):
            del stru_file, need_orb

    class _FlowIrrep:
        raw_name = "-A"
        name = "A"
        characters = np.array([2.0 + 0.0j], dtype=complex)
        active_ops = np.array([True], dtype=bool)

    class _FlowEntry:
        name = "GM"
        irreps = [_FlowIrrep()]

    class _FlowResolution:
        entry = _FlowEntry()
        k_conv = np.array([0.0, 0.0, 0.0], dtype=float)

    report_path = tmp_path / "Out" / "CHARACTER" / "symmetry_character_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "Magnetic symmetry detection:\n\n********************************************************************************\n\nknum =  1    kname= \n",
        encoding="utf-8",
    )

    analysis_operation = {
        "rotation": np.eye(3, dtype=int),
        "translation": np.array([0.0, 0.0, 0.25], dtype=float),
        "cart_rotation": np.eye(3, dtype=float),
        "spin_matrix": np.eye(2, dtype=complex),
    }

    monkeypatch.setattr(
        module.SymmStructureAnalyzer,
        "analyze_magnetic",
        lambda *args, **kwargs: {
            "resolved_group": 1,
            "operations": [analysis_operation],
            "source_operations": [analysis_operation],
            "kpoint_records": [
                {
                    "k_index": 1,
                    "k_direct": np.array([0.0, 0.0, 0.0], dtype=float),
                    "active_operation_indices": [0],
                    "resolution": _FlowResolution(),
                    "k_name": "GM",
                }
            ],
            "need_rebuild_hs": False,
            "canonical_kpoints_direct": np.array([[0.0, 0.0, 0.0]], dtype=float),
        },
    )
    class _FakeMetadata:
        basis_num = 1
        species_by_atom = ["H"]
        positions_frac = np.array([[0.0, 0.0, 0.0]], dtype=float)
        lattice_vector = np.eye(3, dtype=float)

    monkeypatch.setattr(
        module,
        "load_abacus_hs_blocks",
        lambda *args, **kwargs: (_FakeMetadata(), {(0, 0, 0): np.eye(1, dtype=complex)}, {(0, 0, 0): np.eye(1, dtype=complex)}),
    )
    monkeypatch.setattr(
        module,
        "get_symmetry_operations_from_metadata",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not query nonmagnetic spglib ops in magnetic path")),
    )
    monkeypatch.setattr(module, "prepare_operation_contexts", lambda metadata, operations, map_tol=1.0e-5: [])
    monkeypatch.setattr(
        module,
        "self_covariance_statistics",
        lambda *args, **kwargs: {"global_max_abs": 0.0, "mean_abs_over_operations": 0.0},
    )
    monkeypatch.setattr(module, "sequential_symmetrize_hs", lambda **kwargs: (kwargs["hr_blocks"], kwargs["sr_blocks"], []))
    monkeypatch.setattr(module, "write_symmetrized_hs", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "build_multixr_from_dense_blocks", lambda **kwargs: (object(), object()))
    monkeypatch.setattr(module, "build_dk_matrix", lambda *args, **kwargs: np.eye(1, dtype=complex), raising=False)

    class _SymmTB:
        def __init__(self, nspin, lattice_constant, lattice_vector, max_kpoint_num):
            self.nspin = nspin
            self.lattice_constant = lattice_constant
            self.lattice_vector = lattice_vector
            self.max_kpoint_num = max_kpoint_num
            self.HSR_iSsparse = False
            self.basis_num = 1
            self.tb_solver = _FakeSolver()

        def set_solver_HSR(self, hr_obj, sr_obj, sparse_flag):
            del hr_obj, sr_obj, sparse_flag

        def read_stru(self, stru_file, need_orb=True):
            del stru_file, need_orb

    monkeypatch.setattr(module, "TBModel", _SymmTB)

    tb = _MagTB()
    cal = module.Character(tb)
    cal.calculate_character(
        stru_file="STRU",
        kpoint_mode="direct",
        kpoint_num=1,
        kpoint_direct_coor=[[0.0, 0.0, 0.0]],
        group=1,
        symm_prec=1e-5,
        occ_band=1,
        band=[1, 1],
        mag_tag=1,
        mag=[0.0, 0.0, 1.0],
        data_symmetrize=1,
    )


def test_character_raises_when_covariance_error_exceeds_threshold(
    load_pyatb, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_pyatb("pyatb.symmetry.character")

    class _TB:
        nspin = 1
        max_kpoint_num = 10

        def read_stru(self, stru_file, need_orb=False):
            del stru_file, need_orb

    monkeypatch.setattr(
        module.SymmStructureAnalyzer,
        "analyze_nonmagnetic",
        lambda *args, **kwargs: {
            "resolved_group": 1,
            "need_rebuild_hs": False,
            "operations": [],
            "source_operations": [
                {
                    "rotation": np.eye(3, dtype=int),
                    "translation": np.zeros(3, dtype=float),
                    "cart_rotation": np.eye(3, dtype=float),
                }
            ],
            "kpoint_records": [],
            "canonical_kpoints_direct": np.array([[0.0, 0.0, 0.0]], dtype=float),
        },
    )

    class _FakeMetadata:
        basis_num = 1
        species_by_atom = ["H"]
        positions_frac = np.array([[0.0, 0.0, 0.0]], dtype=float)
        lattice_vector = np.eye(3, dtype=float)

    hr_blocks = {(0, 0, 0): np.array([[1.0 + 0.0j]], dtype=complex)}
    sr_blocks = {(0, 0, 0): np.array([[1.0 + 0.0j]], dtype=complex)}
    monkeypatch.setattr(
        module,
        "load_abacus_hs_blocks",
        lambda *args, **kwargs: (_FakeMetadata(), hr_blocks, sr_blocks),
    )
    monkeypatch.setattr(module, "prepare_operation_contexts", lambda *args, **kwargs: [("op", {"ctx": True})])

    def _fake_cov_stats(blocks, *args, **kwargs):
        del args, kwargs
        if blocks is hr_blocks:
            return {"global_max_abs": 2.0e-1, "mean_abs_over_operations": 1.0e-3}
        return {"global_max_abs": 5.0e-2, "mean_abs_over_operations": 1.0e-4}

    monkeypatch.setattr(module, "self_covariance_statistics", _fake_cov_stats)
    monkeypatch.setattr(module.Character, "_calculate_character_rows", lambda self, *args, **kwargs: [])

    cal = module.Character(_TB())
    with pytest.raises(ValueError, match="哈密顿数据或者交叠矩阵数据在对称性操作下误差过大"):
        cal.calculate_character(
            stru_file="STRU",
            kpoint_mode="direct",
            kpoint_num=1,
            kpoint_direct_coor=[[0.0, 0.0, 0.0]],
            group="auto",
            symm_prec=1e-5,
            occ_band=1,
            band=[1, 1],
            mag_tag=0,
            mag="auto",
            data_symmetrize=0,
        )


def test_character_warns_when_covariance_error_is_moderate(
    load_pyatb, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_pyatb("pyatb.symmetry.character")

    class _TB:
        nspin = 1
        max_kpoint_num = 10

        def read_stru(self, stru_file, need_orb=False):
            del stru_file, need_orb

    monkeypatch.setattr(
        module.SymmStructureAnalyzer,
        "analyze_nonmagnetic",
        lambda *args, **kwargs: {
            "resolved_group": 1,
            "need_rebuild_hs": False,
            "operations": [],
            "source_operations": [
                {
                    "rotation": np.eye(3, dtype=int),
                    "translation": np.zeros(3, dtype=float),
                    "cart_rotation": np.eye(3, dtype=float),
                }
            ],
            "kpoint_records": [],
            "canonical_kpoints_direct": np.array([[0.0, 0.0, 0.0]], dtype=float),
        },
    )

    class _FakeMetadata:
        basis_num = 1
        species_by_atom = ["H"]
        positions_frac = np.array([[0.0, 0.0, 0.0]], dtype=float)
        lattice_vector = np.eye(3, dtype=float)

    hr_blocks = {(0, 0, 0): np.array([[1.0 + 0.0j]], dtype=complex)}
    sr_blocks = {(0, 0, 0): np.array([[1.0 + 0.0j]], dtype=complex)}
    monkeypatch.setattr(
        module,
        "load_abacus_hs_blocks",
        lambda *args, **kwargs: (_FakeMetadata(), hr_blocks, sr_blocks),
    )
    monkeypatch.setattr(module, "prepare_operation_contexts", lambda *args, **kwargs: [("op", {"ctx": True})])

    def _fake_cov_stats(blocks, *args, **kwargs):
        del args, kwargs
        if blocks is hr_blocks:
            return {"global_max_abs": 2.0e-4, "mean_abs_over_operations": 1.0e-6}
        return {"global_max_abs": 8.0e-6, "mean_abs_over_operations": 5.0e-7}

    monkeypatch.setattr(module, "self_covariance_statistics", _fake_cov_stats)
    monkeypatch.setattr(module.Character, "_calculate_character_rows", lambda self, *args, **kwargs: [])

    cal = module.Character(_TB())
    with pytest.warns(
        UserWarning,
        match="covariance of the input HR/SR data is not high enough; set data_symmetrize = 1",
    ):
        cal.calculate_character(
            stru_file="STRU",
            kpoint_mode="direct",
            kpoint_num=1,
            kpoint_direct_coor=[[0.0, 0.0, 0.0]],
            group="auto",
            symm_prec=1e-5,
            occ_band=1,
            band=[1, 1],
            mag_tag=0,
            mag="auto",
            data_symmetrize=0,
        )


def test_magnetic_requested_group_with_more_ops_than_unitary_group_fails(
    load_pyatb, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_pyatb("pyatb.symmetry.symm_stru")
    analyzer = module.SymmStructureAnalyzer(_FakeTB(), output_path=str(tmp_path / "Out" / "CHARACTER"))
    analyzer._output_path.mkdir(parents=True, exist_ok=True)

    def _fake_load_input_stru():
        from ase import Atoms

        return (
            Atoms(numbers=[1], scaled_positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=True),
            tmp_path / "STRU",
        )

    class _TwoOperationDB:
        doubnum = 4
        symops = [
            type("Op", (), {"rotation": np.eye(3, dtype=int), "translation": np.zeros(3, dtype=float)})(),
        ]
        path = Path("/tmp/kLG_2.data")

    monkeypatch.setattr(analyzer, "_load_input_stru", _fake_load_input_stru)
    monkeypatch.setattr(
        module.SymmStructureAnalyzer,
        "_get_magnetic_unitary_symmetry_data",
        staticmethod(
            lambda atoms, magnetic_moments, symm_prec: (
                {"rotations": np.array([np.eye(3, dtype=int)]), "translations": np.zeros((1, 3), dtype=float)},
                1,
            )
        ),
    )
    monkeypatch.setattr(module.KLittleGroupsDB, "load", staticmethod(lambda path: _TwoOperationDB()))
    monkeypatch.setattr(
        module.SymmStructureAnalyzer,
        "_reorder_operations_with_database",
        lambda self, operations, db: (operations, []),
    )

    with pytest.raises(ValueError, match="Requested space group has more operations than the detected magnetic unitary group"):
        analyzer.analyze_magnetic(2, 1e-5, np.array([[0.0, 0.0, 1.0]]), kpoints_direct=np.zeros((1, 3)))


def test_nonmagnetic_reuses_source_when_standardization_is_only_origin_shift(
    load_pyatb, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from ase import Atoms

    module = load_pyatb("pyatb.symmetry.symm_stru")
    analyzer = module.SymmStructureAnalyzer(_FakeTB(), output_path=str(tmp_path / "Out" / "CHARACTER"))
    analyzer._output_path.mkdir(parents=True, exist_ok=True)

    lattice = np.eye(3, dtype=float)
    source_atoms = Atoms(
        numbers=[1, 1],
        scaled_positions=[[0.25, 0.75, 0.0], [0.75, 0.25, 0.0]],
        cell=lattice,
        pbc=True,
    )
    std_atoms = Atoms(
        numbers=[1, 1],
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]],
        cell=lattice,
        pbc=True,
    )
    rotation = np.diag([-1, -1, 1]).astype(int)
    source_sym_data = {
        "rotations": np.array([np.eye(3, dtype=int), rotation], dtype=int),
        "translations": np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]], dtype=float),
    }
    std_sym_data = {
        "rotations": np.array([np.eye(3, dtype=int), rotation], dtype=int),
        "translations": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float),
    }

    class _Dataset:
        number = 129

    class _DB:
        doubnum = 4
        path = Path("/tmp/kLG_129.data")
        symops = [
            type("Op", (), {"rotation": np.eye(3, dtype=int), "translation": np.zeros(3, dtype=float)})(),
            type("Op", (), {"rotation": rotation, "translation": np.array([0.5, 0.5, 0.0], dtype=float)})(),
        ]

        def resolve_kpoint_from_star(self, *args, **kwargs):
            raise AssertionError("kpoint resolution is not needed in this regression test")

    monkeypatch.setattr(analyzer, "_load_input_stru", lambda: (source_atoms, tmp_path / "STRU"))
    monkeypatch.setattr(
        analyzer,
        "_standardize_nonmagnetic_cell",
        lambda atoms, symm_prec: (_Dataset(), std_atoms, source_atoms, std_sym_data),
    )
    monkeypatch.setattr(analyzer, "_get_symmetry_data", lambda atoms, symm_prec: source_sym_data)
    monkeypatch.setattr(module.KLittleGroupsDB, "load", staticmethod(lambda path: _DB()))

    result = analyzer.analyze_nonmagnetic(129, 1.0e-5, kpoints_direct=np.zeros((0, 3)))

    assert result["need_rebuild_hs"] is False
    assert result["rebuild_reason"] == "reuse-original-origin-shift"
    np.testing.assert_allclose(result["operations"][1].translation, np.array([0.5, 0.5, 0.0]))


def test_kpoint_record_uses_detected_structure_little_group_not_database_active_ops(
    load_pyatb, tmp_path: Path
) -> None:
    module = load_pyatb("pyatb.symmetry.symm_stru")
    analyzer = module.SymmStructureAnalyzer(_FakeTB(), output_path=str(tmp_path / "Out" / "CHARACTER"))

    identity = module.SymmetryOperation(
        rotation=np.eye(3, dtype=int),
        translation=np.zeros(3, dtype=float),
        inverse_rotation=np.eye(3, dtype=int),
        cart_rotation=np.eye(3, dtype=float),
        euler_zyz=np.zeros(3, dtype=float),
        spin_matrix=np.eye(2, dtype=complex),
        symbol="E",
        description="unity op.",
        axis=np.array([1.0, 0.0, 0.0], dtype=float),
    )
    mirror_x = module.SymmetryOperation(
        rotation=np.diag([-1, 1, 1]).astype(int),
        translation=np.zeros(3, dtype=float),
        inverse_rotation=np.diag([-1, 1, 1]).astype(int),
        cart_rotation=np.diag([-1, 1, 1]).astype(float),
        euler_zyz=np.zeros(3, dtype=float),
        spin_matrix=np.eye(2, dtype=complex),
        symbol="M",
        description="mirror op.",
        axis=np.array([1.0, 0.0, 0.0], dtype=float),
    )

    class _Entry:
        name = "GP"
        irreps = []

        @property
        def little_group_ops(self):
            return np.array([0, 1], dtype=int)

    class _Resolution:
        entry = _Entry()
        entry_index = 1
        rotated_k_prim = np.array([0.25, 0.0, 0.0], dtype=float)
        k_conv = np.array([0.25, 0.0, 0.0], dtype=float)
        rotation_index = 1
        variable_count = 0

    class _DB:
        def resolve_kpoint_from_star(self, *args, **kwargs):
            return _Resolution()

    records = analyzer._resolve_kpoint_records(
        [identity, mirror_x],
        _DB(),
        np.array([[0.25, 0.0, 0.0]], dtype=float),
    )

    assert records[0]["little_group_indices"] == [1]
    assert records[0]["active_operation_indices"] == [0]


def test_star_rotated_x_point_uses_representative_table_operation_order(load_pyatb, tmp_path: Path) -> None:
    module = load_pyatb("pyatb.symmetry.symm_stru")
    db_module = load_pyatb("pyatb.symmetry.k_little_groups")
    analyzer = module.SymmStructureAnalyzer(_FakeTB(), output_path=str(tmp_path / "Out" / "CHARACTER"))
    db_path = Path(db_module.__file__).resolve().parents[1] / "kLittleGroups" / "kLG_129.data"
    db = db_module.KLittleGroupsDB.load(db_path)

    operations = []
    for db_op in db.symops[: db.doubnum // 2]:
        rotation = np.asarray(db_op.rotation, dtype=int)
        operations.append(
            module.SymmetryOperation(
                rotation=rotation,
                translation=np.asarray(db_op.translation, dtype=float),
                inverse_rotation=np.rint(np.linalg.inv(rotation)).astype(int),
                cart_rotation=rotation.astype(float),
                euler_zyz=np.zeros(3, dtype=float),
                spin_matrix=np.eye(2, dtype=complex),
                symbol="E",
                description="",
                axis=np.array([1.0, 0.0, 0.0], dtype=float),
            )
        )

    records = analyzer._resolve_kpoint_records(
        operations,
        db,
        np.array([[0.5, 0.0, 0.0]], dtype=float),
    )

    assert records[0]["k_name"] == "X"
    assert records[0]["active_operation_indices"] == [0, 1, 4, 5, 8, 9, 12, 13]
    assert records[0]["table_operation_indices"] == [0, 1, 5, 4, 8, 9, 13, 12]


def test_star_rotation_table_mapping_uses_full_seitz_translation(load_pyatb, tmp_path: Path) -> None:
    module = load_pyatb("pyatb.symmetry.symm_stru")
    analyzer = module.SymmStructureAnalyzer(_FakeTB(), output_path=str(tmp_path / "Out" / "CHARACTER"))

    def _operation(rotation, translation):
        rotation = np.asarray(rotation, dtype=int)
        return module.SymmetryOperation(
            rotation=rotation,
            translation=np.asarray(translation, dtype=float),
            inverse_rotation=np.rint(np.linalg.inv(rotation)).astype(int),
            cart_rotation=rotation.astype(float),
            euler_zyz=np.zeros(3, dtype=float),
            spin_matrix=np.eye(2, dtype=complex),
            symbol="E",
            description="",
            axis=np.array([1.0, 0.0, 0.0], dtype=float),
        )

    star_rotation = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=int)
    active_rotation = np.diag([-1, 1, 1]).astype(int)
    representative_rotation = np.diag([1, -1, 1]).astype(int)
    operations = [
        _operation(np.eye(3, dtype=int), [0.0, 0.0, 0.0]),
        _operation(star_rotation, [0.25, 0.0, 0.0]),
        _operation(active_rotation, [0.5, 0.0, 0.0]),
        _operation(representative_rotation, [0.0, 0.0, 0.0]),
        _operation(representative_rotation, [0.0, 0.5, 0.0]),
    ]

    table_ops = analyzer._representative_table_operation_indices(
        operations,
        active_operation_indices=[2],
        rotation_index=2,
    )

    assert table_ops == [4]


def test_star_table_mapping_general_point_strictly_resolves_without_database_fallback(load_pyatb, tmp_path: Path) -> None:
    module = load_pyatb("pyatb.symmetry.symm_stru")
    db_module = load_pyatb("pyatb.symmetry.k_little_groups")
    analyzer = module.SymmStructureAnalyzer(_FakeTB(), output_path=str(tmp_path / "Out" / "CHARACTER"))
    db_path = Path(db_module.__file__).resolve().parents[1] / "kLittleGroups" / "kLG_166.data"
    db = db_module.KLittleGroupsDB.load(db_path)

    operations = []
    for db_op in db.symops[: db.doubnum // 2]:
        candidates = analyzer._database_operation_candidates(db_op, db)
        rotation = np.asarray(candidates[1][0] if len(candidates) > 1 else candidates[0][0], dtype=int)
        translation = np.asarray(candidates[1][1] if len(candidates) > 1 else candidates[0][1], dtype=float)
        operations.append(
            module.SymmetryOperation(
                rotation=rotation,
                translation=translation,
                inverse_rotation=np.rint(np.linalg.inv(rotation)).astype(int),
                cart_rotation=rotation.astype(float),
                euler_zyz=np.zeros(3, dtype=float),
                spin_matrix=np.eye(2, dtype=complex),
                symbol="E",
                description="",
                axis=np.array([1.0, 0.0, 0.0], dtype=float),
            )
        )

    records = analyzer._resolve_kpoint_records(
        operations,
        db,
        np.array([[0.413, 0.271, 0.197]], dtype=float),
    )

    assert records[0]["k_name"] == "GP"
    assert records[0]["little_group_indices"] == [1]
    assert records[0]["active_operation_indices"] == [0]
    assert records[0]["table_operation_indices"] == [0]


def test_nonsymmorphic_boundary_point_resolves_to_y_when_k_basis_mapping_is_correct(load_pyatb, tmp_path: Path) -> None:
    module = load_pyatb("pyatb.symmetry.symm_stru")
    db_module = load_pyatb("pyatb.symmetry.k_little_groups")
    analyzer = module.SymmStructureAnalyzer(_FakeTB(), output_path=str(tmp_path / "Out" / "CHARACTER"))
    db_path = Path(db_module.__file__).resolve().parents[1] / "kLittleGroups" / "kLG_166.data"
    db = db_module.KLittleGroupsDB.load(db_path)

    operations = []
    for db_op in db.symops[: db.doubnum // 2]:
        candidates = analyzer._database_operation_candidates(db_op, db)
        rotation = np.asarray(candidates[1][0] if len(candidates) > 1 else candidates[0][0], dtype=int)
        translation = np.asarray(candidates[1][1] if len(candidates) > 1 else candidates[0][1], dtype=float)
        operations.append(
            module.SymmetryOperation(
                rotation=rotation,
                translation=translation,
                inverse_rotation=np.rint(np.linalg.inv(rotation)).astype(int),
                cart_rotation=rotation.astype(float),
                euler_zyz=np.zeros(3, dtype=float),
                spin_matrix=np.eye(2, dtype=complex),
                symbol="E",
                description="",
                axis=np.array([1.0, 0.0, 0.0], dtype=float),
            )
        )

    records = analyzer._resolve_kpoint_records(
        operations,
        db,
        np.array([[0.516145426055, 0.483854573945, 0.5]], dtype=float),
    )
    assert records[0]["k_name"] == "Y"
    assert records[0]["little_group_indices"] == [1, 5]
    assert records[0]["active_operation_indices"] == [0, 4]
    assert records[0]["table_operation_indices"] == [0, 3]



def test_dk_matrix_identity(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.Dk_matrix")
    matrix = module.spin_half_matrix_from_cartesian_rotation([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert matrix.shape == (2, 2)
    assert abs(matrix[0, 0] - 1.0) < 1e-12
    assert abs(matrix[1, 1] - 1.0) < 1e-12
    assert abs(matrix[0, 1]) < 1e-12
    assert abs(matrix[1, 0]) < 1e-12


def test_build_dk_matrix_returns_identity_for_single_s_orbital(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.Dk_matrix")

    class _AtomType:
        species = "H"
        atom_num = 1
        orbital_num = [1]
        cartesian_coor = np.array([[0.0, 0.0, 0.0]], dtype=float)

    class _TB:
        nspin = 1
        basis_num = 1
        lattice_vector = np.eye(3)
        lattice_constant = 1.0
        stru_atom = [_AtomType()]

    op = {
        "rotation": np.eye(3, dtype=int),
        "translation": np.zeros(3, dtype=float),
        "cart_rotation": np.eye(3, dtype=float),
    }

    matrix = module.build_dk_matrix(_TB(), np.array([0.0, 0.0, 0.0], dtype=float), op)

    assert matrix.shape == (1, 1)
    assert abs(matrix[0, 0] - 1.0) < 1.0e-12


def test_build_dk_matrix_supports_soc_interleaved_spin_basis(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.Dk_matrix")

    class _AtomType:
        species = "H"
        atom_num = 1
        orbital_num = [1]
        cartesian_coor = np.array([[0.0, 0.0, 0.0]], dtype=float)

    class _TB:
        nspin = 4
        basis_num = 2
        lattice_vector = np.eye(3)
        lattice_constant = 1.0
        stru_atom = [_AtomType()]

    op = {
        "rotation": np.eye(3, dtype=int),
        "translation": np.zeros(3, dtype=float),
        "cart_rotation": np.eye(3, dtype=float),
        "spin_matrix": np.eye(2, dtype=complex),
    }

    matrix = module.build_dk_matrix(_TB(), np.array([0.0, 0.0, 0.0], dtype=float), op)

    assert matrix.shape == (2, 2)
    assert np.allclose(matrix, np.eye(2, dtype=complex))


def test_dk_find_atom_mapping_canonicalizes_boundary_equivalent_positions(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.Dk_matrix")

    metadata = module.BasisMetadata(
        basis_num=1,
        spinless_basis_num=1,
        spin_factor=1,
        lattice_vector=np.eye(3, dtype=float),
        positions_frac=np.array([[0.0, 1.0, 0.0]], dtype=float),
        species_by_atom=["Se"],
        shells=[],
        atom_ranges={0: (0, 1)},
    )
    operation = {
        "rotation": np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=int),
        "translation": np.zeros(3, dtype=float),
        "cart_rotation": np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
    }

    mapping = module.find_atom_mapping(metadata, operation)

    assert len(mapping) == 1
    np.testing.assert_array_equal(mapping[0]["cell_shift"], np.array([0, 0, 0], dtype=int))


def test_build_atom_local_rotation_supports_active_and_passive_conventions(
    load_pyatb, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_pyatb("pyatb.symmetry.Dk_matrix")

    class _Meta:
        spin_factor = 2

    captured = {}

    def _fake_orbital(metadata, atom_index, op):
        del metadata, atom_index
        captured.setdefault("orbital_rotations", []).append(np.asarray(op["cart_rotation"], dtype=float))
        return np.eye(1, dtype=complex)

    def _fake_spin(cart_rotation):
        captured.setdefault("spin_rotations", []).append(np.asarray(cart_rotation, dtype=float))
        return np.eye(2, dtype=complex)

    monkeypatch.setattr(module, "atom_orbital_rotation", _fake_orbital)
    monkeypatch.setattr(module, "spin_half_matrix_from_cartesian_rotation", _fake_spin)

    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    inverse_rotation = np.linalg.inv(rotation)

    active = module.build_atom_local_rotation(_Meta(), atom_index=0, cart_rotation=rotation, passive_basis=False)
    passive = module.build_atom_local_rotation(_Meta(), atom_index=0, cart_rotation=rotation, passive_basis=True)

    assert active.shape == (2, 2)
    assert passive.shape == (2, 2)
    np.testing.assert_allclose(captured["orbital_rotations"][0], rotation)
    np.testing.assert_allclose(captured["spin_rotations"][0], rotation)
    np.testing.assert_allclose(captured["orbital_rotations"][1], inverse_rotation)
    np.testing.assert_allclose(captured["spin_rotations"][1], inverse_rotation)



def test_k_little_group_symbolic_coordinate_format_matches_irvsp(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.k_little_groups")
    formatted = module.kreal_to_string(np.array([0.123, -0.123, 0.427], dtype=float))
    assert formatted == "  u   -u    w  "



def test_k_little_group_resolve_f_point_via_star_rotation(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.k_little_groups")
    db_path = Path(module.__file__).resolve().parents[1] / "kLittleGroups" / "kLG_166.data"
    db = module.KLittleGroupsDB.load(db_path)
    inverse_rotations = [
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int),
        np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=int),
        np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=int),
        np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]], dtype=int),
        np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], dtype=int),
        np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=int),
        np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=int),
        np.array([[0, 0, -1], [-1, 0, 0], [0, -1, 0]], dtype=int),
        np.array([[0, -1, 0], [0, 0, -1], [-1, 0, 0]], dtype=int),
        np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=int),
        np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=int),
        np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=int),
    ]

    resolved = db.resolve_kpoint_from_star(
        np.array([0.5, 0.5, 0.0], dtype=float),
        inverse_rotations,
        has_inversion=True,
        little_group_size=4,
        detected_ops=[1, 5, 7, 11],
    )

    assert resolved.entry.name == "F"
    assert np.allclose(resolved.rotated_k_prim, np.array([0.5, 0.5, 0.0], dtype=float))
    assert np.allclose(resolved.k_conv, np.array([0.0, 0.5, 1.0], dtype=float))


def test_k_little_group_centering_kc2p_matches_irvsp_reference(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.k_little_groups")

    np.testing.assert_allclose(
        module._centering_kc2p("B"),
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.0, -0.5, 0.5],
            ],
            dtype=float,
        ),
    )
    np.testing.assert_allclose(
        module._centering_kc2p("A"),
        np.array(
            [
                [0.5, 0.0, 0.5],
                [0.0, 1.0, 0.0],
                [-0.5, 0.0, 0.5],
            ],
            dtype=float,
        ),
    )
    np.testing.assert_allclose(
        module._centering_kc2p("R"),
        np.array(
            [
                [2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0],
                [1.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0],
                [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            ],
            dtype=float,
        ),
    )


def test_k_little_group_explicit_k0_k1_k2_chain_uses_database_primitive_basis(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.k_little_groups")

    class _Entry:
        def __init__(self, name, k_conv, k_prim):
            self.name = name
            self.k_conv = np.asarray(k_conv, dtype=float)
            self.k_prim = np.asarray(k_prim, dtype=float)
            self.antisym = 1
            self.irreps = [type("Ir", (), {"active_ops": np.array([True, True], dtype=bool)})()]

        @property
        def little_group_ops(self):
            return np.array([0, 1], dtype=int)

    star_inv = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=int)

    db = module.KLittleGroupsDB(
        path=Path("/tmp/kLG_166.data"),
        spacegroup_symbol="P",
        doubnum=2,
        symops=[],
        kpoints=[
            _Entry("GP", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            _Entry("X", [0.5, 0.0, 0.0], [0.5, 0.0, 0.0]),
        ],
    )

    current_to_db = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    inverse_rotations = [np.eye(3, dtype=int), star_inv]

    resolved = db.resolve_kpoint_from_star(
        np.array([0.5, 0.0, 0.0], dtype=float),
        inverse_rotations,
        has_inversion=False,
        little_group_size=2,
        detected_ops=[1, 4],
        current_to_db_prim=current_to_db,
    )

    assert resolved.entry.name == "X"
    np.testing.assert_allclose(
        resolved.mapped_k_prim,
        np.array([0.0, 0.5, 0.0], dtype=float),
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        resolved.rotated_k_prim,
        np.array([0.5, 0.0, 0.0], dtype=float),
        atol=1.0e-6,
    )
    assert resolved.rotation_index == 2


def test_k_little_group_table_characters_conjugate_raw_table_on_nonsymmorphic_boundary(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.k_little_groups")
    db = module.KLittleGroupsDB(
        path=Path("/tmp/kLG_205.data"),
        spacegroup_symbol="P",
        doubnum=4,
        symops=[],
        kpoints=[],
    )
    irrep = module.IrrepEntry(
        raw_name="T5",
        name="T5",
        reality=1,
        dimension=1,
        characters=np.array([1.0 + 0.0j, 0.0 + 1.0j, 0.0 + 1.0j, -1.0 + 0.0j], dtype=complex),
        active_ops=np.array([True, True, True, True], dtype=bool),
        phase_kinds=np.array([1, 2, 2, 1], dtype=int),
        coeff_uvw=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        factor_strings=["            "] * 4,
    )
    entry = module.KPointEntry(
        name="T",
        k_conv=np.array([0.5, 0.5, 0.026316], dtype=float),
        k_prim=np.array([0.5, 0.5, 0.026316], dtype=float),
        antisym=1,
        irreps=[irrep],
    )
    resolution = module.KPointResolution(
        entry=entry,
        entry_index=6,
        rotated_k_prim=entry.k_prim,
        k_conv=entry.k_conv,
        rotation_index=1,
        variable_count=1,
        cornwell_satisfied=False,
    )

    traces = db.irrep_table_characters(resolution, irrep)

    assert np.allclose(
        traces,
        np.array([1.0 + 0.0j, -0.0826 - 0.9966j, -0.0826 - 0.9966j, -1.0 + 0.0j], dtype=complex),
        atol=5.0e-5,
    )



def test_transformations_use_irvsp_column_vector_layout(load_pyatb) -> None:
    from ase import Atoms

    module = load_pyatb("pyatb.symmetry.symm_stru")
    analyzer = module.SymmStructureAnalyzer(_FakeTB(), output_path="Out/CHARACTER")
    atoms = Atoms(numbers=[1], scaled_positions=[[0.0, 0.0, 0.0]], cell=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]], pbc=True)
    buf = StringIO()

    analyzer._write_transformations(buf, atoms, atoms, Path("/tmp/kLG_1.data"), {"need_rebuild_hs": False})
    text = buf.getvalue()

    assert "a1       1.00000000      2.00000000      3.00000000" in text
    assert "a2       4.00000000      5.00000000      6.00000000" in text
    assert "a3       7.00000000      8.00000000     10.00000000" in text



def test_k_little_group_table_uses_star_resolution_and_irvsp_header(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.symm_stru")
    analyzer = module.SymmStructureAnalyzer(_FakeTB(), output_path="Out/CHARACTER")

    irrep = _FakeIrrep(
        raw_name="F1+",
        name="F1+",
        reality=1,
        characters=[1 + 0j] * 12,
        active_ops=[True, False, False, False, True, False, True, False, False, False, True, False],
    )
    entry = _FakeEntry("F", [0.0, 0.5, 1.0], [irrep], antisym=1)
    resolution = _FakeResolution(entry, [0.5, 0.5, 0.5], [0.0, 0.5, 1.0], entry_index=4)
    db = _FakeDB(resolution, [1 + 0j] * 12)

    operations = []
    for symbol, axis, rotation in [
        ("E", [1.0, 0.0, 0.0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        ("C3", [0.0, 0.0, 1.0], [[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
        ("C3", [0.0, 0.0, 1.0], [[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
        ("C2", [0.5, 0.866, 0.0], [[0, 0, -1], [0, -1, 0], [-1, 0, 0]]),
        ("C2", [1.0, 0.0, 0.0], [[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),
        ("C2", [0.5, -0.866, 0.0], [[-1, 0, 0], [0, 0, -1], [0, -1, 0]]),
        ("I", [1.0, 0.0, 0.0], [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]),
        ("IC3", [0.0, 0.0, 1.0], [[0, 0, -1], [-1, 0, 0], [0, -1, 0]]),
        ("IC3", [0.0, 0.0, 1.0], [[0, -1, 0], [0, 0, -1], [-1, 0, 0]]),
        ("IC2", [0.5, 0.866, 0.0], [[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
        ("IC2", [1.0, 0.0, 0.0], [[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
        ("IC2", [0.5, -0.866, 0.0], [[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
    ]:
        rotation = np.array(rotation, dtype=int)
        operations.append(
            module.SymmetryOperation(
                rotation=rotation,
                translation=np.zeros(3, dtype=float),
                inverse_rotation=np.rint(np.linalg.inv(rotation)).astype(int),
                cart_rotation=np.eye(3, dtype=float),
                euler_zyz=np.zeros(3, dtype=float),
                spin_matrix=np.eye(2, dtype=complex),
                symbol=symbol,
                description="desc",
                axis=np.array(axis, dtype=float),
            )
        )

    buf = StringIO()
    analyzer._write_k_little_group_table(buf, operations, db, np.array([[0.5, 0.5, 0.0]], dtype=float))
    text = buf.getvalue()

    assert db.calls and np.allclose(db.calls[0]["k_prim"], [0.5, 0.5, 0.0])
    assert db.calls[0]["detected_ops"] == [1, 5, 7, 11]
    assert "knum =  1    kname=" in text
    assert "The k-point name is F" in text
    assert "    4    F  : kname      0.00 0.50 1.00 :  given in the conventional basis" in text
    assert " Reality           1           5           7          11" in text
    assert "F1+" in text
    assert text.count("1.00+0.00i") >= 4



def test_k_little_group_table_omits_phase_factor_column(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.symm_stru")
    analyzer = module.SymmStructureAnalyzer(_FakeTB(), output_path="Out/CHARACTER")

    irrep = _FakeIrrep(
        raw_name="G1",
        name="G1",
        reality=1,
        characters=[1 + 0j],
        active_ops=[True],
    )
    entry = _FakeEntry("G", [0.0, 0.0, 0.0], [irrep], antisym=1)
    resolution = _FakeResolution(entry, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], entry_index=1)
    db = _FakeDB(resolution, [1 + 0j])
    op = module.SymmetryOperation(
        rotation=np.eye(3, dtype=int),
        translation=np.array([0.25, 0.0, 0.0], dtype=float),
        inverse_rotation=np.eye(3, dtype=int),
        cart_rotation=np.eye(3, dtype=float),
        euler_zyz=np.zeros(3, dtype=float),
        spin_matrix=np.eye(2, dtype=complex),
        symbol="E",
        description="unity op.",
        axis=np.array([1.0, 0.0, 0.0], dtype=float),
    )

    buf = StringIO()
    analyzer._write_k_little_group_table(buf, [op], db, np.array([[0.0, 0.0, 0.0]], dtype=float))
    text = buf.getvalue()

    assert "exp(-i*k*taui)" not in text
    assert "(+1.00 0.00i)" not in text
    assert "element" in text
    assert "symmetry ops" in text
    assert "main axes" in text



def test_build_symmetry_operations_suppresses_gimbal_lock_warning(load_pyatb) -> None:
    from ase import Atoms

    module = load_pyatb("pyatb.symmetry.symm_stru")
    atoms = Atoms(numbers=[1], scaled_positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=True)
    sym_data = {
        "rotations": np.array([np.eye(3, dtype=int)]),
        "translations": np.array([[0.0, 0.0, 0.0]], dtype=float),
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        operations = module.SymmStructureAnalyzer._build_symmetry_operations(atoms, sym_data)

    assert len(operations) == 1
    assert not any("Gimbal lock detected" in str(item.message) for item in caught)



def test_sort_operations_matches_irvsp_like_order(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.symm_stru")
    analyzer = module.SymmStructureAnalyzer(_FakeTB(), output_path="Out/CHARACTER")

    def op(symbol, axis, rotation, translation=(0.0, 0.0, 0.0)):
        return module.SymmetryOperation(
            rotation=np.array(rotation, dtype=int),
            translation=np.array(translation, dtype=float),
            inverse_rotation=np.array(rotation, dtype=int),
            cart_rotation=np.eye(3, dtype=float),
            euler_zyz=np.zeros(3, dtype=float),
            spin_matrix=np.eye(2, dtype=complex),
            symbol=symbol,
            description=symbol,
            axis=np.array(axis, dtype=float),
        )

    unordered = [
        op("IC2", [1.0, 0.0, 0.0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        op("C2", [1.0, 0.0, 0.0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        op("C3", [0.0, 0.0, 1.0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        op("I", [1.0, 0.0, 0.0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        op("E", [1.0, 0.0, 0.0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        op("IC3", [0.0, 0.0, 1.0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        op("C2", [0.5, 0.866, 0.0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        op("IC2", [0.5, -0.866, 0.0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    ]

    ordered = analyzer._sort_operations_irvsp_like(unordered)

    assert [item.symbol for item in ordered] == ["E", "C3", "C2", "C2", "I", "IC3", "IC2", "IC2"]
    assert np.allclose(ordered[2].axis, [0.5, 0.866, 0.0])
    assert np.allclose(ordered[3].axis, [1.0, 0.0, 0.0])


def test_standardization_reuses_original_files_when_only_permutation(load_pyatb, tmp_path: Path) -> None:
    module = load_pyatb("pyatb.symmetry.symm_stru")
    analyzer = module.SymmStructureAnalyzer(_FakeTB(), output_path=str(tmp_path))

    result = analyzer._finalize_standardization_result(
        detected_group=166,
        resolved_group=166,
        lattice_changed=False,
        atom_permutation_only=True,
        source_stru="STRU",
        source_hr="data-HR-sparse_SPIN0.csr",
        source_sr="data-SR-sparse_SPIN0.csr",
        atom_mapping=[],
        lattice_old=np.eye(3),
        lattice_new=np.eye(3),
        lattice_transform_fractional=np.eye(3),
        xyz_axis_transform_cartesian=np.eye(3),
        rebuild_reason="reuse-original",
    )

    assert result["need_rebuild_hs"] is False
    assert result["target_stru"] == "STRU"
    assert result["target_hr"].endswith("data-HR-sparse_SPIN0.csr")
    assert result["target_sr"].endswith("data-SR-sparse_SPIN0.csr")


def test_standardization_requires_rebuild_when_atoms_move_beyond_permutation(load_pyatb, tmp_path: Path) -> None:
    module = load_pyatb("pyatb.symmetry.symm_stru")
    analyzer = module.SymmStructureAnalyzer(_FakeTB(), output_path=str(tmp_path))

    result = analyzer._finalize_standardization_result(
        detected_group=166,
        resolved_group=166,
        lattice_changed=False,
        atom_permutation_only=False,
        source_stru="STRU",
        source_hr="data-HR-sparse_SPIN0.csr",
        source_sr="data-SR-sparse_SPIN0.csr",
        atom_mapping=[],
        lattice_old=np.eye(3),
        lattice_new=np.eye(3),
        lattice_transform_fractional=np.eye(3),
        xyz_axis_transform_cartesian=np.eye(3),
        rebuild_reason="atom-position-changed",
    )

    assert result["need_rebuild_hs"] is True
    assert result["target_stru"] == "STRU-symm"
    assert result["target_hr"].endswith("data-HR-sparse_SPIN0-symm.csr")
    assert result["target_sr"].endswith("data-SR-sparse_SPIN0-symm.csr")


def test_standardization_result_is_json_serializable_with_atom_shift_arrays(load_pyatb, tmp_path: Path) -> None:
    module = load_pyatb("pyatb.symmetry.symm_stru")
    analyzer = module.SymmStructureAnalyzer(_FakeTB(), output_path=str(tmp_path))

    result = analyzer._finalize_standardization_result(
        detected_group=166,
        resolved_group=166,
        lattice_changed=True,
        atom_permutation_only=False,
        source_stru="STRU",
        source_hr="data-HR-sparse_SPIN0.csr",
        source_sr="data-SR-sparse_SPIN0.csr",
        atom_mapping=[{"old_atom": 0, "new_atom": 1, "shift": np.array([1, 0, -1], dtype=int)}],
        lattice_old=np.eye(3),
        lattice_new=np.eye(3),
        lattice_transform_fractional=np.eye(3),
        xyz_axis_transform_cartesian=np.eye(3),
        rebuild_reason="lattice-changed",
    )

    payload = json.loads(json.dumps(result))
    assert payload["atom_mapping"][0]["shift"] == [1, 0, -1]


def test_build_atom_mapping_supports_supercell_to_primitive_basis_change(load_pyatb) -> None:
    from ase import Atoms

    module = load_pyatb("pyatb.symmetry.symm_stru")

    source = Atoms(
        numbers=[34, 34],
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        cell=[[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        pbc=True,
    )
    target = Atoms(
        numbers=[34],
        scaled_positions=[[0.0, 0.0, 0.0]],
        cell=np.eye(3),
        pbc=True,
    )

    mapping = module.SymmStructureAnalyzer._build_atom_mapping(source, target, tol=1.0e-8)

    assert len(mapping) == 2
    assert mapping[0][1] == 0
    assert mapping[1][1] == 0
    np.testing.assert_array_equal(mapping[0][2], np.array([0, 0, 0], dtype=int))
    np.testing.assert_array_equal(mapping[1][2], np.array([1, 0, 0], dtype=int))


def test_build_atom_mapping_wraps_boundary_equivalent_positions_without_extra_shift(load_pyatb) -> None:
    from ase import Atoms

    module = load_pyatb("pyatb.symmetry.symm_stru")

    source = Atoms(
        numbers=[34],
        scaled_positions=[[1.0, 0.0, 0.0]],
        cell=np.eye(3),
        pbc=True,
    )
    target = Atoms(
        numbers=[34],
        scaled_positions=[[0.0, 0.0, 0.0]],
        cell=np.eye(3),
        pbc=True,
    )

    mapping = module.SymmStructureAnalyzer._build_atom_mapping(source, target, tol=1.0e-8)

    assert len(mapping) == 1
    assert mapping[0][1] == 0
    np.testing.assert_array_equal(mapping[0][2], np.array([0, 0, 0], dtype=int))


def test_standardize_nonmagnetic_cell_separates_mapping_and_canonical_primitive_cells(load_pyatb, tmp_path: Path) -> None:
    from ase import Atoms

    module = load_pyatb("pyatb.symmetry.symm_stru")
    analyzer = module.SymmStructureAnalyzer(_FakeTB(), output_path=str(tmp_path))

    atoms = Atoms(
        numbers=[83, 83, 83, 83, 34, 34, 34, 34, 34, 34],
        scaled_positions=[
            [0.6019999577, 0.5, 0.1019999577],
            [0.3980000423, 0.5, 0.8980000423],
            [0.6019999577, 0.0, 0.6019999577],
            [0.3980000423, 0.0, 0.3980000423],
            [0.0, 0.5, 0.5],
            [0.2080000060, 0.5, 0.7080000060],
            [0.7919999940, 0.5, 0.2919999940],
            [0.0, 0.0, 0.0],
            [0.2080000060, 0.0, 0.2080000060],
            [0.7919999940, 0.0, 0.7919999940],
        ],
        cell=[
            [1.7957136, 2.67980484, -9.29321703],
            [-2.25329615, -3.18991417, -1.35524888],
            [-2.4096709, 8.87889606, -16.89225834],
        ],
        pbc=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        dataset, canonical_atoms, mapping_atoms, _ = analyzer._standardize_nonmagnetic_cell(atoms, 1.0e-5)

    assert int(dataset.number) == 166
    assert len(mapping_atoms) == 5
    assert len(canonical_atoms) == 5
    assert not np.allclose(mapping_atoms.cell.array, canonical_atoms.cell.array)
    diff = (
        np.asarray(mapping_atoms.get_scaled_positions(), dtype=float)
        - np.asarray(canonical_atoms.get_scaled_positions(), dtype=float)
    )
    diff -= np.rint(diff)
    np.testing.assert_allclose(diff, np.zeros_like(diff), atol=1.0e-8)

    mapping = analyzer._build_atom_mapping(atoms, mapping_atoms, tol=1.0e-6)
    assert len(mapping) == 10


def test_write_standardized_stru_preserves_pseudopotential_and_orbital_entries(load_pyatb, tmp_path: Path) -> None:
    from ase import Atoms

    module = load_pyatb("pyatb.symmetry.symm_stru")
    analyzer = module.SymmStructureAnalyzer(_FakeTB(), output_path=str(tmp_path))

    source_path = tmp_path / "STRU"
    target_path = tmp_path / "STRU-symm"
    source_path.write_text(
        "\n".join(
            [
                "ATOMIC_SPECIES",
                "Bi   208.9804 Bi.upf upf201",
                "Se    78.9710 Se.upf upf201",
                "",
                "NUMERICAL_ORBITAL",
                "Bi/Orbital_Bi_DZP/Bi_gga_10au_100Ry_2s2p2d1f.orb",
                "Se/Orbital_Se_DZP/Se_gga_10au_100Ry_2s2p2d1f.orb",
                "",
                "LATTICE_CONSTANT",
                "1.8897162",
                "",
                "LATTICE_VECTORS",
                "1.0 0.0 0.0",
                "0.0 1.0 0.0",
                "0.0 0.0 1.0",
                "",
                "ATOMIC_POSITIONS",
                "Direct",
                "",
                "Bi",
                "0.0",
                "1",
                "0.0 0.0 0.0 0 0 0",
                "",
                "Se",
                "0.0",
                "1",
                "0.5 0.5 0.5 0 0 0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    std_atoms = Atoms(
        numbers=[83, 34],
        scaled_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        cell=np.eye(3),
        pbc=True,
    )

    analyzer._write_standardized_stru(source_path, std_atoms, target_path)
    text = target_path.read_text(encoding="utf-8")

    assert "Bi   208.9804 Bi.upf upf201" in text
    assert "Se    78.9710 Se.upf upf201" in text
    assert "Bi/Orbital_Bi_DZP/Bi_gga_10au_100Ry_2s2p2d1f.orb" in text
    assert "Se/Orbital_Se_DZP/Se_gga_10au_100Ry_2s2p2d1f.orb" in text


def test_compose_two_stage_mapping_applies_intermediate_lattice_transform(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.symm_stru")

    mapping12 = [
        (0, 1, np.array([1, 0, 0], dtype=int)),
        (1, 0, np.array([0, 1, 0], dtype=int)),
    ]
    mapping23 = [
        (0, 2, np.array([2, 0, 0], dtype=int)),
        (1, 3, np.array([0, 3, 0], dtype=int)),
    ]
    b23 = np.array(
        [
            [2, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=int,
    )

    composed = module.SymmStructureAnalyzer._compose_two_stage_mapping(mapping12, mapping23, b23)

    assert composed[0][0] == 0
    assert composed[0][1] == 3
    np.testing.assert_array_equal(composed[0][2], np.array([2, 3, 0], dtype=int))
    assert composed[1][0] == 1
    assert composed[1][1] == 2
    np.testing.assert_array_equal(composed[1][2], np.array([2, 1, 0], dtype=int))


def test_build_rotated_atom_mapping_matches_rotated_primitive_atoms(load_pyatb) -> None:
    from ase import Atoms

    module = load_pyatb("pyatb.symmetry.symm_stru")

    source = Atoms(
        numbers=[1, 2],
        positions=np.array([[0.1, 0.2, 0.0], [0.7, 0.2, 0.0]], dtype=float),
        cell=np.eye(3),
        pbc=True,
    )
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    target = Atoms(
        numbers=[1, 2],
        positions=np.asarray(source.positions, dtype=float) @ rotation,
        cell=np.eye(3),
        pbc=True,
    )

    mapping, _, max_err = module.SymmStructureAnalyzer._build_rotated_atom_mapping(source, target, rotation, tol=1.0e-8)

    assert max_err <= 1.0e-8
    assert [(old_idx, new_idx) for old_idx, new_idx, _ in mapping] == [(0, 0), (1, 1)]
    for _, _, shift in mapping:
        np.testing.assert_array_equal(np.asarray(shift, dtype=int), np.zeros(3, dtype=int))


def test_lattice_transform_fractional_uses_row_vector_convention(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.symm_stru")

    lattice_old = np.array(
        [
            [1.7957136, 2.67980484, -9.29321703],
            [-2.25329615, -3.18991417, -1.35524888],
            [-2.4096709, 8.87889606, -16.89225834],
        ],
        dtype=float,
    )
    lattice_new = np.array(
        [
            [2.33148353, -2.84449094, 9.12375361],
            [-1.7957136, -2.67980484, 9.29321703],
            [0.07818738, -6.03440511, 7.76850473],
        ],
        dtype=float,
    )

    transform = module.SymmStructureAnalyzer._lattice_transform_fractional(lattice_old, lattice_new)

    np.testing.assert_allclose(
        transform,
        np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, -1.0]], dtype=float),
        atol=1.0e-8,
    )


def test_synchronize_block_keys_aligns_hr_and_sr_r_sets(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.hs_standardize")

    hr_blocks = {(0, 0, 0): np.eye(1, dtype=complex)}
    sr_blocks = {(1, 0, 0): np.eye(1, dtype=complex)}

    synced_hr, synced_sr = module._synchronize_block_keys(hr_blocks, sr_blocks, basis_num=1)

    assert set(synced_hr) == {(0, 0, 0), (1, 0, 0)}
    assert set(synced_sr) == {(0, 0, 0), (1, 0, 0)}
    assert synced_hr[(1, 0, 0)].shape == (1, 1)
    assert synced_sr[(0, 0, 0)].shape == (1, 1)


def test_data_symmetrized_multixr_aligns_hr_and_sr_r_sets(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.data_covariance_constraint")

    hr_obj, sr_obj = module.build_multixr_from_dense_blocks(
        hr_blocks={(0, 0, 0): np.eye(1, dtype=complex), (1, 0, 0): np.eye(1, dtype=complex)},
        sr_blocks={(0, 0, 0): np.eye(1, dtype=complex)},
        basis_num=1,
        nspin=1,
        hr_unit="Ry",
    )

    assert hr_obj.R_direct_coor.shape == sr_obj.R_direct_coor.shape
    np.testing.assert_array_equal(hr_obj.R_direct_coor, sr_obj.R_direct_coor)


def test_sequential_symmetrize_uses_full_block_average(load_pyatb, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_pyatb("pyatb.symmetry.data_covariance_constraint")

    calls: list[str] = []

    class _Summary:
        def __init__(self):
            self.max_abs = 0.0
            self.mean_abs = 0.0
            self.rms_abs = 0.0
            self.rel_fro = 0.0
            self.element_count = 1
            self.missing_predicted_R_count = 0
            self.extra_predicted_R_count = 0

        def to_dict(self):
            return {
                "max_abs": self.max_abs,
                "mean_abs": self.mean_abs,
                "rms_abs": self.rms_abs,
                "rel_fro": self.rel_fro,
                "element_count": self.element_count,
                "missing_predicted_R_count": self.missing_predicted_R_count,
                "extra_predicted_R_count": self.extra_predicted_R_count,
            }

    def _fake_transform_blocks_with_context(
        source_blocks,
        metadata,
        context,
        zero_tol=1.0e-14,
        nonzero_block_tol=1.0e-9,
        return_touched_pairs=False,
        active_pair_index=None,
    ):
        del metadata, context, zero_tol, nonzero_block_tol, active_pair_index
        transformed = {key: np.asarray(value, dtype=complex).copy() for key, value in source_blocks.items()}
        transformed[(0, 0, 0)] = np.array([[2.0 + 0.0j]], dtype=complex)
        touched = {((0, 0, 0), 0, 0)}
        if return_touched_pairs:
            return transformed, touched
        return transformed

    def _fake_average_block_sets(lhs_blocks, rhs_blocks, basis_num):
        del rhs_blocks, basis_num
        calls.append("full")
        return {key: np.asarray(value, dtype=complex).copy() for key, value in lhs_blocks.items()}

    monkeypatch.setattr(module, "transform_blocks_with_context", _fake_transform_blocks_with_context)
    monkeypatch.setattr(module, "average_block_sets", _fake_average_block_sets)
    monkeypatch.setattr(
        module,
        "average_block_sets_on_touched_pairs",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("touched-pair average should not be used")),
    )
    monkeypatch.setattr(module, "build_active_pair_index", lambda *args, **kwargs: {(0, 0, 0): {(0, 0)}})
    monkeypatch.setattr(module, "merge_active_pair_index", lambda active_index, touched_pairs: active_index)
    monkeypatch.setattr(module, "compare_block_sets_on_candidate_pairs", lambda *args, **kwargs: _Summary())
    monkeypatch.setattr(module, "_self_covariance_statistics_with_contexts", lambda *args, **kwargs: {"global_max_abs": 0.0})

    metadata = type("Meta", (), {"basis_num": 1})()
    operations = [{"index": 1}]
    operation_contexts = [({"index": 1}, {"rotation": np.eye(3, dtype=int), "translation": np.zeros(3), "pair_rows": [], "pair_row_by_source": {}})]
    hr_blocks = {(0, 0, 0): np.array([[1.0 + 0.0j]], dtype=complex)}
    sr_blocks = {(0, 0, 0): np.array([[1.0 + 0.0j]], dtype=complex)}

    hr_symm, sr_symm, history = module.sequential_symmetrize_hs(
        hr_blocks,
        sr_blocks,
        metadata,
        operations,
        operation_target_max_abs_ry=1.0e-8,
        max_iter_per_operation=1,
        nonzero_block_tol=1.0e-9,
        operation_contexts=operation_contexts,
    )

    assert calls == ["full", "full"]
    assert hr_symm[(0, 0, 0)][0, 0] == 1.0 + 0.0j
    assert sr_symm[(0, 0, 0)][0, 0] == 1.0 + 0.0j
    assert history[-1]["final_symmetry_error"]["HR"]["global_max_abs"] == 0.0


def test_triangular_vector_to_dense_keeps_abacus_upper_triangle_convention(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.hs_standardize")

    dense = module._triangular_vector_to_dense(np.array([1.0, 2.0, 3.0], dtype=float), basis_num=2)

    np.testing.assert_allclose(dense, np.array([[1.0, 2.0], [0.0, 3.0]], dtype=complex))


def test_local_rotation_includes_spin_rotation_for_soc(load_pyatb, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_pyatb("pyatb.symmetry.hs_standardize")

    class _Meta:
        spin_factor = 2

    monkeypatch.setattr(
        module,
        "build_atom_local_rotation",
        lambda metadata, atom_index, cart_rotation, passive_basis=False: np.eye(2, dtype=complex),
    )
    rotation = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)

    local = module._local_rotation(_Meta(), atom_index=0, xyz_axis_transform_cartesian=rotation)

    assert local.shape == (2, 2)
    assert np.allclose(local, np.eye(2, dtype=complex))


def test_local_rotation_uses_inverse_cartesian_transform(load_pyatb, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_pyatb("pyatb.symmetry.hs_standardize")

    class _Meta:
        spin_factor = 2

    captured = {}

    def _fake_local(metadata, atom_index, cart_rotation, passive_basis=False):
        del metadata, atom_index
        captured["cart_rotation"] = np.asarray(cart_rotation, dtype=float)
        captured["passive_basis"] = bool(passive_basis)
        return np.eye(2, dtype=complex)

    monkeypatch.setattr(module, "build_atom_local_rotation", _fake_local)

    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    _ = module._local_rotation(_Meta(), atom_index=0, xyz_axis_transform_cartesian=rotation)

    np.testing.assert_allclose(captured["cart_rotation"], rotation)
    assert captured["passive_basis"] is True


def test_character_reads_canonicalized_stru_when_rebuild_is_required(
    load_pyatb, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_pyatb("pyatb.symmetry.character")

    class _FakeSolver:
        def diago_H(self, k_direct_coor):
            del k_direct_coor
            return np.eye(1, dtype=complex).reshape(1, 1, 1), np.array([[0.25]], dtype=float)

        def get_Sk(self, k_direct_coor):
            del k_direct_coor
            return np.eye(1, dtype=complex).reshape(1, 1, 1)

    class _FlowTB:
        nspin = 1
        max_kpoint_num = 10
        basis_num = 1
        lattice_vector = np.eye(3)
        lattice_constant = 2.0
        stru_atom = []

        def __init__(self):
            self.read_calls = []
            self.tb_solver = _FakeSolver()

        def read_stru(self, stru_file, need_orb=False):
            self.read_calls.append((stru_file, need_orb))

    tb = _FlowTB()
    cal = module.Character(tb)
    canonical_init = {}

    class _CanonicalTB:
        def __init__(self, nspin, lattice_constant, lattice_vector, max_kpoint_num):
            canonical_init["nspin"] = nspin
            canonical_init["lattice_constant"] = lattice_constant
            canonical_init["lattice_vector"] = np.asarray(lattice_vector, dtype=float)
            canonical_init["max_kpoint_num"] = max_kpoint_num
            self.nspin = 1
            self.max_kpoint_num = 10
            self.basis_num = 1
            self.lattice_vector = np.eye(3)
            self.lattice_constant = 1.0
            self.stru_atom = []
            self.tb_solver = _FakeSolver()

        def set_solver_HSR(self, hr, sr, isSparse=False):
            del hr, sr, isSparse

        def read_stru(self, stru_file, need_orb=False):
            tb.read_calls.append((stru_file, need_orb))

    monkeypatch.setattr(
        module.SymmStructureAnalyzer,
        "analyze_nonmagnetic",
        lambda *args, **kwargs: {
            "resolved_group": 1,
            "need_rebuild_hs": True,
                "target_stru": "STRU-symm",
                "target_hr": "data-HR-sparse_SPIN0-symm.csr",
                "target_sr": "data-SR-sparse_SPIN0-symm.csr",
                "atom_mapping": [],
                "lattice_new": 2.0 * np.eye(3),
                "lattice_transform_fractional": np.eye(3),
                "xyz_axis_transform_cartesian": np.eye(3),
                "full_matrix_from_hermitian": True,
            "operations": [],
            "source_operations": [],
            "kpoint_records": [],
        },
    )
    canonical_call = {}
    monkeypatch.setattr(
        module,
        "canonicalize_abacus_hs",
        lambda **kwargs: (
            canonical_call.update(kwargs),
            {"hr": object(), "sr": object()},
        )[1],
    )
    class _FakeMetadata:
        basis_num = 1
        species_by_atom = ["H"]
        positions_frac = np.array([[0.0, 0.0, 0.0]], dtype=float)
        lattice_vector = np.eye(3, dtype=float)

    monkeypatch.setattr(
        module,
        "load_abacus_hs_blocks",
        lambda *args, **kwargs: (_FakeMetadata(), {(0, 0, 0): np.eye(1, dtype=complex)}, {(0, 0, 0): np.eye(1, dtype=complex)}),
    )
    monkeypatch.setattr(module, "prepare_operation_contexts", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        module,
        "self_covariance_statistics",
        lambda *args, **kwargs: {"global_max_abs": 0.0, "mean_abs_over_operations": 0.0},
    )
    monkeypatch.setattr(module, "TBModel", _CanonicalTB)

    cal.calculate_character(
        stru_file="STRU",
        kpoint_mode="direct",
        kpoint_num=1,
        kpoint_direct_coor=[[0.0, 0.0, 0.0]],
        group="auto",
        symm_prec=1e-5,
        occ_band=1,
        band=[1, 1],
        mag_tag=0,
        mag="auto",
    )

    assert tb.read_calls[-1] == ("STRU-symm", True)
    assert tb.read_calls[0] == ("STRU", True)
    np.testing.assert_allclose(canonical_init["lattice_vector"], np.eye(3))
    assert canonical_call["full_matrix_from_hermitian"] is True


def test_abacus_read_stru_wraps_boundary_coordinates_like_abacus(load_pyatb, tmp_path: Path) -> None:
    module = load_pyatb("pyatb.io.abacus_read_stru")
    stru_path = tmp_path / "STRU"
    orb_path = tmp_path / "H.orb"
    orb_path.write_text("dummy\n", encoding="utf-8")
    stru_path.write_text(
        """
ATOMIC_SPECIES
H 1 H.upf
NUMERICAL_ORBITAL
H.orb
LATTICE_CONSTANT
1.8897162
LATTICE_VECTORS
1 0 0
0 1 0
0 0 1
ATOMIC_POSITIONS
Direct
H
0.0
3
-0.0000002 0.0 0.0
0.0 1.0000002 0.0
0.5 0.5 0.5
""".strip()
        + "\n",
        encoding="utf-8",
    )

    atoms = module.read_stru(str(stru_path))
    coords = np.asarray(atoms[0].cartesian_coor, dtype=float)
    scale = 1.8897162 / module.Ang_to_Bohr

    np.testing.assert_allclose(coords[0], np.array([0.0, 0.0, 0.0]), atol=1.0e-12)
    np.testing.assert_allclose(coords[1], np.array([0.0, 0.0, 0.0]), atol=1.0e-12)
    np.testing.assert_allclose(coords[2], np.array([0.5, 0.5, 0.5]) * scale, atol=1.0e-12)


def test_symmetry_analyzer_loads_abacus_stru_with_wrapped_positions(
    load_pyatb, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_pyatb("pyatb.symmetry.symm_stru")
    from ase import Atoms

    (tmp_path / "STRU").write_text("placeholder\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "ase_read",
        lambda *args, **kwargs: Atoms(
            numbers=[1, 1],
            scaled_positions=[[-0.0000002, 0.0, 0.0], [0.0, 1.0000002, 0.0]],
            cell=np.eye(3),
            pbc=True,
        ),
    )
    analyzer = module.SymmStructureAnalyzer(_FakeTB(), output_path=str(tmp_path / "Out" / "CHARACTER"))

    atoms, _ = analyzer._load_input_stru()
    scaled = np.asarray(atoms.get_scaled_positions(wrap=False), dtype=float)

    np.testing.assert_allclose(scaled, np.zeros((2, 3), dtype=float), atol=1.0e-12)
