from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


class _FakeTB:
    nspin = 1
    max_kpoint_num = 10


def test_calculate_kpoint_irreps_runs_character_once_per_kpoint(load_pyatb, monkeypatch, tmp_path: Path) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    calls = []

    class _FakeCharacter:
        def __init__(self, tb):
            self.tb = tb
            self.output_path = str(tmp_path / "Out" / "CHARACTER")

        def calculate_character(self, **kwargs):
            calls.append(kwargs)
            output_path = Path(self.output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            band_start, band_stop = kwargs["band"]
            label = "GM" if kwargs["kpoint_direct_coor"][0][0] == pytest.approx(0.0) else "Z"
            irrep = "GM8" if label == "GM" else "Z5"
            output_path.joinpath("band_irrep.txt").write_text(
                (
                    f"knum=  1   kname={label}\n"
                    "band    degency   energy          irrrp\n"
                    f"{band_start:<8d}{(band_stop - band_start + 1):<10d}{1.25:<16.6f}{irrep:<24s}\n"
                    "\n"
                ),
                encoding="utf-8",
            )

    monkeypatch.setattr(module, "Character", _FakeCharacter)

    results = module.calculate_kpoint_irreps(
        _FakeTB(),
        [
            module.KPointBandSelection(kpoint=[0.0, 0.0, 0.0], band=(77, 78), label="Gamma target"),
            {"kpoint": [0.5, 0.5, 0.5], "band": (79, 80), "label": "Z target"},
        ],
        stru_file="Bi2Se3.STRU",
        group=166,
        symm_prec=1.0e-6,
        occ_band=80,
        mag_tag=0,
        mag="auto",
        data_symmetrize=1,
        data_symm_target_max_abs_ry=1.0e-9,
    )

    assert len(calls) == 2
    assert calls[0]["kpoint_mode"] == "direct"
    assert calls[0]["kpoint_num"] == 1
    assert calls[0]["kpoint_direct_coor"].tolist() == [[0.0, 0.0, 0.0]]
    assert calls[0]["band"] == (77, 78)
    assert calls[0]["occ_band"] == 80
    assert calls[0]["group"] == 166
    assert calls[0]["symm_prec"] == pytest.approx(1.0e-6)
    assert calls[0]["stru_file"] == "Bi2Se3.STRU"
    assert calls[0]["data_symmetrize"] == 1
    assert calls[0]["data_symm_target_max_abs_ry"] == pytest.approx(1.0e-9)
    assert calls[1]["kpoint_direct_coor"].tolist() == [[0.5, 0.5, 0.5]]
    assert calls[1]["band"] == (79, 80)

    assert results[0]["label"] == "Gamma target"
    assert results[0]["kpoint"] == [0.0, 0.0, 0.0]
    assert results[0]["band"] == [77, 78]
    assert results[0]["rows"] == [
        {
            "k_index": 1,
            "k_name": "GM",
            "band": 77,
            "degeneracy": 2,
            "energy": pytest.approx(1.25),
            "irrep": "GM8",
        }
    ]
    assert results[1]["label"] == "Z target"
    assert results[1]["rows"][0]["irrep"] == "Z5"


def test_kpoint_band_selection_validates_per_kpoint_band_ranges(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    with pytest.raises(ValueError, match="kpoint"):
        module.KPointBandSelection(kpoint=[0.0, 0.0], band=(1, 2))

    with pytest.raises(ValueError, match="band"):
        module.KPointBandSelection(kpoint=[0.0, 0.0, 0.0], band=(4, 3))

    with pytest.raises(ValueError, match="selection"):
        module.calculate_kpoint_irreps(_FakeTB(), [])


def test_calculate_kp_irreps_creates_kp_directory_without_copying_character_outputs(
    load_pyatb, monkeypatch, tmp_path: Path
) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    class _FakeCharacter:
        call_count = 0

        def __init__(self, tb):
            self.tb = tb
            self.output_path = str(tmp_path / "Out" / "CHARACTER")

        def calculate_character(self, **kwargs):
            type(self).call_count += 1
            output_path = Path(self.output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            band_start, band_stop = kwargs["band"]
            label = f"K{type(self).call_count}"
            irrep = f"IR{type(self).call_count}"
            output_path.joinpath("band_irrep.txt").write_text(
                (
                    f"knum=  1   kname={label}\n"
                    "band    degency   energy          irrrp\n"
                    f"{band_start:<8d}{(band_stop - band_start + 1):<10d}{2.5:<16.6f}{irrep:<24s}\n"
                    "\n"
                ),
                encoding="utf-8",
            )
            output_path.joinpath("trace.txt").write_text(f"trace {label}\n", encoding="utf-8")

    monkeypatch.setattr(module, "Character", _FakeCharacter)

    results = module.calculate_kp_irreps(
        _FakeTB(),
        kpoint_direct_coor=[[0.0, 0.0, 0.0]],
        band=[[77, 80]],
        label=["GM bands"],
        output_path=tmp_path / "Out" / "KP",
        group=166,
        occ_band=80,
    )

    assert [result["band"] for result in results] == [[77, 80]]
    assert results[0]["archive_path"] is None
    assert results[0]["output_path"] == str(tmp_path / "Out" / "CHARACTER")
    assert (tmp_path / "Out" / "CHARACTER" / "band_irrep.txt").exists()
    assert (tmp_path / "Out" / "CHARACTER" / "trace.txt").read_text(encoding="utf-8") == "trace K1\n"
    assert (tmp_path / "Out" / "KP").is_dir()
    assert not (tmp_path / "Out" / "KP" / "selection_001").exists()
    assert not (tmp_path / "Out" / "KP" / "band_irrep.txt").exists()
    assert not (tmp_path / "Out" / "KP" / "kp_irrep.json").exists()


def test_calculate_kp_irreps_honors_korder_and_zeeman_toggle(load_pyatb, monkeypatch, tmp_path: Path) -> None:
    module = load_pyatb("pyatb.symmetry.kp")
    calls: dict[str, object] = {}

    class _FakeCharacter:
        def __init__(self, tb):
            self.tb = tb
            self.output_path = str(tmp_path / "Out" / "CHARACTER")

        def calculate_character(self, **kwargs):
            output_path = Path(self.output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            output_path.joinpath("band_irrep.txt").write_text(
                "knum=  1   kname=GM\n"
                "band    degency   energy          irrrp\n"
                "77      2         9.117456        A\n",
                encoding="utf-8",
            )
            return {
                "active_stru_path": str(tmp_path / "STRU"),
                "active_hr_path": str(tmp_path / "HR.csr"),
                "active_sr_path": str(tmp_path / "SR.csr"),
                "analysis_result": {},
            }

    def _fake_k_polynomial(_spgrep_info, _character_table, *, orders, lattice=None):
        del lattice
        calls["orders"] = tuple(orders)
        return []

    monkeypatch.setattr(module, "Character", _FakeCharacter)
    monkeypatch.setattr(module, "_collect_kp_symmetry_info", lambda *args, **kwargs: {"spgrep_operations": {}, "character_table": {}})
    monkeypatch.setattr(module, "_operator_irrep_analyses_for_results", lambda *args, **kwargs: [])
    monkeypatch.setattr(module, "_k_polynomial_irrep_analyses", _fake_k_polynomial)
    monkeypatch.setattr(
        module,
        "_zeeman_field_irrep_analyses",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Zeeman should be disabled")),
    )
    monkeypatch.setattr(module, "_representation_alignment_analyses_for_results", lambda *args, **kwargs: [])

    class _NoRR:
        has_rR = False

    monkeypatch.setattr(module, "_lowdin_tb_from_character_payload", lambda *args, **kwargs: _NoRR())

    module.calculate_kp_irreps(
        _FakeTB(),
        kpoint_direct_coor=[[0.0, 0.0, 0.0]],
        band=[[77, 80]],
        output_path=tmp_path / "Out" / "KP",
        korder=1,
        zeeman_term="no",
    )

    assert calls["orders"] == (0, 1)


def test_kpoint_records_use_requested_band_range_not_degeneracy_representatives(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    records = module._kp_model_kpoint_records(
        [
            {
                "selection_index": 1,
                "kpoint": [0.0, 0.0, 0.0],
                "label": "GM",
                "band": [75, 82],
                "rows": [
                    {"k_index": 1, "k_name": "GM", "band": 75, "irrep": "GM6 + GM7"},
                    {"k_index": 1, "k_name": "GM", "band": 77, "irrep": "GM8"},
                    {"k_index": 1, "k_name": "GM", "band": 79, "irrep": "GM9"},
                    {"k_index": 1, "k_name": "GM", "band": 81, "irrep": "GM8"},
                ],
                "character_payload": {"analysis_result": {}},
            }
        ]
    )

    assert records[0]["band_range"] == [75, 82]
    assert records[0]["band_rep"] == "GM6 + GM7 + GM8 + GM9 + GM8"


def test_calculate_kp_irreps_writes_kp_symmetry_info(load_pyatb, monkeypatch, tmp_path: Path) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    active_stru = tmp_path / "STRU"
    active_hr = tmp_path / "Out" / "CHARACTER" / "data-HR-sparse_SPIN0-covsymm.csr"
    active_sr = tmp_path / "Out" / "CHARACTER" / "data-SR-sparse_SPIN0-covsymm.csr"

    class _FakeCharacter:
        def __init__(self, tb):
            self.tb = tb
            self.output_path = str(tmp_path / "Out" / "CHARACTER")

        def calculate_character(self, **kwargs):
            output_path = Path(self.output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            output_path.joinpath("band_irrep.txt").write_text(
                "knum=  1   kname=GM\n"
                "band    degency   energy          irrrp\n"
                "77      2         9.117456        GM8\n"
                "79      2         9.625764        GM9\n",
                encoding="utf-8",
            )
            return {
                "active_stru_path": str(active_stru),
                "active_hr_path": str(active_hr),
                "active_sr_path": str(active_sr),
                "analysis_result": {
                    "detected_group": 166,
                    "resolved_group": 166,
                    "operation_count": 12,
                    "magnetic": False,
                },
            }

    expected_info = {
        "nspin": 4,
        "active_stru_path": str(active_stru),
        "active_hr_path": str(active_hr),
        "active_sr_path": str(active_sr),
        "lattice_vectors": [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 4.0],
        ],
        "reciprocal_lattice_vectors": [
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.25],
        ],
        "kpoint_records": [
            {
                "k_index": 1,
                "k_direct": [0.0, 0.0, 0.0],
                "k_name": "GM",
                "star_transform": "The k-point is transformed by Identity operation to k-star",
                "primitive_basis": [0.0, 0.0, 0.0],
                "conventional_basis": [0.0, 0.0, 0.0],
                "band_range": [77, 79],
                "band_rep": "GM8 + GM9",
            }
        ],
        "spgrep_operations": {
            "call": "get_spacegroup_spinor_irreps_from_primitive_symmetry(time_reversals=magnetic)",
            "coordinate_convention": "x' = R x + tau",
            "kpoint": [0.0, 0.0, 0.0],
            "little_group_mapping_one_based": [1],
            "corep_shapes": [[1, 1, 1]],
            "operations": [
                {
                    "operation_index": 1,
                    "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "translation": [0.0, 0.0, 0.0],
                    "time_reversal": False,
                    "anti_linear": False,
                    "in_little_group": True,
                }
            ],
            "coreps": [
                {
                    "corep_index": 1,
                    "shape": [1, 2, 2],
                    "operation_matrices": [
                        {
                            "operation_index": 1,
                            "matrix": [
                                [[1.0, 0.0], [0.0, 0.0]],
                                [[0.0, 0.0], [1.0, 0.0]],
                            ],
                        }
                    ],
                }
            ],
        },
        "magnetic_moments": [[0.0, 0.0, 0.0]],
        "spglib_spacegroup_dataset": {
            "number": 166,
            "international": "R-3m",
            "hall_symbol": "-R 3 2\"",
        },
        "spglib_magnetic_dataset": {"uni_number": 1328},
        "magnetic_spacegroup_type": {
            "uni_number": 1328,
            "litvin_number": 1328,
            "type": 2,
            "type_label": "type-II",
            "bns_number": "166.98",
            "og_number": "166.2.1328",
            "bns_setting": "166.98",
            "og_setting": "166.2.1328",
        },
        "spin_group": {"enabled": True},
    }
    identity4 = [
        [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
    ]
    c3_blocked = [
        [[0.5, 0.866025], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.5, -0.866025], [0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0], [0.5, 0.866025], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.5, -0.866025]],
    ]
    phase_matrix = [
        [[0.968912, 0.247404], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.968912, 0.247404], [0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0], [0.877583, -0.479426], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.877583, -0.479426]],
    ]
    expected_info["representation_alignment_analyses"] = [
        {
            "selection_index": 1,
            "target_label": "GM8 + GM9",
            "target_corep_indices": [1, 2],
            "target_corep_dimensions": [2, 2],
            "band_range": [77, 79],
            "operation_indices": [1, 2],
            "antiunitary_operation_indices": [1],
            "numeric_representation_matrices": [identity4, c3_blocked],
            "spgrep_representation_matrices": [identity4, c3_blocked],
            "numeric_antiunitary_representation_matrices": [identity4],
            "spgrep_antiunitary_representation_matrices": [identity4],
            "schur_alignment": {
                "max_abs_representation_difference": 0.0,
                "unitary_max_abs_representation_difference": 0.0,
                "antiunitary_max_abs_representation_difference": 0.0,
                "frobenius_representation_difference": 0.0,
                "unitarity_error": 0.0,
                "unitary_numeric_to_standard": identity4,
                "antiunitary_phase_refinement": {
                    "status": "applied",
                    "block_dimensions": [2, 2],
                    "block_phases_rad": [0.25, -0.5],
                    "phase_matrix": phase_matrix,
                },
            },
        }
    ]
    expected_info["operator_irrep_analyses"] = [
        {
            "target_label": "GM8 + GM9",
            "operator_representation": "(GM8 + GM9)^* x (GM8 + GM9)",
            "operation_indices": [1, 2],
            "band_characters": [[4.0, 0.0], [2.0, 0.0]],
            "operator_characters": [[16.0, 0.0], [4.0, 0.0]],
            "decomposition": [{"label": "GM1+", "multiplicity": 1}],
            "hermitian_basis": [{"basis_index": 1, "label": "diag(1)"}],
            "projected_basis": [
                {
                    "irrep_label": "GM1+",
                    "rank": 1,
                    "matrices": [
                        {
                            "label": "GM1+_1",
                            "linear_combination": [
                                {
                                    "basis_index": 1,
                                    "basis_label": "diag(1)",
                                    "coefficient": [1.0, 0.0],
                                }
                            ],
                            "matrix": [
                                [[0.707107, 0.0], [0.0, 0.0]],
                                [[0.0, 0.0], [0.0, 0.0]],
                            ],
                        }
                    ],
                }
            ],
        }
    ]
    expected_info["k_polynomial_irrep_analyses"] = [
        {
            "order": 1,
            "operation_indices": [1, 2],
            "characters": [[3.0, 0.0], [0.0, 0.0]],
            "decomposition": [{"label": "GM3-", "multiplicity": 1}],
            "monomial_basis": [{"basis_index": 1, "label": "kx", "exponents": [1, 0, 0]}],
            "projected_basis": [
                {
                    "irrep_label": "GM3-",
                    "rank": 1,
                    "functions": [
                        {
                            "label": "GM3-_k_basis_1",
                            "linear_combination": [
                                {
                                    "basis_index": 1,
                                    "basis_label": "kx",
                                    "coefficient": [1.0, 0.0],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    ]
    expected_info["zeeman_field_irrep_analyses"] = [
        {
            "order": 1,
            "operation_indices": [1, 2],
            "characters": [[3.0, 0.0], [0.0, 0.0]],
            "decomposition": [{"label": "GM3+", "multiplicity": 1}],
            "monomial_basis": [{"basis_index": 1, "label": "Bx", "exponents": [1, 0, 0]}],
            "projected_basis": [
                {
                    "irrep_label": "GM3+",
                    "rank": 1,
                    "functions": [
                        {
                            "label": "GM3+_k_basis_1",
                            "linear_combination": [
                                {
                                    "basis_index": 1,
                                    "basis_label": "Bx",
                                    "coefficient": [1.0, 0.0],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    monkeypatch.setattr(module, "Character", _FakeCharacter)
    monkeypatch.setattr(module, "_collect_kp_symmetry_info", lambda *args, **kwargs: expected_info)

    class _SpinorTB(_FakeTB):
        nspin = 4

    module.calculate_kp_irreps(
        _SpinorTB(),
        kpoint_direct_coor=[[0.0, 0.0, 0.0]],
        band=[[77, 80]],
        output_path=tmp_path / "Out" / "KP",
        group=166,
        occ_band=80,
    )

    info_path = tmp_path / "Out" / "KP" / "kp_symmetry_info.json"
    assert json.loads(info_path.read_text(encoding="utf-8")) == expected_info
    text = (tmp_path / "Out" / "KP" / "kp_symmetry_info.txt").read_text(encoding="utf-8")
    assert "active_stru_path" in text
    assert "spglib_magnetic_dataset.uni_number = 1328" in text
    model_text = (tmp_path / "Out" / "KP" / "kp_model.txt").read_text(encoding="utf-8")
    star_line = "*" * 93
    assert model_text.startswith(
        f"{star_line}\n"
        f"{star_line}\n"
        "                                   KP      MODEL \n"
    )
    assert "Spglib determined space group No. 166 (R-3m), Hall symbol -R 3 2\"." in model_text
    assert f"calculated stru: {active_stru}" in model_text
    assert "Symmetry operations Pi={Ri|taui+tm}   note: defined in Symmetrized Stru" in model_text
    assert "1 (E): unity op." in model_text
    assert "time reversal: no" in model_text
    assert "    Ri       taui   inv(Ri)   Ri(Cartesian coord)" in model_text
    assert "    Ri       taui   inv(Ri)   Ri(Cartesian coord)         spin matrix" not in model_text
    assert "  1  0  0   0.000   1  0  0   1.000  0.000  0.000" in model_text
    assert "Band rep:    GM8 + GM9" in model_text
    symmetry_index = model_text.index("Symmetry operations Pi={Ri|taui+tm}")
    kpoint_index = model_text.index("knum =  1    k = 0.000000 0.000000 0.000000")
    matrix_index = model_text.index("Symmetry representation matrices")
    hermitian_index = model_text.index("Hermitian matrix irreducible decomposition")
    assert symmetry_index < kpoint_index < matrix_index < hermitian_index
    assert "Target-subspace linear symmetry representation matrices" not in model_text
    assert "Operator indices: 1" in model_text
    assert "Operator indices: 2" in model_text
    representation_header = module._format_representation_block_header(["GM8", "GM9"], [2, 2])
    assert representation_header in model_text
    assert "Symmetrized basis: spgrep target corep matrices in CHARACTER operation order" not in model_text
    assert "Numerical basis: ABACUS/pyatb wavefunction representation matrices" not in model_text
    assert "  Symmetrized basis:" in model_text
    assert "  Numerical basis:" in model_text
    assert "Through Schur Lemma : D_symm(g) ~= U^dagger D_num(g) U" in model_text
    assert "U = U0 P_alpha" in model_text
    assert "P_alpha = diag(e^{i alpha_GM8} I_2, e^{i alpha_GM9} I_2)" in model_text
    assert "Anti-linear symmetry representation matrices" in model_text
    assert "e^{i alpha_GM8} =" in model_text
    assert "e^{i alpha_GM9} =" in model_text
    assert "  Operator indices: 3 " in model_text
    assert "U0 =" in model_text
    assert "U =" in model_text
    assert "Transformed representation matrix errors:" in model_text
    assert "linear max_abs_difference =" in model_text
    assert "anti-linear max_abs_difference =" in model_text
    assert "(   0.969  -0.247i)" in model_text
    assert "( 0.968912-0.247404i)" not in model_text
    error_index = model_text.index("Transformed representation matrix errors:")
    k_decomp_index = model_text.index("k polynomial irreducible decomposition")
    zeeman_index = model_text.index("Zeeman magnetic-field irreducible decomposition")
    assert error_index < hermitian_index < k_decomp_index < zeeman_index
    assert "Operator representation: (GM8 + GM9)^* x (GM8 + GM9)" in model_text
    assert "Decomposition: GM1+" in model_text
    assert "(   0.707  +0.000i)" in model_text
    assert "( 0.707107+0.000000i)" not in model_text
    schur_line_index = model_text.index("Through Schur Lemma")
    anti_linear_index = model_text.index("Anti-linear symmetry representation matrices")
    phase_index = model_text.index("e^{i alpha_GM8} =")
    u0_index = model_text.index("U0 =")
    u_index = model_text.index("U =\n", u0_index)
    assert schur_line_index < anti_linear_index < phase_index < u0_index < u_index < error_index
    assert "antiunitary operation 1" not in model_text
    assert "Spgrep little-group representation trace table" not in model_text
    assert "Spgrep trace check against CHARACTER table" not in model_text
    assert "Target-subspace symmetry representation alignment" not in model_text
    assert "Final real Hermitian time-reversal-even k.p basis" not in model_text
    assert "Final real Hermitian time-reversal-even Zeeman basis" not in model_text
    spgrep_text = (tmp_path / "Out" / "KP" / "spgrep_operations.txt").read_text(encoding="utf-8")
    assert "Spgrep magnetic unitary symmetry operations" in spgrep_text
    assert "operation 1" in spgrep_text
    assert "Spgrep little-group representation trace table" in spgrep_text
    assert "Spgrep representation matrices for GM8" in spgrep_text


def test_magnetic_spacegroup_type_summary_derives_settings(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    class _SpaceGroupType:
        uni_number = 1328
        litvin_number = 1328
        type = 2
        bns_number = "166.98"
        og_number = "166.2.1328"

    summary = module._magnetic_spacegroup_type_summary(_SpaceGroupType())

    assert summary["bns_setting"] == "166.98"
    assert summary["og_setting"] == "166.2.1328"
    assert summary["type"] == 2
    assert summary["type_label"] == "type-II"


def test_magnetic_spacegroup_type_labels_four_types(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    assert module._magnetic_spacegroup_type_label(1) == "type-I"
    assert module._magnetic_spacegroup_type_label(2) == "type-II"
    assert module._magnetic_spacegroup_type_label(3) == "type-III"
    assert module._magnetic_spacegroup_type_label(4) == "type-IV"


def test_spgrep_unitary_operations_are_sorted_by_character_order(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    class _CharacterOperation:
        def __init__(self, rotation):
            self.rotation = rotation

    op_a = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    op_b = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    operation_by_source_index = {
        0: {"rotation": op_a},
        1: {"rotation": op_b},
    }

    assert module._sort_spgrep_unitary_operations_by_character_order(
        [0, 1],
        operation_by_source_index,
        [_CharacterOperation(op_b), _CharacterOperation(op_a)],
    ) == [1, 0]


def test_spgrep_trace_table_checks_character_table(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    spgrep_info = {
        "operations": [
            {"operation_index": 1},
            {"operation_index": 2},
        ],
        "coreps": [
            {
                "corep_index": 1,
                "operation_matrices": [
                    {
                        "operation_index": 1,
                        "matrix": [
                            [[1.0, 0.0], [0.0, 0.0]],
                            [[0.0, 0.0], [1.0, 0.0]],
                        ],
                    },
                    {
                        "operation_index": 2,
                        "matrix": [
                            [[0.0, 1.0], [0.0, 0.0]],
                            [[0.0, 0.0], [0.0, -1.0]],
                        ],
                    },
                ],
            }
        ],
    }
    character_table = {
        "display_operation_indices": [1, 2],
        "irreps": [
            {
                "label": "GM8",
                "dimension": 2,
                "double_valued": True,
                "characters": [[2.0, 0.0], [0.0, 0.0]],
            }
        ],
    }

    checks = module._spgrep_trace_character_checks(spgrep_info, character_table)

    assert checks[0]["label"] == "GM8"
    assert checks[0]["corep_index"] == 1
    assert checks[0]["status"] == "ok"
    assert checks[0]["max_abs_diff"] == pytest.approx(0.0)

    text = "\n".join(
        module._format_spgrep_trace_tables(
            spgrep_info,
            character_table=character_table,
            star_line="***",
        )
    )
    assert "Spgrep little-group representation trace table" in text
    assert "corep 1 dim=2 traces:" in text
    assert "Spgrep trace check against CHARACTER table" in text
    assert "GM8 -> corep 1 max_abs_diff = 0.000000e+00 status = ok" in text
    assert "op 2 spgrep =  0.000000 character =  0.000000 diff =  0.000000" in text


def test_spgrep_corep_indices_group_conjugate_irrep_terms(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    spgrep_info = {
        "coreps": [
            {
                "corep_index": 1,
                "operation_matrices": [
                    {"operation_index": 1, "matrix": [[[2.0, 0.0]]]},
                    {"operation_index": 2, "matrix": [[[1.0, 0.0]]]},
                ],
            },
            {
                "corep_index": 4,
                "operation_matrices": [
                    {
                        "operation_index": 1,
                        "matrix": [
                            [[1.0, 0.0], [0.0, 0.0]],
                            [[0.0, 0.0], [1.0, 0.0]],
                        ],
                    },
                    {
                        "operation_index": 2,
                        "matrix": [
                            [[0.0, 0.0], [0.0, 0.0]],
                            [[0.0, 0.0], [0.0, 0.0]],
                        ],
                    },
                ],
            },
        ],
    }
    character_table = {
        "irreps": [
            {"label": "GM6", "characters": [[1.0, 0.0], [0.0, 1.0]]},
            {"label": "GM7", "characters": [[1.0, 0.0], [0.0, -1.0]]},
            {"label": "GM8", "characters": [[2.0, 0.0], [1.0, 0.0]]},
        ],
    }

    assert module._spgrep_corep_indices_for_irrep_terms(
        ["GM6", "GM7", "GM8"],
        spgrep_info=spgrep_info,
        character_table=character_table,
    ) == [4, 1]


def test_spgrep_all_operations_output_includes_antiunitary_operations(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    spgrep_info = {
        "call": "get_spacegroup_spinor_irreps_from_primitive_symmetry(time_reversals=magnetic)",
        "coordinate_convention": "x' = R x + tau",
        "kpoint": [0.0, 0.0, 0.0],
        "spgrep_little_group_mapping_one_based": [1, 2],
        "operations": [
            {
                "operation_index": 1,
                "spgrep_operation_index": 1,
                "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "translation": [0.0, 0.0, 0.0],
                "time_reversal": False,
                "anti_linear": False,
                "in_little_group": True,
            }
        ],
        "antiunitary_operations": [
            {
                "antiunitary_operation_index": 1,
                "spgrep_operation_index": 2,
                "rotation": [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
                "translation": [0.0, 0.0, 0.0],
                "time_reversal": True,
                "anti_linear": True,
                "in_little_group": True,
            }
        ],
    }

    text = "\n".join(module._format_spgrep_all_operations(spgrep_info, star_line="***"))

    assert "Spgrep all little-group symmetry operations" in text
    assert "spgrep little group mapping (1-based): 1 2" in text
    assert "operation 1" in text
    assert "kind: unitary" in text
    assert "unitary operation index: 1" in text
    assert "operation 2" in text
    assert "kind: antiunitary" in text
    assert "antiunitary operation index: 1" in text
    assert "spgrep operation index: 2" in text
    assert "time reversal: yes" in text
    assert "anti-linear: yes" in text


def test_spgrep_character_style_operations_include_time_reversal_flag(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    spgrep_info = {
        "call": "get_spacegroup_spinor_irreps_from_primitive_symmetry(time_reversals=magnetic)",
        "coordinate_convention": "x' = R x + tau",
        "spgrep_little_group_mapping_one_based": [1, 2],
        "operations": [
            {
                "operation_index": 1,
                "spgrep_operation_index": 1,
                "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "translation": [0.0, 0.0, 0.0],
                "time_reversal": False,
                "anti_linear": False,
                "in_little_group": True,
            }
        ],
        "antiunitary_operations": [
            {
                "antiunitary_operation_index": 1,
                "spgrep_operation_index": 2,
                "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "translation": [0.0, 0.0, 0.0],
                "time_reversal": True,
                "anti_linear": True,
                "in_little_group": True,
            }
        ],
    }

    text = "\n".join(
        module._format_spgrep_character_style_operations(
            spgrep_info,
            lattice=np.eye(3),
            star_line="***",
        )
    )

    assert "Symmetry operations Pi={Ri|taui+tm}   note: defined in Symmetrized Stru" in text
    assert "1 (E): unity op." in text
    assert "2 (E*T): unity op. times time reversal" in text
    assert text.index("main axes: ( 1.000,  0.000,  0.000)") < text.index("time reversal: no")
    assert "time reversal: no" in text
    assert "time reversal: yes" in text
    assert "anti-linear: yes" in text
    assert "    Ri       taui   inv(Ri)   Ri(Cartesian coord)" in text
    assert "spin matrix" not in text
    assert "( 1.000 0.000)( 0.000 0.000)" not in text


def test_numeric_lowdin_kp_coefficients_use_velocity_formula(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    eigenvalues = np.asarray([1.0, 5.0], dtype=float)
    velocity = np.zeros((3, 2, 2), dtype=complex)
    velocity[0, 0, 0] = 0.25
    velocity[1, 0, 0] = -0.50
    velocity[0, 0, 1] = 2.0
    velocity[0, 1, 0] = 2.0
    velocity[1, 0, 1] = 3.0
    velocity[1, 1, 0] = 3.0

    analysis = module._build_numeric_lowdin_kp_analysis(
        eigenvalues,
        velocity,
        target_indices=[0],
        reference_k_direct=[0.0, 0.0, 0.0],
        reference_k_cartesian=[0.0, 0.0, 0.0],
        band_range=[1, 1],
        selection_index=1,
        free_electron_coefficient=10.0,
    )

    linear = np.asarray(module._complex_array_from_pairs(analysis["linear_matrices"][0]["matrix"]))
    assert linear.shape == (1, 1)
    assert linear[0, 0] == pytest.approx(0.25)

    lowdin_xx = np.asarray(
        module._complex_array_from_pairs(analysis["lowdin_quadratic_tensor"][0][0])
    )
    assert lowdin_xx[0, 0] == pytest.approx(-1.0)

    quadratic_by_label = {
        item["label"]: np.asarray(module._complex_array_from_pairs(item["matrix"]))
        for item in analysis["quadratic_monomial_matrices"]
    }
    assert quadratic_by_label["kx^2"][0, 0] == pytest.approx(9.0)
    assert quadratic_by_label["kx*ky"][0, 0] == pytest.approx(-3.0)
    assert quadratic_by_label["ky^2"][0, 0] == pytest.approx(7.75)
    assert quadratic_by_label["kz^2"][0, 0] == pytest.approx(10.0)


def test_numeric_lowdin_kp_coefficients_include_third_order_remote_chains(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    eigenvalues = np.asarray([0.0, 2.0, 5.0], dtype=float)
    velocity = np.zeros((3, 3, 3), dtype=complex)
    velocity[0, 0, 1] = 2.0
    velocity[1, 1, 2] = 3.0
    velocity[2, 2, 0] = 4.0

    analysis = module._build_numeric_lowdin_kp_analysis(
        eigenvalues,
        velocity,
        target_indices=[0],
        reference_k_direct=[0.0, 0.0, 0.0],
        reference_k_cartesian=[0.0, 0.0, 0.0],
        band_range=[1, 1],
        selection_index=1,
        free_electron_coefficient=0.0,
    )

    cubic_by_label = {
        item["label"]: np.asarray(module._complex_array_from_pairs(item["matrix"]))
        for item in analysis["cubic_monomial_matrices"]
    }
    # Six tensor permutations contribute equally to kx*ky*kz after symmetrization.
    expected = 0.5 * 2.0 * 3.0 * 4.0 * (
        1.0 / ((0.0 - 2.0) * (0.0 - 5.0))
        + 1.0 / ((0.0 - 2.0) * (0.0 - 5.0))
    )
    assert cubic_by_label["kx*ky*kz"][0, 0] == pytest.approx(expected)


def test_numeric_lowdin_kp_coefficients_include_third_order_target_remote_target_chains(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    eigenvalues = np.asarray([0.0, 1.0, 4.0], dtype=float)
    velocity = np.zeros((3, 3, 3), dtype=complex)
    velocity[0, 0, 2] = 2.0
    velocity[1, 2, 1] = 3.0
    velocity[2, 1, 0] = 5.0

    analysis = module._build_numeric_lowdin_kp_analysis(
        eigenvalues,
        velocity,
        target_indices=[0, 1],
        reference_k_direct=[0.0, 0.0, 0.0],
        reference_k_cartesian=[0.0, 0.0, 0.0],
        band_range=[1, 2],
        selection_index=1,
        free_electron_coefficient=0.0,
    )

    cubic_by_label = {
        item["label"]: np.asarray(module._complex_array_from_pairs(item["matrix"]))
        for item in analysis["cubic_monomial_matrices"]
    }
    expected_tensor_component = -0.5 * 2.0 * 3.0 * 5.0 / ((0.0 - 4.0) * (1.0 - 4.0))
    assert cubic_by_label["kx*ky*kz"][0, 0] == pytest.approx(expected_tensor_component)


def test_numeric_lowdin_kp_formatter_uses_report_style_and_three_decimal_matrices(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    matrix = [
        [[1.23456, -2.34567], [0.0, 0.0]],
        [[0.0, 0.0], [-3.45678, 4.56789]],
    ]
    analysis = {
        "selection_index": 1,
        "basis_convention": "debug basis",
        "hamiltonian_data_convention": "debug H/S",
        "reference_k_direct": [0.0, 0.0, 0.0],
        "reference_k_cartesian": [0.0, 0.0, 0.0],
        "band_range": [1, 2],
        "target_band_indices_one_based": [1, 2],
        "remote_band_count": 4,
        "free_electron_coefficient_eV_A2": 3.80998212,
        "hamiltonian_data_paths": {"HR": "debug-HR"},
        "constant_energies_eV": [1.0, 2.0],
        "linear_matrices": [{"direction": "kx", "matrix": matrix}],
        "lowdin_quadratic_monomial_matrices": [{"label": "kx^2", "matrix": matrix}],
        "quadratic_monomial_matrices": [{"label": "kx^2", "matrix": matrix}],
        "cubic_monomial_matrices": [{"label": "kx^3", "matrix": matrix}],
    }

    text = "\n".join(module._format_numeric_lowdin_kp_analyses([analysis], star_line="***"))

    assert "Numerical Lowdin k.p model coefficients" in text
    assert "hbar^2/(2m_e) = 3.80998212 eV Angstrom^2" in text
    assert "constant energies E0 (eV):" in text
    assert "E1  =  1.0000000000" in text
    assert "V_kx =" in text
    assert "C_kx^2 =" in text
    assert "D_kx^3 =" in text
    assert "(   1.235  -2.346i)" in text
    assert "( 1.234560-2.345670i)" not in text
    assert "selection =" not in text
    assert "basis convention: debug basis" not in text
    assert "HR file:" not in text
    assert "remote band count" not in text
    assert "Second-order Lowdin-only coefficient matrices L_Q" not in text
    assert "L_kx^2" not in text


def test_numeric_lowdin_kp_formatter_respects_max_order(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    matrix = [[[1.0, 0.0]]]
    analysis = {
        "selection_index": 1,
        "free_electron_coefficient_eV_A2": 3.80998212,
        "constant_energies_eV": [1.0],
        "linear_matrices": [{"direction": "kx", "matrix": matrix}],
        "quadratic_monomial_matrices": [{"label": "kx^2", "matrix": matrix}],
        "cubic_monomial_matrices": [{"label": "kx^3", "matrix": matrix}],
    }

    text = "\n".join(
        module._format_numeric_lowdin_kp_analyses(
            [analysis],
            star_line="***",
            max_order=2,
        )
    )

    assert "Linear coefficient matrices V_i" in text
    assert "Second-order total coefficient matrices C_Q" in text
    assert "Third-order Lowdin coefficient matrices D_R" not in text
    assert "D_kx^3" not in text


def test_numeric_lowdin_zeeman_coefficients_include_orbital_and_spin_parts(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    eigenvalues = np.asarray([0.0, 2.0], dtype=float)
    velocity = np.zeros((3, 2, 2), dtype=complex)
    velocity[0, 0, 1] = 2.0
    velocity[1, 1, 0] = 3.0
    spin_target = np.zeros((3, 1, 1), dtype=complex)
    spin_target[2, 0, 0] = 1.0

    analysis = module._build_numeric_lowdin_kp_analysis(
        eigenvalues,
        velocity,
        target_indices=[0],
        reference_k_direct=[0.0, 0.0, 0.0],
        reference_k_cartesian=[0.0, 0.0, 0.0],
        band_range=[1, 1],
        selection_index=1,
        free_electron_coefficient=10.0,
        spin_target_matrices=spin_target,
    )

    orbital_by_label = {
        item["field"]: np.asarray(module._complex_array_from_pairs(item["matrix"]))
        for item in analysis["zeeman_orbital_matrices"]
    }
    spin_by_label = {
        item["field"]: np.asarray(module._complex_array_from_pairs(item["matrix"]))
        for item in analysis["zeeman_spin_matrices"]
    }
    total_by_label = {
        item["field"]: np.asarray(module._complex_array_from_pairs(item["matrix"]))
        for item in analysis["zeeman_matrices"]
    }

    assert orbital_by_label["Bz"][0, 0] == pytest.approx(0.15j)
    assert spin_by_label["Bz"][0, 0] == pytest.approx(1.0)
    assert total_by_label["Bz"][0, 0] == pytest.approx(1.0 + 0.15j)
    assert analysis["zeeman_convention"] == "H_Z = (mu_B / 2) sum_i G_i B_i"


def test_numeric_zeeman_formatter_uses_three_decimal_matrices(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    matrix = [
        [[1.23456, -2.34567], [0.0, 0.0]],
        [[0.0, 0.0], [-3.45678, 4.56789]],
    ]
    analysis = {
        "selection_index": 1,
        "basis_convention": "debug basis",
        "hamiltonian_data_convention": "debug H/S",
        "zeeman_formula": "G formula",
        "zeeman_orbital_matrices": [{"field": "Bx", "matrix": matrix}],
        "zeeman_spin_matrices": [{"field": "Bx", "matrix": matrix}],
        "zeeman_matrices": [{"field": "Bx", "matrix": matrix}],
    }

    text = "\n".join(module._format_numeric_zeeman_analyses([analysis], star_line="***"))

    assert "Numerical Zeeman matrices from Lowdin perturbation" in text
    assert "Formula: G formula" in text
    assert "G_orb_Bx =" in text
    assert "sigma_Bx =" in text
    assert "G_Bx =" in text
    assert "(   1.235  -2.346i)" in text
    assert "( 1.234560-2.345670i)" not in text
    assert "selection =" not in text
    assert "basis convention: debug basis" not in text
    assert "H/S data convention: debug H/S" not in text


def test_three_decimal_complex_formatter_aligns_columns_and_suppresses_negative_zero(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    values = [
        [70.2471, 0.0],
        [0.0, 0.0],
        [-24.3311, 0.0],
        [-1.0e-8, -1.0e-8],
        [0.0, -103.068],
    ]
    formatted = [module._format_complex_pair_3(value) for value in values]

    assert formatted == [
        "(  70.247  +0.000i)",
        "(   0.000  +0.000i)",
        "( -24.331  +0.000i)",
        "(   0.000  +0.000i)",
        "(   0.000-103.068i)",
    ]
    assert len({len(item) for item in formatted}) == 1
    assert len({item.index(".") for item in formatted}) == 1
    assert len({item.rindex(".") for item in formatted}) == 1


def test_schur_kp_fit_formatter_outputs_transformed_numeric_and_fit_side_by_side(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    raw_numeric = [
        [[9.87654, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [8.76543, 0.0]],
    ]
    numeric = [
        [[1.23456, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [2.34567, -3.45678]],
    ]
    fitted = [
        [[1.2, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [2.3, -3.4]],
    ]
    analysis = {
        "selection_index": 1,
        "absolute_residual": 0.1,
        "relative_residual": 0.01,
        "max_abs_difference_all": 0.05,
        "formal_parameter_labels": ["K1_1"],
        "formal_parameters": [3.5],
        "records": [
            {
                "monomial": "kx",
                "raw_numeric_matrix": raw_numeric,
                "transformed_numeric_matrix": numeric,
                "formal_fit_matrix": fitted,
                "max_abs_difference": 0.01,
            },
            {
                "monomial": "kx^2",
                "raw_numeric_matrix": raw_numeric,
                "transformed_numeric_matrix": numeric,
                "formal_fit_matrix": fitted,
                "max_abs_difference": 0.01,
            },
        ],
    }

    final_model = [
        {
            "order": 1,
            "terms": [
                {
                    "label": "K1_1",
                    "linear_combination": [
                        {
                            "coefficient": [1.0, 0.0],
                            "hermitian_basis_label": "diag(1)",
                            "k_basis_label": "kx",
                        }
                    ],
                }
            ],
        }
    ]
    numeric_analyses = [
        {
            "selection_index": 1,
            "free_electron_coefficient_eV_A2": 3.80998212,
        }
    ]

    text = "\n".join(
        module._format_schur_kp_parameter_fit_analyses(
            [analysis],
            star_line="***",
            final_model_analyses=final_model,
            numeric_analyses=numeric_analyses,
        )
    )

    assert "Numerical Lowdin k.p model coefficients" in text
    assert "hbar^2/(2m_e) = 3.80998212 eV Angstrom^2" in text
    assert "Symmetrized formual for k.p Hamiltonian:" in text
    assert "K1_1 = 1 * diag(1) * kx" in text
    assert "Parameters fitted:" in text
    assert "Detailed information:" in text
    assert "Linear coefficient matrices V_i (eV Angstrom)" in text
    assert "V_kx" in text
    assert "Numerical result:" in text
    assert "Fitted result:" in text
    assert "(   9.877  +0.000i)" in text
    assert "(   1.235  +0.000i)" not in text
    assert text.index("Symmetrized formual for k.p Hamiltonian:") < text.index("Parameters fitted:")
    assert text.index("Parameters fitted:") < text.index("Detailed information:")
    assert text.index("Detailed information:") < text.index("Linear coefficient matrices V_i")
    assert "---------------------------------------------------------------------------------------\nSecond-order total coefficient matrices C_Q (eV Angstrom^2)\nC_kx^2\n" in text
    assert "( 1.234560+0.000000i)" not in text
    assert "Schur-aligned numerical Lowdin k.p parameter fit" not in text
    assert "Formal parameters:" not in text
    assert " V_kx " not in text
    assert "transformed numeric matrix:" not in text
    assert "formal fit matrix:" not in text
    assert "difference matrix:" not in text


def test_schur_zeeman_fit_formatter_outputs_transformed_numeric_and_fit_side_by_side(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    raw_numeric = [
        [[9.87654, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [8.76543, 0.0]],
    ]
    numeric = [
        [[1.23456, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [2.34567, -3.45678]],
    ]
    fitted = [
        [[1.2, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [2.3, -3.4]],
    ]
    analysis = {
        "selection_index": 1,
        "absolute_residual": 0.1,
        "relative_residual": 0.01,
        "max_abs_difference_all": 0.05,
        "formal_parameter_labels": ["Z1_1"],
        "formal_parameters": [3.5],
        "records": [
            {
                "field": "Bx",
                "raw_numeric_matrix": raw_numeric,
                "transformed_numeric_matrix": numeric,
                "formal_fit_matrix": fitted,
                "max_abs_difference": 0.01,
            }
        ],
    }

    final_model = [
        {
            "order": 1,
            "terms": [
                {
                    "label": "Z1_1",
                    "linear_combination": [
                        {
                            "coefficient": [1.0, 0.0],
                            "hermitian_basis_label": "diag(1)",
                            "k_basis_label": "Bx",
                        }
                    ],
                }
            ],
        }
    ]
    numeric_analyses = [
        {
            "selection_index": 1,
            "zeeman_formula": "G formula",
        }
    ]

    text = "\n".join(
        module._format_schur_zeeman_parameter_fit_analyses(
            [analysis],
            star_line="***",
            final_zeeman_model_analyses=final_model,
            numeric_analyses=numeric_analyses,
        )
    )

    assert "Numerical Zeeman matrices from Lowdin perturbation" in text
    assert "G formula" in text
    assert "Symmetrized formual for Zeeman Hamiltonian:" in text
    assert "Z1_1 = 1 * diag(1) * Bx" in text
    assert "Parameters fitted:" in text
    assert "Detailed information:" in text
    assert "Zeeman coefficient matrices G_i" in text
    assert "G_Bx" in text
    assert "Numerical result:" in text
    assert "Fitted result:" in text
    assert "(   9.877  +0.000i)" in text
    assert "(   1.235  +0.000i)" not in text
    assert text.index("Symmetrized formual for Zeeman Hamiltonian:") < text.index("Parameters fitted:")
    assert text.index("Parameters fitted:") < text.index("Detailed information:")
    assert text.index("Detailed information:") < text.index("Zeeman coefficient matrices G_i")
    assert "( 1.234560+0.000000i)" not in text
    assert "Formal parameters:" not in text
    assert "transformed numeric matrix:" not in text
    assert "formal fit matrix:" not in text
    assert "difference matrix:" not in text


def test_summary_formatter_expands_kp_and_zeeman_hamiltonians_with_ordered_parameter_names(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    final_kp = [
        {
            "order": 0,
            "terms": [
                {
                    "label": "K0_1",
                    "linear_combination": [
                        {
                            "coefficient": [1.0 / np.sqrt(2.0), 0.0],
                            "hermitian_basis_label": "diag(1)",
                            "k_basis_label": "1",
                        }
                    ],
                }
            ],
        },
        {
            "order": 1,
            "terms": [
                {
                    "label": "K1_1",
                    "linear_combination": [
                        {
                            "coefficient": [1.0, 0.0],
                            "hermitian_basis_label": "re(1,2)",
                            "k_basis_label": "kx",
                        }
                    ],
                }
            ],
        },
    ]
    kp_fit = [
        {
            "formal_parameter_labels": ["K0_1", "K1_1"],
            "formal_parameters": [1.25, -2.5],
        }
    ]
    final_zeeman = [
        {
            "order": 1,
            "terms": [
                {
                    "label": "Z1_1",
                    "linear_combination": [
                        {
                            "coefficient": [1.0, 0.0],
                            "hermitian_basis_label": "diag(2)",
                            "k_basis_label": "Bz",
                        }
                    ],
                }
            ],
        }
    ]
    zeeman_fit = [
        {
            "formal_parameter_labels": ["Z1_1"],
            "formal_parameters": [3.5],
        }
    ]

    text = "\n".join(
        module._format_final_hamiltonian_summary(
            kp_fit,
            final_kp,
            zeeman_fit,
            final_zeeman,
            star_line="***",
        )
    )

    assert "Summary" in text
    assert "Symmetrized H(k) Hamiltonian:" in text
    assert "H(k)11 = a1" in text
    assert "H(k)12 = b1*kx" in text
    assert "Parameters:\na1:  0.883883476483\nb1: -1.767766952966" in text
    assert "Symmetrized Zeeman Hamiltonian:" in text
    assert "H(z)22 = g1*Bz" in text
    assert "g1:  3.500000000000" in text
    assert text.rstrip().endswith("***\nFinished")


def test_spin_pauli_target_matrices_use_orbital_major_soc_basis(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    wavefunctions = np.eye(4, dtype=complex)
    overlap = np.eye(4, dtype=complex)

    spin_target = module._spin_pauli_target_matrices_from_overlap(
        wavefunctions,
        overlap,
        target_indices=[0, 1],
        nspin=4,
    )

    assert spin_target.shape == (3, 2, 2)
    assert np.allclose(spin_target[0], [[0.0, 1.0], [1.0, 0.0]], atol=1.0e-12, rtol=0.0)
    assert np.allclose(spin_target[1], [[0.0, -1.0j], [1.0j, 0.0]], atol=1.0e-12, rtol=0.0)
    assert np.allclose(spin_target[2], [[1.0, 0.0], [0.0, -1.0]], atol=1.0e-12, rtol=0.0)


def test_numeric_lowdin_kp_reuses_character_eigenvectors(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    velocity_basis = np.zeros((1, 3, 2, 2), dtype=complex)
    velocity_basis[0, 0, 0, 0] = 0.25
    velocity_basis[0, 0, 0, 1] = 2.0
    velocity_basis[0, 0, 1, 0] = 2.0

    class _Solver:
        def diago_H(self, _kpoint):
            raise AssertionError("Lowdin must reuse CHARACTER eigenvectors")

        def get_velocity_basis_k(self, _kpoint):
            return velocity_basis

    class _TB:
        has_rR = True
        tb_solver = _Solver()

    analyses = module._numeric_lowdin_kp_analyses_for_results(
        _TB(),
        [
            {
                "selection_index": 1,
                "kpoint": [0.0, 0.0, 0.0],
                "band": [1, 1],
                "character_payload": {
                    "character_rows": [
                        {
                            "target_band_range": [1, 1],
                            "target_eigenvalues_full": np.asarray([1.0, 5.0], dtype=float),
                            "target_eigenvectors_full": np.eye(2, dtype=complex),
                            "target_representation_matrices": [np.eye(1, dtype=complex)],
                        }
                    ]
                },
            }
        ],
    )

    assert len(analyses) == 1
    assert analyses[0]["basis_convention"] == "CHARACTER eigenvector basis at the reference k point"
    linear = np.asarray(module._complex_array_from_pairs(analyses[0]["linear_matrices"][0]["matrix"]))
    assert linear[0, 0] == pytest.approx(0.25)
    lowdin_xx = np.asarray(
        module._complex_array_from_pairs(analyses[0]["lowdin_quadratic_tensor"][0][0])
    )
    assert lowdin_xx[0, 0] == pytest.approx(-1.0)


def test_lowdin_tb_from_character_payload_uses_symmetrized_hs_and_rr(load_pyatb, monkeypatch) -> None:
    module = load_pyatb("pyatb.symmetry.kp")
    calls: dict[str, object] = {}

    class _ReferenceTB:
        nspin = 4
        HSR_iSsparse = True
        lattice_constant = 1.0
        lattice_vector = np.eye(3)
        max_kpoint_num = 20

    class _ActiveTB:
        def __init__(self, nspin, lattice_constant, lattice_vector, max_kpoint_num):
            calls["init"] = (nspin, lattice_constant, np.asarray(lattice_vector).tolist(), max_kpoint_num)

        def set_solver_HSR(self, hr, sr, is_sparse):
            calls["hsr"] = (hr, sr, is_sparse)

        def set_solver_rR(self, rR_x, rR_y, rR_z, is_sparse):
            calls["rr"] = (rR_x, rR_y, rR_z, is_sparse)

        def read_stru(self, stru_path, need_orb=False):
            calls["stru"] = (stru_path, need_orb)

    monkeypatch.setattr(module, "TBModel", _ActiveTB)
    monkeypatch.setattr(module, "abacus_readHR", lambda nspin, path, unit: ("HR", nspin, path, unit))
    monkeypatch.setattr(module, "abacus_readSR", lambda nspin, path: ("SR", nspin, path))
    monkeypatch.setattr(module, "abacus_readrR", lambda path, unit: (("rRx", path, unit), "rRy", "rRz"))

    active = module._lowdin_tb_from_character_payload(
        _ReferenceTB(),
        {
            "active_stru_path": "/tmp/symmetrized/STRU",
            "active_hr_path": "/tmp/symmetrized/data-HR-sparse_SPIN0-covsymm.csr",
            "active_sr_path": "/tmp/symmetrized/data-SR-sparse_SPIN0-covsymm.csr",
            "lattice_constant": 2.0,
            "lattice_vector": np.eye(3) * 3.0,
            "HR_unit": "Ry",
        },
        rR_route="/tmp/source/data-rR-sparse.csr",
        rR_unit="Bohr",
    )

    assert isinstance(active, _ActiveTB)
    assert calls["init"] == (4, 2.0, (np.eye(3) * 3.0).tolist(), 20)
    assert calls["hsr"] == (
        ("HR", 4, "/tmp/symmetrized/data-HR-sparse_SPIN0-covsymm.csr", "Ry"),
        ("SR", 4, "/tmp/symmetrized/data-SR-sparse_SPIN0-covsymm.csr"),
        True,
    )
    assert calls["rr"] == (("rRx", "/tmp/source/data-rR-sparse.csr", "Bohr"), "rRy", "rRz", True)
    assert calls["stru"] == ("/tmp/symmetrized/STRU", True)
    assert active.kp_lowdin_data_convention == "CHARACTER active symmetrized H/S data with input rR matrix"
    assert active.kp_lowdin_data_paths == {
        "stru": "/tmp/symmetrized/STRU",
        "HR": "/tmp/symmetrized/data-HR-sparse_SPIN0-covsymm.csr",
        "SR": "/tmp/symmetrized/data-SR-sparse_SPIN0-covsymm.csr",
        "rR": "/tmp/source/data-rR-sparse.csr",
    }


def test_numeric_lowdin_kp_skips_without_loaded_rr(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    class _Solver:
        def get_velocity_matrix(self, _kpoint):
            raise AssertionError("velocity solver must not be called without rR")

    class _TB:
        has_rR = False
        tb_solver = _Solver()

    analyses = module._numeric_lowdin_kp_analyses_for_results(
        _TB(),
        [{"selection_index": 1, "kpoint": [0.0, 0.0, 0.0], "band": [1, 1]}],
    )

    assert analyses == []


def test_schur_intertwiner_aligns_equivalent_representations(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    theta = 2.0 * np.pi / 3.0
    rotation = np.asarray(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=complex,
    )
    std = [
        np.eye(2, dtype=complex),
        rotation,
        rotation @ rotation,
    ]
    angle = 0.37
    gauge = np.asarray(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ],
        dtype=complex,
    )
    numeric = [gauge @ matrix @ gauge.conj().T for matrix in std]

    alignment = module._schur_representation_alignment(
        operation_indices=[1, 2, 3],
        numeric_matrices=numeric,
        standard_matrices=std,
    )
    transform = np.asarray(module._complex_array_from_pairs(alignment["unitary_numeric_to_standard"]))

    assert alignment["max_abs_representation_difference"] <= 1.0e-12
    for num, ref in zip(numeric, std):
        assert np.allclose(transform.conj().T @ num @ transform, ref, atol=1.0e-12, rtol=0.0)


def test_spgrep_antiunitary_irrep_matrices_are_formatted(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    spgrep_info = {
        "antiunitary_operations": [
            {
                "antiunitary_operation_index": 1,
                "spgrep_operation_index": 2,
                "time_reversal": True,
                "anti_linear": True,
            }
        ],
        "coreps": [
            {
                "corep_index": 1,
                "antiunitary_operation_matrices": [
                    {
                        "antiunitary_operation_index": 1,
                        "spgrep_operation_index": 2,
                        "matrix": [
                            [[0.0, 0.0], [1.0, 0.0]],
                            [[-1.0, 0.0], [0.0, 0.0]],
                        ],
                    }
                ],
            }
        ],
    }

    text = "\n".join(
        module._format_spgrep_antiunitary_irrep_matrices(
            spgrep_info,
            target_label="GM8",
            star_line="***",
        )
    )

    assert "Spgrep antiunitary representation matrices for GM8" in text
    assert "corep index: 1" in text
    assert "matrix convention: anti-linear operations are written as D K" in text
    assert "antiunitary operation 1" in text
    assert "spgrep operation index: 2" in text
    assert "time reversal: yes" in text
    assert "anti-linear: yes" in text
    assert "D =" in text
    assert "( 1.000000+0.000000i)" in text


def test_spgrep_operations_file_writes_selected_irrep_matrix_sections(load_pyatb, tmp_path: Path) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    identity = [
        [[1.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [1.0, 0.0]],
    ]
    time_reversal_matrix = [
        [[0.0, 0.0], [1.0, 0.0]],
        [[-1.0, 0.0], [0.0, 0.0]],
    ]
    info = {
        "spgrep_operations": {
            "call": "get_spacegroup_spinor_irreps_from_primitive_symmetry(time_reversals=magnetic)",
            "coordinate_convention": "x' = R x + tau",
            "kpoint": [0.0, 0.0, 0.0],
            "spgrep_little_group_mapping_one_based": [1, 2],
            "operations": [
                {
                    "operation_index": 1,
                    "spgrep_operation_index": 1,
                    "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "translation": [0.0, 0.0, 0.0],
                    "time_reversal": False,
                    "anti_linear": False,
                    "in_little_group": True,
                }
            ],
            "antiunitary_operations": [
                {
                    "antiunitary_operation_index": 1,
                    "spgrep_operation_index": 2,
                    "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "translation": [0.0, 0.0, 0.0],
                    "time_reversal": True,
                    "anti_linear": True,
                    "in_little_group": True,
                }
            ],
            "coreps": [
                {
                    "corep_index": 1,
                    "operation_matrices": [
                        {"operation_index": 1, "spgrep_operation_index": 1, "matrix": identity}
                    ],
                    "antiunitary_operation_matrices": [
                        {
                            "antiunitary_operation_index": 1,
                            "spgrep_operation_index": 2,
                            "matrix": time_reversal_matrix,
                        }
                    ],
                },
                {
                    "corep_index": 2,
                    "operation_matrices": [
                        {"operation_index": 1, "spgrep_operation_index": 1, "matrix": identity}
                    ],
                    "antiunitary_operation_matrices": [
                        {
                            "antiunitary_operation_index": 1,
                            "spgrep_operation_index": 2,
                            "matrix": time_reversal_matrix,
                        }
                    ],
                },
            ],
        },
        "character_table": {
            "display_operation_indices": [1],
            "irreps": [
                {"label": "A", "dimension": 2, "characters": [[2.0, 0.0]]},
                {"label": "B", "dimension": 2, "characters": [[2.0, 0.0]]},
            ],
        },
        "representation_alignment_analyses": [
            {
                "target_label": "A + B",
                "target_corep_indices": [1, 2],
                "target_corep_dimensions": [2, 2],
            }
        ],
    }

    module._write_kp_symmetry_info(info, tmp_path)

    text = (tmp_path / "spgrep_operations.txt").read_text(encoding="utf-8")
    assert "Spgrep representation matrices for A" in text
    assert "Spgrep antiunitary representation matrices for A" in text
    assert "Spgrep representation matrices for B" in text
    assert "Spgrep antiunitary representation matrices for B" in text
    assert "Spgrep representation matrices for GM8" not in text
    assert "Spgrep representation matrices for GM9" not in text
    assert text.count("antiunitary operation 1") >= 3


def test_operator_irrep_analysis_decomposes_and_projects_hermitian_basis(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    identity = [
        [[1.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [1.0, 0.0]],
    ]
    c2_matrix = [
        [[1.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [-1.0, 0.0]],
    ]
    spgrep_info = {
        "operations": [{"operation_index": 1}, {"operation_index": 2}],
        "coreps": [
            {
                "corep_index": 1,
                "operation_matrices": [
                    {"operation_index": 1, "matrix": identity},
                    {"operation_index": 2, "matrix": c2_matrix},
                ],
            }
        ],
    }
    character_table = {
        "irreps": [
            {
                "label": "A",
                "dimension": 1,
                "double_valued": False,
                "characters": [[[1.0, 0.0]], [[1.0, 0.0]]],
            },
            {
                "label": "B",
                "dimension": 1,
                "double_valued": False,
                "characters": [[[1.0, 0.0]], [[-1.0, 0.0]]],
            },
        ]
    }

    operation_indices, target_matrices = module._target_matrices_from_corep_indices(spgrep_info, [1])
    analysis = module._operator_irrep_analysis(
        character_table,
        target_label="GM8",
        operation_indices=operation_indices,
        target_matrices=target_matrices,
        target_corep_indices=[1],
    )

    assert analysis["operator_representation"] == "GM8^* x GM8"
    assert [entry["label"] for entry in analysis["hermitian_basis"]] == [
        "diag(1)",
        "diag(2)",
        "re(1,2)",
        "im(1,2)",
    ]
    assert [(item["label"], item["multiplicity"]) for item in analysis["decomposition"]] == [
        ("A", 2),
        ("B", 2),
    ]
    projected = {item["irrep_label"]: item for item in analysis["projected_basis"]}
    assert projected["A"]["rank"] == 2
    assert projected["B"]["rank"] == 2
    a_labels = {
        term["basis_label"]
        for matrix in projected["A"]["matrices"]
        for term in matrix["linear_combination"]
    }
    b_labels = {
        term["basis_label"]
        for matrix in projected["B"]["matrices"]
        for term in matrix["linear_combination"]
    }
    assert a_labels == {"diag(1)", "diag(2)"}
    assert b_labels == {"re(1,2)", "im(1,2)"}

    text = "\n".join(
        module._format_operator_irrep_analyses(
            [analysis],
            character_table=character_table,
            star_line="***",
        )
    )
    assert "Hermitian matrix irreducible decomposition" in text
    assert "Operator representation: GM8^* x GM8" in text
    assert "Decomposition: 2A + 2B" in text
    assert "Projected Hermitian matrix basis for A" in text
    assert "diag(1)" in text
    assert "X1  = diag(1)" in text
    assert "X3  = re(1,2) = 1/sqrt(2) * (E_12 + E_21)" in text
    assert "X4  = im(1,2) = 1/sqrt(2) * (-iE_12 + iE_21)" in text
    assert module._format_complex_symbolic([2.0 ** -0.5, 0.0]) == "1/sqrt(2)"
    assert module._format_complex_symbolic([1.0 / 3.0, 0.0]) == "1/3"
    assert module._format_complex_symbolic([0.0, 2.0]) == "2i"
    assert module._format_complex_symbolic([0.0, -3.0]) == "-3i"
    assert "Projected Hermitian matrix basis for B" in text
    assert "re(1,2)" in text
    assert module._format_complex_symbolic([1.0 / (2.0 * 2.0 ** 0.5), 0.0]) == "sqrt(2)/4"
    assert module._format_complex_symbolic([1.0 / (2.0 * 3.0 ** 0.5), 0.0]) == "sqrt(3)/6"


def test_operator_irrep_analyses_use_combined_band_representation(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    spgrep_info = {
        "operations": [{"operation_index": 1}, {"operation_index": 2}],
        "coreps": [
            {
                "corep_index": 1,
                "operation_matrices": [
                    {"operation_index": 1, "matrix": [[[1.0, 0.0]]]},
                    {"operation_index": 2, "matrix": [[[1.0, 0.0]]]},
                ],
            },
            {
                "corep_index": 2,
                "operation_matrices": [
                    {"operation_index": 1, "matrix": [[[1.0, 0.0]]]},
                    {"operation_index": 2, "matrix": [[[-1.0, 0.0]]]},
                ],
            },
        ],
    }
    character_table = {
        "irreps": [
            {
                "label": "A",
                "dimension": 1,
                "double_valued": False,
                "characters": [[1.0, 0.0], [1.0, 0.0]],
            },
            {
                "label": "B",
                "dimension": 1,
                "double_valued": False,
                "characters": [[1.0, 0.0], [-1.0, 0.0]],
            },
        ]
    }
    info = {
        "spgrep_operations": spgrep_info,
        "character_table": character_table,
    }
    results = [
        {
            "rows": [
                {
                    "irrep": "A + B",
                }
            ]
        }
    ]

    analyses = module._operator_irrep_analyses_for_results(info, results)

    assert len(analyses) == 1
    analysis = analyses[0]
    assert analysis["target_label"] == "A + B"
    assert analysis["operator_representation"] == "(A + B)^* x (A + B)"
    assert analysis["dimension"] == 2
    assert [(item["label"], item["multiplicity"]) for item in analysis["decomposition"]] == [
        ("A", 2),
        ("B", 2),
    ]


def test_k_polynomial_irrep_analysis_decomposes_linear_functions(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    shear_operation = {
        "rotation": [[1, 1, 0], [0, 1, 0], [0, 0, 1]],
    }
    assert module._reciprocal_rotation_from_operation(shear_operation).reshape(-1).tolist() == pytest.approx(
        [1.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    )
    scaled_lattice = [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    swap_operation = {
        "rotation": [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
    }
    assert module._reciprocal_rotation_from_operation(
        swap_operation,
        lattice=scaled_lattice,
    ).reshape(-1).tolist() == pytest.approx(
        [0.0, 2.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0]
    )

    spgrep_info = {
        "operations": [
            {
                "operation_index": 1,
                "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
            {
                "operation_index": 2,
                "rotation": [[1, 0, 0], [0, -1, 0], [0, 0, 1]],
            },
        ]
    }
    character_table = {
        "irreps": [
            {
                "label": "A",
                "dimension": 1,
                "double_valued": False,
                "characters": [[1.0, 0.0], [1.0, 0.0]],
            },
            {
                "label": "B",
                "dimension": 1,
                "double_valued": False,
                "characters": [[1.0, 0.0], [-1.0, 0.0]],
            },
        ]
    }

    analyses = module._k_polynomial_irrep_analyses(
        spgrep_info,
        character_table,
        orders=[1],
    )

    assert len(analyses) == 1
    analysis = analyses[0]
    assert analysis["order"] == 1
    assert [item["label"] for item in analysis["monomial_basis"]] == ["kx", "ky", "kz"]
    assert [(item["label"], item["multiplicity"]) for item in analysis["decomposition"]] == [
        ("A", 2),
        ("B", 1),
    ]
    projected = {item["irrep_label"]: item for item in analysis["projected_basis"]}
    a_labels = {
        term["basis_label"]
        for basis in projected["A"]["functions"]
        for term in basis["linear_combination"]
    }
    b_labels = {
        term["basis_label"]
        for basis in projected["B"]["functions"]
        for term in basis["linear_combination"]
    }
    assert a_labels == {"kx", "kz"}
    assert b_labels == {"ky"}

    text = "\n".join(module._format_k_polynomial_irrep_analyses(analyses, star_line="***"))
    assert "k polynomial irreducible decomposition" in text
    assert "Order = 1" in text
    assert "Decomposition: 2A + B" in text
    assert "Monomial basis:" in text
    assert "operations:" not in text
    assert "characters:" not in text
    assert "rank =" not in text
    assert "Projected k-polynomial basis for A" in text
    assert "kx" in text
    assert "Projected k-polynomial basis for B" in text
    assert "ky" in text


def test_k_polynomial_irrep_analysis_includes_constant_order(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    spgrep_info = {
        "operations": [
            {
                "operation_index": 1,
                "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
            {
                "operation_index": 2,
                "rotation": [[1, 0, 0], [0, -1, 0], [0, 0, 1]],
            },
        ]
    }
    character_table = {
        "irreps": [
            {"label": "A", "dimension": 1, "double_valued": False, "characters": [[1.0, 0.0], [1.0, 0.0]]},
            {"label": "B", "dimension": 1, "double_valued": False, "characters": [[1.0, 0.0], [-1.0, 0.0]]},
        ]
    }

    analyses = module._k_polynomial_irrep_analyses(spgrep_info, character_table, orders=[0])

    assert len(analyses) == 1
    assert analyses[0]["order"] == 0
    assert [item["label"] for item in analyses[0]["monomial_basis"]] == ["1"]
    assert [(item["label"], item["multiplicity"]) for item in analyses[0]["decomposition"]] == [("A", 1)]


def test_zeeman_field_irrep_analysis_uses_cartesian_pseudovector(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    inversion = {
        "rotation": [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
    }
    assert module._zeeman_field_transform_from_operation(inversion).reshape(-1).tolist() == pytest.approx(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    )

    mirror_y = {
        "rotation": [[1, 0, 0], [0, -1, 0], [0, 0, 1]],
    }
    assert module._zeeman_field_transform_from_operation(mirror_y).reshape(-1).tolist() == pytest.approx(
        [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
    )

    spgrep_info = {
        "operations": [
            {
                "operation_index": 1,
                "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
            {
                "operation_index": 2,
                "rotation": [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
            },
        ]
    }
    character_table = {
        "irreps": [
            {"label": "Ag", "dimension": 1, "double_valued": False, "characters": [[1.0, 0.0], [1.0, 0.0]]},
            {"label": "Au", "dimension": 1, "double_valued": False, "characters": [[1.0, 0.0], [-1.0, 0.0]]},
        ]
    }

    analyses = module._zeeman_field_irrep_analyses(spgrep_info, character_table)

    assert len(analyses) == 1
    assert analyses[0]["order"] == 1
    assert analyses[0]["coordinate_basis"] == "cartesian_pseudovector"
    assert [item["label"] for item in analyses[0]["monomial_basis"]] == ["Bx", "By", "Bz"]
    assert [(item["label"], item["multiplicity"]) for item in analyses[0]["decomposition"]] == [("Ag", 3)]

    text = "\n".join(module._format_zeeman_field_irrep_analyses(analyses, star_line="***"))
    assert "Zeeman magnetic-field irreducible decomposition" in text
    assert "Field basis: Cartesian pseudovector Bx, By, Bz" in text
    assert "Decomposition: 3Ag" in text
    assert "field basis:" in text
    assert "B1  = Bx" in text
    assert "Projected Zeeman-field basis for Ag" in text
    assert "order =" not in text
    assert "operations:" not in text
    assert "characters:" not in text
    assert "rank =" not in text
    assert "exponents=" not in text


def test_zeeman_form_solution_formatter_includes_time_reversal_projected_terms(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    form_solutions = [
        {
            "order": 1,
            "irrep_pair_solutions": [
                {
                    "k_irrep_label": "GM2+",
                    "hermitian_irrep_label": "GM2+",
                    "candidate_count": 2,
                    "terms": [
                        {
                            "label": "H1_GM2+_1",
                            "linear_combination": [
                                {
                                    "hermitian_basis_label": "diag(1)",
                                    "k_basis_label": "Bz",
                                    "coefficient": [1.0, 0.0],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    ]
    final_zeeman = [
        {
            "order": 1,
            "terms": [
                {
                    "label": "Z1_1",
                    "linear_combination": [
                        {
                            "hermitian_basis_label": "diag(1)",
                            "k_basis_label": "Bz",
                            "coefficient": [1.0, 0.0],
                        }
                    ],
                }
            ],
        }
    ]

    text = "\n".join(
        module._format_zeeman_form_solution_analyses(
            form_solutions,
            star_line="***",
            final_zeeman_analyses=final_zeeman,
        )
    )

    assert "Invariant Zeeman form solutions" in text
    assert "Zeeman field order = 1" in text
    assert "B irrep GM2+ ------> Hermitian irrep GM2+" in text
    assert "Candidate_count = 1" in text
    assert "H1_GM2+_1 = 1 * diag(1) * Bz" in text
    assert "Time reversal applied: yes" in text
    assert "Z1_1 = 1 * diag(1) * Bz" in text
    assert "rank =" not in text


def test_kp_form_solutions_pair_conjugate_irreps(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    identity = [
        [[1.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [1.0, 0.0]],
    ]
    c2_matrix = [
        [[1.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [-1.0, 0.0]],
    ]
    spgrep_info = {
        "operations": [
            {"operation_index": 1, "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            {"operation_index": 2, "rotation": [[1, 0, 0], [0, -1, 0], [0, 0, 1]]},
        ],
        "coreps": [
            {
                "corep_index": 1,
                "operation_matrices": [
                    {"operation_index": 1, "matrix": identity},
                    {"operation_index": 2, "matrix": c2_matrix},
                ],
            }
        ],
    }
    character_table = {
        "irreps": [
            {"label": "A", "dimension": 1, "double_valued": False, "characters": [[1.0, 0.0], [1.0, 0.0]]},
            {"label": "B", "dimension": 1, "double_valued": False, "characters": [[1.0, 0.0], [-1.0, 0.0]]},
        ]
    }
    operation_indices, target_matrices = module._target_matrices_from_corep_indices(spgrep_info, [1])
    operator_analysis = module._operator_irrep_analysis(
        character_table,
        target_label="GM8",
        operation_indices=operation_indices,
        target_matrices=target_matrices,
        target_corep_indices=[1],
    )
    k_analyses = module._k_polynomial_irrep_analyses(spgrep_info, character_table, orders=[1])

    form_solutions = module._kp_form_solution_analyses(
        [operator_analysis],
        k_analyses,
        character_table,
    )

    assert len(form_solutions) == 1
    assert form_solutions[0]["order"] == 1
    pairs = {(item["k_irrep_label"], item["hermitian_irrep_label"]) for item in form_solutions[0]["irrep_pair_solutions"]}
    assert pairs == {("A", "A"), ("B", "B")}
    assert sum(item["rank"] for item in form_solutions[0]["irrep_pair_solutions"]) == 6

    text = "\n".join(module._format_kp_form_solution_analyses(form_solutions, star_line="***"))
    assert "Invariant k.p form solutions" in text
    assert "k polynomial order = 1" in text
    assert "k irrep A ------> Hermitian irrep A" in text
    assert "k irrep B ------> Hermitian irrep B" in text
    assert "Candidate_count =" in text
    assert "Time reversal applied: no" in text
    assert "rank =" not in text
    assert "* diag(1) * kx" in text or "* diag(2) * kx" in text
    assert "* re(1,2) * ky" in text or "* im(1,2) * ky" in text

    first_term = form_solutions[0]["irrep_pair_solutions"][0]["terms"][0]
    tr_text = "\n".join(
        module._format_kp_form_solution_analyses(
            form_solutions,
            star_line="***",
            time_reversal_analyses=[
                {
                    "term_checks": [
                        {"label": first_term["label"], "status": "ok"},
                    ],
                }
            ],
        )
    )
    assert "Time reversal applied: yes" in tr_text
    assert "K1_1 =" in tr_text


def test_kp_form_solution_formatter_uses_final_time_reversal_basis_by_block(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    form_solutions = [
        {
            "order": 2,
            "irrep_pair_solutions": [
                {
                    "k_irrep_label": "GM1+",
                    "hermitian_irrep_label": "GM1+",
                    "candidate_count": 4,
                    "terms": [
                        {
                            "label": "H2_GM1+_1",
                            "linear_combination": [
                                {
                                    "hermitian_basis_label": "diag(1)",
                                    "k_basis_label": "kz^2",
                                    "coefficient": [1.0, 0.0],
                                }
                            ],
                        }
                    ],
                },
                {
                    "k_irrep_label": "GM3+",
                    "hermitian_irrep_label": "GM3+",
                    "candidate_count": 16,
                    "terms": [
                        {
                            "label": "H2_GM3+_1",
                            "linear_combination": [
                                {
                                    "hermitian_basis_label": "re(1,2)",
                                    "k_basis_label": "kx*ky",
                                    "coefficient": [1.0, 0.0],
                                }
                            ],
                        },
                        {
                            "label": "H2_GM3+_2",
                            "linear_combination": [
                                {
                                    "hermitian_basis_label": "im(1,2)",
                                    "k_basis_label": "kx^2",
                                    "coefficient": [1.0, 0.0],
                                }
                            ],
                        },
                    ],
                },
            ],
        }
    ]
    final_kp = [
        {
            "order": 2,
            "terms": [
                {
                    "label": "K2_1",
                    "linear_combination": [
                        {
                            "hermitian_basis_label": "diag(1)",
                            "k_basis_label": "kz^2",
                            "coefficient": [1.0, 0.0],
                        }
                    ],
                }
            ],
        }
    ]

    text = "\n".join(
        module._format_kp_form_solution_analyses(
            form_solutions,
            star_line="***",
            final_kp_model_analyses=final_kp,
        )
    )

    assert "k irrep GM1+ ------> Hermitian irrep GM1+" in text
    assert "K2_1 = 1 * diag(1) * kz^2" in text
    assert "k irrep GM3+ ------> Hermitian irrep GM3+" in text
    assert "Candidate_count = 2" in text
    assert "K2: none" in text
    assert "Candidate_count = 16" not in text


def test_kp_form_solution_formatter_renumbers_final_k_terms_in_output_order(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    form_solutions = [
        {
            "order": 1,
            "irrep_pair_solutions": [
                {
                    "k_irrep_label": "GM2-",
                    "hermitian_irrep_label": "GM2-",
                    "terms": [
                        {
                            "label": "H1_GM2-_1",
                            "linear_combination": [
                                {
                                    "hermitian_basis_label": "re(1,3)",
                                    "k_basis_label": "kz",
                                    "coefficient": [1.0, 0.0],
                                }
                            ],
                        }
                    ],
                },
                {
                    "k_irrep_label": "GM3-",
                    "hermitian_irrep_label": "GM3-",
                    "terms": [
                        {
                            "label": "H1_GM3-_1",
                            "linear_combination": [
                                {
                                    "hermitian_basis_label": "re(1,4)",
                                    "k_basis_label": "ky",
                                    "coefficient": [1.0, 0.0],
                                }
                            ],
                        }
                    ],
                },
            ],
        }
    ]
    final_kp = [
        {
            "order": 1,
            "terms": [
                {
                    "label": "K1_2",
                    "linear_combination": [
                        {
                            "hermitian_basis_label": "re(1,3)",
                            "k_basis_label": "kz",
                            "coefficient": [1.0, 0.0],
                        }
                    ],
                },
                {
                    "label": "K1_1",
                    "linear_combination": [
                        {
                            "hermitian_basis_label": "re(1,4)",
                            "k_basis_label": "ky",
                            "coefficient": [1.0, 0.0],
                        }
                    ],
                },
            ],
        }
    ]

    text = "\n".join(
        module._format_kp_form_solution_analyses(
            form_solutions,
            star_line="***",
            final_kp_model_analyses=final_kp,
        )
    )

    assert "K1_1 = 1 * re(1,3) * kz" in text
    assert "K1_2 = 1 * re(1,4) * ky" in text
    assert text.index("K1_1 = 1 * re(1,3) * kz") < text.index("K1_2 = 1 * re(1,4) * ky")
    assert "K1_2 = 1 * re(1,3) * kz" not in text


def test_kp_form_solutions_project_only_identity_contractions(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    theta = 2.0 * np.pi / 3.0
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=float,
    )
    reflection = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float)
    e_matrices = [
        np.eye(2),
        rotation,
        rotation @ rotation,
        reflection,
        reflection @ rotation,
        reflection @ rotation @ rotation,
    ]
    zero = np.zeros((2, 2), dtype=float)
    x_rep_matrices = [
        np.block([[matrix, zero], [zero, matrix]])
        for matrix in e_matrices
    ]
    # The k-polynomial matrices follow the function-basis convention.  The
    # coefficient projector must therefore use their inverse representation.
    k_rep_matrices = [np.linalg.inv(matrix) for matrix in e_matrices]
    character_table = {
        "irreps": [
            {
                "label": "E",
                "dimension": 2,
                "double_valued": False,
                "characters": [
                    module._complex_scalar_to_pair(np.trace(matrix))
                    for matrix in e_matrices
                ],
            }
        ]
    }
    operator_analysis = {
        "hermitian_basis": [{"label": f"X{index + 1}"} for index in range(4)],
        "representation_matrices": [
            module._complex_matrix_to_pairs(matrix)
            for matrix in x_rep_matrices
        ],
        "projected_basis": [
            {
                "irrep_label": "E",
                "matrices": [
                    {
                        "linear_combination": [
                            {
                                "basis_index": index + 1,
                                "basis_label": f"X{index + 1}",
                                "coefficient": [1.0, 0.0],
                            }
                        ]
                    }
                    for index in range(4)
                ],
            }
        ],
    }
    k_analysis = {
        "order": 1,
        "monomial_basis": [{"label": "kx"}, {"label": "ky"}],
        "representation_matrices": [
            module._complex_matrix_to_pairs(matrix)
            for matrix in k_rep_matrices
        ],
        "projected_basis": [
            {
                "irrep_label": "E",
                "functions": [
                    {
                        "linear_combination": [
                            {
                                "basis_index": index + 1,
                                "basis_label": ["kx", "ky"][index],
                                "coefficient": [1.0, 0.0],
                            }
                        ]
                    }
                    for index in range(2)
                ],
            }
        ],
    }

    form_solutions = module._kp_form_solution_analyses(
        [operator_analysis],
        [k_analysis],
        character_table,
    )

    pair = form_solutions[0]["irrep_pair_solutions"][0]
    assert pair["candidate_count"] == 8
    assert pair["rank"] == 2
    assert len(pair["terms"]) == 2

    text = "\n".join(module._format_kp_form_solution_analyses(form_solutions, star_line="***"))
    assert "Candidate_count = 2" in text
    assert "Candidate_count = 8" not in text


def test_time_reversal_constraint_flags_odd_spinless_k_terms(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    spgrep_info = {
        "antiunitary_operations": [
            {
                "antiunitary_operation_index": 1,
                "spgrep_operation_index": 2,
                "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "translation": [0.0, 0.0, 0.0],
                "time_reversal": True,
                "anti_linear": True,
            }
        ],
        "coreps": [
            {
                "corep_index": 1,
                "antiunitary_operation_matrices": [
                    {
                        "antiunitary_operation_index": 1,
                        "spgrep_operation_index": 2,
                        "matrix": [[[1.0, 0.0]]],
                    }
                ],
            }
        ],
    }
    operator_analysis = {
        "target_label": "A",
        "dimension": 1,
        "target_corep_indices": [1],
        "hermitian_basis": [{"basis_index": 1, "label": "diag(1)"}],
    }
    form_solutions = [
        {
            "order": 1,
            "irrep_pair_solutions": [
                {
                    "k_irrep_label": "B",
                    "hermitian_irrep_label": "A",
                    "terms": [
                        {
                            "label": "H1_B_1",
                            "linear_combination": [
                                {
                                    "hermitian_basis_index": 1,
                                    "hermitian_basis_label": "diag(1)",
                                    "k_basis_index": 1,
                                    "k_basis_label": "kx",
                                    "coefficient": [1.0, 0.0],
                                }
                            ],
                        }
                    ],
                }
            ],
        },
        {
            "order": 2,
            "irrep_pair_solutions": [
                {
                    "k_irrep_label": "A",
                    "hermitian_irrep_label": "A",
                    "terms": [
                        {
                            "label": "H2_A_1",
                            "linear_combination": [
                                {
                                    "hermitian_basis_index": 1,
                                    "hermitian_basis_label": "diag(1)",
                                    "k_basis_index": 1,
                                    "k_basis_label": "kx^2",
                                    "coefficient": [1.0, 0.0],
                                }
                            ],
                        }
                    ],
                }
            ],
        },
    ]

    analyses = module._time_reversal_constraint_analyses(
        {"spgrep_operations": spgrep_info},
        [operator_analysis],
        form_solutions,
    )

    checks = {
        item["label"]: item
        for item in analyses[0]["term_checks"]
    }
    assert checks["H1_B_1"]["status"] == "violation"
    assert checks["H1_B_1"]["parity"] == -1
    assert checks["H2_A_1"]["status"] == "ok"
    assert checks["H2_A_1"]["parity"] == 1

    text = "\n".join(
        module._format_time_reversal_constraint_analyses(
            analyses,
            star_line="***",
        )
    )
    assert "Time-reversal constraint check for k.p invariant terms" in text
    assert "H1_B_1   order = 1   parity = -1   status = violation" in text
    assert "H2_A_1   order = 2   parity = 1   status = ok" in text


def test_final_kp_model_uses_real_tr_even_linear_combinations(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.kp")

    spgrep_info = {
        "antiunitary_operations": [
            {
                "antiunitary_operation_index": 1,
                "spgrep_operation_index": 2,
                "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "translation": [0.0, 0.0, 0.0],
                "time_reversal": True,
                "anti_linear": True,
            }
        ],
        "coreps": [
            {
                "corep_index": 1,
                "antiunitary_operation_matrices": [
                    {
                        "antiunitary_operation_index": 1,
                        "spgrep_operation_index": 2,
                        "matrix": [
                            [[0.0, 0.0], [1.0, 0.0]],
                            [[1.0, 0.0], [0.0, 0.0]],
                        ],
                    }
                ],
            }
        ],
    }
    operator_analysis = {
        "target_label": "A",
        "dimension": 2,
        "target_corep_indices": [1],
        "hermitian_basis": [
            {"basis_index": 1, "label": "diag(1)"},
            {"basis_index": 2, "label": "diag(2)"},
            {"basis_index": 3, "label": "re(1,2)"},
            {"basis_index": 4, "label": "im(1,2)"},
        ],
    }
    form_solutions = [
        {
            "order": 1,
            "irrep_pair_solutions": [
                {
                    "k_irrep_label": "B",
                    "hermitian_irrep_label": "A",
                    "terms": [
                        {
                            "label": "H1_B_1",
                            "linear_combination": [
                                {
                                    "hermitian_basis_index": 1,
                                    "hermitian_basis_label": "diag(1)",
                                    "k_basis_index": 1,
                                    "k_basis_label": "kx",
                                    "coefficient": [1.0, 0.0],
                                }
                            ],
                        },
                        {
                            "label": "H1_B_2",
                            "linear_combination": [
                                {
                                    "hermitian_basis_index": 2,
                                    "hermitian_basis_label": "diag(2)",
                                    "k_basis_index": 1,
                                    "k_basis_label": "kx",
                                    "coefficient": [1.0, 0.0],
                                }
                            ],
                        },
                    ],
                }
            ],
        },
        {
            "order": 2,
            "irrep_pair_solutions": [
                {
                    "k_irrep_label": "A",
                    "hermitian_irrep_label": "A",
                    "terms": [
                        {
                            "label": "H2_A_1",
                            "linear_combination": [
                                {
                                    "hermitian_basis_index": 1,
                                    "hermitian_basis_label": "diag(1)",
                                    "k_basis_index": 1,
                                    "k_basis_label": "kx^2",
                                    "coefficient": [0.0, 1.0],
                                }
                            ],
                        },
                        {
                            "label": "H2_A_2",
                            "linear_combination": [
                                {
                                    "hermitian_basis_index": 2,
                                    "hermitian_basis_label": "diag(2)",
                                    "k_basis_index": 1,
                                    "k_basis_label": "kx^2",
                                    "coefficient": [0.0, 1.0],
                                }
                            ],
                        },
                    ],
                }
            ],
        },
    ]

    analyses = module._final_kp_model_analyses(
        {"spgrep_operations": spgrep_info},
        [operator_analysis],
        form_solutions,
    )

    by_order = {item["order"]: item for item in analyses}
    assert by_order[1]["real_candidate_rank"] == 2
    assert by_order[1]["tr_even_rank"] == 1
    assert by_order[1]["tr_odd_rank"] == 1
    assert len(by_order[1]["terms"]) == 1
    order1_terms = by_order[1]["terms"][0]["linear_combination"]
    assert {term["hermitian_basis_label"] for term in order1_terms} == {"diag(1)", "diag(2)"}
    assert all(abs(term["coefficient"][1]) < 1.0e-10 for term in order1_terms)

    assert by_order[2]["real_candidate_rank"] == 2
    assert by_order[2]["tr_even_rank"] == 1
    assert by_order[2]["terms"][0]["linear_combination"][0]["coefficient"][1] == pytest.approx(0.0)

    text = "\n".join(module._format_final_kp_model_analyses(analyses, star_line="***"))
    assert "Final real Hermitian time-reversal-even k.p basis" in text
    assert "order = 1" in text
    assert "tr_even_rank = 1" in text
    assert "K1_1" in text


def test_parse_abacus_stru_magnetic_moments_from_atom_lines(load_pyatb, tmp_path: Path) -> None:
    module = load_pyatb("pyatb.symmetry.kp")
    stru = tmp_path / "STRU"
    stru.write_text(
        "\n".join(
            [
                "ATOMIC_SPECIES",
                "X 1 X.upf",
                "NUMERICAL_ORBITAL",
                "X.orb",
                "LATTICE_CONSTANT",
                "1.0",
                "LATTICE_VECTORS",
                "1 0 0",
                "0 1 0",
                "0 0 1",
                "ATOMIC_POSITIONS",
                "Direct",
                "X",
                "0.5",
                "2",
                "0 0 0 1 1 1 mag 2.0",
                "0.5 0.5 0.5 1 1 1 mag 1.0 2.0 3.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    moments = module._parse_abacus_stru_magnetic_moments(stru, atom_count=2)

    assert moments.tolist() == [[0.0, 0.0, 2.0], [1.0, 2.0, 3.0]]
