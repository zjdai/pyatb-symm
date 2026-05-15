from __future__ import annotations

from pathlib import Path

import pytest


def _write_input(path: Path, character_block: str) -> None:
    path.write_text(
        "\n".join(
            [
                "INPUT_PARAMETERS",
                "{",
                "    nspin 1",
                "    package ABACUS",
                "    fermi_energy 0.0",
                "    fermi_energy_unit eV",
                "    HR_route data-HR-sparse_SPIN0.csr",
                "    SR_route data-SR-sparse_SPIN0.csr",
                "}",
                "",
                "LATTICE",
                "{",
                "    lattice_constant 1.0",
                "    lattice_constant_unit Angstrom",
                "    lattice_vector 1 0 0 0 1 0 0 0 1",
                "}",
                "",
                "CHARACTER",
                "{",
                character_block,
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_character_block_accepts_direct_kpoints_auto_group_and_auto_mag(load_pyatb, tmp_path: Path) -> None:
    input_mod = load_pyatb("pyatb.io.input")
    input_file = tmp_path / "Input"
    _write_input(
        input_file,
        "\n".join(
            [
                "    kpoint_mode direct",
                "    kpoint_num 2",
                "    kpoint_direct_coor 0.0 0.0 0.0 0.5 0.0 0.0",
                "    group auto",
                "    symm_prec 1e-5",
                "    occ_band 8",
                "    band 7 10",
                "    mag_tag 0",
                "    mag auto",
                "    data_symmetrize 1",
                "    data_symm_target_max_abs_ry 1e-8",
                "    data_symm_max_iter_per_operation 5",
                "    data_symm_nonzero_block_tol 1e-9",
                "    data_symm_verbose 1",
            ]
        ),
    )

    input_data, function_switch, _ = input_mod.read_input(str(input_file))

    assert function_switch["CHARACTER"] is True
    assert input_data["CHARACTER"]["kpoint_mode"] == "direct"
    assert input_data["CHARACTER"]["kpoint_direct_coor"].tolist() == [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
    assert input_data["CHARACTER"]["group"] == "auto"
    assert input_data["CHARACTER"]["mag"] == "auto"
    assert input_data["CHARACTER"]["data_symmetrize"] == 1
    assert input_data["CHARACTER"]["data_symm_target_max_abs_ry"] == pytest.approx(1e-8)
    assert input_data["CHARACTER"]["data_symm_max_iter_per_operation"] == 5
    assert input_data["CHARACTER"]["data_symm_nonzero_block_tol"] == pytest.approx(1e-9)
    assert input_data["CHARACTER"]["data_symm_verbose"] == 1


def test_character_block_accepts_mp_kpoints_integer_group_and_manual_mag(load_pyatb, tmp_path: Path) -> None:
    input_mod = load_pyatb("pyatb.io.input")
    input_file = tmp_path / "Input"
    _write_input(
        input_file,
        "\n".join(
            [
                "    kpoint_mode mp",
                "    mp_grid 4 4 1",
                "    group 166",
                "    symm_prec 1e-6",
                "    occ_band 24",
                "    band 23 28",
                "    mag_tag 1",
                "    mag 0 0 5 0 0 -5",
            ]
        ),
    )

    input_data, _, _ = input_mod.read_input(str(input_file))

    assert input_data["CHARACTER"]["kpoint_mode"] == "mp"
    assert input_data["CHARACTER"]["mp_grid"].tolist() == [4, 4, 1]
    assert input_data["CHARACTER"]["group"] == 166
    assert input_data["CHARACTER"]["mag_tag"] == 1
    assert input_data["CHARACTER"]["mag"].tolist() == [0.0, 0.0, 5.0, 0.0, 0.0, -5.0]


def test_character_block_rejects_invalid_manual_mag_length(load_pyatb, tmp_path: Path) -> None:
    input_mod = load_pyatb("pyatb.io.input")
    input_file = tmp_path / "Input"
    _write_input(
        input_file,
        "\n".join(
            [
                "    kpoint_mode direct",
                "    kpoint_num 1",
                "    kpoint_direct_coor 0.0 0.0 0.0",
                "    group auto",
                "    symm_prec 1e-5",
                "    occ_band 8",
                "    band 7 10",
                "    mag_tag 1",
                "    mag 0 0 5 1",
            ]
        ),
    )

    with pytest.raises(ValueError):
        input_mod.read_input(str(input_file))
