from __future__ import annotations

from pathlib import Path

import pytest


def _write_kp_input(path: Path, kp_block: str) -> None:
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
                "KP",
                "{",
                kp_block,
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_kp_block_accepts_direct_kpoints_with_per_kpoint_bands(load_pyatb, tmp_path: Path) -> None:
    input_mod = load_pyatb("pyatb.io.input")
    input_file = tmp_path / "Input"
    _write_kp_input(
        input_file,
        "\n".join(
            [
                "    stru_file STRU",
                "    kpoint_mode direct",
                "    kpoint_num 2",
                "    kpoint_direct_coor 0.0 0.0 0.0 0.5 0.5 0.5",
                "    band 77 78 79 80",
                "    group 166",
                "    symm_prec 1e-6",
                "    occ_band 80",
                "    mag_tag 0",
                "    mag auto",
            ]
        ),
    )

    input_data, function_switch, _ = input_mod.read_input(str(input_file))

    assert function_switch["KP"] is True
    assert input_data["KP"]["kpoint_mode"] == "direct"
    assert input_data["KP"]["kpoint_num"] == 2
    assert input_data["KP"]["kpoint_direct_coor"].tolist() == [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    assert input_data["KP"]["band"].tolist() == [[77, 78], [79, 80]]
    assert input_data["KP"]["group"] == 166
    assert input_data["KP"]["symm_prec"] == pytest.approx(1.0e-6)
    assert input_data["KP"]["occ_band"] == 80
    assert input_data["KP"]["mag"] == "auto"


def test_kp_block_rejects_band_count_not_matching_kpoints(load_pyatb, tmp_path: Path) -> None:
    input_mod = load_pyatb("pyatb.io.input")
    input_file = tmp_path / "Input"
    _write_kp_input(
        input_file,
        "\n".join(
            [
                "    kpoint_mode direct",
                "    kpoint_num 2",
                "    kpoint_direct_coor 0.0 0.0 0.0 0.5 0.5 0.5",
                "    band 77 78",
            ]
        ),
    )

    with pytest.raises(ValueError, match="KP.band"):
        input_mod.read_input(str(input_file))
