from __future__ import annotations

from pathlib import Path


def test_read_stru_resolves_numerical_orbital_relative_to_stru(load_pyatb, tmp_path: Path) -> None:
    module = load_pyatb("pyatb.io.abacus_read_stru")
    orb_dir = tmp_path / "orb"
    orb_dir.mkdir()
    orb_path = orb_dir / "H.orb"
    orb_path.write_text(
        "\n".join(
            [
                "line1",
                "line2",
                "line3",
                "line4",
                "lmax 0",
                "norb 1",
                "line7",
                "line8",
                "line9",
                "mesh 4",
                "dr 0.1",
                "orbital 1",
                "comment",
                "1.0 2.0 3.0 4.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    stru_path = tmp_path / "STRU"
    stru_path.write_text(
        "\n".join(
            [
                "ATOMIC_SPECIES",
                "H 1.0 H.upf",
                "NUMERICAL_ORBITAL",
                "orb/H.orb",
                "LATTICE_CONSTANT",
                "1.0",
                "LATTICE_VECTORS",
                "1.0 0.0 0.0",
                "0.0 1.0 0.0",
                "0.0 0.0 1.0",
                "ATOMIC_POSITIONS",
                "Direct",
                "H",
                "0.0",
                "1",
                "0.0 0.0 0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    atoms = module.read_stru(str(stru_path))
    atoms[0].read_numerical_orb()

    assert Path(atoms[0].orb_file) == orb_path
    assert atoms[0].l_max == 0
    assert atoms[0].mesh == 4
