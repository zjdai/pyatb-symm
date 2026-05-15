from __future__ import annotations

from pathlib import Path

import pytest


class _FakeTB:
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


def test_calculate_kp_irreps_writes_summary_and_archives_character_outputs(
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
        kpoint_direct_coor=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        band=[[77, 78], [79, 80]],
        label=["GM bands", "Z bands"],
        output_path=tmp_path / "Out" / "KP",
        group=166,
        occ_band=80,
    )

    assert [result["band"] for result in results] == [[77, 78], [79, 80]]
    assert (tmp_path / "Out" / "KP" / "kp_irrep.json").exists()
    summary = (tmp_path / "Out" / "KP" / "band_irrep.txt").read_text(encoding="utf-8")
    assert "selection=  1" in summary
    assert "GM bands" in summary
    assert "IR1" in summary
    assert "selection=  2" in summary
    assert "Z bands" in summary
    assert "IR2" in summary
    assert (tmp_path / "Out" / "KP" / "selection_001" / "band_irrep.txt").exists()
    assert (tmp_path / "Out" / "KP" / "selection_002" / "trace.txt").read_text(encoding="utf-8") == "trace K2\n"
