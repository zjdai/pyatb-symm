from __future__ import annotations

import numpy as np


class _FakeIrrep:
    def __init__(self, name, characters, raw_name=None, phase_kinds=None, coeff_uvw=None):
        self.raw_name = raw_name if raw_name is not None else name
        self.name = name
        self.reality = 1
        self.characters = np.array(characters, dtype=complex)
        self.active_ops = np.array([True] * len(characters), dtype=bool)
        self.phase_kinds = np.array(
            phase_kinds if phase_kinds is not None else [1] * len(characters), dtype=int
        )
        self.coeff_uvw = np.array(
            coeff_uvw if coeff_uvw is not None else [[0.0, 0.0, 0.0]] * len(characters),
            dtype=float,
        )


class _FakeEntry:
    def __init__(self, irreps):
        self.irreps = irreps


class _FakeResolution:
    def __init__(self, entry, k_conv=None, cornwell_satisfied=True):
        self.entry = entry
        self.k_conv = np.array(k_conv if k_conv is not None else [0.0, 0.0, 0.0], dtype=float)
        self.cornwell_satisfied = cornwell_satisfied



def test_group_degenerate_bands_respects_energy_tolerance(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.character_core")
    energies = np.array([0.0, 1.0e-7, 0.2, 0.200000008, 0.6], dtype=float)

    groups = module.group_degenerate_bands(energies, tol=1.0e-6)

    assert groups == [(0, 1), (2, 3), (4, 4)]



def test_group_degenerate_bands_uses_character_energy_tolerance_by_default(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.character_core")
    energies = np.array([0.0, 4.9e-4, 1.2e-3], dtype=float)

    groups = module.group_degenerate_bands(energies)

    assert groups == [(0, 1), (2, 2)]


def test_assign_irrep_matches_trace_vector(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.character_core")
    resolution = _FakeResolution(
        _FakeEntry(
            [
                _FakeIrrep("A", [1.0 + 0.0j, 1.0 + 0.0j]),
                _FakeIrrep("B", [1.0 + 0.0j, -1.0 + 0.0j]),
            ]
        )
    )

    matched = module.assign_irrep_from_characters(
        np.array([1.0 + 0.0j, -1.0 + 0.0j], dtype=complex),
        resolution,
        active_operation_indices=[0, 1],
        tol=1.0e-8,
    )

    assert matched == "B"


def test_assign_irrep_combination_uses_representative_table_columns(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.character_core")
    resolution = _FakeResolution(
        _FakeEntry(
            [
                _FakeIrrep("X1", [1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]),
                _FakeIrrep("X2", [1.0 + 0.0j, 3.0 + 0.0j, 2.0 + 0.0j]),
            ]
        )
    )

    matched = module.assign_irrep_combination(
        np.array([1.0 + 0.0j, 3.0 + 0.0j], dtype=complex),
        resolution,
        active_operation_indices=[0, 1],
        table_operation_indices=[0, 2],
        max_terms=1,
        tol=1.0e-8,
    )

    assert matched == "X1"


def test_calculate_subspace_characters_takes_trace_in_degenerate_subspace(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.character_core")
    eigenvectors = np.eye(3, dtype=complex)
    overlap = np.eye(3, dtype=complex)
    operations = [
        np.diag([1.0, -1.0, 1.0]).astype(complex),
        np.diag([1.0, 1.0, -1.0]).astype(complex),
    ]

    chars = module.calculate_subspace_characters(
        eigenvectors=eigenvectors,
        overlap=overlap,
        operation_matrices=operations,
        band_range=(0, 1),
    )

    assert np.allclose(chars, np.array([0.0 + 0.0j, 2.0 + 0.0j], dtype=complex))


def test_assign_irrep_combination_prefers_double_valued_irreps_for_soc(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.character_core")
    resolution = _FakeResolution(
        _FakeEntry(
            [
                _FakeIrrep("F1+", [1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j]),
                _FakeIrrep("F2+", [1.0 + 0.0j, -1.0 + 0.0j, 1.0 + 0.0j, -1.0 + 0.0j]),
                _FakeIrrep("F3", [1.0 + 0.0j, 0.0 - 1.0j, 1.0 + 0.0j, 0.0 - 1.0j], raw_name="-F3"),
                _FakeIrrep("F4", [1.0 + 0.0j, 0.0 + 1.0j, 1.0 + 0.0j, 0.0 + 1.0j], raw_name="-F4"),
            ]
        )
    )

    matched = module.assign_irrep_combination(
        np.array([2.0 + 0.0j, 0.0 + 0.0j, 2.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
        resolution,
        active_operation_indices=[0, 1, 2, 3],
        max_terms=2,
        tol=1.0e-8,
        spinful=True,
    )

    assert matched == "F3 + F4"


def test_nonsymmorphic_boundary_table_conjugates_raw_characters_for_direct_matching(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.character_core")
    phase_kinds = [1, 2, 2, 1]
    coeff_uvw = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ]
    resolution = _FakeResolution(
        _FakeEntry(
            [
                _FakeIrrep("T4", [1.0 + 0.0j, 0.0 - 1.0j, 0.0 - 1.0j, -1.0 + 0.0j], phase_kinds=phase_kinds, coeff_uvw=coeff_uvw),
                _FakeIrrep("T5", [1.0 + 0.0j, 0.0 + 1.0j, 0.0 + 1.0j, -1.0 + 0.0j], phase_kinds=phase_kinds, coeff_uvw=coeff_uvw),
            ]
        ),
        k_conv=[0.5, 0.5, 0.026316],
        cornwell_satisfied=False,
    )

    matched = module.assign_irrep_combination(
        np.array([2.0 + 0.0j, -0.1652 - 1.9932j, -0.1652 - 1.9932j, -2.0 + 0.0j], dtype=complex),
        resolution,
        active_operation_indices=[0, 1, 2, 3],
        max_terms=2,
        tol=1.0e-2,
        spinful=False,
    )

    assert matched == "T5 + T5"


def test_assign_irrep_combination_applies_delta_tau_phase_to_table(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.character_core")

    class _Op:
        def __init__(self, translation):
            self.translation = np.array(translation, dtype=float)

    resolution = _FakeResolution(
        _FakeEntry(
            [
                _FakeIrrep("Y3", [1.0 + 0.0j, 1.0 + 0.0j], raw_name="-Y3"),
                _FakeIrrep("Y4", [1.0 + 0.0j, -1.0 + 0.0j], raw_name="-Y4"),
            ]
        ),
        k_conv=[0.0, 0.5, 0.0],
        cornwell_satisfied=False,
    )
    phase_operations = [_Op([0.0, 0.0, 0.0]), _Op([0.0, 0.5, 0.0])]

    matched = module.assign_irrep_combination(
        np.array([1.0 + 0.0j, 0.0 + 1.0j], dtype=complex),
        resolution,
        active_operation_indices=[0, 1],
        table_operation_indices=[0, 1],
        phase_k_direct=np.array([0.0, 0.5, 0.0], dtype=float),
        phase_operations=phase_operations,
        table_operation_translations=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float),
        max_terms=1,
        tol=1.0e-8,
        spinful=False,
    )

    assert matched == "Y3"


def test_assign_irrep_combination_does_not_double_count_table_phase(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.character_core")

    class _Op:
        def __init__(self, translation):
            self.translation = np.array(translation, dtype=float)

    phase = np.exp(0.25j * np.pi)
    resolution = _FakeResolution(
        _FakeEntry(
            [
                _FakeIrrep("Z1", [1.0 + 0.0j, 0.0 + 1.0j, phase, phase**3]),
                _FakeIrrep("Z3", [1.0 + 0.0j, 0.0 - 1.0j, np.conj(phase), np.conj(phase**3)]),
            ]
        ),
        k_conv=[0.0, 0.0, 0.5],
        cornwell_satisfied=True,
    )
    translations = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.25], [0.0, 0.0, 0.75]],
        dtype=float,
    )
    phase_operations = [_Op(tau) for tau in translations]

    matched = module.assign_irrep_combination(
        np.array([2.0 + 0.0j, 0.0 + 0.0j, 2.0**0.5 + 0.0j, -(2.0**0.5) + 0.0j], dtype=complex),
        resolution,
        active_operation_indices=[0, 1, 2, 3],
        table_operation_indices=[0, 1, 2, 3],
        phase_k_direct=np.array([0.0, 0.0, 0.5], dtype=float),
        phase_operations=phase_operations,
        table_operation_translations=translations,
        max_terms=2,
        tol=1.0e-8,
        spinful=False,
    )

    assert matched == "Z1 + Z3"


def test_assign_irrep_combination_applies_coeff_phase_on_cornwell_line(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.character_core")
    coeff_uvw = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.5], [0.0, 0.0, 1.5]]
    phase_kinds = [1, 2, 2, 2]
    resolution = _FakeResolution(
        _FakeEntry(
            [
                _FakeIrrep("LD5", [1.0 + 0.0j, 0.0 - 1.0j, -(2.0**-0.5) + (2.0**-0.5) * 1.0j, -(2.0**-0.5) - (2.0**-0.5) * 1.0j], phase_kinds=phase_kinds, coeff_uvw=coeff_uvw),
                _FakeIrrep("LD6", [1.0 + 0.0j, 0.0 - 1.0j, (2.0**-0.5) - (2.0**-0.5) * 1.0j, (2.0**-0.5) + (2.0**-0.5) * 1.0j], phase_kinds=phase_kinds, coeff_uvw=coeff_uvw),
            ]
        ),
        k_conv=[0.0, 0.0, 1.0 / 12.0],
        cornwell_satisfied=True,
    )

    matched = module.assign_irrep_combination(
        np.array([1.0 + 0.0j, -0.258819 - 0.965926j, 0.608761 - 0.793353j, 0.923880 + 0.382683j], dtype=complex),
        resolution,
        active_operation_indices=[0, 1, 2, 3],
        table_operation_indices=[0, 1, 2, 3],
        max_terms=1,
        tol=1.0e-5,
        spinful=True,
    )

    assert matched == "LD6"


def test_spinless_assignment_prefers_irvsp_single_valued_decomposition(load_pyatb) -> None:
    module = load_pyatb("pyatb.symmetry.character_core")
    resolution = _FakeResolution(
        _FakeEntry(
            [
                _FakeIrrep("R1", [1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j]),
                _FakeIrrep("R2", [1.0 + 0.0j, -1.0 + 0.0j, 1.0 + 0.0j, -1.0 + 0.0j]),
                _FakeIrrep("R3", [1.0 + 0.0j, -1.0 + 0.0j, -1.0 + 0.0j, 1.0 + 0.0j]),
                _FakeIrrep("R4", [1.0 + 0.0j, 1.0 + 0.0j, -1.0 + 0.0j, -1.0 + 0.0j]),
                _FakeIrrep("R5", [2.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], raw_name="-R5"),
            ]
        )
    )

    matched = module.assign_irrep_combination(
        np.array([4.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
        resolution,
        active_operation_indices=[0, 1, 2, 3],
        max_terms=4,
        tol=1.0e-8,
        spinful=False,
    )

    assert matched == "R1 + R2 + R3 + R4"
