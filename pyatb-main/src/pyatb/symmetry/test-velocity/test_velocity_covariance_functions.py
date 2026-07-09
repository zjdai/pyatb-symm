from __future__ import annotations

import sys
import unittest
import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "velocity_covariance.py"
SPEC = importlib.util.spec_from_file_location("velocity_covariance", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Failed to load {MODULE_PATH}.")
velocity_covariance = importlib.util.module_from_spec(SPEC)
sys.modules["velocity_covariance"] = velocity_covariance
SPEC.loader.exec_module(velocity_covariance)

band_gauge_switch_matrix = velocity_covariance.band_gauge_switch_matrix
switch_band_velocity_gauge = velocity_covariance.switch_band_velocity_gauge
time_reversal_basis_matrix = velocity_covariance.time_reversal_basis_matrix
transform_band_velocity_matrix = velocity_covariance.transform_band_velocity_matrix
transform_band_velocity_to_target_gauge = velocity_covariance.transform_band_velocity_to_target_gauge
transform_orbital_velocity_matrix = velocity_covariance.transform_orbital_velocity_matrix
transform_wavefunction_coefficients = velocity_covariance.transform_wavefunction_coefficients


class VelocityCovarianceFunctionTests(unittest.TestCase):
    def test_wavefunction_unitary_nspin1(self) -> None:
        representation = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        coefficients = np.array([[1.0 + 2.0j, 0.5], [3.0 - 1.0j, -2.0j]], dtype=complex)

        transformed = transform_wavefunction_coefficients(
            representation,
            coefficients,
            nspin=1,
            antiunitary=False,
        )

        np.testing.assert_allclose(transformed, representation @ coefficients)

    def test_wavefunction_antiunitary_soc(self) -> None:
        representation = np.eye(4, dtype=complex)
        coefficients = np.array(
            [
                [1.0 + 2.0j, 2.0 - 0.5j],
                [0.5 - 1.0j, -1.0 + 0.25j],
                [3.0 + 0.0j, -2.0j],
                [-0.25 + 0.75j, 1.5 + 1.0j],
            ],
            dtype=complex,
        )
        time_reversal = time_reversal_basis_matrix(4, basis_dim=4)

        transformed = transform_wavefunction_coefficients(
            representation,
            coefficients,
            nspin=4,
            antiunitary=True,
        )

        np.testing.assert_allclose(transformed, representation @ time_reversal @ coefficients.conj())

    def test_orbital_velocity_unitary(self) -> None:
        representation = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        rotation = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        velocity = np.zeros((3, 2, 2), dtype=complex)
        velocity[0] = np.array([[1.0, 2.0j], [-2.0j, 3.0]], dtype=complex)
        velocity[1] = np.array([[4.0, 1.0 - 1.0j], [1.0 + 1.0j, 5.0]], dtype=complex)
        velocity[2] = np.array([[6.0, -1.0j], [1.0j, 7.0]], dtype=complex)

        transformed = transform_orbital_velocity_matrix(
            representation,
            rotation,
            velocity,
            nspin=1,
            antiunitary=False,
        )

        mixed = np.asarray([velocity[1], velocity[0], velocity[2]], dtype=complex)
        expected = np.asarray([representation @ mixed[a] @ representation.conj().T for a in range(3)])
        np.testing.assert_allclose(transformed, expected)

    def test_orbital_velocity_antiunitary_soc(self) -> None:
        representation = np.eye(4, dtype=complex)
        rotation = np.eye(3, dtype=float)
        velocity = np.zeros((3, 4, 4), dtype=complex)
        for alpha in range(3):
            base = np.arange(16, dtype=float).reshape(4, 4) + alpha
            velocity[alpha] = base + 1j * base.T
        time_reversal = time_reversal_basis_matrix(4, basis_dim=4)

        transformed = transform_orbital_velocity_matrix(
            representation,
            rotation,
            velocity,
            nspin=4,
            antiunitary=True,
        )

        expected = np.asarray(
            [time_reversal @ (-np.conj(velocity[a])) @ time_reversal.conj().T for a in range(3)],
            dtype=complex,
        )
        np.testing.assert_allclose(transformed, expected)

    def test_band_velocity_transported_and_target_gauge(self) -> None:
        rotation = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        velocity = np.zeros((3, 2, 2), dtype=complex)
        velocity[0] = np.array([[1.0, 2.0j], [-2.0j, 2.0]], dtype=complex)
        velocity[1] = np.array([[3.0, 1.0], [1.0, 4.0]], dtype=complex)
        velocity[2] = np.array([[5.0, -1.0j], [1.0j, 6.0]], dtype=complex)
        phase_switch = np.diag([1.0, 1.0j]).astype(complex)

        transported = transform_band_velocity_matrix(rotation, velocity, antiunitary=True)
        expected_transport = -np.conj(np.asarray([-velocity[1], velocity[0], velocity[2]], dtype=complex))
        np.testing.assert_allclose(transported, expected_transport)

        switched = switch_band_velocity_gauge(transported, phase_switch)
        direct = transform_band_velocity_to_target_gauge(
            rotation,
            velocity,
            phase_switch,
            antiunitary=True,
        )
        expected_switch = np.asarray(
            [phase_switch @ transported[a] @ phase_switch.conj().T for a in range(3)],
            dtype=complex,
        )
        np.testing.assert_allclose(switched, expected_switch)
        np.testing.assert_allclose(direct, expected_switch)

    def test_band_gauge_switch_matrix(self) -> None:
        representation = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        source_coefficients = np.eye(2, dtype=complex)
        target_coefficients = np.eye(2, dtype=complex)
        target_overlap = np.eye(2, dtype=complex)

        switch = band_gauge_switch_matrix(
            target_coefficients,
            target_overlap,
            representation,
            source_coefficients,
            nspin=1,
            antiunitary=False,
        )

        np.testing.assert_allclose(switch, representation)


if __name__ == "__main__":
    unittest.main()
