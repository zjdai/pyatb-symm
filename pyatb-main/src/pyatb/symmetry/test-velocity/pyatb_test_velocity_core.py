from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import json

import numpy as np
import spglib
from ase.data import atomic_numbers

from pyatb.io.abacus_read_xr import abacus_readHR, abacus_readSR, abacus_readrR
from pyatb.symmetry.Dk_matrix import build_dk_matrix, extract_abacus_basis_metadata
from pyatb.symmetry.data_covariance_constraint import _cart_rotation_from_fractional
from pyatb.symmetry.hs_covariance import read_stru_lattice_vector
from pyatb.tb.tb import tb as TBModel


@dataclass
class VelocityMatrixError:
    max_abs: float
    mean_abs: float
    rms_abs: float
    rel_fro: float
    element_count: int
    worst_direction: str
    worst_row: int
    worst_col: int
    worst_reference: complex
    worst_predicted: complex
    worst_difference: complex
    direction_errors: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_abs": float(self.max_abs),
            "mean_abs": float(self.mean_abs),
            "rms_abs": float(self.rms_abs),
            "rel_fro": float(self.rel_fro),
            "element_count": int(self.element_count),
            "worst_direction": str(self.worst_direction),
            "worst_row": int(self.worst_row),
            "worst_col": int(self.worst_col),
            "worst_reference": _complex_to_pair(self.worst_reference),
            "worst_predicted": _complex_to_pair(self.worst_predicted),
            "worst_difference": _complex_to_pair(self.worst_difference),
            "direction_errors": self.direction_errors,
        }


@dataclass
class KPointData:
    k_direct: np.ndarray
    eigenvectors: np.ndarray
    eigenvalues: np.ndarray
    velocity_basis: np.ndarray


def _complex_to_pair(value: complex) -> list[float]:
    z = complex(value)
    return [float(z.real), float(z.imag)]


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, complex):
        return _complex_to_pair(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _as_path(path_value: str | Path | None, *, name: str) -> Path:
    if path_value is None:
        raise ValueError(f"{name} is required.")
    path = Path(path_value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path.resolve()


def _resolve_hsr_paths(
    hsr_path: str | Path | Sequence[str | Path] | Mapping[str, str | Path] | None,
    *,
    hr_path: str | Path | None,
    sr_path: str | Path | None,
    rR_path: str | Path | None,
) -> tuple[Path, Path, Path]:
    if hsr_path is not None:
        if isinstance(hsr_path, Mapping):
            hr_path = hr_path or hsr_path.get("HR") or hsr_path.get("hr") or hsr_path.get("H")
            sr_path = sr_path or hsr_path.get("SR") or hsr_path.get("sr") or hsr_path.get("S")
            rR_path = (
                rR_path
                or hsr_path.get("rR")
                or hsr_path.get("RR")
                or hsr_path.get("rr")
                or hsr_path.get("r")
            )
        elif isinstance(hsr_path, (str, Path)):
            base = Path(hsr_path).expanduser()
            if base.is_dir():
                hr_path = hr_path or base / "data-HR-sparse_SPIN0.csr"
                sr_path = sr_path or base / "data-SR-sparse_SPIN0.csr"
                rR_path = rR_path or base / "data-rR-sparse.csr"
            elif hr_path is None:
                hr_path = base
        else:
            values = list(hsr_path)
            if len(values) != 3:
                raise ValueError("hsr_path sequence must contain exactly HR, SR, and rR paths.")
            hr_path = hr_path or values[0]
            sr_path = sr_path or values[1]
            rR_path = rR_path or values[2]

    return (
        _as_path(hr_path, name="hr_path"),
        _as_path(sr_path, name="sr_path"),
        _as_path(rR_path, name="rR_path"),
    )


def _wrap_k(k_direct: Sequence[float] | np.ndarray) -> np.ndarray:
    wrapped = np.mod(np.asarray(k_direct, dtype=float).reshape(3), 1.0)
    wrapped[np.abs(wrapped - 1.0) < 1.0e-12] = 0.0
    return wrapped


def _k_key(k_direct: Sequence[float] | np.ndarray, decimals: int = 10) -> tuple[float, float, float]:
    return tuple(float(value) for value in np.round(_wrap_k(k_direct), int(decimals)).tolist())


def _k_equivalent(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
    tol: float,
) -> bool:
    diff = np.asarray(left, dtype=float).reshape(3) - np.asarray(right, dtype=float).reshape(3)
    diff = diff - np.rint(diff)
    return bool(np.max(np.abs(diff)) <= float(tol))


def _mapped_kpoint(
    k_direct: Sequence[float] | np.ndarray,
    rotation: np.ndarray,
    *,
    time_reversal: bool,
) -> np.ndarray:
    k = np.asarray(k_direct, dtype=float).reshape(3)
    inv_rotation = np.linalg.inv(np.asarray(rotation, dtype=float))
    mapped = k @ inv_rotation
    if bool(time_reversal):
        mapped = -mapped
    return np.asarray(mapped, dtype=float)


def _matrix_error(
    reference: np.ndarray,
    predicted: np.ndarray,
    *,
    direction_names: Sequence[str] = ("x", "y", "z"),
) -> VelocityMatrixError:
    ref = np.asarray(reference, dtype=complex)
    pred = np.asarray(predicted, dtype=complex)
    if ref.shape != pred.shape:
        raise ValueError(f"Cannot compare matrices with different shapes: {ref.shape} vs {pred.shape}.")
    diff = pred - ref
    abs_diff = np.abs(diff)
    max_abs = float(np.max(abs_diff)) if abs_diff.size else 0.0
    mean_abs = float(np.mean(abs_diff)) if abs_diff.size else 0.0
    rms_abs = float(np.sqrt(np.mean(abs_diff**2))) if abs_diff.size else 0.0
    ref_norm = float(np.linalg.norm(ref.reshape(-1)))
    rel_fro = float(np.linalg.norm(diff.reshape(-1)) / ref_norm) if ref_norm > 0.0 else 0.0

    if abs_diff.size:
        flat_index = int(np.argmax(abs_diff.reshape(-1)))
        direction_index, row, col = np.unravel_index(flat_index, abs_diff.shape)
    else:
        direction_index = row = col = 0
    direction_errors = []
    for idx, direction_name in enumerate(direction_names):
        if idx >= diff.shape[0]:
            break
        direction_diff = diff[idx]
        direction_ref = ref[idx]
        direction_abs = np.abs(direction_diff)
        direction_ref_norm = float(np.linalg.norm(direction_ref.reshape(-1)))
        direction_errors.append(
            {
                "direction": str(direction_name),
                "max_abs": float(np.max(direction_abs)) if direction_abs.size else 0.0,
                "mean_abs": float(np.mean(direction_abs)) if direction_abs.size else 0.0,
                "rms_abs": float(np.sqrt(np.mean(direction_abs**2))) if direction_abs.size else 0.0,
                "rel_fro": (
                    float(np.linalg.norm(direction_diff.reshape(-1)) / direction_ref_norm)
                    if direction_ref_norm > 0.0
                    else 0.0
                ),
                "element_count": int(direction_diff.size),
            }
        )
    name = str(direction_names[int(direction_index)]) if int(direction_index) < len(direction_names) else str(direction_index)
    return VelocityMatrixError(
        max_abs=max_abs,
        mean_abs=mean_abs,
        rms_abs=rms_abs,
        rel_fro=rel_fro,
        element_count=int(diff.size),
        worst_direction=name,
        worst_row=int(row),
        worst_col=int(col),
        worst_reference=complex(ref[direction_index, row, col]) if ref.size else 0.0 + 0.0j,
        worst_predicted=complex(pred[direction_index, row, col]) if pred.size else 0.0 + 0.0j,
        worst_difference=complex(diff[direction_index, row, col]) if diff.size else 0.0 + 0.0j,
        direction_errors=direction_errors,
    )


def _time_reversal_basis_matrix(metadata) -> np.ndarray:
    if int(metadata.spin_factor) == 1:
        return np.eye(int(metadata.basis_num), dtype=complex)
    if int(metadata.spin_factor) != 2:
        raise ValueError(f"Unsupported spin_factor for time reversal: {metadata.spin_factor}.")
    spin_time_reversal = np.array(
        [
            [0.0, -1.0j],
            [1.0j, 0.0],
        ],
        dtype=complex,
    )
    return np.kron(np.eye(int(metadata.spinless_basis_num), dtype=complex), spin_time_reversal)


class VelocityCovarianceTester:
    """
    Validate basis-representation velocity matrices under magnetic symmetry operations.

    The convention follows ``build_dk_matrix``: a unitary operation maps a source
    k point to ``k @ inv(R)``.  A time-reversal operation maps it to
    ``-k @ inv(R)`` and is represented with the target-k phase ``D(g, k') U_T K``
    in the basis space.
    Velocity is treated as a time-reversal-odd Cartesian vector operator.
    """

    def __init__(
        self,
        structure_path: str | Path,
        hsr_path: str | Path | Sequence[str | Path] | Mapping[str, str | Path] | None = None,
        *,
        hr_path: str | Path | None = None,
        sr_path: str | Path | None = None,
        rR_path: str | Path | None = None,
        nspin: int = 4,
        soc: bool | None = None,
        kpoints: Sequence[Sequence[float]] | np.ndarray | None = None,
        hr_unit: str = "Ry",
        rR_unit: str = "Bohr",
        is_sparse: bool = False,
        max_kpoint_num: int = 8000,
        pseudo_dir: str | Path = "./",
        orbital_dir: str | Path = "./",
        symprec: float = 1.0e-5,
        mag_symprec: float | None = None,
        k_tol: float = 1.0e-6,
        map_tol: float = 1.0e-6,
        magnetic_moments: Sequence[float] | np.ndarray | None = None,
        with_time_reversal: bool = True,
        is_axial: bool | None = None,
    ) -> None:
        self.structure_path = _as_path(structure_path, name="structure_path")
        self.hr_path, self.sr_path, self.rR_path = _resolve_hsr_paths(
            hsr_path,
            hr_path=hr_path,
            sr_path=sr_path,
            rR_path=rR_path,
        )
        self.nspin = int(nspin)
        if self.nspin not in (1, 4):
            raise ValueError("VelocityCovarianceTester currently supports nspin=1 or nspin=4.")
        self.soc = bool(self.nspin == 4 if soc is None else soc)
        self.kpoints = self._normalize_kpoints(kpoints)
        self.hr_unit = str(hr_unit)
        self.rR_unit = str(rR_unit)
        self.is_sparse = bool(is_sparse)
        self.max_kpoint_num = int(max_kpoint_num)
        self.pseudo_dir = pseudo_dir
        self.orbital_dir = orbital_dir
        self.symprec = float(symprec)
        self.mag_symprec = self.symprec if mag_symprec is None else float(mag_symprec)
        self.k_tol = float(k_tol)
        self.map_tol = float(map_tol)
        self.magnetic_moments_input = None if magnetic_moments is None else np.asarray(magnetic_moments, dtype=float)
        self.with_time_reversal = bool(with_time_reversal)
        self.is_axial = bool(self.soc) if is_axial is None else bool(is_axial)

        self.lattice_vector = np.asarray(read_stru_lattice_vector(self.structure_path), dtype=float)
        self.tb = self._build_tb_model()
        self.metadata = extract_abacus_basis_metadata(self.tb)
        self.operations = self._load_magnetic_operations()
        self._k_cache: dict[tuple[float, float, float], KPointData] = {}

    @staticmethod
    def _normalize_kpoints(kpoints: Sequence[Sequence[float]] | np.ndarray | None) -> np.ndarray:
        if kpoints is None:
            return np.zeros((1, 3), dtype=float)
        values = np.asarray(kpoints, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, 3)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError(f"kpoints must have shape (n, 3), got {values.shape}.")
        return values

    def _build_tb_model(self):
        model = TBModel(
            self.nspin,
            1.0,
            np.asarray(self.lattice_vector, dtype=float),
            max_kpoint_num=max(int(self.max_kpoint_num), int(len(self.kpoints)) * 2),
        )
        hr = abacus_readHR(self.nspin, str(self.hr_path), self.hr_unit)
        sr = abacus_readSR(self.nspin, str(self.sr_path))
        model.set_solver_HSR(hr, sr, self.is_sparse)
        rr = abacus_readrR(str(self.rR_path), self.rR_unit)
        model.set_solver_rR(rr[0], rr[1], rr[2], self.is_sparse)
        model.read_stru(
            str(self.structure_path),
            need_orb=True,
            pseudo_dir=self.pseudo_dir,
            orbital_dir=self.orbital_dir,
        )
        return model

    def _magnetic_moments(self) -> np.ndarray:
        atom_count = int(len(self.metadata.species_by_atom))
        if self.magnetic_moments_input is not None:
            moments = np.asarray(self.magnetic_moments_input, dtype=float)
            if moments.shape[0] != atom_count:
                raise ValueError(
                    f"magnetic_moments first dimension must match atom count {atom_count}, got {moments.shape}."
                )
            return moments
        if self.soc:
            return np.zeros((atom_count, 3), dtype=float)
        return np.zeros(atom_count, dtype=float)

    def _load_magnetic_operations(self) -> list[dict[str, Any]]:
        numbers = np.asarray([atomic_numbers[str(symbol)] for symbol in self.metadata.species_by_atom], dtype=int)
        cell = (
            np.asarray(self.metadata.lattice_vector, dtype=float),
            np.asarray(self.metadata.positions_frac, dtype=float),
            numbers,
            self._magnetic_moments(),
        )
        dataset = spglib.get_magnetic_symmetry(
            cell,
            symprec=self.symprec,
            mag_symprec=self.mag_symprec,
            is_axial=self.is_axial,
            with_time_reversal=self.with_time_reversal,
        )
        if dataset is None:
            raise RuntimeError("spglib.get_magnetic_symmetry failed.")

        rotations = np.asarray(dataset["rotations"], dtype=int)
        translations = np.asarray(dataset["translations"], dtype=float)
        time_reversals = np.asarray(dataset.get("time_reversals", np.zeros(len(rotations), dtype=bool)), dtype=bool)

        operations: list[dict[str, Any]] = []
        for index, (rotation, translation, time_reversal) in enumerate(
            zip(rotations, translations, time_reversals, strict=True),
            start=1,
        ):
            cart_rotation = _cart_rotation_from_fractional(rotation, self.metadata.lattice_vector)
            operations.append(
                {
                    "index": int(index),
                    "rotation": np.asarray(rotation, dtype=int),
                    "translation": np.asarray(translation, dtype=float),
                    "cart_rotation": np.asarray(cart_rotation, dtype=float),
                    "time_reversal": bool(time_reversal),
                    "anti_linear": bool(time_reversal),
                    "kind": "antiunitary" if bool(time_reversal) else "unitary",
                }
            )
        return operations

    def _kpoint_data(self, k_direct: Sequence[float] | np.ndarray) -> KPointData:
        key = _k_key(k_direct)
        cached = self._k_cache.get(key)
        if cached is not None:
            return cached

        k_array = np.asarray(k_direct, dtype=float).reshape(1, 3)
        eigenvectors, eigenvalues = self.tb.tb_solver.diago_H(k_array)
        velocity_basis = self.tb.tb_solver.get_velocity_basis_k(k_array)
        data = KPointData(
            k_direct=np.asarray(k_array[0], dtype=float),
            eigenvectors=np.asarray(eigenvectors[0], dtype=complex),
            eigenvalues=np.asarray(eigenvalues[0], dtype=float),
            velocity_basis=np.asarray(velocity_basis[0], dtype=complex),
        )
        self._k_cache[key] = data
        return data

    def _operator_matrix(self, operation: Mapping[str, Any], representation_k: np.ndarray) -> np.ndarray:
        dk = build_dk_matrix(
            self.tb,
            np.asarray(representation_k, dtype=float),
            operation,
            map_tol=self.map_tol,
        )
        if not bool(operation.get("time_reversal", False)):
            return np.asarray(dk, dtype=complex)
        return np.asarray(dk, dtype=complex) @ _time_reversal_basis_matrix(self.metadata)

    def _predicted_velocity(
        self,
        source_velocity: np.ndarray,
        operation: Mapping[str, Any],
        target_k: np.ndarray,
    ) -> np.ndarray:
        velocity = np.asarray(source_velocity, dtype=complex)
        cart_rotation = np.asarray(operation["cart_rotation"], dtype=float)
        mixed = np.zeros_like(velocity, dtype=complex)
        for alpha in range(3):
            for beta in range(3):
                coeff = float(cart_rotation[alpha, beta])
                if abs(coeff) > 1.0e-14:
                    mixed[alpha] += coeff * velocity[beta]

        op_matrix = self._operator_matrix(operation, target_k)
        if bool(operation.get("time_reversal", False)):
            mixed = -np.conj(mixed)
        return np.asarray([op_matrix @ mixed[alpha] @ op_matrix.conj().T for alpha in range(3)], dtype=complex)

    def _operation_record(self, k_index: int, source_k: np.ndarray, operation: Mapping[str, Any]) -> dict[str, Any]:
        target_k = _mapped_kpoint(
            source_k,
            np.asarray(operation["rotation"], dtype=int),
            time_reversal=bool(operation.get("time_reversal", False)),
        )
        source = self._kpoint_data(source_k)
        target = self._kpoint_data(target_k)
        predicted = self._predicted_velocity(source.velocity_basis, operation, target.k_direct)
        error = _matrix_error(target.velocity_basis, predicted)
        relation = "little_group" if _k_equivalent(target_k, source_k, self.k_tol) else "k_star"
        return {
            "k_index": int(k_index),
            "source_k_direct": np.asarray(source_k, dtype=float).tolist(),
            "target_k_direct": np.asarray(target_k, dtype=float).tolist(),
            "target_k_wrapped": _wrap_k(target_k).tolist(),
            "relation": relation,
            "operation_index": int(operation["index"]),
            "operation_kind": str(operation["kind"]),
            "time_reversal": bool(operation.get("time_reversal", False)),
            "rotation": np.asarray(operation["rotation"], dtype=int).tolist(),
            "translation": np.asarray(operation["translation"], dtype=float).tolist(),
            "cart_rotation": np.asarray(operation["cart_rotation"], dtype=float).tolist(),
            "energy_max_abs_difference": float(np.max(np.abs(np.sort(source.eigenvalues) - np.sort(target.eigenvalues)))),
            "velocity_error": error.to_dict(),
        }

    @staticmethod
    def _summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not records:
            return {
                "count": 0,
                "global_max_abs": 0.0,
                "mean_abs_over_records": 0.0,
                "rms_abs_over_records": 0.0,
                "max_rel_fro_over_records": 0.0,
                "worst_record": None,
            }
        max_values = [float(record["velocity_error"]["max_abs"]) for record in records]
        mean_values = [float(record["velocity_error"]["mean_abs"]) for record in records]
        rms_values = [float(record["velocity_error"]["rms_abs"]) for record in records]
        rel_values = [float(record["velocity_error"]["rel_fro"]) for record in records]
        worst = max(records, key=lambda record: float(record["velocity_error"]["max_abs"]))
        return {
            "count": int(len(records)),
            "global_max_abs": float(max(max_values)),
            "mean_abs_over_records": float(np.mean(mean_values)),
            "rms_abs_over_records": float(np.sqrt(np.mean(np.asarray(rms_values, dtype=float) ** 2))),
            "max_rel_fro_over_records": float(max(rel_values)),
            "worst_record": {
                "k_index": int(worst["k_index"]),
                "operation_index": int(worst["operation_index"]),
                "operation_kind": str(worst["operation_kind"]),
                "relation": str(worst["relation"]),
                "source_k_direct": worst["source_k_direct"],
                "target_k_direct": worst["target_k_direct"],
                "velocity_error": worst["velocity_error"],
            },
        }

    def k_stars(self) -> list[dict[str, Any]]:
        stars = []
        for k_index, source_k in enumerate(self.kpoints, start=1):
            targets: dict[tuple[float, float, float], dict[str, Any]] = {}
            for operation in self.operations:
                target_k = _mapped_kpoint(
                    source_k,
                    np.asarray(operation["rotation"], dtype=int),
                    time_reversal=bool(operation.get("time_reversal", False)),
                )
                key = _k_key(target_k)
                row = targets.setdefault(
                    key,
                    {
                        "target_k_wrapped": _wrap_k(target_k).tolist(),
                        "operation_indices": [],
                        "antiunitary_operation_indices": [],
                    },
                )
                row["operation_indices"].append(int(operation["index"]))
                if bool(operation.get("time_reversal", False)):
                    row["antiunitary_operation_indices"].append(int(operation["index"]))
            stars.append(
                {
                    "k_index": int(k_index),
                    "source_k_direct": np.asarray(source_k, dtype=float).tolist(),
                    "star_size": int(len(targets)),
                    "star": list(targets.values()),
                }
            )
        return stars

    def run(self) -> dict[str, Any]:
        records = [
            self._operation_record(k_index, np.asarray(source_k, dtype=float), operation)
            for k_index, source_k in enumerate(self.kpoints, start=1)
            for operation in self.operations
        ]
        little_group_records = [record for record in records if record["relation"] == "little_group"]
        k_star_records = [record for record in records if record["relation"] == "k_star"]
        unitary_records = [record for record in records if not bool(record["time_reversal"])]
        antiunitary_records = [record for record in records if bool(record["time_reversal"])]
        return {
            "convention": {
                "k_mapping_unitary": "k' = k @ inv(R)",
                "k_mapping_antiunitary": "k' = -k @ inv(R)",
                "unitary_velocity": "V_alpha(k') = D(k',g) [sum_beta O_alpha_beta V_beta(k)] D(k',g)^dag",
                "antiunitary_velocity": (
                    "V_alpha(k') = D(k',g) U_T [-conj(sum_beta O_alpha_beta V_beta(k))] "
                    "U_T^dag D(k',g)^dag"
                ),
                "matrix_space": "basis representation returned by tb_solver.get_velocity_basis_k",
            },
            "inputs": {
                "structure_path": str(self.structure_path),
                "hr_path": str(self.hr_path),
                "sr_path": str(self.sr_path),
                "rR_path": str(self.rR_path),
                "nspin": int(self.nspin),
                "soc": bool(self.soc),
                "hr_unit": str(self.hr_unit),
                "rR_unit": str(self.rR_unit),
                "symprec": float(self.symprec),
                "mag_symprec": float(self.mag_symprec),
                "k_tol": float(self.k_tol),
                "map_tol": float(self.map_tol),
                "kpoints": np.asarray(self.kpoints, dtype=float).tolist(),
            },
            "operation_count": int(len(self.operations)),
            "unitary_operation_count": int(sum(1 for operation in self.operations if not operation["time_reversal"])),
            "antiunitary_operation_count": int(sum(1 for operation in self.operations if operation["time_reversal"])),
            "k_stars": self.k_stars(),
            "summary": {
                "all": self._summary(records),
                "little_group": self._summary(little_group_records),
                "k_star": self._summary(k_star_records),
                "unitary": self._summary(unitary_records),
                "antiunitary": self._summary(antiunitary_records),
            },
            "records": records,
        }

    def write_report(self, output_path: str | Path) -> dict[str, Any]:
        report = self.run()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        return report


def run_velocity_covariance_test(
    structure_path: str | Path,
    hsr_path: str | Path | Sequence[str | Path] | Mapping[str, str | Path] | None = None,
    **kwargs,
) -> dict[str, Any]:
    return VelocityCovarianceTester(structure_path, hsr_path, **kwargs).run()


def _read_json_array(path: str | Path | None) -> Any:
    if path is None:
        return None
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate pyatb velocity matrices against magnetic symmetry covariance.",
    )
    parser.add_argument("--structure", required=True, help="ABACUS STRU path.")
    parser.add_argument(
        "--hsr-path",
        help=(
            "Directory containing data-HR-sparse_SPIN0.csr, data-SR-sparse_SPIN0.csr, "
            "and data-rR-sparse.csr. Explicit --hr-path/--sr-path/--rR-path override this."
        ),
    )
    parser.add_argument("--hr-path", help="Explicit HR csr path.")
    parser.add_argument("--sr-path", help="Explicit SR csr path.")
    parser.add_argument("--rR-path", help="Explicit rR csr path.")
    parser.add_argument(
        "--kpoint",
        action="append",
        nargs=3,
        type=float,
        metavar=("KX", "KY", "KZ"),
        help="Direct-coordinate source k point. May be passed multiple times. Default: 0 0 0.",
    )
    parser.add_argument("--output", default="velocity_covariance_report.json", help="Output JSON report path.")
    parser.add_argument("--nspin", type=int, default=4, choices=(1, 4), help="pyatb spin channel setting.")
    parser.add_argument("--soc", action=argparse.BooleanOptionalAction, default=None, help="Override SOC mode.")
    parser.add_argument("--hr-unit", default="Ry", help="HR energy unit passed to abacus_readHR.")
    parser.add_argument("--rR-unit", default="Bohr", help="rR length unit passed to abacus_readrR.")
    parser.add_argument("--is-sparse", action="store_true", help="Use sparse H/S/rR solver data.")
    parser.add_argument("--max-kpoint-num", type=int, default=8000, help="TB model k-point allocation.")
    parser.add_argument("--pseudo-dir", default="./", help="Pseudopotential directory used by TBModel.read_stru.")
    parser.add_argument("--orbital-dir", default="./", help="Orbital directory used by TBModel.read_stru.")
    parser.add_argument("--symprec", type=float, default=1.0e-5, help="spglib symmetry tolerance.")
    parser.add_argument("--mag-symprec", type=float, default=None, help="spglib magnetic moment tolerance.")
    parser.add_argument("--k-tol", type=float, default=1.0e-6, help="Tolerance for little-group k equivalence.")
    parser.add_argument("--map-tol", type=float, default=1.0e-6, help="Tolerance passed to build_dk_matrix.")
    parser.add_argument(
        "--magnetic-moments-json",
        help="Optional JSON file containing magnetic moments, shaped as spglib expects.",
    )
    parser.add_argument(
        "--no-time-reversal",
        action="store_true",
        help="Disable time-reversal operations in spglib.get_magnetic_symmetry.",
    )
    parser.add_argument(
        "--is-axial",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override spglib magnetic moment axial-vector mode.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    kpoints = args.kpoint if args.kpoint is not None else [[0.0, 0.0, 0.0]]
    tester = VelocityCovarianceTester(
        args.structure,
        args.hsr_path,
        hr_path=args.hr_path,
        sr_path=args.sr_path,
        rR_path=args.rR_path,
        nspin=args.nspin,
        soc=args.soc,
        kpoints=kpoints,
        hr_unit=args.hr_unit,
        rR_unit=args.rR_unit,
        is_sparse=args.is_sparse,
        max_kpoint_num=args.max_kpoint_num,
        pseudo_dir=args.pseudo_dir,
        orbital_dir=args.orbital_dir,
        symprec=args.symprec,
        mag_symprec=args.mag_symprec,
        k_tol=args.k_tol,
        map_tol=args.map_tol,
        magnetic_moments=_read_json_array(args.magnetic_moments_json),
        with_time_reversal=not args.no_time_reversal,
        is_axial=args.is_axial,
    )
    report = tester.write_report(args.output)
    summary = report["summary"]["all"]
    print(f"Wrote velocity covariance report: {args.output}")
    print(
        "All operations: "
        f"count={summary['count']} "
        f"max_abs={summary['global_max_abs']:.12e} "
        f"mean_abs={summary['mean_abs_over_records']:.12e} "
        f"rms_abs={summary['rms_abs_over_records']:.12e} "
        f"max_rel_fro={summary['max_rel_fro_over_records']:.12e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
