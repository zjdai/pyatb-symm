from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np
from scipy.io import FortranFile


EPSIL = 1.0e-7
SU = 0.123
S1_U = 0.877
S_2U = -0.246
SV = 0.313
S1_V = 0.687
SW = 0.427
S_U = -0.123
S1U = 1.123


def _frac_diff(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d -= np.rint(d)
    return float(np.sum(np.abs(d)))


def _centering_kc2p(symbol: str) -> np.ndarray:
    s = symbol.strip().upper()[:1]
    if s == "P":
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
    if s == "C":
        return np.array(
            [
                [0.5, 0.5, 0.0],
                [-0.5, 0.5, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
    if s == "B":
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.0, -0.5, 0.5],
            ],
            dtype=float,
        )
    if s == "A":
        return np.array(
            [
                [0.5, 0.0, 0.5],
                [0.0, 1.0, 0.0],
                [-0.5, 0.0, 0.5],
            ],
            dtype=float,
        )
    if s == "R":
        return np.array(
            [
                [2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0],
                [1.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0],
                [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            ],
            dtype=float,
        )
    if s == "F":
        return np.array(
            [
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ],
            dtype=float,
        )
    if s == "I":
        return np.array(
            [
                [-0.5, 0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, -0.5],
            ],
            dtype=float,
        )
    raise ValueError(f"Unsupported centering symbol: {symbol}")


@dataclass
class IrrepEntry:
    raw_name: str
    name: str
    reality: int
    dimension: int
    characters: np.ndarray  # shape (doubnum,), complex
    active_ops: np.ndarray  # shape (doubnum,), bool
    phase_kinds: np.ndarray  # shape (doubnum,), int
    coeff_uvw: np.ndarray  # shape (doubnum, 3), float
    factor_strings: list[str]


@dataclass
class KPointEntry:
    name: str
    k_conv: np.ndarray
    k_prim: np.ndarray
    antisym: int
    irreps: list[IrrepEntry]

    @property
    def little_group_ops(self) -> np.ndarray:
        if not self.irreps:
            return np.zeros((0,), dtype=int)
        return np.where(self.irreps[0].active_ops)[0]


@dataclass
class SymmetryElement:
    rotation: np.ndarray
    translation: np.ndarray
    spin: np.ndarray


@dataclass
class KPointResolution:
    entry: KPointEntry
    entry_index: int
    k_conv: np.ndarray
    rotation_index: int
    variable_count: int
    mapped_k_prim: np.ndarray | None = None
    rotated_k_prim: np.ndarray | None = None
    cornwell_satisfied: bool = True


def _count_variables(kstr: str) -> int:
    return int("  u  " in kstr) + int("  v  " in kstr) + int("  w  " in kstr)


def _token_to_value(token: str) -> float:
    token = token.rjust(5)
    if token == "  u  ":
        return SU
    if token == "  v  ":
        return SV
    if token == "  w  ":
        return SW
    if token == " 1-u ":
        return S1_U
    if token == "-2u  ":
        return S_2U
    if token == " -u  ":
        return S_U
    if token == " 1+u ":
        return S1U
    if token == " 1-v ":
        return S1_V
    value = float(token)
    if abs(value - 0.33) < 1.0e-8:
        return 1.0 / 3.0
    return value


def _format_fixed_token(value: float) -> str:
    if abs(value - SU) < EPSIL:
        return "  u  "
    if abs(value - SV) < EPSIL:
        return "  v  "
    if abs(value - SW) < EPSIL:
        return "  w  "
    if abs(value - S1_U) < EPSIL:
        return " 1-u "
    if abs(value - S_2U) < EPSIL:
        return "-2u  "
    if abs(value - S_U) < EPSIL:
        return " -u  "
    if abs(value - S1U) < EPSIL:
        return " 1+u "
    if abs(value - S1_V) < EPSIL:
        return " 1-v "
    return f"{value:5.2f}"


def kreal_to_string(coorkp: np.ndarray) -> str:
    coorkp = np.asarray(coorkp, dtype=float)
    return "".join(_format_fixed_token(float(coorkp[i])) for i in range(3))


class KLittleGroupsDB:
    def __init__(self, path: Path, spacegroup_symbol: str, doubnum: int, symops: list[SymmetryElement], kpoints: list[KPointEntry]):
        self.path = Path(path)
        self.spacegroup_symbol = spacegroup_symbol
        self.doubnum = int(doubnum)
        self.symops = symops
        self.kpoints = kpoints
        self.kc2p = _centering_kc2p(spacegroup_symbol)
        self.p2c = np.linalg.inv(self.kc2p)

    @classmethod
    def load(cls, path: str | Path) -> "KLittleGroupsDB":
        p = Path(path)
        f = FortranFile(str(p), "r")

        head = bytes(f.read_record(np.uint8))
        doubnum = struct.unpack("<i", head[:4])[0]
        sg_symbol = head[4:14].decode("ascii").strip()

        symops: list[SymmetryElement] = []
        for _ in range(doubnum):
            rec = bytes(f.read_record(np.uint8))
            ints = np.frombuffer(rec[:36], dtype=np.int32).reshape(3, 3)
            vals = np.frombuffer(rec[36:], dtype=np.float64)
            tau = vals[:3]
            df = vals[3:].reshape(2, 4)
            spin = np.array(
                [
                    [df[0, 0] * np.exp(1j * np.pi * df[1, 0]), df[0, 1] * np.exp(1j * np.pi * df[1, 1])],
                    [df[0, 2] * np.exp(1j * np.pi * df[1, 2]), df[0, 3] * np.exp(1j * np.pi * df[1, 3])],
                ],
                dtype=complex,
            )
            symops.append(SymmetryElement(rotation=ints, translation=tau, spin=spin))

        numk, tnir = f.read_record(np.int32)

        kpoint_map: dict[tuple[str, tuple[float, float, float], int], KPointEntry] = {}
        kc2p = _centering_kc2p(sg_symbol)

        for _ in range(int(tnir)):
            line = bytes(f.read_record(np.uint8)).decode("ascii", errors="replace").strip()
            toks = line.split()
            if len(toks) < 9:
                raise ValueError(f"Unexpected kLittleGroups record format: '{line}'")

            k_conv = np.array([float(toks[0]), float(toks[1]), float(toks[2])], dtype=float)
            antisym = int(toks[3])
            ir_name = toks[4]
            ir_dim = int(toks[5])
            _nele = int(toks[6])
            k_name = toks[7]
            reality = int(toks[8])

            active = np.zeros((doubnum,), dtype=bool)
            chars = np.zeros((doubnum,), dtype=complex)
            phase_kinds = np.zeros((doubnum,), dtype=int)
            coeff_uvw = np.zeros((doubnum, 3), dtype=float)
            factor_strings = ["            "] * doubnum

            for j in range(doubnum):
                pair = f.read_record(np.int32)
                flag = int(pair[-1])
                if flag == 1:
                    active[j] = True
                    phase_tag = int(f.read_record(np.int32)[-1])
                    vals = f.read_record(np.float64)
                    phase_kinds[j] = phase_tag
                    amp = float(vals[0])
                    phase_pi = float(vals[1])
                    chars[j] = amp * np.exp(1j * np.pi * phase_pi)
                    if phase_tag not in (1, 2):
                        raise ValueError(f"Unexpected phase tag in kLittleGroups file: {phase_tag}")
                    if phase_tag == 2:
                        coeff_uvw[j, :] = np.asarray(vals[2:5], dtype=float)
                        factor_strings[j] = f"{vals[2]:4.1f}{vals[3]:4.1f}{vals[4]:4.1f}"
                elif flag == 0:
                    continue
                else:
                    raise ValueError(f"Unexpected active flag in kLittleGroups file: {flag}")

            key = (k_name, tuple(np.round(k_conv, 8)), antisym)
            if key not in kpoint_map:
                kpoint_map[key] = KPointEntry(
                    name=k_name,
                    k_conv=k_conv,
                    k_prim=np.asarray(k_conv @ kc2p, dtype=float),
                    antisym=antisym,
                    irreps=[],
                )
            kpoint_map[key].irreps.append(
                IrrepEntry(
                    raw_name=ir_name,
                    name=ir_name.lstrip("-"),
                    reality=reality,
                    dimension=ir_dim,
                    characters=chars,
                    active_ops=active,
                    phase_kinds=phase_kinds,
                    coeff_uvw=coeff_uvw,
                    factor_strings=factor_strings,
                )
            )

        kpoints = list(kpoint_map.values())
        if len(kpoints) != int(numk):
            # Keep running but preserve explicit warning signal for caller.
            pass

        return cls(path=p, spacegroup_symbol=sg_symbol, doubnum=doubnum, symops=symops, kpoints=kpoints)

    @staticmethod
    def _shift_negative_to_unit_interval(k_prim: np.ndarray) -> np.ndarray:
        k = np.asarray(k_prim, dtype=float).copy()
        for i in range(3):
            if k[i] < 0.0 and abs(k[i]) > 1.0e-6:
                k[i] += 1.0
        return k

    def _reference_kpoint_matches(
        self,
        k_prim: np.ndarray,
        tol: float = 1.0e-5,
    ) -> list[tuple[KPointEntry, int, np.ndarray, int, float]]:
        tkk = np.asarray(k_prim, dtype=float)
        tkkc = tkk @ self.p2c

        matches: list[tuple[KPointEntry, int, np.ndarray, int, float]] = []

        for entry_index, entry in enumerate(self.kpoints, start=1):
            ckpoint = kreal_to_string(entry.k_conv)
            refkpoint = np.full(3, 9999.0, dtype=float)
            is_variable = [False, False, False]

            token1 = ckpoint[0:5]
            token2 = ckpoint[5:10]
            token3 = ckpoint[10:15]

            if token1 == "  u  ":
                is_variable[0] = True
                refkpoint[0] = tkkc[0]
                if token2 == "  u  ":
                    refkpoint[1] = refkpoint[0]
                if token2 == " 1-u ":
                    refkpoint[1] = 1.0 - refkpoint[0]
                if token2 == "-2u  ":
                    refkpoint[1] = -2.0 * refkpoint[0]
                if token2 == " -u  ":
                    refkpoint[1] = -refkpoint[0]
                if token2 == " 1+u ":
                    refkpoint[1] = 1.0 + refkpoint[0]
                if token3 == "  u  ":
                    refkpoint[2] = refkpoint[0]
                if token3 == " 1-u ":
                    refkpoint[2] = 1.0 - refkpoint[0]
                if token3 == "-2u  ":
                    refkpoint[2] = -2.0 * refkpoint[0]
                if token3 == " -u  ":
                    refkpoint[2] = -refkpoint[0]
                if token3 == " 1+u ":
                    refkpoint[2] = 1.0 + refkpoint[0]

            if token2 == "  v  ":
                is_variable[1] = True
                refkpoint[1] = tkkc[1]
                if token1 == " 1-v ":
                    refkpoint[0] = 1.0 - refkpoint[1]
                if token3 == " 1-v ":
                    refkpoint[2] = 1.0 - refkpoint[1]
                if token3 == "  v  ":
                    refkpoint[2] = refkpoint[1]

            if token3 == "  w  ":
                is_variable[2] = True
                refkpoint[2] = tkkc[2]

            # IRVSP exceptions around off-diagonal u templates.
            if token1 != "  u  " and token2 == "  u  ":
                is_variable[0] = True
                refkpoint[1] = tkkc[1]
            if token1 == " 1+u " and token2 == " 1-u ":
                is_variable[0] = True
                refkpoint[0] = tkkc[0]
                refkpoint[1] = 2.0 - refkpoint[0]

            for j, token in enumerate((token1, token2, token3)):
                if abs(refkpoint[j] - 9999.0) < tol:
                    refkpoint[j] = _token_to_value(token)

            refkpointp = refkpoint @ self.kc2p
            diff = np.abs((refkpointp - np.floor(refkpointp)) - (tkk - np.floor(tkk))).sum()
            if diff < tol:
                varnum = _count_variables(ckpoint)
                matches.append((entry, varnum, tkkc.copy(), entry_index, _frac_diff(tkk, entry.k_prim)))

        matches.sort(key=lambda item: (item[1], item[4], item[3]))
        return matches

    def _match_reference_kpoint(self, k_prim: np.ndarray, tol: float = 1.0e-5) -> tuple[KPointEntry | None, int, np.ndarray, int]:
        matches = self._reference_kpoint_matches(k_prim, tol=tol)
        if not matches:
            return None, 9999, np.asarray(k_prim, dtype=float) @ self.p2c, -1
        entry, varnum, k_conv, entry_index, _ = matches[0]
        return entry, varnum, k_conv, entry_index

    def resolve_kpoint_from_star(
        self,
        k_prim: np.ndarray,
        inverse_rotations: list[np.ndarray],
        has_inversion: bool,
        little_group_size: int | None = None,
        detected_ops: list[int] | None = None,
        current_to_db_prim: np.ndarray | None = None,
        tol: float = 1.0e-5,
    ) -> KPointResolution:
        k_prim = np.asarray(k_prim, dtype=float)
        if current_to_db_prim is None:
            mapped_k_prim = np.asarray(k_prim, dtype=float)
        else:
            mapped_k_prim = np.asarray(k_prim, dtype=float) @ np.asarray(current_to_db_prim, dtype=float)
            mapped_k_prim -= np.floor(mapped_k_prim)
        candidate_rotations = [np.asarray(rot, dtype=int) for rot in inverse_rotations]
        if not has_inversion:
            candidate_rotations.append(-np.eye(3, dtype=int))

        best = None
        detected_set = None if detected_ops is None else {int(i) for i in detected_ops}
        for index, inv_rot in enumerate(candidate_rotations, start=1):
            rotated = self._shift_negative_to_unit_interval(mapped_k_prim @ inv_rot)
            entry, varnum, k_conv, entry_index = self._match_reference_kpoint(rotated, tol=tol)
            rotation_index = 0 if (index == len(candidate_rotations) and not has_inversion) else index

            if entry is not None:
                candidate = KPointResolution(
                    entry=entry,
                    entry_index=entry_index,
                    mapped_k_prim=mapped_k_prim.copy(),
                    rotated_k_prim=rotated,
                    k_conv=k_conv,
                    rotation_index=rotation_index,
                    variable_count=varnum,
                )
                entry_lg_size = int(entry.irreps[0].active_ops[: self.doubnum // 2].sum())
                entry_lg_set = {int(i) + 1 for i in entry.little_group_ops if int(i) < self.doubnum // 2}
                score = (
                    len(entry_lg_set.symmetric_difference(detected_set)) if detected_set is not None else 0,
                    abs(entry_lg_size - little_group_size) if little_group_size is not None else 0,
                    0,
                    varnum,
                    _frac_diff(rotated, entry.k_prim),
                )
                if best is None or score < best[0]:
                    best = (score, candidate)

        if best is None:
            raise ValueError(
                "Nonsymmorphic kpoint is NOT found for primitive k="
                f"{k_prim.tolist()} after mapping to database primitive basis "
                f"{mapped_k_prim.tolist()}."
            )
        return best[1]

    def irrep_table_characters(self, resolution: KPointResolution, irrep: IrrepEntry) -> np.ndarray:
        traces = np.zeros((self.doubnum,), dtype=complex)
        raw_characters = np.asarray(irrep.characters, dtype=complex)
        if not bool(getattr(resolution, "cornwell_satisfied", True)):
            raw_characters = np.conj(raw_characters)
        for j in range(self.doubnum):
            if not irrep.active_ops[j]:
                continue
            if irrep.phase_kinds[j] == 1:
                traces[j] = raw_characters[j]
            elif irrep.phase_kinds[j] == 2:
                # Keep phase convention consistent with D(k,g) and irrep matching:
                # exp(-i*pi*uvw·k_conv)
                angle = -np.pi * float(np.dot(irrep.coeff_uvw[j, :], resolution.k_conv))
                traces[j] = raw_characters[j] * np.exp(1j * angle)
            else:
                raise ValueError(f"Unexpected phase kind {irrep.phase_kinds[j]} for irrep {irrep.name}.")
        return traces

    def match_kpoint(self, k_prim: np.ndarray, detected_ops: list[int] | None = None, tol: float = 1e-4) -> KPointEntry | None:
        k = np.asarray(k_prim, dtype=float)
        detected_set = None
        if detected_ops is not None:
            detected_set = {int(i) - 1 for i in detected_ops}

        best = None
        for entry in self.kpoints:
            lg_set = {int(i) for i in entry.little_group_ops if int(i) < self.doubnum // 2}
            op_mismatch = 0
            if detected_set is not None:
                op_mismatch = len(detected_set.symmetric_difference(lg_set))
            diff = _frac_diff(k, entry.k_prim)
            score = (op_mismatch, diff)
            if best is None or score < best[0]:
                best = (score, entry)
        if best is None:
            return None
        if best[0][0] > 0 and best[0][1] > tol:
            return None
        if best[0][0] == 0 and best[0][1] > 0.55:
            return None
        return best[1]
