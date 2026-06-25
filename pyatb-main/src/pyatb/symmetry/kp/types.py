from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


KP_FREE_ELECTRON_COEFFICIENT_EV_A2 = 3.80998212
KP_DIRECTIONS = ("kx", "ky", "kz")
ZEEMAN_FIELD_DIRECTIONS = ("Bx", "By", "Bz")


@dataclass(frozen=True)
class KPointBandSelection:
    """One kp target: a direct-coordinate k point and its band range."""

    kpoint: Sequence[float]
    band: Sequence[int]
    label: str | None = None
    occ_band: int | None = None

    def __post_init__(self) -> None:
        kpoint = tuple(float(value) for value in self.kpoint)
        if len(kpoint) != 3:
            raise ValueError("kp selection kpoint must contain exactly three direct coordinates.")

        band = tuple(int(value) for value in self.band)
        if len(band) != 2 or band[0] <= 0 or band[1] <= 0 or band[0] > band[1]:
            raise ValueError("kp selection band must be two positive integers with band[0] <= band[1].")

        if self.occ_band is not None and int(self.occ_band) <= 0:
            raise ValueError("kp selection occ_band must be a positive integer.")

        object.__setattr__(self, "kpoint", kpoint)
        object.__setattr__(self, "band", band)
        if self.occ_band is not None:
            object.__setattr__(self, "occ_band", int(self.occ_band))

    @property
    def band_start(self) -> int:
        return int(self.band[0])

    @property
    def band_stop(self) -> int:
        return int(self.band[1])


def _coerce_selection(selection: KPointBandSelection | Mapping[str, Any]) -> KPointBandSelection:
    if isinstance(selection, KPointBandSelection):
        return selection
    if isinstance(selection, Mapping):
        try:
            kpoint = selection["kpoint"]
            band = selection["band"]
        except KeyError as exc:
            raise ValueError("kp selection mapping must contain 'kpoint' and 'band'.") from exc
        return KPointBandSelection(
            kpoint=kpoint,
            band=band,
            label=selection.get("label"),
            occ_band=selection.get("occ_band"),
        )
    raise TypeError("kp selection must be a KPointBandSelection or a mapping.")


def _normalize_selections(
    selections: Sequence[KPointBandSelection | Mapping[str, Any]],
) -> list[KPointBandSelection]:
    normalized = [_coerce_selection(selection) for selection in selections]
    if not normalized:
        raise ValueError("kp calculation requires at least one kpoint/band selection.")
    return normalized


def _kp_zeeman_enabled(value: str | bool | int) -> bool:
    if isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1"}:
        return True
    if text in {"no", "n", "false", "0"}:
        return False
    raise ValueError("KP.zeeman_term must be yes or no.")
