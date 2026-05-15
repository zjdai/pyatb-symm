from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _purge_pyatb_modules() -> None:
    for name in list(sys.modules):
        if name == "pyatb" or name.startswith("pyatb."):
            del sys.modules[name]


def _ensure_fake_mpi4py() -> None:
    class _FakeComm:
        def Get_size(self) -> int:
            return 1

        def Get_rank(self) -> int:
            return 0

        def Barrier(self) -> None:
            return None

        def reduce(self, value, root=0, op=None):
            return value

    class _FakeOpFactory:
        @staticmethod
        def Create(func, commute=False):
            return func

    fake_mpi = types.SimpleNamespace(
        COMM_WORLD=_FakeComm(),
        Op=_FakeOpFactory(),
        SUM="sum",
    )
    sys.modules["mpi4py"] = types.SimpleNamespace(MPI=fake_mpi)


def _ensure_fake_interface_python() -> None:
    if "pyatb.interface_python" in sys.modules:
        return

    class _FakeInterfacePython:
        def __init__(self, *args, **kwargs) -> None:
            pass

    fake_module = types.ModuleType("pyatb.interface_python")
    fake_module.interface_python = _FakeInterfacePython
    sys.modules["pyatb.interface_python"] = fake_module


@pytest.fixture
def load_pyatb(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    _purge_pyatb_modules()
    _ensure_fake_mpi4py()
    _ensure_fake_interface_python()

    def _loader(module_name: str):
        return importlib.import_module(module_name)

    return _loader
