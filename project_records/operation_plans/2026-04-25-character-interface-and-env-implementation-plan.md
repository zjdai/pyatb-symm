# CHARACTER Interface And Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first-stage `CHARACTER` input interface and executable skeleton to `pyatb`, then create a dedicated `symm` conda environment with `ase-abacus`, install editable `pyatb`, and build `irvsp` / `ir2tb`.

**Architecture:** Keep the existing `pyatb` input system intact and add `CHARACTER` as a narrow extension. Handle the mixed-type `group` and variable-length `mag` parameters with `CHARACTER`-specific post-processing in `input.py`, dispatch from `main.py`, and isolate future logic in a new `pyatb.symmetry.character.Character` class that validates inputs, creates output/logging, and exits with a clear `NotImplementedError`.

**Tech Stack:** Python, pytest, conda, pybind11/setuptools editable install, mpi4py, oneAPI `ifort`, IRVSP Makefiles.

---

### Task 1: Add a Minimal Python Test Harness For CHARACTER Input Parsing

**Files:**
- Create: `pyatb-main/tests/conftest.py`
- Create: `pyatb-main/tests/test_character_input.py`
- Test: `pyatb-main/tests/test_character_input.py`

- [ ] **Step 1: Write the failing tests for CHARACTER input parsing**

Create `tests/conftest.py` to reload the mutable input globals between tests:

```python
import importlib

import pytest

import pyatb.io.default_input as default_input
import pyatb.io.input as input_mod


@pytest.fixture(autouse=True)
def _reset_input_modules():
    importlib.reload(default_input)
    importlib.reload(input_mod)
    yield
    importlib.reload(default_input)
    importlib.reload(input_mod)
```

```python
from pathlib import Path

import pytest

from pyatb.io.input import read_input


def _write_input(path: Path, character_block: str) -> None:
    path.write_text(
        "\n".join(
            [
                "INPUT_PARAMETERS",
                "{",
                "    nspin 1",
                "    package ABACUS",
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


def test_character_block_accepts_auto_group_and_auto_mag(tmp_path: Path):
    input_file = tmp_path / "Input"
    _write_input(
        input_file,
        "\n".join(
            [
                "    nspin 1",
                "    group auto",
                "    symm_prec 1e-5",
                "    occ_band 8",
                "    band 7 10",
                "    mag_tag 0",
                "    mag auto",
            ]
        ),
    )

    input_data, function_switch, _ = read_input(str(input_file))

    assert function_switch["CHARACTER"] is True
    assert input_data["CHARACTER"]["group"] == "auto"
    assert input_data["CHARACTER"]["mag"] == "auto"


def test_character_block_accepts_integer_group_and_manual_mag(tmp_path: Path):
    input_file = tmp_path / "Input"
    _write_input(
        input_file,
        "\n".join(
            [
                "    nspin 4",
                "    group 166",
                "    symm_prec 1e-6",
                "    occ_band 24",
                "    band 23 28",
                "    mag_tag 1",
                "    mag 0 0 5 0 0 -5",
            ]
        ),
    )

    input_data, _, _ = read_input(str(input_file))

    assert input_data["CHARACTER"]["group"] == 166
    assert input_data["CHARACTER"]["mag_tag"] == 1
    assert input_data["CHARACTER"]["mag"].tolist() == [0.0, 0.0, 5.0, 0.0, 0.0, -5.0]


def test_character_block_rejects_invalid_manual_mag_length(tmp_path: Path):
    input_file = tmp_path / "Input"
    _write_input(
        input_file,
        "\n".join(
            [
                "    nspin 1",
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
        read_input(str(input_file))
```

- [ ] **Step 2: Run the parser tests and verify they fail**

Run:

```bash
cd /home/zjdai/file-test/pyatb_symm/pyatb-main && pytest tests/test_character_input.py -q
```

Expected:

- Fail because `CHARACTER` is not yet registered in the input system.

- [ ] **Step 3: Commit the red test scaffold**

```bash
git add pyatb-main/tests/conftest.py pyatb-main/tests/test_character_input.py
git commit -m "test: add CHARACTER input parsing cases"
```

### Task 2: Implement CHARACTER Input Parsing And Main Dispatch

**Files:**
- Modify: `pyatb-main/src/pyatb/io/default_input.py`
- Modify: `pyatb-main/src/pyatb/io/input.py`
- Modify: `pyatb-main/src/pyatb/main.py`
- Test: `pyatb-main/tests/test_character_input.py`

- [ ] **Step 1: Register CHARACTER in the default input tables**

Add the new switch and block definition in `default_input.py`:

```python
function_switch = {
    # ...
    "CHARACTER": False,
}

INPUT = {
    # ...
    "CHARACTER": {
        "nspin": [int, 1, None],
        "group": [str, 1, "auto"],
        "symm_prec": [float, 1, 1e-5],
        "occ_band": [int, 1, None],
        "band": [int, 2, None],
        "mag_tag": [int, 1, 0],
        "mag": [str, 1, "auto"],
    },
}
```

- [ ] **Step 2: Add CHARACTER-specific post-processing in `input.py`**

Implement helpers that:

```python
def _parse_character_group(raw_value):
    if raw_value == "auto":
        return "auto"
    value = int(raw_value)
    if value <= 0:
        raise ValueError("CHARACTER.group must be a positive integer or 'auto'")
    return value


def _parse_character_mag(block_data, known_parameter_names):
    # Locate "mag" and consume tokens until the next known parameter name or block end.
    # Return "auto" or np.ndarray(dtype=float).
    ...
```

And call them from `parameter_require_additional_operations()` plus `check()` to enforce:

```python
if function_switch["CHARACTER"]:
    params = INPUT["CHARACTER"]
    if params["nspin"] not in (1, 2, 4):
        raise ValueError(...)
    if params["nspin"] != INPUT["INPUT_PARAMETERS"]["nspin"]:
        raise ValueError(...)
    if params["symm_prec"] <= 0:
        raise ValueError(...)
    if params["occ_band"] <= 0:
        raise ValueError(...)
    if params["band"][0] <= 0 or params["band"][1] <= 0 or params["band"][0] > params["band"][1]:
        raise ValueError(...)
    if params["mag_tag"] not in (0, 1):
        raise ValueError(...)
    if isinstance(params["mag"], np.ndarray) and params["mag"].size % 3 != 0:
        raise ValueError(...)
```

- [ ] **Step 3: Wire the main dispatch**

Add import and dispatch in `main.py`:

```python
from pyatb.symmetry import Character

# ...
if function_switch["CHARACTER"]:
    character_parameters = INPUT["CHARACTER"]
    cal_character = Character(m_tb)
    cal_character.calculate_character(**character_parameters, **input_parameters)
```

- [ ] **Step 4: Re-run the parser tests**

Run:

```bash
cd /home/zjdai/file-test/pyatb_symm/pyatb-main && pytest tests/test_character_input.py -q
```

Expected:

- All parsing tests pass.

- [ ] **Step 5: Commit the parser implementation**

```bash
git add pyatb-main/src/pyatb/io/default_input.py pyatb-main/src/pyatb/io/input.py pyatb-main/src/pyatb/main.py
git commit -m "feat: add CHARACTER input parsing and dispatch"
```

### Task 3: Add the CHARACTER Skeleton Module With Explicit Placeholder Exit

**Files:**
- Create: `pyatb-main/src/pyatb/symmetry/__init__.py`
- Create: `pyatb-main/src/pyatb/symmetry/character.py`
- Create: `pyatb-main/tests/test_character_module.py`
- Modify: `pyatb-main/src/pyatb/main.py`
- Test: `pyatb-main/tests/test_character_module.py`

- [ ] **Step 1: Write the failing skeleton tests**

```python
from pathlib import Path

import pytest

from pyatb.symmetry.character import Character


class _FakeTB:
    nspin = 1
    max_kpoint_num = 10


def test_character_creates_output_dir_and_raises_not_implemented(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tb_model = _FakeTB()
    cal = Character(tb_model)

    with pytest.raises(NotImplementedError):
        cal.calculate_character(
            nspin=1,
            group="auto",
            symm_prec=1e-5,
            occ_band=8,
            band=[7, 10],
            mag_tag=0,
            mag="auto",
            package="ABACUS",
        )

    assert (tmp_path / "Out" / "CHARACTER").exists()
```

- [ ] **Step 2: Run the skeleton test and verify it fails**

Run:

```bash
cd /home/zjdai/file-test/pyatb_symm/pyatb-main && pytest tests/test_character_module.py -q
```

Expected:

- Fail because `pyatb.symmetry.character.Character` does not exist yet.

- [ ] **Step 3: Implement the skeleton module**

Create `src/pyatb/symmetry/__init__.py`:

```python
from pyatb.symmetry.character import Character
```

Create `src/pyatb/symmetry/character.py` with a class that:

```python
class Character:
    def __init__(self, tb):
        self._tb = tb
        self.output_path = os.path.join(OUTPUT_PATH, "CHARACTER")
        ...

    def calculate_character(self, **kwargs):
        self._validate_parameters(...)
        self._write_log(...)
        raise NotImplementedError("CHARACTER calculation kernel is not implemented yet.")
```

Behavior requirements:

- create or recreate `Out/CHARACTER`
- write module header + parameter summary to `running.log`
- validate the already-parsed parameters again locally

- [ ] **Step 4: Re-run the module tests**

Run:

```bash
cd /home/zjdai/file-test/pyatb_symm/pyatb-main && pytest tests/test_character_module.py -q
```

Expected:

- Skeleton tests pass and confirm the explicit `NotImplementedError`.

- [ ] **Step 5: Commit the module skeleton**

```bash
git add pyatb-main/src/pyatb/symmetry/__init__.py pyatb-main/src/pyatb/symmetry/character.py pyatb-main/tests/test_character_module.py
git commit -m "feat: add CHARACTER skeleton module"
```

### Task 4: Create And Validate The `symm` Conda Environment

**Files:**
- Create: `project_records/change_logs/2026-04-25-character-interface-and-env-implementation-log.md`
- Create: `test_workspace/character_input_smoke/Input.auto`
- Create: `test_workspace/character_input_smoke/Input.manual`
- Create: `/home/zjdai/software/miniconda3/envs/symm/etc/conda/activate.d/irvspdata.sh`
- Create: `/home/zjdai/software/miniconda3/envs/symm/etc/conda/deactivate.d/irvspdata.sh`

- [ ] **Step 1: Create the conda environment**

Run:

```bash
conda create -y -n symm python=3.11 numpy scipy matplotlib mpi4py pybind11 spglib openblas lapacke pytest pip
```

Expected:

- New environment `symm` exists.

- [ ] **Step 2: Install `ase-abacus` and editable `pyatb`**

Run:

```bash
conda run -n symm python -m pip install ase-abacus
conda run -n symm python -m pip install -e /home/zjdai/file-test/pyatb_symm/pyatb-main
```

Expected:

- `import ase` and `import pyatb` both succeed in `symm`.

- [ ] **Step 3: Add IRVSPDATA activation hooks**

Create:

```bash
export IRVSPDATA=/home/zjdai/file-test/pyatb_symm/IRVSP-master/src_irvsp_v2_release
```

in `activate.d/irvspdata.sh`, and an unset in `deactivate.d/irvspdata.sh`.

- [ ] **Step 4: Build IRVSP and IR2TB with oneAPI available**

Run:

```bash
bash -lc '. /home/zjdai/software/oneapi/2024.1/setvars.sh --force >/tmp/symm_irvsp_setvars.log 2>&1 && conda run -n symm make -C /home/zjdai/file-test/pyatb_symm/IRVSP-master/src_irvsp_v2_release'
bash -lc '. /home/zjdai/software/oneapi/2024.1/setvars.sh --force >/tmp/symm_ir2tb_setvars.log 2>&1 && conda run -n symm make -C /home/zjdai/file-test/pyatb_symm/IRVSP-master/src_ir2tb_v2'
```

Expected:

- `/home/zjdai/file-test/pyatb_symm/IRVSP-master/src_irvsp_v2_release/irvsp` exists
- `/home/zjdai/file-test/pyatb_symm/IRVSP-master/src_ir2tb_v2/ir2tb` exists

- [ ] **Step 5: Run environment smoke verification**

Run:

```bash
conda run -n symm python -c "import ase, pyatb, spglib; print('ok')"
conda run -n symm bash -lc 'echo $IRVSPDATA'
conda run -n symm /home/zjdai/file-test/pyatb_symm/IRVSP-master/src_irvsp_v2_release/irvsp | head -n 2 || true
conda run -n symm /home/zjdai/file-test/pyatb_symm/IRVSP-master/src_ir2tb_v2/ir2tb | head -n 2 || true
```

Expected:

- Python prints `ok`
- `IRVSPDATA` resolves to the expected path
- both binaries launch and print usage / startup text rather than missing-library errors

- [ ] **Step 6: Commit implementation and environment records**

```bash
git add pyatb-main project_records/change_logs/2026-04-25-character-interface-and-env-implementation-log.md test_workspace/character_input_smoke
git commit -m "feat: implement CHARACTER interface and symm environment"
```
