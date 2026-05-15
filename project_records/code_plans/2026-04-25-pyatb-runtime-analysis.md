# PyATB Runtime Analysis For Future Symmetry Work

Date: `2026-04-25`

Scope:
- Clarify the end-to-end runtime flow of `pyatb` when the user provides an `Input` file.
- Identify the existing initialization, k-point, diagonalization, band, and wavefunction-coefficient APIs that should be reused for a future `symmetry/` module.

## 1. Overall Runtime Flow From `Input`

### 1.1 Program entry

- The console entry is declared in `pyatb-main/setup.py`:
  - `pyatb = pyatb.main:main`
- This means a normal run starts from `src/pyatb/main.py::main()`.

### 1.2 Package initialization before `main()`

Importing `pyatb` triggers `src/pyatb/__init__.py`, which does the following:

- Sets `INPUT_PATH = os.getcwd()`
- Sets `OUTPUT_PATH = os.path.join(INPUT_PATH, 'Out')`
- Creates `Out/` if it does not exist
- Creates `running.log`
- Initializes MPI globals from `src/pyatb/parallel.py`
  - `COMM`
  - `SIZE`
  - `RANK`
- Starts the global timer

Important consequence:

- `pyatb` expects to be launched in the directory that contains the `Input` file.
- The runtime workspace is the current working directory, not the package source directory.

### 1.3 Input parsing

`src/pyatb/main.py::main()` starts with:

- `read_input(os.path.join(INPUT_PATH, 'Input'))`

The parsing pipeline is in `src/pyatb/io/input.py`:

1. `get_file_block()`
   - Removes comments with `skip_notes()`
   - Splits the text file into named blocks like `INPUT_PARAMETERS { ... }`

2. `update_INPUT()`
   - Reads the raw block content into the default schema from `src/pyatb/io/default_input.py`
   - Expands option-dependent parameters through `parameter_options`
     - `package`
     - `kpoint_mode`
     - `integrate_mode`
     - `cal_surface_method`
   - Expands multiline parameters through `parameter_multigroups`
     - `lattice_vector`
     - `high_symmetry_kpoint`
     - `kpoint_direct_coor`
   - Expands dependent parameters through `parameter_dependence`
     - `HR_route`
     - `kpoint_label`
     - `valence_e`

3. `parameter_require_additional_operations()`
   - Converts `fermi_energy` units if needed
   - Converts `lattice_constant` from `Bohr` to `Angstrom` if needed
   - Extracts `kpoint_num_in_line` from the last column of `high_symmetry_kpoint`

4. `check()`
   - Requires `INPUT_PARAMETERS`
   - Requires `LATTICE`
   - Requires a `FERMI_ENERGY` block when `fermi_energy` is `Auto`
   - Requires `rR_route` if the selected function needs the `rR` matrix

5. `read_input()`
   - Writes `Out/input.json`
   - Converts Python lists to `numpy.ndarray`
   - Returns:
     - `new_INPUT`
     - `function_switch`
     - `bool_need_rR`

### 1.4 Tight-binding model initialization

After parsing, `main()` reads:

- `nspin`
- `lattice_constant`
- `lattice_vector`
- `package`
- `sparse_format`
- `max_kpoint_num`

Then it creates:

- `m_tb = tb(nspin, lattice_constant, lattice_vector, max_kpoint_num)`

The `tb` class is defined in `src/pyatb/tb/tb.py`.

Internally:

- For `nspin != 2`, one solver is created:
  - `tb.tb_solver`
- For `nspin == 2`, two solvers are created:
  - `tb.tb_solver_up`
  - `tb.tb_solver_dn`

The Python wrapper class is `src/pyatb/tb/solver.py::solver`.
Its backend is the compiled pybind interface:

- `pyatb.interface_python`

### 1.5 Reading `HR`, `SR`, and optional `rR`

`main()` then loads the matrix data according to `package`.

For `ABACUS`:

- `abacus_readHR(...)`
- `abacus_readSR(...)`
- `abacus_readrR(...)` if needed

For `WANNIER90`:

- `wannier90_readTB(...)` if the `.tb` file contains `r`
- otherwise `wannier90_readHR(...)`

### 1.6 Binding the data into the solver

Still in `main()`:

- `m_tb.set_solver_HSR(HR, SR, isSparse)`
- or `m_tb.set_solver_HSR_spin2(HR_up, HR_dn, SR, isSparse)`
- optional:
  - `m_tb.set_solver_rR(rR_x, rR_y, rR_z, isSparse)`

This is the real point where the TB model becomes numerically usable.

After this step, the solver can produce:

- `H(k)`
- `S(k)`
- eigenvalues
- eigenvectors
- `r(k)`
- velocity-related matrices

### 1.7 Module dispatch

After initialization, `main()` dispatches the selected calculation blocks in a fixed order using `function_switch`.

The order in `src/pyatb/main.py` starts with:

1. `FERMI_ENERGY`
2. `BAND_STRUCTURE`
3. `BANDUNFOLDING`
4. `BANDUNFOLDING_SPIN_TEXTURE`
5. `FAT_BAND`
6. `FERMI_SURFACE`
7. `FIND_NODES`
8. `JDOS`
9. `PDOS`
10. `SPIN_TEXTURE`
11. `SURFACE_STATE`
12. Berry / transport related modules

Important detail:

- If `FERMI_ENERGY` is run, the result is written back into:
  - `input_parameters['fermi_energy']`
- Later modules reuse this updated value.

### 1.8 Common execution pattern inside each module

Most modules follow the same structure:

1. Construct a module object with `m_tb`
2. Choose a k-point mode
3. Build a k-point generator
4. Loop over k-point batches
5. Distribute the batch across MPI ranks
6. Call the low-level solver on each rank
7. Gather or reduce the results
8. Write results into `Out/<Module_Name>/`

This common pattern is important because a future `symmetry/` module should probably reuse it rather than invent a separate runtime model.

## 2. Existing APIs Relevant To Future Symmetry Work

The future goal is:

- for an arbitrary k point
- and an arbitrary selected band
- obtain the character / irrep-like symmetry information

For that goal, the existing code already provides most of the numerical prerequisites except the symmetry operator construction itself.

### 2.1 Initialization API

There are two equivalent ways to initialize the TB model.

#### CLI path

- `src/pyatb/main.py::main()`

This path is driven by an `Input` file and dispatches a full calculation.

#### Python API path

- `src/pyatb/init_tb.py::init_tb(...)`

This function is especially useful for a future `symmetry/` module because it directly returns a ready-to-use `tb` object:

- reads `HR`, `SR`, and optional `rR`
- constructs `tb(...)`
- calls `set_solver_HSR(...)` / `set_solver_HSR_spin2(...)`
- optionally calls `set_solver_rR(...)`
- returns `m_tb`

Conclusion:

- For a reusable `symmetry` Python module, `init_tb(...)` is the cleanest existing initialization entry point.

### 2.2 K-point APIs

The k-point generators live in:

- `src/pyatb/kpt/kpoint_generator.py`

Core generators:

- `mp_generator`
- `line_generator`
- `array_generater`
- `string_generator`
- `string_generator_3d`

MPI splitter:

- `kpoints_in_different_process`
- `string_in_different_process`

For the future symmetry task, the most relevant APIs are:

- `array_generater`
  - best for arbitrary user-specified k points
- `kpoints_in_different_process`
  - best if the new module wants to stay MPI-compatible

### 2.3 High-level module entry for band and wavefunction reuse

The existing high-level band entry is:

- `src/pyatb/fermi/band_structure.py::Band_Structure.calculate_band_structure(...)`

It does:

- select `band_range`
- select `kpoint_mode`
- call one of:
  - `set_k_mp(...)`
  - `set_k_line(...)`
  - `set_k_direct(...)`
- call `get_band_structure()`

The key internal dispatch is:

- if `wf_collect == True`
  - use `diago_H(...)` or `diago_H_range(...)`
- else
  - use `diago_H_eigenvaluesOnly(...)` or `diago_H_eigenvaluesOnly_range(...)`

This is the existing standard path for:

- band energies
- optional wavefunction coefficients

### 2.4 Low-level diagonalization APIs

The actual diagonalization APIs are in:

- `src/pyatb/tb/solver.py`

Most relevant functions:

- `get_Hk(k_direct_coor)`
- `get_Sk(k_direct_coor)`
- `diago_H(k_direct_coor)`
- `diago_H_range(k_direct_coor, lower_band_index, upper_band_index)`
- `diago_H_eigenvaluesOnly(k_direct_coor)`
- `diago_H_eigenvaluesOnly_range(k_direct_coor, lower_band_index, upper_band_index)`

Important detail:

- `lower_band_index` and `upper_band_index` are counted from `1`, not from `0`.

Return shapes:

- `diago_H(...)`
  - eigenvectors: `[nk, basis_num, basis_num]`
  - eigenvalues: `[nk, basis_num]`
- `diago_H_range(...)`
  - eigenvectors: `[nk, basis_num, selected_band_num]`
  - eigenvalues: `[nk, selected_band_num]`

Interpretation:

- The returned eigenvectors are the basis-expansion coefficients of the Bloch states.
- This is exactly the data a symmetry-character module will need.

### 2.5 Existing evidence that the basis is not treated as trivially orthonormal

Several existing modules explicitly combine eigenvectors with `S(k)`:

- `Fat_Band`
  - uses `diago_H_range(...)` and `get_Sk(...)`
- `PDOS`
  - uses `diago_H(...)` and `get_Sk(...)`
- `Spin_Texture`
  - uses `diago_H_range(...)` and `get_Sk(...)`

Representative formulas in the current code:

- `((C^dagger S)^T * C).real`
- `C^dagger S sigma C`
- `M = C^dagger S`

This strongly suggests:

- the wavefunction coefficients should be interpreted together with `S(k)`
- future symmetry-character calculations should be careful with overlap-matrix normalization and matrix elements

This is a key design constraint for the future `symmetry/` implementation.

### 2.6 Where band and wavefunction coefficients are already exposed

If `wf_collect=True`, `Band_Structure` writes:

- `Out/Band_Structure/wfc.dat`
- or for `nspin == 2`
  - `wfc_up.dat`
  - `wfc_dn.dat`

It also writes:

- `band.dat`
- or `band_up.dat` / `band_dn.dat`
- `kpt.dat`

Also, if:

- `SIZE == 1`
- and `total_kpoint_num <= max_kpoint_num`

then `get_band_structure()` can directly return:

- `(kvec_d, eig, wf)`

Conclusion:

- The existing code already knows how to expose the eigenvector coefficients.
- A symmetry module does not need to reimplement diagonalization.

### 2.7 Recommended existing call path for the target problem

For the final goal "character representation of an arbitrary chosen band at an arbitrary k point", the cleanest current call path is:

1. Initialize a `tb` object
   - preferred: `init_tb(...)`
2. Prepare the target k-point array
   - example shape: `[1, 3]`
3. Use the low-level solver directly
   - `diago_H_range(k, band, band)`
4. Also get the overlap matrix if needed
   - `get_Sk(k)`
5. Build/apply the future symmetry operator in the same basis
6. Evaluate the band-resolved character from the selected eigenvector

If `nspin == 2`:

- the new module must choose between:
  - `tb.tb_solver_up`
  - `tb.tb_solver_dn`

If `nspin != 2`:

- use:
  - `tb.tb_solver`

## 3. Direct Answer To The Two Requested Questions

### Question 1: What is the full runtime flow of `pyatb` given an `Input` file?

Short answer:

1. Start from `pyatb.main:main`
2. Use current working directory as the runtime workspace
3. Parse `Input`
4. Expand defaults and option-dependent parameters
5. Validate the required blocks
6. Build `tb`
7. Read `HR`, `SR`, and optional `rR`
8. Bind the matrices into the low-level solver
9. Instantiate the requested module(s)
10. Build the relevant k-point generator
11. Split k points across MPI ranks
12. Call the low-level solver for each batch
13. Gather results and write them into `Out/...`

### Question 2: Which existing functions should be reused for initialization, k points, diagonalization, bands, and wavefunction coefficients?

Short answer:

- Initialization:
  - `pyatb.init_tb.init_tb(...)`
  - or internally:
    - `tb.set_solver_HSR(...)`
    - `tb.set_solver_HSR_spin2(...)`
    - `tb.set_solver_rR(...)`

- K points:
  - `kpoint_generator.array_generater`
  - `kpoint_generator.mp_generator`
  - `kpoint_generator.line_generator`
  - `kpoint_generator.kpoints_in_different_process`

- Diagonalization:
  - `solver.diago_H(...)`
  - `solver.diago_H_range(...)`
  - `solver.diago_H_eigenvaluesOnly(...)`
  - `solver.diago_H_eigenvaluesOnly_range(...)`

- Band + wavefunction collection:
  - `Band_Structure.calculate_band_structure(...)`
  - `Band_Structure.get_band_structure()`
  - `wf_collect=True`

- Matrix access that will likely matter for symmetry:
  - `solver.get_Hk(...)`
  - `solver.get_Sk(...)`

## 4. Immediate Implication For The Planned `symmetry/` Folder

The codebase already provides:

- initialization
- k-point generation
- MPI batch distribution
- diagonalization
- eigenvector coefficients
- overlap matrices

The main missing piece is:

- construction of symmetry operators in the same basis as the TB coefficients
- band-resolved character evaluation logic on top of the existing solver outputs

So the future `symmetry/` work should focus on:

- basis/symmetry operator construction
- degeneracy handling
- overlap-aware character evaluation

and should avoid reimplementing:

- input parsing
- TB loading
- k-point batching
- diagonalization
