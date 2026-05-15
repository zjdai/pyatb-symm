# 2026-04-28 HS Covariance Validation Summary

## Purpose

This document summarizes two classes of HS covariance problems that were repeatedly mixed together during debugging:

1. How to relate a supercell Hamiltonian to a primitive-cell Hamiltonian in the same Cartesian frame using an integer supercell matrix `M`;
2. How to relate two Hamiltonians defined for the same physical lattice when the Cartesian axes are different, i.e. when the issue is a real-space rotation `Q`, not a supercell remapping.

The goal is not to replay the entire debugging history, but to record the final conventions, formulas, edge cases, and failure modes that are actually reusable.

---

## 1. Notation and ABACUS Conventions

### 1.1 Row-vector lattice convention

In this project and in ABACUS `STRU`, lattice vectors are stored row by row:

```text
A =
[ a1 ]
[ a2 ]
[ a3 ]
```

Each row is a Cartesian lattice vector.

If the fractional coordinate is written as a row vector `f = (f1, f2, f3)`, then the Cartesian coordinate is:

```text
r_cart = f @ A
```

This convention must be fixed first. Otherwise all later formulas involving `M`, `Q`, and `R` can end up transposed or reversed.

### 1.2 ABACUS real-space Hamiltonian convention

ABACUS / pyatb uses:

```text
H_{νμ}(R) = <0, ν | H | R, μ>
S_{νμ}(R) = <0, ν | R, μ>
```

So:

- `R` acts on the orbital on the right;
- a block `(R, atom1, atom2)` corresponds to the real-space pair vector

```text
d(R; atom1, atom2) = R @ A + tau_2 - tau_1
```

where `tau_1` and `tau_2` are the Cartesian positions of `atom1` and `atom2` inside the `R = 0` home cell.

If `tau_1` and `tau_2` are swapped, the later `R'` mapping acquires the wrong sign.

### 1.3 ABACUS stores only the upper triangle explicitly

For a fixed `R`, the files `data-HR-sparse_SPIN0.csr` and `data-SR-sparse_SPIN0.csr` store only the upper triangle of that matrix.

Therefore:

- reading the file into a dense matrix with the upper triangle filled and the lower triangle left as zero is only a partial representation;
- reconstructing the full matrix requires the Hermitian relation

```text
H(R) = H(-R)^dagger
S(R) = S(-R)^dagger
```

More precisely:

- the upper part of the full matrix at `R` comes from the explicit storage at `R`;
- the lower part comes from the explicit storage at `-R`, followed by conjugate transpose.

The same issue appears at the atom-block level. If a requested block `(R, atom_a, atom_b)` lies on the lower-triangular side of the full matrix, it must not be treated as a zero block. Its Hermitian storage partner is:

```text
X_eff(R, a, b) = X_store(-R, b, a)^dagger
```

where `X` is either `H` or `S`.

This was one of the most important pitfalls in the entire debugging process.

---

## 2. Case A: supercell -> primitive mapping in the same Cartesian frame

### 2.1 When this case applies

This case applies when:

- the source and target structures live in the same Cartesian frame;
- no real-space rotation of the orbital/spin basis is needed;
- the difference is only supercell vs primitive cell, or a pure relabeling within the same frame;
- the target is to connect the two HS datasets by an integer supercell matrix `M`, atom mapping, and `R` relabeling.

This is the core logic behind `test-abacus-3 -> test-abacus-4`.

### 2.2 Lattice relation

Let:

- `A_sc` be the supercell lattice;
- `A_pr` be the primitive-cell lattice.

Under the row-vector convention:

```text
A_sc = M @ A_pr
```

So the correct inference is:

```text
M_raw = A_sc @ inv(A_pr)
M = round(M_raw)
```

The residual from the nearest integer matrix must be checked explicitly.

Writing `M` on the wrong side is a common failure mode.

### 2.3 Single-atom mapping

Suppose atom `a` in the source maps to atom `a'` in the target primitive cell, with an integer cell shift `shift_a` needed to bring it back to the target home cell. Then:

```text
tau_a = shift_a @ A_pr + tau'_a'
```

Here:

- `tau_a` is the Cartesian position of source atom `a`;
- `tau'_a'` is the Cartesian position of target atom `a'`;
- `shift_a` is an integer row vector.

### 2.4 Correct mapping for `(R, atom1, atom2)`

For a source pair:

```text
d = R @ A_sc + tau_2 - tau_1
```

Using `A_sc = M @ A_pr` and the single-atom mapping:

```text
d
= R @ (M @ A_pr) + (shift_2 @ A_pr + tau'_2) - (shift_1 @ A_pr + tau'_1)
= (R @ M + shift_2 - shift_1) @ A_pr + tau'_2 - tau'_1
```

Therefore the target primitive-cell label must be:

```text
R' = R @ M + shift_2 - shift_1
```

This is the central formula for the no-rotation supercell-to-primitive problem.

### 2.5 How the Hamiltonian blocks are related when there is no rotation

If both structures are in the same Cartesian frame and the local orbital axes, angular-momentum convention, and spin basis are unchanged, then no additional orbital/spin rotation is needed:

```text
D_a = I
D_b = I
```

The corresponding blocks should then satisfy:

```text
H_pr(R', a', b') = H_sc(R, a, b)
S_pr(R', a', b') = S_sc(R, a, b)
```

The only remaining requirements are:

- find the correct block mapping `(R, a, b) -> (R', a', b')`;
- handle the ABACUS upper-triangular storage correctly.

### 2.6 Common mistakes in this case

#### Mistake 1: letting `R` act on the left atom

The wrong form is usually:

```text
d_wrong = R @ A + tau_1 - tau_2
```

This flips the roles of `shift_1` and `shift_2`, and the later `R'` mapping becomes wrong.

#### Mistake 2: multiplying `M` on the wrong side

With row-vector fractional coordinates, the correct relation is:

```text
R' = R @ M + shift_2 - shift_1
```

not `M @ R + ...`.

#### Mistake 3: summing repeated source samples

This is wrong in validation of a supercell-to-primitive reduction.

If several supercell copies map to the same primitive target key, they should be treated as repeated equivalent samples that must agree with each other. They should not be added together during validation.

#### Mistake 4: ignoring the `1.0 -> 0.0` boundary wrap

ABACUS may wrap a boundary atom back into the home cell, e.g.

```text
(0, 1, 0) -> (0, 0, 0)
```

If the structure file, the mapping code, and the DFT output are not using the same wrapped convention, the inferred shifts and later `R'` values will be wrong.

#### Mistake 5: treating a lower-triangular target block as zero

This was one of the key implementation errors during the debugging cycle.

If the geometrically correct target block lies in the lower-triangular side of the full matrix, but the comparison path uses only the bare upper-triangular dense representation, that block is misread as zero. Brute-force matching then appears to prefer some unrelated upper-triangular block.

In that situation the geometric mapping looks wrong, but the real problem is only the storage convention.

---

## 3. Case B: the same physical lattice in different Cartesian frames

### 3.1 When this case applies

This case applies when:

- the source and target structures describe the same primitive cell;
- the real-space connectivity is the same;
- the issue is not supercell vs primitive cell;
- the issue is that the Cartesian axes differ, i.e. the whole structure has been rotated into a different frame.

This is the core logic behind `test-abacus-2/soc -> test-abacus-4`.

### 3.2 The first task is to find the physical real-space rotation `Q`

With row-vector Cartesian coordinates, the geometric relation should be written as:

```text
r_target ~= r_source @ Q_row
A_target ~= A_source @ Q_row
```

where:

- `Q_row` is the physical real-space rotation acting on row vectors from the right;
- if the downstream code expects a column-vector rotation, it must be converted as

```text
Q_col = Q_row^T
```

This conversion must match the internal convention of the `D`-matrix construction code.

### 3.3 The `D` matrix must come from a physical rotation

The basis transformation needed here is a real-space representation:

```text
D(Q) = D_orbital(Q) ⊗ D_spin(Q)
```

It is incorrect to feed any of the following directly into the basis transform:

- the supercell matrix `M`;
- a pure lattice-basis change matrix;
- a fractional-coordinate transformation;
- an unchecked choice among `Q`, `Q^T`, and `Q^{-1}`.

These mistakes often produce a fake sense of partial agreement:

- the lattice may look “close enough”;
- a few atomic positions may also fit;
- but HS covariance stays at the `1e-1 ~ 1e0` level instead of converging to `1e-5` or better.

### 3.4 Correct block covariance relation

In this case, there is no supercell-style `R -> R'` relabeling. One compares the same `R` and the same atom pair, but rotates the block internally.

If `D` is defined as the basis transformation from the source basis to the target basis, then:

```text
H_target(R) = D H_source(R) D^dagger
S_target(R) = D S_source(R) D^dagger
```

The reverse form is:

```text
H_source(R) = D^dagger H_target(R) D
S_source(R) = D^dagger S_target(R) D
```

This is exactly the relation that was finally validated for `test-abacus-2/soc -> test-abacus-4`.

However, the direction of the daggers depends on how `D` is defined in code. The formula must never be discussed independently of the implementation convention.

### 3.5 Common mistakes in this case

#### Mistake 1: fitting the rotation from the atomic point cloud only

This was the main source of false solutions.

Crystals can have point-group symmetry. A rotation can fit the wrapped atomic coordinates fairly well and still fail to represent the actual orientation of the full structure because it does not match the lattice vectors.

The correct fit should use both:

- lattice vectors;
- wrapped nonzero atomic Cartesian coordinates.

#### Mistake 2: mixing `Q`, `Q^T`, and `Q^{-1}`

This error appears repeatedly when row-vector and column-vector conventions are not fixed first.

The sequence must be:

1. fix whether coordinates are row vectors or column vectors;
2. fix whether rotations act from the right or from the left;
3. fix whether the `D` builder expects `Q_row` or `Q_col`;
4. keep the same convention all the way through.

#### Mistake 3: rotating orbitals but not spin

For SOC, `D` must include both the orbital and the spin-1/2 representation.

Otherwise the geometry can be correct while the HS covariance still fails systematically.

#### Mistake 4: confusing a k-space representation with a local real-space basis representation

The required object here is a local orbital-plus-spin real-space representation matrix. A k-space little-group representation cannot be substituted for it.

#### Mistake 5: ignoring the atom permutation induced by the rotation

A real-space rotation usually not only rotates the local basis, but also sends atom `a` to another atom `a'`.

So the correct relation is not only an internal similarity transformation; the atom correspondence must be resolved first, and then the block-internal `D` rotation must be applied.

---

## 4. The essential difference between the two cases

### 4.1 Case A is controlled by `M` and `shift`

If both structures live in the same Cartesian frame, the essential questions are:

- what integer relation connects the two lattices;
- which source atom maps to which target home-cell atom;
- what is the correct `R'`.

The geometric condition is:

```text
R @ A_source + tau_2 - tau_1
=
R' @ A_target + tau'_2 - tau'_1
```

### 4.2 Case B is controlled by `Q` and `D(Q)`

If both structures describe the same primitive cell but in different Cartesian frames, the key object is the physical rotation:

```text
r_target ~= r_source @ Q_row
A_target ~= A_source @ Q_row
```

and then the basis rotation:

```text
H_target(R) = D H_source(R) D^dagger
```

### 4.3 What to do when both effects appear together

In a real structure-standardization + HS-standardization workflow, both effects may coexist:

1. use `M`, `shift`, and atom mapping to determine which source block corresponds to which target block;
2. use the real-space rotation `Q` to determine how the matrix elements inside that block transform;
3. apply the ABACUS Hermitian storage rule when reconstructing the actual compared block.

In other words:

- `M` labels the block;
- `Q` and `D(Q)` transform the local basis inside the block;
- `R <-> -R` Hermitian pairing makes the stored data complete.

---

## 5. Recommended debugging order

If a similar problem appears again, the safest order is:

### Step 1: validate geometry before touching HS

Check:

- whether `M_raw` is close enough to an integer matrix;
- the residual in `A_source - M @ A_target`;
- single-atom mapping errors;
- pair-vector mapping errors.

If geometry fails, HS covariance checks are meaningless.

### Step 2: verify that the axis change is a physical rotation

For the different-frame case, verify:

- `det(Q) ≈ 1`;
- small lattice residuals;
- small wrapped atomic Cartesian residuals.

Do not trust an atom-only fit.

### Step 3: verify the definition and side of `D`

After fixing `Q`, verify:

- that `D` includes both orbital and spin sectors;
- whether the code uses `D H D^dagger` or `D^dagger H D`;
- whether that order is consistent with how `D` is defined.

### Step 4: only then check the ABACUS storage convention

If the geometry and the rotation are both correct but some blocks suddenly appear as zero or get “replaced” by unrelated blocks, the first suspicion should be:

- bare upper-triangle use;
- failure to switch lower blocks to `(-R, b, a)^dagger`;
- failure to reconstruct the missing lower part for diagonal/self blocks.

---

## 6. Final lessons from this debugging cycle

The essential lessons can be compressed into four statements:

1. `R` acts on the orbital on the right, so the pair vector must be written as `R @ A + tau_2 - tau_1`.
2. In the same Cartesian frame, the core supercell-to-primitive formula is `R' = R @ M + shift_2 - shift_1`.
3. In different Cartesian frames, the core object is not `M` but the physical real-space rotation `Q` and its orbital-plus-spin representation `D(Q)`.
4. ABACUS stores only the upper triangle explicitly; if the Hermitian storage partner is not restored, many apparent “mapping errors” are actually readout errors.

Only when all four points are handled consistently does the HS covariance problem really close.
