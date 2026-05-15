from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix

from pyatb.io.abacus_read_stru import read_stru
from pyatb.io.abacus_read_xr import abacus_readHR, abacus_readSR
from pyatb.symmetry.Dk_matrix import (
    build_atom_local_rotation,
    extract_abacus_basis_metadata,
)


def canonicalize_fractional_coordinates(frac: np.ndarray, tol: float = 1.0e-9) -> np.ndarray:
    wrapped = np.asarray(frac, dtype=float) - np.floor(np.asarray(frac, dtype=float))
    wrapped[np.abs(wrapped) <= tol] = 0.0
    wrapped[np.abs(wrapped - 1.0) <= tol] = 0.0
    return wrapped


def map_target_r_vector(b_matrix, r_old, shift_a, shift_b, tol: float = 1.0e-8) -> np.ndarray:
    raw = np.asarray(r_old, dtype=float) @ np.asarray(b_matrix, dtype=float)
    raw = raw + np.asarray(shift_b, dtype=float) - np.asarray(shift_a, dtype=float)
    rounded = np.rint(raw).astype(int)
    if float(np.max(np.abs(raw - rounded))) > tol:
        raise ValueError("Mapped R vector is not integral within tolerance.")
    return rounded


def accumulate_block(blocks: dict, key, value: np.ndarray) -> None:
    block = np.asarray(value, dtype=complex)
    if key in blocks:
        blocks[key] = blocks[key] + block
    else:
        blocks[key] = np.array(block, dtype=complex, copy=True)


def standardize_sparse_blocks(source_blocks, atom_mapping, lattice_transform, orbital_rotations):
    target_blocks = {}
    for (old_a, old_b, r_old), block in source_blocks:
        map_a = atom_mapping[old_a]
        map_b = atom_mapping[old_b]
        r_new = map_target_r_vector(lattice_transform, r_old, map_a["shift"], map_b["shift"])
        d_a = np.asarray(orbital_rotations[old_a], dtype=complex)
        d_b = np.asarray(orbital_rotations[old_b], dtype=complex)
        block_new = d_a.conj().T @ np.asarray(block, dtype=complex) @ d_b
        accumulate_block(target_blocks, (map_a["new_atom"], map_b["new_atom"], tuple(r_new.tolist())), block_new)
    return target_blocks


def compute_xyz_axis_transform(lattice_old: np.ndarray, lattice_new: np.ndarray, tol: float = 1.0e-12) -> np.ndarray:
    def _frame(lattice: np.ndarray) -> np.ndarray:
        a1 = np.asarray(lattice[0], dtype=float)
        a2 = np.asarray(lattice[1], dtype=float)
        a3 = np.asarray(lattice[2], dtype=float)

        e1 = a1 / np.linalg.norm(a1)
        normal = np.cross(a1, a2)
        if np.linalg.norm(normal) <= tol:
            normal = np.cross(a1, a3)
        e3 = normal / np.linalg.norm(normal)
        e2 = np.cross(e3, e1)
        e2 = e2 / np.linalg.norm(e2)
        return np.vstack([e1, e2, e3])

    frame_old = _frame(np.asarray(lattice_old, dtype=float))
    frame_new = _frame(np.asarray(lattice_new, dtype=float))
    rotation = frame_new.T @ frame_old
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def build_standardization_atom_mapping(source_metadata, target_metadata, tol: float = 1.0e-6) -> list[dict]:
    inv_target_lattice = np.linalg.inv(np.asarray(target_metadata.lattice_vector, dtype=float))
    mapping: list[dict] = []

    for old_atom, (source_frac, species) in enumerate(
        zip(source_metadata.positions_frac, source_metadata.species_by_atom, strict=True)
    ):
        cart = np.asarray(source_frac, dtype=float) @ np.asarray(source_metadata.lattice_vector, dtype=float)
        frac_target = cart @ inv_target_lattice

        best = None
        for new_atom, (target_frac, target_species) in enumerate(
            zip(target_metadata.positions_frac, target_metadata.species_by_atom, strict=True)
        ):
            if species != target_species:
                continue
            shift = np.rint(frac_target - target_frac).astype(int)
            residual = frac_target - (target_frac + shift)
            residual -= np.rint(residual)
            err = float(np.max(np.abs(residual)))
            if best is None or err < best[0]:
                best = (err, new_atom, shift)

        if best is None or best[0] > tol:
            raise ValueError(f"Failed to map source atom {old_atom + 1} into standardized structure.")

        mapping.append(
            {
                "old_atom": old_atom,
                "new_atom": int(best[1]),
                "shift": np.asarray(best[2], dtype=int),
                "species": species,
            }
        )

    return mapping


def _build_metadata_from_stru(stru_path: Path, lattice_vector: np.ndarray, nspin: int):
    stru_atom = read_stru(str(stru_path))
    spin_factor = 2 if int(nspin) == 4 else 1
    basis_num = 0
    for atom_type in stru_atom:
        atom_type.read_numerical_orb()
        local_dim = 0
        for l_value, count in enumerate(atom_type.orbital_num):
            local_dim += (2 * l_value + 1) * int(count)
        basis_num += int(atom_type.atom_num) * local_dim
    basis_num *= spin_factor

    fake_tb = SimpleNamespace(
        lattice_vector=np.asarray(lattice_vector, dtype=float),
        basis_num=int(basis_num),
        nspin=int(nspin),
        stru_atom=stru_atom,
    )
    return extract_abacus_basis_metadata(fake_tb)


def _triangular_vector_to_dense(vector: np.ndarray, basis_num: int) -> np.ndarray:
    dense = np.zeros((basis_num, basis_num), dtype=complex)
    index = 0
    for row in range(basis_num):
        for col in range(row, basis_num):
            dense[row, col] = vector[index]
            index += 1
    return dense


def _iter_dense_xr_blocks(xr) -> list[tuple[np.ndarray, np.ndarray]]:
    rows = []
    for i_r, r_vector in enumerate(np.asarray(xr.R_direct_coor, dtype=int)):
        row = np.asarray(xr.XR.getrow(i_r).toarray()).ravel()
        dense = _triangular_vector_to_dense(row, int(xr.basis_num))
        rows.append((np.asarray(r_vector, dtype=int), dense))
    return rows


def _dense_blocks_by_r(xr) -> dict[tuple[int, int, int], np.ndarray]:
    return {tuple(int(value) for value in r_vector.tolist()): dense for r_vector, dense in _iter_dense_xr_blocks(xr)}


def _full_dense_blocks_by_r_from_hermitian_partners(
    dense_by_r: dict[tuple[int, int, int], np.ndarray],
) -> dict[tuple[int, int, int], np.ndarray]:
    full_by_r: dict[tuple[int, int, int], np.ndarray] = {}
    for r_key, upper_dense in dense_by_r.items():
        partner_key = tuple(int(-value) for value in r_key)
        partner_upper = dense_by_r.get(partner_key)
        full_dense = np.asarray(upper_dense, dtype=complex).copy()
        if partner_upper is None:
            full_by_r[r_key] = full_dense
            continue
        nrow = full_dense.shape[0]
        lower_indices = np.tril_indices(nrow, k=-1)
        full_dense[lower_indices] = np.conj(np.asarray(partner_upper, dtype=complex).T[lower_indices])
        full_by_r[r_key] = full_dense
    return full_by_r


def _dense_blocks_by_r_with_optional_full_reconstruction(
    xr,
    full_matrix_from_hermitian: bool = False,
) -> dict[tuple[int, int, int], np.ndarray]:
    dense_by_r = _dense_blocks_by_r(xr)
    if not bool(full_matrix_from_hermitian):
        return dense_by_r
    return _full_dense_blocks_by_r_from_hermitian_partners(dense_by_r)


def _atom_slice(metadata, atom_index: int) -> slice:
    offset, dim = metadata.atom_ranges[int(atom_index)]
    start = offset * metadata.spin_factor
    stop = start + dim * metadata.spin_factor
    return slice(start, stop)


def _local_rotation(source_metadata, atom_index: int, xyz_axis_transform_cartesian: np.ndarray) -> np.ndarray:
    return build_atom_local_rotation(
        source_metadata,
        atom_index,
        np.asarray(xyz_axis_transform_cartesian, dtype=float),
        passive_basis=True,
    )


def _assemble_target_dense_blocks(
    source_xr,
    source_metadata,
    target_metadata,
    atom_mapping: list[dict],
    lattice_transform_fractional: np.ndarray,
    xyz_axis_transform_cartesian: np.ndarray,
    full_matrix_from_hermitian: bool = False,
):
    orbital_rotations = {
        item["old_atom"]: _local_rotation(source_metadata, item["old_atom"], xyz_axis_transform_cartesian)
        for item in atom_mapping
    }
    atom_mapping_by_old = {item["old_atom"]: item for item in atom_mapping}

    target_r_blocks: dict[tuple[int, int, int], np.ndarray] = {}
    mapping_lines: list[str] = []

    source_dense_upper_by_r = _dense_blocks_by_r(source_xr)
    if bool(full_matrix_from_hermitian):
        source_dense_by_r = _full_dense_blocks_by_r_from_hermitian_partners(source_dense_upper_by_r)
    else:
        source_dense_by_r = source_dense_upper_by_r

    for r_old in sorted(source_dense_by_r.keys()):
        dense = np.asarray(source_dense_by_r[r_old], dtype=complex)
        partner_key = tuple(int(-value) for value in r_old)
        pair_weight = 0.5 if bool(full_matrix_from_hermitian) and (partner_key in source_dense_upper_by_r) else 1.0
        for map_a in atom_mapping:
            old_a = int(map_a["old_atom"])
            source_slice_a = _atom_slice(source_metadata, old_a)
            d_a = orbital_rotations[old_a]
            target_a = int(map_a["new_atom"])
            target_slice_a = _atom_slice(target_metadata, target_a)

            for old_b, map_b in atom_mapping_by_old.items():
                source_slice_b = _atom_slice(source_metadata, old_b)
                d_b = orbital_rotations[old_b]
                target_b = int(map_b["new_atom"])
                target_slice_b = _atom_slice(target_metadata, target_b)

                local_block = dense[source_slice_a, source_slice_b]
                if not np.any(np.abs(local_block) > 1.0e-14):
                    continue

                r_new = map_target_r_vector(
                    lattice_transform_fractional,
                    r_old,
                    np.asarray(map_a["shift"], dtype=int),
                    np.asarray(map_b["shift"], dtype=int),
                )
                key = tuple(int(value) for value in r_new.tolist())
                target_dense = target_r_blocks.setdefault(
                    key,
                    np.zeros((target_metadata.basis_num, target_metadata.basis_num), dtype=complex),
                )
                rotated = pair_weight * (d_a.conj().T @ local_block @ d_b)
                target_dense[target_slice_a, target_slice_b] += rotated
                mapping_lines.append(
                    "R %4d %4d %4d atom%d ------> R %4d %4d %4d atom%d\n"
                    % (
                        int(r_old[0]),
                        int(r_old[1]),
                        int(r_old[2]),
                        old_a + 1,
                        int(r_new[0]),
                        int(r_new[1]),
                        int(r_new[2]),
                        target_a + 1,
                    )
                )

    return target_r_blocks, mapping_lines


def _unit_scale_from_hr_unit(hr_unit: str) -> float:
    if hr_unit == "eV":
        return 1.0
    if hr_unit == "Ry":
        from pyatb.constants import Ry_to_eV

        return float(Ry_to_eV)
    raise ValueError(f"Unsupported HR unit: {hr_unit}")


def _write_abacus_sparse_xr(path: Path, matrices_by_r: dict, basis_num: int, nspin: int, unit_scale: float = 1.0) -> None:
    r_keys = sorted(matrices_by_r.keys())
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"Matrix Dimension of XR {basis_num}\n")
        handle.write(f"R Number {len(r_keys)}\n")

        for r_key in r_keys:
            dense = np.asarray(matrices_by_r[r_key], dtype=complex) / float(unit_scale)
            upper = np.triu(dense)
            if int(nspin) != 4:
                if np.max(np.abs(upper.imag)) > 1.0e-8:
                    raise ValueError("Non-spinor XR writer received complex values beyond tolerance.")
                upper = upper.real
            csr = csr_matrix(upper)
            handle.write(f"{r_key[0]} {r_key[1]} {r_key[2]} {csr.nnz}\n")
            if csr.nnz == 0:
                continue
            if int(nspin) == 4:
                handle.write(" ".join(f"({value.real:.16e},{value.imag:.16e})" for value in csr.data) + "\n")
            else:
                handle.write(" ".join(f"{float(value):.16e}" for value in csr.data) + "\n")
            handle.write(" ".join(str(int(value)) for value in csr.indices) + "\n")
            handle.write(" ".join(str(int(value)) for value in csr.indptr) + "\n")


def _synchronize_block_keys(
    hr_blocks: dict[tuple[int, int, int], np.ndarray],
    sr_blocks: dict[tuple[int, int, int], np.ndarray],
    basis_num: int,
) -> tuple[dict[tuple[int, int, int], np.ndarray], dict[tuple[int, int, int], np.ndarray]]:
    all_keys = sorted(set(hr_blocks) | set(sr_blocks))
    for key in all_keys:
        if key not in hr_blocks:
            hr_blocks[key] = np.zeros((basis_num, basis_num), dtype=complex)
        if key not in sr_blocks:
            sr_blocks[key] = np.zeros((basis_num, basis_num), dtype=complex)
    return hr_blocks, sr_blocks


def canonicalize_abacus_hs(
    tb,
    target_stru_path,
    hr_route,
    sr_route,
    hr_unit,
    atom_mapping,
    lattice_new,
    lattice_transform_fractional,
    xyz_axis_transform_cartesian,
    output_hr_path,
    output_sr_path,
    mapping_output_path=None,
    full_matrix_from_hermitian: bool = True,
):
    source_metadata = extract_abacus_basis_metadata(tb)
    target_metadata = _build_metadata_from_stru(Path(target_stru_path), np.asarray(lattice_new, dtype=float), int(tb.nspin))
    if atom_mapping is None:
        atom_mapping = build_standardization_atom_mapping(source_metadata, target_metadata)
    else:
        atom_mapping = [
            {
                "old_atom": int(item["old_atom"]),
                "new_atom": int(item["new_atom"]),
                "shift": np.asarray(item["shift"], dtype=int),
                "species": str(item.get("species", source_metadata.species_by_atom[int(item["old_atom"])])),
            }
            for item in atom_mapping
        ]

    source_hr = abacus_readHR(int(tb.nspin), str(hr_route), hr_unit)
    source_sr = abacus_readSR(int(tb.nspin), str(sr_route))

    hr_blocks, mapping_lines = _assemble_target_dense_blocks(
        source_hr,
        source_metadata,
        target_metadata,
        atom_mapping,
        np.asarray(lattice_transform_fractional, dtype=float),
        np.asarray(xyz_axis_transform_cartesian, dtype=float),
        full_matrix_from_hermitian=full_matrix_from_hermitian,
    )
    sr_blocks, _ = _assemble_target_dense_blocks(
        source_sr,
        source_metadata,
        target_metadata,
        atom_mapping,
        np.asarray(lattice_transform_fractional, dtype=float),
        np.asarray(xyz_axis_transform_cartesian, dtype=float),
        full_matrix_from_hermitian=full_matrix_from_hermitian,
    )
    hr_blocks, sr_blocks = _synchronize_block_keys(hr_blocks, sr_blocks, int(target_metadata.basis_num))

    _write_abacus_sparse_xr(
        Path(output_hr_path),
        hr_blocks,
        int(target_metadata.basis_num),
        int(tb.nspin),
        unit_scale=_unit_scale_from_hr_unit(hr_unit),
    )
    _write_abacus_sparse_xr(
        Path(output_sr_path),
        sr_blocks,
        int(target_metadata.basis_num),
        int(tb.nspin),
        unit_scale=1.0,
    )

    if mapping_output_path is not None:
        Path(mapping_output_path).write_text("".join(mapping_lines), encoding="utf-8")

    canonical_hr = abacus_readHR(int(tb.nspin), str(output_hr_path), hr_unit)
    canonical_sr = abacus_readSR(int(tb.nspin), str(output_sr_path))

    return {
        "hr": canonical_hr,
        "sr": canonical_sr,
        "atom_mapping": atom_mapping,
        "target_basis_num": int(target_metadata.basis_num),
        "r_block_mapping_lines": mapping_lines,
        "full_matrix_from_hermitian": bool(full_matrix_from_hermitian),
    }
