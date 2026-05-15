# 2026-04-28 Dk 边界原子 / HS 验证记录

## 本轮目标

- 把 ABACUS 边界点规范 `0 1 0 == 0 0 0` 正式并入仓库里的对称映射路径。
- 统一局域轨道+自旋旋转矩阵的构造，使 `build_dk_matrix()` 和 `hs_standardize.py` 使用同一套 D 矩阵约定。
- 重新执行单元测试和真实数据的 HS 自对称协变性验证。

## 代码修改

- `pyatb-main/src/pyatb/symmetry/Dk_matrix.py`
  - 新增 `_canonicalize_fractional_coordinates()`。
  - 在 `extract_abacus_basis_metadata()` 中把分数坐标规范到 `[0, 1)`。
  - 在 `find_atom_mapping()` 中先规范原子分数坐标，再做原子映射。
  - 新增共享的 `build_atom_local_rotation()`，统一局域轨道+自旋表示块构造。
  - `build_dk_matrix()` 改为调用共享的局域旋转构造函数。

- `pyatb-main/src/pyatb/symmetry/hs_standardize.py`
  - `_local_rotation()` 改为统一走 `build_atom_local_rotation(..., passive_basis=True)`。

- `pyatb-main/tests/test_character_module.py`
  - 新增 `Dk_matrix.find_atom_mapping()` 的边界等价点回归测试。
  - 新增 `build_atom_local_rotation()` 主动/被动约定回归测试。
  - 把 `hs_standardize._local_rotation()` 的测试同步到新的共享实现路径。

## 修正的根因

- 仓库中的 `Dk_matrix.find_atom_mapping()` 路径，在收到原始边界分数坐标时，对 `0 1 0` 这种与 `0 0 0` 等价的原子仍然不够稳健。
- 这会产生假的晶格平移，例如 `(0, -1, 1)`，从而污染后续的相位因子和原子映射逻辑。
- 同时，局域 D 矩阵的构造此前分散在多个文件中。本轮把这条路径合并成一个共享实现，避免再次分叉。

## 单元测试

### 定点回归测试

运行：

```bash
conda run -n symm bash -lc 'cd /home/zjdai/file-test/pyatb_symm/pyatb-main && pytest -q tests/test_character_module.py -k "dk_find_atom_mapping_canonicalizes_boundary_equivalent_positions or build_atom_local_rotation_supports_active_and_passive_conventions"'
```

结果：

- `2 passed`

### 相关对称性 / HS 子集测试

运行：

```bash
conda run -n symm bash -lc 'cd /home/zjdai/file-test/pyatb_symm/pyatb-main && pytest -q tests/test_hs_standardize.py tests/test_character_module.py -k "build_atom_mapping_wraps_boundary_equivalent_positions_without_extra_shift or dk_find_atom_mapping_canonicalizes_boundary_equivalent_positions or build_atom_local_rotation_supports_active_and_passive_conventions or local_rotation_uses_inverse_cartesian_transform or local_rotation_includes_spin_rotation_for_soc or build_dk_matrix_returns_identity_for_single_s_orbital or build_dk_matrix_supports_soc_interleaved_spin_basis or canonicalize_fractional_coordinates_wraps_boundary_points or standardize_sparse_blocks_uses_passive_basis_rotation_order"'
```

结果：

- `9 passed, 25 deselected`

### 两个相关测试文件全集

运行：

```bash
conda run -n symm bash -lc 'cd /home/zjdai/file-test/pyatb_symm/pyatb-main && pytest -q tests/test_hs_standardize.py tests/test_character_module.py'
```

结果：

- `34 passed, 4 warnings`

## 真实数据 HS 自对称协变性验证

运行：

```bash
conda run -n symm python /tmp/validate_self_symmetry_axis_angle_canon.py
```

### `test_workspace/test-abacus-2/soc`

- `all_HR_passed = True`
- `all_SR_passed = True`
- `max_HR_rel_fro = 7.348293095132547e-09`
- `max_HR_abs = 8.502953857211339e-09`
- `max_SR_rel_fro = 1.408314006298417e-08`
- `max_SR_abs = 7.105726340661533e-09`

### `test_workspace/test-abacus-4`

- `all_HR_passed = True`
- `all_SR_passed = True`
- `max_HR_rel_fro = 1.0461610608790815e-08`
- `max_HR_abs = 1.6186617213421666e-08`
- `max_SR_rel_fro = 1.9900195723509407e-08`
- `max_SR_abs = 1.1202022191791894e-08`

## 结论

- 仓库代码现在已经正式包含 ABACUS 边界原子规范化行为。
- k 空间表示与 HS 标准化现在共享同一条局域 D 矩阵构造路径。
- 修正边界点处理后，两套真实数据的 HS 自对称协变性都继续通过，误差保持在 `1e-8` 量级。
