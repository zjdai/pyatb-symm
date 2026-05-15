# 2026-04-28 跨结构真实空间协变性验证记录

## 目标

重新验证下列两个结构之间的 HS 协变性：

- 源结构：`test_workspace/test-abacus-2/soc`
- 目标结构：`test_workspace/test-abacus-4`

这次严格采用以下规则：

- 先求能够映射实际笛卡尔结构的真实空间旋转矩阵；
- 用这个真实空间旋转矩阵构造轨道+自旋 `D` 矩阵；
- 最后在“相同 `R`、相同原子对”的前提下验证协变性，不再额外引入 `R -> R'` 重标记层。

## 调试中的关键错误

### 错误尝试 1：只用原子点云拟合旋转

- 我先只用原子笛卡尔坐标做刚体拟合，原子位置误差看起来很小；
- 但晶格矢量误差非常大。

根因是：

- Bi2Se3 本身有点群对称；
- 只看原子点云时，会落到一个“把原子位置也能拟合好、但不是整个结构真实取向”的假旋转上；
- 这个旋转不能用来构造正确的 `D` 矩阵。

### 修正后的正确做法

- 联合使用以下真实空间信息做拟合：
  - 晶格矢量
  - 非零原子笛卡尔坐标
- 先解行向量形式的正交 Procrustes 问题：
  - `cart_source @ R_row ~= cart_target`
- 再转换为 `D` 矩阵代码实际使用的列向量旋转矩阵：
  - `R_col = R_row^T`

最终构造轨道+自旋 `D` 矩阵时使用的就是这个 `R_col`。

## 输出文件

验证报告输出在：

- `/home/zjdai/file-test/pyatb_symm/test_workspace/test-abacus-4/test2_to_test4_realspace_rotation_covariance_report.json`

## 真实空间旋转的几何质量

这次用于构造 `D` 的真实空间旋转矩阵满足：

- `det = 0.9999999999999999`
- `max_abs_lattice_error = 9.351880272845392e-05`
- `fro_lattice_error = 1.7146014411770493e-04`
- `max_abs_atom_error = 2.0869794914801787e-04`
- `max_atom_error_norm = 2.2817837937748366e-04`

也就是说，这个旋转矩阵同时与晶格和 wrap 后的原子结构相容。

## HS 协变性结果

### 源结构 -> 目标结构

验证公式：

```text
H_target(R) ?= D(R_cart) H_source(R) D(R_cart)^dagger
S_target(R) ?= D(R_cart) S_source(R) D(R_cart)^dagger
```

结果：

- `HR rel_fro_global = 1.0759581974675406e-05`
- `HR max_abs_diff = 1.819333167396575e-05`
- `SR rel_fro_global = 1.1825975213669202e-05`
- `SR max_abs_diff = 4.597218759620075e-06`
- `common_R_count = 165`
- `only_reference_R_count = 0`
- `only_predicted_R_count = 0`

最坏 HR 元素：

- `R = [0, 0, 0]`
- `(row, col) = (74, 74)`
- `abs_diff = 1.819333167396575e-05`

最坏 SR 元素：

- `R = [0, 0, -1]`
- `(row, col) = (11, 155)`
- `abs_diff = 4.597218759620075e-06`

### 目标结构 -> 源结构

验证公式：

```text
H_source(R) ?= D(R_cart)^dagger H_target(R) D(R_cart)
S_source(R) ?= D(R_cart)^dagger S_target(R) D(R_cart)
```

结果：

- `HR rel_fro_global = 1.0759616399590502e-05`
- `HR max_abs_diff = 1.8220270389002202e-05`
- `SR rel_fro_global = 1.1826031612177153e-05`
- `SR max_abs_diff = 4.188333590348509e-06`
- `common_R_count = 165`
- `only_reference_R_count = 0`
- `only_predicted_R_count = 0`

反向结果和正向结果数值上一致。

## 错误旋转的对照结果

作为对照，我故意用“转置后错误的旋转矩阵”再跑同样的 `source -> target` 验证，结果明显变坏：

- `HR rel_fro_global = 0.2810372368264005`
- `HR max_abs_diff = 0.30521995408348407`
- `SR rel_fro_global = 0.6279282790459534`
- `SR max_abs_diff = 0.35113247180691337`

这说明之前的大误差，确实就是因为构造 `D` 矩阵时用了错误的笛卡尔旋转矩阵。

## 结论

- 这次已经按照“真实空间旋转 -> 轨道+自旋 `D` -> 同 `R` 协变性验证”的正确流程，重新完成了 `test-abacus-2/soc -> test-abacus-4` 的验证。
- `D` 矩阵必须由真实的笛卡尔旋转矩阵构造，不能把解释错误的晶格基变换矩阵直接塞进去。
- 修正后，跨结构 HS 协变性误差从原来的 `1e-1 ~ 1e0` 量级下降到 `1e-5` 量级。

## 2026-04-28 主流程补齐接入后的复验（2 -> 4，含旋转）

为与主流程“先补齐每个 `R` 的完整矩阵”保持一致，新增一次复验：

- 源结构采用：`test_workspace/test-abacus-4/verification_tmp/STRU_S2_abs`
  - 与 `test-abacus-2/soc` 同一 5 原子原胞，但轨道路径已改成绝对路径，便于直接读取轨道信息。
- 目标结构采用：`test_workspace/test-abacus-4/verification_tmp/STRU_S1_abs`
  - 与 `test-abacus-4` 同取向原胞，轨道路径同样为绝对路径。
- HS 数据仍使用：
  - 源：`test-abacus-2/soc/OUT.ABACUS/data-HR/SR-sparse_SPIN0.csr`
  - 目标：`test-abacus-4/OUT.ABACUS/data-HR/SR-sparse_SPIN0.csr`

输出文件：

- `test_workspace/test-abacus-4/test2_to_test4_realspace_rotation_covariance_report_rerun_20260428_full_matrix_integrated.json`

结果：

- `full_matrix_from_hermitian = true`
- `common_R_count = 165`
- `atom_mapping_max_error = 4.217539478323928e-06`
- `HR rel_fro_global = 1.0759581974911338e-05`
- `SR rel_fro_global = 1.1825975213684855e-05`
- `HR max_abs_diff = 1.8193331674631885e-05`
- `SR max_abs_diff = 4.597218761509189e-06`

结论：在“补齐每个 `R` 的完整矩阵”已接入主流程后，`test-abacus-2 -> test-abacus-4` 的旋转协变性依旧维持 `1e-5` 量级，通过。

## 2026-04-28 `test-abacus-3/pyatb` 主流程修正与复验（补齐+旋转）

### 根因定位

1. `symm_stru.py::_build_atom_mapping()` 过早对 `old_pos_in_std_basis` 做了 wrap，导致超胞到原胞映射中的整格矢 `shift` 被抹成 `0`。  
2. `hs_standardize.py::_assemble_target_dense_blocks()` 在启用 `full_matrix_from_hermitian=True` 且 `R/-R` 成对存在时，旋转后块被双计数（幅值约放大 2 倍）。

### 代码修正

- `pyatb-main/src/pyatb/symmetry/symm_stru.py`
  - 在 `_build_atom_mapping()` 中保留未 wrap 的 `old_pos_in_std_basis = cart_old @ inv_std_lattice`，只在残差判断时做周期归约。
- `pyatb-main/src/pyatb/symmetry/hs_standardize.py`
  - 在 `_assemble_target_dense_blocks()` 中：
    - 区分原始上三角块与补齐后的全矩阵块；
    - 若 `full_matrix_from_hermitian=True` 且 `-R` 存在，对该 `R` 的旋转贡献乘 `0.5`，消除双计数。
- `pyatb-main/tests/test_character_module.py`
  - 更新超胞->原胞映射测试期望：第二个原子的 `shift` 从 `[0,0,0]` 更正为 `[1,0,0]`。

### 回归测试

命令：

```bash
cd /home/zjdai/file-test/pyatb_symm/pyatb-main
pytest -q tests/test_hs_standardize.py tests/test_hs_covariance.py tests/test_character_module.py
```

结果：`40 passed, 4 warnings`。

### `test-abacus-3/pyatb` 复跑

命令：

```bash
cd /home/zjdai/file-test/pyatb_symm/test_workspace/test-abacus-3/pyatb
pyatb > rerun_after_pair_weight_fix_20260428.log 2>&1
```

`Out/CHARACTER/standardization_summary.json` 中 `atom_mapping.shift` 已恢复为非零整格矢（例如 `[-1,-1,0]`、`[-2,-1,-1]` 等）。

### HS 协变性复验（`HS-symm` vs `test-abacus-2/soc` 参考）

输出文件：

- `test_workspace/test-abacus-3/pyatb/integrated_covariance_3pyatb_to_2soc_full_report_after_pair_weight_fix_20260428.json`

关键指标：

- `HR rel_fro_diff = 8.923393464487352e-05`
- `SR rel_fro_diff = 1.1664759139084118e-05`
- `HR max_abs_diff = 0.0053413193317481955`
- `SR max_abs_diff = 4.131167610664163e-06`
- `target_missing_count = 0`（HR/SR）

结论：`test-abacus-3/pyatb` 主流程修正后，标准化结构与 `HS-symm` 对参考结果恢复到 `1e-4 ~ 1e-5` 量级协变误差，通过。
