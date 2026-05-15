# PYATB 特征标流程总结（结构标准化 / 数据对称化 / 群表示）

## 0. 范围与目标
本文总结当前 `pyatb-main/src/pyatb/symmetry` 中 CHARACTER 流程的三块核心能力，并给出公式与代码实现映射：

1. 结构标准化与 HS（HR/SR）标准化
2. 数据对称化（covariance constraint）
3. 群表示与特征标（小群构建、特征标计算、不可约表示标识）

文中的符号与代码变量保持一致：
- 晶格行向量矩阵：`A`（每行为 `a1,a2,a3`）
- 晶格整数映射：`M`, `B`
- 空间群操作：`g = {R|τ}`
- 原子映射：`a -> a'`，胞移 `w_a`（整数量）
- 轨道/自旋局域表示：`D_a`

---

## 1. 结构标准化与 HS 标准化

### 1.1 三个结构与两步映射
当前实现使用三层结构关系（对应你之前确认的逻辑）：

- `stru1`：输入原始结构（可能是超胞、非标准轴）
- `stru2`：`spglib.standardize_cell(..., to_primitive=True, no_idealize=True)` 得到的 primitive（与 `stru1` 尽量同笛卡尔系）
- `stru3`：`spglib.standardize_cell(..., to_primitive=True, no_idealize=False)` 得到的标准 primitive

代码入口：
- `SymmStructureAnalyzer.analyze_nonmagnetic` in `symm_stru.py`

核心矩阵：
- `M12 = round(A1 * A2^{-1})`
- `Q23`：通过 Procrustes 拟合得到 `A2 @ Q23 ~= A3`
- `B23 = round((A2 @ Q23) * A3^{-1})`
- `B13 = M12 @ B23 = round((A1 @ Q23) * A3^{-1})`

实现位置：
- `_fit_row_rotation`, `_round_integer_matrix`, `_compose_two_stage_mapping`

### 1.2 原子映射与 R 映射
对每个原子 `a`，求 `stru1 -> stru3` 的映射：
- 原子索引映射 `π(a)`
- 整数胞移 `w_a`

对于原子对 `(R, a, b)`，目标索引为：

\[
R' = R B_{13} + w_b - w_a
\]

这正是 `map_target_r_vector` + `_compose_two_stage_mapping` + `canonicalize_abacus_hs` 中实际使用的映射规则。

### 1.3 HS 块旋转协变关系
对每个原子对块，当前 HS 标准化使用的变换为：

\[
H'_{\pi(a)\pi(b)}(R') = D_a^{\dagger} \, H_{ab}(R) \, D_b
\]
\[
S'_{\pi(a)\pi(b)}(R') = D_a^{\dagger} \, S_{ab}(R) \, D_b
\]

对应实现：
- `hs_standardize.py::_assemble_target_dense_blocks`
- 代码形式：`rotated = d_a.conj().T @ local_block @ d_b`

说明：
- `D_a` 来自 `build_atom_local_rotation(..., passive_basis=True)`，是按被动坐标变换约定构造。
- `xyz_axis_transform_cartesian` 在 `symm_stru.py` 中传入 `Q23^T`，使本地 `D` 构造与被动约定一致。

### 1.4 下三角补齐与 Hermitian 伙伴
ABACUS 稀疏 XR 文件默认存上三角。为了严格做块映射与协变性验证，实现支持：
- 用 `R` 与 `-R` 伙伴补齐完整 dense 块（含下三角）
- 再做旋转与映射

关键函数：
- `_full_dense_blocks_by_r_from_hermitian_partners`
- `_dense_blocks_by_r_with_optional_full_reconstruction`

这一步是此前排查“映射看似错但本质是矩阵不完整”问题的关键修复点。

### 1.5 输出与主流程接管
若结构需要标准化（晶格变或非纯置换）：
- 生成 `STRU-symm`（放当前工作目录）
- 生成 `data-HR-sparse_SPIN0-symm.csr` 与 `data-SR-sparse_SPIN0-symm.csr`（放 `Out/CHARACTER`）
- 主流程切换到标准化后的 TB 对象继续算特征标

入口：
- `character.py::calculate_character`
- 调用 `canonicalize_abacus_hs` 后重建 `TBModel`

---

## 2. 数据对称化（Covariance Constraint）

### 2.1 目标
在结构与映射固定后，对数值噪声较大的 HS 数据做群平均约束，使其更接近群协变形式，减少特征标判定中的 `??`。

开关与参数：
- `data_symmetrize`（0/1）
- `data_symm_target_max_abs_ry`（每操作收敛阈值，默认 `1e-8`）
- `data_symm_max_iter_per_operation`（每操作最多迭代次数，默认 5）
- `data_symm_nonzero_block_tol`（非零块判定阈值）

入口：
- `character.py::calculate_character`
- `data_covariance_constraint.py::sequential_symmetrize_hs`

### 2.2 单个对称操作的变换
对 `g={R_g|τ_g}`，在当前实现下（主动约定）对块变换为：

\[
R' = R_g R + (\delta_b - \delta_a)
\]
\[
T_g(H)_{a'b'}(R') = D_a \, H_{ab}(R) \, D_b^{\dagger}
\]

其中 `a->a'` 和 `\delta_a` 来自 `find_atom_mapping`。

对应实现：
- `_prepare_operation_context`
- `transform_blocks_with_context`

### 2.3 迭代平均策略
每个操作最多迭代 `N` 次（默认 5）：

\[
H \leftarrow \frac{1}{2}\left(H + T_g(H)\right),\quad
S \leftarrow \frac{1}{2}\left(S + T_g(S)\right)
\]

迭代时只处理“非零原子对块”，阈值由 `nonzero_block_tol` 控制，以提高速度并避免零块污染。

实现细节：
- 活跃对索引：`build_active_pair_index`
- 仅对 touched pairs 平均：`average_block_sets_on_touched_pairs`
- 误差统计：`compare_block_sets_on_candidate_pairs`

### 2.4 最终统计与输出
流程结束后，做全操作扫描并输出：
- 每操作 max/mean/rms/rel_fro
- 全局最大误差元素位置 `(R,row,col)`
- 对称化前后 HR/SR 误差对比

实现：
- `self_covariance_statistics`
- `compare_block_sets_with_detail`
- `write_symmetrized_hs`

说明：
- 当前实现默认不做最后 Hermitize（你后续要求里已改成不做）。
- 参数 `hr_max_abs_threshold_ry` 当前仅在接口中保留，未参与强制中断逻辑。

---

## 3. 群表示与特征标

### 3.1 小群构建
给定 `k`，先从实空间操作中判定小群：

\[
\mathbf{k}R^{-1} - \mathbf{k} = \mathbf{G},\quad \mathbf{G}\in\mathbb{Z}^3
\]

实现：
- `symm_stru.py::_little_group_operation_indices`

然后结合 `kLittleGroups/kLG_*.data` 数据库做条目匹配（含 IRVSP 的变量 `u,v,w` 规则与 fallback 逻辑）：
- `KLittleGroupsDB.resolve_kpoint_from_star`

### 3.2 k 点约定
当前逻辑已经按你的最新要求改为：
- `k` 点直接使用输入最小原胞 `k` 点
- 不再做标准化过程中的 `k` 坐标变换

实现位置：
- `symm_stru.py::analyze_nonmagnetic` 中 `canonical_kpoints = np.asarray(kpoints_direct, dtype=float)`

### 3.3 构造 D_k(g)
对每个操作构造 Bloch 基组表示矩阵：

\[
D_k(g) = e^{-i2\pi k\cdot w_a}\, D_a
\]

- `w_a`：原子映射产生的胞移
- `D_a`：轨道旋转与自旋旋转直积（SOC 时）

实现：
- `Dk_matrix.py::build_dk_matrix`
- 轨道旋转：`atom_orbital_rotation/shell_rotation`
- 自旋旋转：`spin_half_matrix_from_cartesian_rotation`

### 3.4 退化子空间特征标
先对角化得到 `C_{nk}` 与 `E_{nk}`，并取 `S_k`。对退化子空间 `\mathcal{D}` 计算：

\[
\chi_{\mathcal{D}}(g)=\mathrm{Tr}\left[C_{\mathcal{D}}^{\dagger} S_k D_k(g) C_{\mathcal{D}}\right]
\]

实现：
- `character.py::_calculate_character_rows`
- `character_core.py::group_degenerate_bands`
- `character_core.py::calculate_subspace_characters`

### 3.5 不可约表示标识（assign）
对数据库给出的候选 irrep 字符表 `\chi_\alpha(g)`，在活跃操作子集上做组合匹配：

\[
\chi_{\text{calc}}(g) \approx \sum_{\alpha} n_\alpha \chi_\alpha(g),\quad n_\alpha\in\mathbb{N}
\]

当前实现是枚举 `max_terms`（默认 4）以内组合，最小化最大范数误差 `max|\Delta\chi|`。

实现：
- `character_core.py::assign_irrep_combination`
- 失败时返回 `??`

### 3.6 单值/双值表示与 Reality
- 数据库中 `raw_name` 以 `-` 前缀表示双值表示（double-valued）
- `spinful=True` 时优先筛选双值表示
- `reality` 字段直接来自数据库（通常可理解为 FS 指标类别：实/复/伪实）

实现：
- `_filter_irreps_by_spin`
- `symm_stru.py` 中输出 Reality 列

---

## 4. 第二部分：函数级说明（按模块）

> 这里聚焦“你前面反复调试过、且真正影响结果”的函数。

### 4.1 `symm_stru.py`

- `SymmStructureAnalyzer.analyze_nonmagnetic(...)`
  - CHARACTER 总入口（结构标准化 + 对称操作输出 + 小群匹配 + k 点记录）。
  - 产出 `analysis_result`，供后续 HS 标准化、数据对称化、特征标计算共同使用。

- `_standardize_nonmagnetic_cell(...)`
  - 调用 spglib 同时得到 `mapping_atoms(no_idealize)` 与 `std_atoms(no_idealize=False)`，建立两步映射链。

- `_build_atom_mapping(...)`
  - 在不同晶格基下求 atom 对应与整数胞移 `shift`。

- `_build_rotated_atom_mapping(...)`
  - 在 `Q23` 旋转后对 `mapping_atoms -> std_atoms` 做匈牙利匹配，避免原子顺序变化导致错配。

- `_compose_two_stage_mapping(...)`
  - 合成 `stru1->stru2` 与 `stru2->stru3`，得到最终 `stru1->stru3` 原子映射。

- `_fit_row_rotation(...)`
  - 解 `A_from @ Q ≈ A_to` 的行向量 Procrustes 旋转矩阵。

- `_write_standardized_stru(...)`
  - 保留赝势/轨道段与 species 顺序，写出 `STRU-symm`。

- `_build_symmetry_operations(...)`
  - 从 spglib 操作生成 `rotation/translation/cart_rotation/euler/spin/symbol/axis` 全信息。

- `_resolve_kpoint_records(...)`
  - 对每个 k 点求小群操作索引，并在 `kLittleGroups` 中解析对应高对称点条目。

- `_write_transformations/_write_symmetry_operations/_write_k_little_group_table`
  - 生成 `symmetry_character_report.txt` 的结构变换、小群表与操作表。

### 4.2 `hs_standardize.py`

- `canonicalize_abacus_hs(...)`
  - HS 标准化总入口：读取原 HS -> 映射+旋转 -> 写新 HS -> 回读为 `multiXR`。

- `_assemble_target_dense_blocks(...)`
  - 真正执行 `(R,a,b)` 到 `(R',a',b')` 的映射，并做 `D_a^† block D_b`。
  - 支持仅上三角/从 Hermitian 伙伴补齐完整矩阵两种路径。

- `map_target_r_vector(...)`
  - 实现 `R' = R B + shift_b - shift_a` 并做整数性检查。

- `_full_dense_blocks_by_r_from_hermitian_partners(...)`
  - 用 `(-R)` 伙伴恢复当前 `R` 的下三角，保证块运算完整。

- `_write_abacus_sparse_xr(...)`
  - 将 dense 块重新写回 ABACUS CSR 格式（HR/SR）。

### 4.3 `data_covariance_constraint.py`

- `load_abacus_hs_blocks(...)`
  - 读 STRU/HR/SR，并恢复 full dense `R->matrix` 字典。

- `get_symmetry_operations_from_metadata(...)`
  - 从 metadata 调 spglib 得到协变性扫描所需全部操作。

- `_prepare_operation_context(...)`
  - 预计算某个操作的 atom 映射、胞移、局域 `D`、pair 缓存，避免重复构造。

- `transform_blocks_with_context(...)`
  - 对一个操作执行 `T_g(H)` 或 `T_g(S)`。
  - 支持只在 active pairs 上运行（性能优化关键）。

- `build_active_pair_index(...) / merge_active_pair_index(...)`
  - 建立并更新“非零原子对索引”，减少无意义零块处理。

- `sequential_symmetrize_hs(...)`
  - 按操作顺序执行“最多 5 次局部迭代平均”，并记录每步误差历史。

- `self_covariance_statistics(...)`
  - 统计对称化前后协变误差（max/mean/rms/rel_fro）。

- `write_symmetrized_hs(...)`
  - 将对称化后 HR/SR 落盘，供后续特征标计算直接使用。

### 4.4 `Dk_matrix.py`

- `extract_abacus_basis_metadata(tb)`
  - 从 TB/STRU 抽取原子、壳层、基函数偏移、spin_factor 等元数据。

- `find_atom_mapping(metadata, operation, tol)`
  - 对单个群操作计算原子映射与胞移 `cell_shift`。

- `shell_rotation(...)`
  - 在 ABACUS 实球谐基下构造给定 `l` 的轨道旋转块，含反演宇称因子 `(-1)^l`。

- `build_atom_local_rotation(...)`
  - 组装原子局域旋转；SOC 时返回 `orbital ⊗ spin`。

- `build_dk_matrix(tb, k_direct, operation, include_cell_shift_phase=True)`
  - 构造 `D_k(g)`（含 `exp(-i2πk·shift)` 相位和局域旋转矩阵）。

### 4.5 `character_core.py`

- `group_degenerate_bands(energies, tol)`
  - 能量容差分组，形成退化子空间。

- `calculate_subspace_characters(...)`
  - 按 `Tr(C† S D C)` 计算每个操作的子空间特征标。

- `assign_irrep_combination(...)`
  - 在小群字符表上做组合匹配，输出 `GM8`、`L3 + L4` 等标签。

### 4.6 `k_little_groups.py`

- `KLittleGroupsDB.load(path)`
  - 解析 `kLG_*.data`（操作、k 点、irrep 字符表、phase kind、reality）。

- `resolve_kpoint_from_star(...)`
  - 按 k 星、little-group size、操作集匹配条目；含变量点与 fallback 规则。

- `irrep_table_characters(resolution, irrep)`
  - 计算 k 相关的字符值：
    - phase kind 1：直接取字符
    - phase kind 2：乘 `exp(iπ coeff·k_conv)`

---

## 5. 当前流程的工程结论

1. 结构标准化 + HS 标准化路径已可处理“超胞 -> primitive + 轴旋转”的一般情形。  
2. 数据对称化模块能够显著压缩协变误差，并输出可追溯误差历史。  
3. 群表示链路（小群 -> `D_k(g)` -> 子空间特征标 -> irrep 组合）闭环已形成，且可与 IRVSP 风格输出对齐。  
4. 最新逻辑中，`k` 点不再跟随结构标准化变换，而是直接采用输入最小原胞 `k` 点。

