# 2026-04-28 超胞到原胞无旋转协变性验证记录

## 目标

验证以下两套 ABACUS 数据在“同一笛卡尔坐标系、无需任何轨道/自旋旋转”的前提下，是否只靠：

- 整数超胞矩阵 `M`
- 超胞原子到原胞原子的映射
- `R` 指标重标记

就能满足 HS 协变性。

本次验证对象是：

- 超胞源数据：
  - `test_workspace/test-abacus-3/OUT.ABACUS/data-HR-sparse_SPIN0.csr`
  - `test_workspace/test-abacus-3/OUT.ABACUS/data-SR-sparse_SPIN0.csr`
- 原胞目标数据：
  - `test_workspace/test-abacus-4/OUT.ABACUS/data-HR-sparse_SPIN0.csr`
  - `test_workspace/test-abacus-4/OUT.ABACUS/data-SR-sparse_SPIN0.csr`

用于读取轨道基组信息的结构文件是：

- 源结构：`test_workspace/test-abacus-3/pyatb/STRU`
- 目标结构：`test_workspace/test-abacus-4/verification_tmp/STRU_S1_abs`

原因是这两个文件的轨道路径可直接解析。

## 关键结论

### 1. 几何映射是正确的

推断得到的整数超胞矩阵为：

```text
[[ 0, -1,  0],
 [-1,  0,  1],
 [-1,  0, -1]]
```

这与生成 `test-abacus-3` 时使用的随机扩胞矩阵一致。

原子对矢量映射检查结果：

- `pair_vector_error_max_abs = 7.973655637982802e-07`

说明 `(R, atom_a, atom_b)` 到 `(R', atom_a', atom_b')` 的几何映射在数值上是成立的。

### 2. supercell -> primitive 验证时不能把重复来源块直接累加

之前 `canonicalize_abacus_hs()` 的思路是：

- 把所有来源块映射到目标块；
- 如果多个来源块落到同一个目标键，就直接求和。

这对“原胞重排/标准化”是合理的，但对“超胞还原成原胞”的验证不对。

原因是：

- 超胞内部多个原子拷贝会映射到同一个原胞原子；
- 同一个目标块会收到多个等价来源样本；
- 这些样本应当彼此一致，而不是相加。

本次新增的验证模块改为：

- 保留每一个来源块样本；
- 逐个与目标块比较；
- 额外检查同一目标键下的重复样本是否彼此一致。

### 3. 重复来源样本彼此高度一致

样例结果：

- `HR duplicate_consistency_max_abs = 1.0000000502143058e-13`
- `SR duplicate_consistency_max_abs = 9.999999506879345e-17`

这说明：

- 原子映射是稳定的；
- `R` 重标记公式是稳定的；
- 几何等价的来源块在超胞数据内部彼此一致。

### 4. 但直接对目标原胞 HS/SR 比较时仍有显著误差

样例结果：

- `HR max_abs_diff = 0.33128893712493707`
- `HR rel_fro_diff = 0.0555324466596934`
- `SR max_abs_diff = 0.288381729`
- `SR rel_fro_diff = 0.17911222587872075`

因此这一步已经可以明确排除：

- 整数超胞矩阵推断错误；
- 原子映射错误；
- `R` 映射公式错误；
- 边界 wrap 导致的简单配对错误；
- 重复来源块被错误混合这一类程序错误。

后续如果继续查误差来源，重点应当放在：

- 超胞与原胞 DFT 数据本身的规范差异；
- ABACUS 在 supercell / primitive 情况下的 HS/SR 输出约定；
- 是否还存在额外的块规范、相位或基组约定差异。

## 代码改动

新增模块：

- `pyatb-main/src/pyatb/symmetry/hs_covariance.py`

核心接口：

- `infer_integer_lattice_transform()`
- `collect_no_rotation_block_samples_from_dense_blocks()`
- `validate_no_rotation_block_mapping_from_dense_blocks()`
- `validate_abacus_supercell_to_primitive_no_rotation()`

新增测试：

- `pyatb-main/tests/test_hs_covariance.py`

测试覆盖内容：

- 超胞到原胞原子映射允许多对一；
- 重复来源块不会被累加，而是逐样本验证。

## 测试命令

### 单元测试

```bash
pytest /home/zjdai/file-test/pyatb_symm/pyatb-main/tests/test_hs_standardize.py /home/zjdai/file-test/pyatb_symm/pyatb-main/tests/test_hs_covariance.py -q
```

结果：

```text
7 passed in 0.22s
```

### 样例验证

输出报告：

- `test_workspace/test-abacus-4/test3_to_test4_no_rotation_covariance_report.json`

这个 JSON 里包含：

- 超胞矩阵
- 原子映射
- HR/SR 的逐块误差
- 重复来源块一致性
- 实空间原子对矢量误差

## 当前结论

本轮代码已经把“无旋转 supercell -> primitive 映射验证”独立出来，并且确认：

- 映射逻辑本身是对的；
- 现有剩余误差不是简单的 `R`/原子配对错误。

如果下一步继续调试，应直接围绕 `test3_to_test4_no_rotation_covariance_report.json` 中的最坏块开展。

## 边界原子物理重写复查

后续又单独做了一次复查：

- 把 `test_workspace/test-abacus-4/verification_tmp/STRU_S1_abs` 中的边界原子
  - 从 `0.0000000000 1.0000000000 0.0000000000`
  - 物理改写为 `0.0000000000 0.0000000000 0.0000000000`
- 然后用完全相同的验证脚本重新生成报告。

重跑后结果与改写前完全一致：

- `HR max_abs_diff = 0.33128893712493707`
- `HR rel_fro_diff = 0.0555324466596934`
- `SR max_abs_diff = 0.288381729`
- `SR rel_fro_diff = 0.17911222587872075`

这说明：

- 这次无旋转验证内部已经正确处理了边界原子的 wrap；
- 单纯把目标结构文件文本上的 `1.0` 改写成 `0.0`，不会改变验证结果；
- 当前剩余误差来源不在这个边界原子标准化问题上。

## 严格使用顶层 STRU3 / STRU4 的实空间映射复查

按后续要求，又单独做了一轮更严格的验证。这一轮的约束是：

- 几何映射只读取顶层结构文件：
  - `test_workspace/test-abacus-3/STRU`
  - `test_workspace/test-abacus-4/STRU`
- 先对两套结构做 wrap，消除边界原子 `1 -> 0` 的歧义；
- 不做任何坐标轴旋转；
- 单原子映射按真实空间关系求解：
  - 对源结构 `(R, atom)` 先求真实空间位置 `r`
  - 对目标原胞每个 `R=0` 原子位置 `tau`
  - 求解 `(r - tau) = n @ A_target`
  - 其中 `n = (n1, n2, n3)` 就是目标结构中的 `R'`
- 再由单原子映射扩展到原子对 `(R, atom1, atom2)` 的 HS/SR 块验证。

为了让 top-level `STRU` 可以直接读轨道数据，这一轮另外写出了两份 wrap 后的临时绝对路径结构：

- `test_workspace/test-abacus-4/verification_tmp/STRU3_wrapped_abs`
- `test_workspace/test-abacus-4/verification_tmp/STRU4_wrapped_abs`

输出报告：

- `test_workspace/test-abacus-4/test3_to_test4_top_level_wrap_realspace_report.json`

### 结果

按“晶格矢量逐行存储”的 ABACUS 约定，得到：

```text
M =
[[ 0, -1,  0],
 [-1,  0,  1],
 [-1,  0, -1]]
```

数值检查：

- `max_abs(M_raw - round(M_raw)) = 7.263072743235455e-08`
- `max_abs(A_source - M @ A_target) = 1.2266936479932156e-06`

单原子映射误差：

- `single_atom_mapping_max_abs_cart_error = 1.5060593909765885e-06`

最坏单原子映射：

- `old_atom = 10`
- `new_atom = 5`
- `shift = (-1, -1, -1)`
- `max_abs_cart_error = 1.5060593909765885e-06`

HS/SR 协变性结果：

- `HR max_abs_diff = 0.33128893712493707`
- `HR rel_fro_diff = 0.0555324466596934`
- `SR max_abs_diff = 0.288381729`
- `SR rel_fro_diff = 0.17911222587872075`

这与前一轮使用结构副本做映射时的误差完全一致。

### 结论

这说明：

- 严格使用顶层 `STRU3/STRU4`，并按照“先 wrap，再按单原子真实空间方程求 `(R', atom')`”的方式建映射，得到的结果与之前一致；
- 这套 `M`、单原子映射、原子对映射在几何上是成立的；
- 当前 HS/SR 的剩余误差不是由：
  - 顶层结构没有参与映射，
  - 边界原子没有 wrap，
  - `M` 的行列约定写反，
  - 单原子 `(R, atom)` 到 `(R', atom')` 的配对错误
  这些问题导致的。

## 按单原子不变量重新验证

后续又按照修正后的不变量重新验证了一次：

```text
R @ A + tau_a = R' @ A' + tau'_a'
```

这里：

- `A` 是 `test-abacus-3/STRU` 的晶格矩阵；
- `A'` 是 `test-abacus-4/STRU` 的晶格矩阵；
- 晶格矢量按 ABACUS 约定逐行存储；
- `R' = R @ M + shift_a`；
- `tau_a = shift_a @ A' + tau'_a'` 是源原胞原子到目标原胞原子的 home-cell 映射关系。

这一轮输出报告：

- `test_workspace/test-abacus-4/test3_to_test4_single_atom_invariance_report.json`

### M 矩阵与 spglib 交叉检查

得到：

```text
M =
[[ 0, -1,  0],
 [-1,  0,  1],
 [-1,  0, -1]]
```

并且：

- `det(M) = 2`
- `volume_ratio_source_over_target = 2.0000004357334773`

`spglib` 交叉检查结果：

- `find_primitive` 成功；
- primitive 原子数 `= 5`；
- 源结构原子数 `= 10`；
- `source_to_primitive_atom_ratio = 2.0`
- `source_to_primitive_volume_ratio = 1.9999999999810305`

这里 `spglib` 只用于确认：

- `STRU3` 的确可以约成 5 原子的 primitive cell；
- 倍数关系和 `det(M)=2` 一致。

不直接拿 `spglib` 返回的 primitive basis 去定义 `M` 的具体方向，因为 `spglib` 可能选取另一组等价 primitive 基矢。

### 单原子不变量误差

这一轮显式检查了 109 个源 `R`，每个 `R` 上 10 个原子，共 1090 个单原子样本：

- `single_atom_relation_max_abs_diff = 3.629148395134507e-06`
- `single_atom_relation_sum_abs_diff = 0.0024472054923566217`

最坏样本：

- `source_R = (3, 1, 0)`
- `target_R = (-2, -4, 0)`
- `old_atom = 10`
- `new_atom = 5`
- `shift = (-1, -1, -1)`
- `max_abs_cart_diff = 3.629148395134507e-06`

也就是说，按修正后的单原子不变量，几何映射本身是成立的。

### 在此基础上的 HS/SR 结果

使用同一套单原子映射再推到原子对块，结果仍然是：

- `HR max_abs_diff = 0.33128893712493707`
- `HR rel_fro_diff = 0.0555324466596934`
- `SR max_abs_diff = 0.288381729`
- `SR rel_fro_diff = 0.17911222587872075`

### 最终结论

按单原子不变量

```text
R @ A + tau_a = R' @ A' + tau'_a'
```

显式重做验证之后，可以进一步确认：

- `M` 是对的；
- `R' = R @ M + shift_a` 是对的；
- 源单原子到目标单原子的实空间位置保持关系是对的；
- 由此推出的原子对几何映射是对的；
- 当前剩余的 `HR/SR ~ 1e-1` 量级误差不是几何映射定义错误。

## 暴力块匹配与几何映射对照

后续又额外做了一轮“非常直接”的块匹配分析：

- 对每一个源非零局域块 `(R, atom1, atom2)`；
- 在目标结构的**全部** `(R', atom1', atom2')` 块里搜索最小误差块；
- 误差函数定义为：

```text
combined_rel = ||dH||_F / ||H_src||_F + ||dS||_F / ||S_src||_F
```

输出文件：

- `test_workspace/test-abacus-4/block_bruteforce_vs_geometry_report.json`

分析脚本：

- `test_workspace/test-abacus-4/verification_tmp/bruteforce_hs_block_match.py`

### 总体结果

- 源非零块数：`1056`
- 目标全部块数：`4125`
- 暴力最优仍然等于几何映射：`986`
- 暴力最优等于几何映射的 Hermitian 对应键：`0`
- 暴力最优不等于几何映射：`70`

### 70 个不一致块的两类情况

#### 第一类：只是等价重表示

这类块共有 `36` 个。

它们的特点是：

- 暴力最优键和几何键不同；
- 但两者对应的目标原子对实空间矢量完全一致。

也就是说，这一类不是映射错了，而是同一个实空间原子对在目标 HS 里用了另一套等价 `(R', atom1', atom2')` 表示。

这 36 个块全部来自两组源原子对：

- `(atom1, atom2) = (7, 8)` 共 `18` 个
- `(atom1, atom2) = (6, 8)` 共 `18` 个

#### 第二类：真正不等价的块

这类块共有 `34` 个。

它们的特点是：

- 暴力最优键和几何键不同；
- 两者对应的目标原子对实空间矢量也不同；
- 因此不能解释为简单的等价重标记。

这 34 个块全部来自两组源原子对：

- `(atom1, atom2) = (2, 3)` 共 `18` 个
- `(atom1, atom2) = (7, 9)` 共 `16` 个

### 典型问题块

一个典型的 Bi-Bi 问题块：

- 源块：`(R, atom1, atom2) = ([1, 2, 0], 2, 3)`
- 几何映射：`([-1, -1, 2], 2, 1)`
- 暴力最优：`([-2, -1, 2], 1, 2)`
- 两者目标原子对矢量差：`2.0809890516506924`

它的误差对比是：

- 几何映射：`combined_rel = 2.0`
- 暴力最优：`combined_rel = 1.3740818927950933`

一个典型的 Se-Se 等价重表示块：

- 源块：`(R, atom1, atom2) = ([-1, -1, 1], 7, 8)`
- 几何映射：`([1, 2, -2], 5, 3)`
- 暴力最优：`([0, 1, -3], 3, 4)`

这两个目标键的原子对实空间矢量差只有：

- `3.55e-15`

因此它属于“等价重表示”，不是几何错误。

### 这一步说明了什么

暴力搜索的结果比之前更细：

- 大多数源块的最优目标块仍然就是几何映射块；
- 一部分 Se-Se 块的问题只是目标文件采用了另一种等价表示；
- 还剩一部分 Bi-Bi 和 Se-Se 块，暴力最优块和几何映射块既不相同，也不对应同一个实空间原子对；
- 这说明当前问题不仅仅是 `R`/原子索引映射，还有一部分块在目标 HS/SR 中确实更像别的块。

## 2026-04-28 补充诊断：问题实际出在 ABACUS 三角存储比较方式

在 `test_workspace/test-abacus-4/block_mapping_diagnostics_full.json` 和
`test_workspace/test-abacus-4/block_mapping_diagnostics_full.txt` 中，对前面 70 个暴力搜索不一致块做了进一步分解。

新增结论如下：

- 这 `70` 个不一致块全部满足同一个模式：`geometry_target` 落在目标矩阵的 `lower` 块，而暴力最优块落在 `upper` 块；
- 统计结果为：`orientation_counter = {'lower->upper': 70}`；
- 其中 `36` 个只是同一原子对矢量的等价重表示；
- 另 `34` 个暴力最优块几何上其实不对，只是因为当前比较把几何目标块读成了零块，所以它们“数值上更近”。

最关键的验证是：对每一个几何目标块 `(R', a', b')`，改用它在 ABACUS 三角存储中的伙伴块
`(-R', b', a')`，并做共轭转置后再比较：

- 全部 `70` 个块都恢复到 `combined_rel < 1e-2`；
- 最坏一个块也只有 `combined_rel = 2.8898801280926326e-04`；
- 对应最坏块的 `HR` 最大元素误差只有 `2.86803e-06`，`SR` 最大元素误差只有 `2.18e-08`。

这说明：

- `R' = R @ M + shift_2 - shift_1` 的几何映射本身没有被这 70 个异常否定；
- 真正的问题是当前对 ABACUS `HR/SR` 的比较直接使用了上三角展开块，没有在目标几何块落入下三角时切换到它的厄米存储伙伴；
- 因此前面看到的 `combined_rel = 2.0` 主要是假零块比较，而不是几何映射失败。

## 2026-04-28 复跑补充（按“先规则映射，再非零块暴力匹配”）

按用户要求再次复跑 `test-abacus-3 -> test-abacus-4`：

1. 先按结构规则映射 `(R, atom1, atom2) -> (R', atom1', atom2')`；
2. 再对所有非零源块做暴力匹配，检查与结构映射是否一致。

新增输出文件：

- `test_workspace/test-abacus-4/test3_to_test4_no_rotation_covariance_report_rerun_20260428_step1.json`
- `test_workspace/test-abacus-4/block_bruteforce_vs_geometry_report_rerun_20260428_step2.json`
- `test_workspace/test-abacus-4/block_mapping_diagnostics_full_rerun_20260428_step2.json`
- `test_workspace/test-abacus-4/block_mapping_diagnostics_full_rerun_20260428_step2.txt`

复跑结果：

- 规则映射统计与前次一致：
  - `M = [[0,-1,0],[-1,0,1],[-1,0,-1]]`
  - `HR mapped_block_count = 1056`, `SR mapped_block_count = 968`
  - `HR rel_fro_diff = 0.0555324466596934`
  - `SR rel_fro_diff = 0.17911222587872075`
- 非零块暴力匹配统计与前次一致：
  - `best_equals_geometry_count = 986`
  - `other_mismatch_count = 70`
  - `mismatch_count = 70`
- 对 70 个不一致块做方向性与厄米伙伴诊断：
  - `orientation_counter = {'lower->upper': 70}`
  - `partner_match_summary.all_combined_rel_below_1e-2 = true`
  - `worst_combined_rel = 2.8898801280926326e-04`

结论保持不变：结构规则映射本身是自洽的；70 个不一致来源于 ABACUS 三角存储下“几何目标落下三角时未切换厄米伙伴”的比较路径问题，而不是几何映射公式错误。

## 2026-04-28 复跑补充（先补齐每个 R 的下三角，再验平移协变性）

按用户要求，先把 `HR/SR` 在每个 `R` 下补成完整矩阵，再做 `test-abacus-3 -> test-abacus-4` 平移协变性验证。

### 代码更新

- `pyatb-main/src/pyatb/symmetry/hs_covariance.py`
  - 新增 `_full_dense_blocks_by_r_from_hermitian_partners(...)`：
    - 依据 `X(R) = X(-R)^dagger`，对每个 `R` 的下三角块用 `-R` 的上三角补齐。
  - 新增 `_dense_blocks_by_r_with_optional_full_reconstruction(...)`。
  - 在验证入口增加开关 `full_matrix_from_hermitian`：
    - `validate_no_rotation_block_mapping_from_xr(...)`
    - `validate_abacus_supercell_to_primitive_no_rotation(...)`

### 单元测试

- `pyatb-main/tests/test_hs_covariance.py`
  - 新增 `test_full_dense_blocks_are_completed_from_minus_r_partner`
  - 运行结果：`3 passed in 0.22s`

### 新报告

- `test_workspace/test-abacus-4/test3_to_test4_no_rotation_covariance_report_rerun_20260428_full_matrix.json`
  - `full_matrix_from_hermitian = true`

### 数值对比（补齐前 vs 补齐后）

- HR:
  - 补齐前：`mapped_block_count=1056`, `rel_fro_diff=0.0555324466596934`, `max_abs_diff=0.33128893712493707`
  - 补齐后：`mapped_block_count=1922`, `rel_fro_diff=8.539412460957883e-05`, `max_abs_diff=3.6012000000029687e-04`
- SR:
  - 补齐前：`mapped_block_count=968`, `rel_fro_diff=0.17911222587872075`, `max_abs_diff=0.288381729`
  - 补齐后：`mapped_block_count=1806`, `rel_fro_diff=1.6127757588504447e-07`, `max_abs_diff=6.270000000099807e-08`

结论：先补齐下三角后，`test-abacus-3 -> test-abacus-4` 的平移协变性误差显著下降，和“比较路径缺失下三角厄米伙伴”这一判断一致。

## 2026-04-28 主流程接入补充（标准化结构 + 哈密顿流程默认补齐）

### 代码接入

- `pyatb-main/src/pyatb/symmetry/hs_standardize.py`
  - 新增：
    - `_dense_blocks_by_r(...)`
    - `_full_dense_blocks_by_r_from_hermitian_partners(...)`
    - `_dense_blocks_by_r_with_optional_full_reconstruction(...)`
  - `_assemble_target_dense_blocks(...)` 新增参数 `full_matrix_from_hermitian`，并在进入原子块映射前先按 `X(R)=X(-R)^dagger` 补齐每个 `R` 的完整矩阵。
  - `canonicalize_abacus_hs(...)` 新增参数 `full_matrix_from_hermitian`（默认 `True`），并把该状态写入返回字典。
- `pyatb-main/src/pyatb/symmetry/character.py`
  - 在 `calculate_character(...)` 的重建分支中，把 `analysis_result['full_matrix_from_hermitian']` 透传给 `canonicalize_abacus_hs(...)`，默认启用补齐。
- `pyatb-main/src/pyatb/symmetry/symm_stru.py`
  - `StandardizationResult` 新增 `full_matrix_from_hermitian` 字段；
  - 标准化结果默认写入 `full_matrix_from_hermitian=True`；
  - `standardization_summary.txt` 增加对应中英文输出字段。

### 单元测试

- `pytest -q tests/test_hs_standardize.py tests/test_hs_covariance.py tests/test_character_module.py`
- 结果：`40 passed, 4 warnings`

### 集成后重新验证（3 -> 4，无旋转扩胞映射）

- 输出：
  - `test_workspace/test-abacus-4/test3_to_test4_no_rotation_covariance_report_rerun_20260428_integrated_full.json`
- 结果：
  - `full_matrix_from_hermitian = true`
  - `HR rel_fro_diff = 8.539412460957883e-05`
  - `SR rel_fro_diff = 1.6127757588504447e-07`
  - `HR max_abs_diff = 3.6012000000029687e-04`
  - `SR max_abs_diff = 6.270000000099807e-08`

结论：补齐逻辑已进入 pyatb 标准化+HS 主流程，且在 `test-abacus-3 -> test-abacus-4` 上保持通过。
