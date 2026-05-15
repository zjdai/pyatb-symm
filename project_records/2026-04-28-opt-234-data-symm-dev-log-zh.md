# 2026-04-28 仅第2/3/4点优化记录（中文）

## 范围（按你的要求）
不改动 `abacus_readHR/abacus_readSR`。
只实现：
1. 第2点：去掉 data symmetrization 后的写盘再读盘回环。
2. 第3点：对称变换增加非零原子对缓存。
3. 第4点：迭代对称化阶段改增量候选集统计。

测试目录：
- `/home/zjdai/file-test/pyatb_symm/test_workspace/test-abacus-3/pyatb`

## 代码修改
### A) 第2点（去掉读回回环）
文件：
- `pyatb-main/src/pyatb/symmetry/data_covariance_constraint.py`
- `pyatb-main/src/pyatb/symmetry/character.py`

内容：
- 新增块字典到 `multiXR` 的内存转换：
  - `_dense_blocks_to_multixr(...)`
  - `build_multixr_from_dense_blocks(...)`
- `CHARACTER` 在 `data_symmetrize==1` 时：
  - 仍然输出 `*-covsymm.csr` 文件；
  - 但不再从文件重新解析后再喂给 `TBModel`。
- 关键修正：
  - `hr_blocks` 内部已经是 eV，不能再做 `Ry_to_eV` 缩放；
  - 否则特征标会漂移。
- 为保证与“写盘再读盘”完全等价，内存转换增加了与 writer 一致的 `%.16e` 数值量化。

### B) 第3点（非零原子对缓存）
文件：
- `pyatb-main/src/pyatb/symmetry/data_covariance_constraint.py`

内容：
- context 增加：
  - `pair_row_by_source`
  - `pair_rows` 内 source 原子对标签。
- 新增：
  - `build_active_pair_index(...)`
  - `merge_active_pair_index(...)`
- `transform_blocks_with_context(...)` 新增 `active_pair_index` 参数，按 R 块只遍历缓存的活跃原子对。

### C) 第4点（增量统计）
文件：
- `pyatb-main/src/pyatb/symmetry/data_covariance_constraint.py`

内容：
- 新增 `compare_block_sets_on_candidate_pairs(...)`。
- 在 `sequential_symmetrize_hs(...)` 内循环中：
  - 候选集 = 当前 active index + 本轮 touched target pairs；
  - 迭代统计基于候选集，不再每轮全矩阵全元素扫。
- 收敛判据保持：`HR max_abs <= target`。

## 验证
### 1) 基准快照
对比快照目录：
- `Out/CHARACTER_snap_pre_algo`

基准哈希：
- HR: `cd1494c7d74d41e506949ac35b84a2e67c80da24c0101edf317851279aca1280`
- SR: `0225de408ba1799ba12c2576e1cbb71e8a2caaf6ef66dd8666bd0f59b36784a5`
- band_irrep: `3d517f225f0e14b4a185b8ab0b8f2f689e4d758db871a70c1d9a24349f792888`
- symmetry_character_report: `07fa74f93cd759e1607237d43fbbd33eef9c739613b22545c967727d9ddefd03`
- trace: `d78d62da8256522e5e57992ee9f60313fcb559deacbedf09817b1edba4ae8146`

### 2) 修改后运行
命令：
- `pyatb`

日志：
- `run_opt_234_v3.log`
- elapsed: `57.38 s`
- max RSS: `1661832 KB`

### 3) 严格一致性检查
- 对 snapshot 与当前输出做 `diff -q`（HS + 特征标 + trace）：`DIFF_EXIT=0`
- 五个文件哈希全部与基准一致。
- `band_irrep.txt` / `symmetry_character_report.txt` 无 `??`。

## 性能
本轮前参考：
- `run_opt_algo1.log`: `59.89 s`

本轮后：
- `run_opt_234_v3.log`: `57.38 s`

本轮净提升：
- 约 `2.51 s`。
