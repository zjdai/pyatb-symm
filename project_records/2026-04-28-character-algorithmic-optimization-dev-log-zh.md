# 2026-04-28 CHARACTER 算法优化记录（中文）

## 目标
在不改变以下结果的前提下继续做算法级优化：
- HS 输出不变；
- 特征标输出不变；
- 在测试算例提速：
  - `/home/zjdai/file-test/pyatb_symm/test_workspace/test-abacus-3/pyatb`

## 优化前热点分析
命令：
- `python -m cProfile -o prof_opt_algo.pstats pyatb`

主要热点：
- `compare_block_sets`：累计约 `19.10 s`
- `transform_blocks_with_context`：累计约 `18.03 s`
- `sequential_symmetrize_hs`：累计约 `26.95 s`

## 实施的优化
### 1）最终最差矩阵元扫描改为单遍统计
文件：
- `pyatb-main/src/pyatb/symmetry/data_covariance_constraint.py`

修改：
- 新增 `compare_block_sets_with_detail(...)`，在一次遍历中同时给出：
  - 误差统计（max/mean/rms/rel_fro）
  - 最差矩阵元 detail
- 将原先 final 阶段的 `compare_block_sets + _max_error_element_detail` 双遍扫描替换为单遍。

### 2）特征标组合匹配增加组合缓存与字符切片缓存
文件：
- `pyatb-main/src/pyatb/symmetry/character_core.py`

修改：
- 新增 `_cached_combinations_with_replacement(...)`（模块级缓存）。
- 在 `assign_irrep_combination(...)` 中：
  - 对 active operation indices 对应的 irrep character 先切片一次；
  - irrep 名称先缓存一次；
  - 组合枚举复用缓存。

### 3）比较循环的小开销优化
文件：
- `pyatb-main/src/pyatb/symmetry/data_covariance_constraint.py`

修改：
- compare/detail 扫描中去掉不必要的 key 排序（仍保持同样数值逻辑）。

## 校验流程
### A）优化前快照
已保存快照目录：
- `Out/CHARACTER_snap_pre_algo`

快照文件：
- `data-HR-sparse_SPIN0-symm-covsymm.csr`
- `data-SR-sparse_SPIN0-symm-covsymm.csr`
- `band_irrep.txt`
- `symmetry_character_report.txt`
- `trace.txt`

### B）优化后重跑
命令：
- `pyatb`

结果：
- `EXIT_CODE=0`
- 运行时间：`elapsed_sec=59.89`
- 内存峰值：`1658668 KB`

本轮优化前参考时间：
- 上一版：`elapsed_sec=62.31`

### C）严格一致性对比
对 `Out/CHARACTER_snap_pre_algo` 和当前 `Out/CHARACTER` 做对比：
- 5 个关键文件 `diff -q`：`DIFF_EXIT=0`
- SHA256 全部一致：
  - HR：`cd1494c7d74d41e506949ac35b84a2e67c80da24c0101edf317851279aca1280`
  - SR：`0225de408ba1799ba12c2576e1cbb71e8a2caaf6ef66dd8666bd0f59b36784a5`
  - band_irrep：`3d517f225f0e14b4a185b8ab0b8f2f689e4d758db871a70c1d9a24349f792888`
  - symmetry_character_report：`07fa74f93cd759e1607237d43fbbd33eef9c739613b22545c967727d9ddefd03`
  - trace：`d78d62da8256522e5e57992ee9f60313fcb559deacbedf09817b1edba4ae8146`

补充检查：
- `band_irrep.txt` 与 `symmetry_character_report.txt` 中未出现 `??`。

## 结论
- HS 数据与特征标输出保持字节级不变；
- 测试算例运行时间由 `62.31 s` 降到 `59.89 s`。
