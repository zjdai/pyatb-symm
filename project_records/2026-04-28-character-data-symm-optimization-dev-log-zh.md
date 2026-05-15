# 2026-04-28 CHARACTER 数据对称化优化记录（中文）

## 目标
按顺序优化 CHARACTER 的 data-symmetrization 路径：
1. P0：对称操作上下文一次构建、多处复用
2. P1：原子对变换缓存 + touched-pair 原位平均
3. P2：报告格式优化（完整 history 独立 JSON）

测试目录：
- `/home/zjdai/file-test/pyatb_symm/test_workspace/test-abacus-3/pyatb`
- 输入保持 `data_symmetrize = 1`

## 基线（优化前）
- 执行命令：`pyatb`
- 运行时间：`elapsed_sec=75.67`，`max_rss_kb=1658612`
- 特征标输出哈希：
  - `band_irrep.txt`: `3d517f225f0e14b4a185b8ab0b8f2f689e4d758db871a70c1d9a24349f792888`
  - `symmetry_character_report.txt`: `07fa74f93cd759e1607237d43fbbd33eef9c739613b22545c967727d9ddefd03`
  - `trace.txt`: `d78d62da8256522e5e57992ee9f60313fcb559deacbedf09817b1edba4ae8146`
- 检查结果：特征标输出无 `??`。

## P0 修改
文件：
- `pyatb-main/src/pyatb/symmetry/data_covariance_constraint.py`
- `pyatb-main/src/pyatb/symmetry/character.py`

内容：
- 新增 `prepare_operation_contexts(...)`。
- `self_covariance_statistics(...)` 支持传入预构建 `operation_contexts`。
- `sequential_symmetrize_hs(...)` 支持传入预构建 `operation_contexts`。
- CHARACTER 主流程中上下文只构建一次，前后统计和对称化共用。

P0 测试：
- 运行时间：`elapsed_sec=72.02`，`max_rss_kb=1659556`
- 输出一致性：特征标相关文件哈希与基线完全一致。
- 对称化摘要关键数值不变。

## P1 修改
文件：
- `pyatb-main/src/pyatb/symmetry/data_covariance_constraint.py`

内容：
- 在 operation context 中预构建 atom/pair 缓存（`pair_rows`），避免内层循环重复组装 slice/map/shift。
- `transform_blocks_with_context(...)` 改为遍历缓存的原子对。
- `average_block_sets_on_touched_pairs(...)` 增加 `in_place` 模式，并按 `R` 分组更新，避免每轮全量字典拷贝。
- 对称化主循环启用 `in_place=True`。

P1 测试：
- 运行时间：`elapsed_sec=63.70`，`max_rss_kb=1659860`
- 输出一致性：特征标相关文件哈希与基线完全一致。
- 对称化摘要关键数值不变。

## P2 修改
文件：
- `pyatb-main/src/pyatb/symmetry/character.py`

内容：
- 优化 `data_symmetrization_report.txt`：
  - 保留紧凑全局摘要；
  - 增加每个对称操作最终残差表；
  - 增加最终最差矩阵元摘要。
- 将完整 symmetrization history 输出到 `data_symmetrization_history.json`。

P2 测试：
- 运行时间：`elapsed_sec=62.31`，`max_rss_kb=1665108`

### 最终详细一致性校验（P1 对比 P2）
快照目录：
- `Out/CHARACTER_snap_p1`

当前目录：
- `Out/CHARACTER`

精确比对：
- 对 5 个关键文件执行 `diff -q`，退出码 `0`（字节级一致）
  - `data-HR-sparse_SPIN0-symm-covsymm.csr`
  - `data-SR-sparse_SPIN0-symm-covsymm.csr`
  - `band_irrep.txt`
  - `symmetry_character_report.txt`
  - `trace.txt`
- `sha256` 完全一致：
  - HR covsymm CSR：`cd1494c7d74d41e506949ac35b84a2e67c80da24c0101edf317851279aca1280`
  - SR covsymm CSR：`0225de408ba1799ba12c2576e1cbb71e8a2caaf6ef66dd8666bd0f59b36784a5`
  - band_irrep：`3d517f225f0e14b4a185b8ab0b8f2f689e4d758db871a70c1d9a24349f792888`
  - symmetry_character_report：`07fa74f93cd759e1607237d43fbbd33eef9c739613b22545c967727d9ddefd03`
  - trace：`d78d62da8256522e5e57992ee9f60313fcb559deacbedf09817b1edba4ae8146`

结论：
- P0/P1 显著降低运行时间，物理输出未变化。
- P2 仅优化报告格式；HS 与特征标数值输出保持字节级不变。
