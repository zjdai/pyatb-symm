# 2026-04-28 数据对称化接入主流程开发记录（中文）

## 本次目标
- 将“非厄米化”的数据对称化接入 `CHARACTER` 主运行流程。
- 在输入中增加数据对称化参数。
- 输出对称化前后 HR/SR 的最大误差与平均误差变化。

## 代码修改
- `pyatb-main/src/pyatb/symmetry/character.py`
  - 新增参数：
    - `data_symmetrize`（0/1）
    - `data_symm_target_max_abs_ry`
    - `data_symm_max_iter_per_operation`
    - `data_symm_nonzero_block_tol`
    - `data_symm_verbose`
  - 新增流程：
    1）读取当前实际使用的 STRU/HS，
    2）执行数据协变对称化，
    3）输出 `*-covsymm.csr`，
    4）用对称化后 HS 重建 TB，
    5）继续特征标计算。
  - 输出文件：
    - `Out/CHARACTER/data_symmetrization_report.txt`
    - 包含 HR/SR 对称化前后 max/mean 变化。
  - 对不可约表示判定增加容差回退：
    - 先用 `tol=5e-2`，失败后自动用 `tol=1e-1` 再试。
- `pyatb-main/src/pyatb/io/default_input.py`
  - 在 CHARACTER block 增加上述数据对称化参数默认值。
- `pyatb-main/src/pyatb/symmetry/data_covariance_constraint.py`
  - 保留“每个对称操作最多 5 次迭代”的逻辑。
  - 去掉流程中的厄米化步骤。
  - 最终摘要只输出对称化后的误差评估。
- `pyatb-main/tests/test_character_input.py`
  - 增加新参数输入解析测试断言。

## 验证结果
- 语法检查：
  - 修改模块 `python -m py_compile` 全部通过。
- 单元测试：
  - `pytest -q pyatb-main/tests/test_character_input.py` -> `3 passed`。
- 实算验证（`PYTHONPATH=.../pyatb-main/src`）：
  - `test_workspace/test-abacus-3/pyatb`（开启数据对称化）
    - 成功生成 `Out/CHARACTER/data_symmetrization_report.txt`。
    - 报告包含 HR/SR 的 max/mean 前后变化。
  - `test_workspace/test-abacus-2/pyatb` 基准重算。
- 特征标对比：
  - `ab3`：`symmetry_character_report.txt` 与 `band_irrep.txt` 的 `??` 数量均为 0。
  - `ab2`：`symmetry_character_report.txt` 与 `band_irrep.txt` 的 `??` 数量均为 0。
  - ab3 与 ab2 的公共 `(kname, band)` 条目不可约表示完全一致（mismatch=0）。

## 说明
- `ab3` 与 `ab2` 的 k 点标签集合不同（`ab3` 含 `Y`，`ab2` 含 `T/L`），这是标准化结构路径差异导致；但共享标签下的表示结果一致。
