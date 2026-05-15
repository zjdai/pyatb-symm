# 修改记录：ABACUS / IRVSP / IR2TB 阅读笔记

日期：2026-04-25

## 本次完成内容

1. 阅读 `test-abacus` 下与 ABACUS 对称矩阵构造相关的主要脚本：
   - `generate_feo_hs_rotations.py`
   - `verify_feo_hs2_rotation.py`
   - `verify_feo_soc_hs2_rotation.py`
   - `Bi2Se3-periodic/extract_symmetry_operations.py`
   - `Bi2Se3-periodic/verify_nsoc_symmetry_covariance.py`
   - `Bi2Se3-periodic/verify_soc_symmetry_covariance.py`
   - `Bi2Se3-periodic/verify_bi2se3_kspace_symmetry.py`
2. 阅读 `test-abacus-2` 下与 ABACUS-VASP 特征标对齐相关的主要脚本：
   - `validate_bi2se3_abacus_vasp.py`
   - `export_abacus_tqc.py`
3. 阅读 `IRVSP-master/src_irvsp_v2_release` 的主流程与核心 Fortran 模块：
   - `irrep.f90`
   - `symm.f90`
   - `init.f90`
   - `wave_data.f90`
   - `kgroup.f90`
   - `rotkv.f`
   - `chrct.f90`
   - `wrtir.f`
   - `rmprop.f`
   - `pntgrp.f`
   - `pntgrpb.f`
   - `nonsymm.f90`
   - `addsign.f90`
4. 阅读 `IRVSP-master/src_ir2tb_v2` 的 TB 版本工作流：
   - `main.f90`
   - `init.f90`
   - `wave_data.f90`
   - `file_util.f90`
   - `comms.f90`
5. 形成一份长笔记，写入：
   - `project_records/code_plans/2026-04-25-abacus-irvsp-ir2tb-reading-notes.md`

## 本次最重要结论

1. `pyatb` 后续若先做 character，不必先照搬 IRVSP 全部群表。
2. 真正必须优先复现的是 ABACUS 基底下 `D_g(k)` 的构造。
3. character 的数值公式应保留 `S(k)` 度量。
4. SOC 下的自旋旋转约定应沿用 ABACUS 验证脚本已确认的 half-angle 方案。

## 对后续实现的直接价值

这份笔记已经把后续实现所需的“代码地图”整理完成，下一阶段可以直接据此拆出 `pyatb-main/src/.../symmetry` 的最小模块，而不必重新全量读这四套代码。

