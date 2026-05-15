# ABACUS / IRVSP / IR2TB 阅读笔记

日期：2026-04-25  
工作区：`/home/zjdai/file-test/pyatb_symm`

## 0. 范围与结论

这份笔记覆盖四组代码：

1. `/home/zjdai/file-test/d-matrix/test-abacus`
2. `/home/zjdai/file-test/d-matrix/test-abacus-2`
3. `/home/zjdai/file-test/pyatb_symm/IRVSP-master/src_irvsp_v2_release`
4. `/home/zjdai/file-test/pyatb_symm/IRVSP-master/src_ir2tb_v2`

用户消息里写的是 `test-abacus2`，实际目录名是 `test-abacus-2`，本笔记按实际目录记录。

目前已经明确三件对后续 `pyatb/src/pyatb/symmetry/` 最关键的事情：

1. ABACUS 侧真正要复用的核心不是“整套脚本”，而是“如何在 ABACUS 轨道基底里构造对称操作矩阵”。
2. 若目标只是“任意 k 点任意指定能带的特征标”，最直接路线并不是照搬 IRVSP 的整套群论表，而是优先复用 ABACUS 这边已经验证过的
   `D_g(k)` 构造方式，再结合 `pyatb` 已有的广义本征问题求解结果，用
   `C^\dagger S(k) D_g(k) C`
   计算角色。
3. IRVSP / IR2TB 的强项不是基底矩阵构造，而是“如何把角色识别成不可约表示、如何组织 little-group 数据与输出格式”。后续如果要做 irrep 标签，而不仅是字符，这两套程序尤其是 IRVSP 的 little-group 数据组织方式非常值得继续借鉴。

## 1. 四套代码的大图景

### 1.1 `test-abacus`

这一组主要是“ABACUS 对称性验证工具”，不是生产程序。它有两条主线：

1. 真实空间 `H(R)` / `S(R)` 协变性验证
2. 倒空间 `H(k)` / `S(k)` 协变性验证，以及基底表示矩阵 `D_g(k)` 的显式构造

它解决的问题是：

- ABACUS 的轨道基顺序到底是什么；
- 非 SOC / SOC 下轨道与自旋如何一起转动；
- 空间群操作在 ABACUS 原子轨道基组里的矩阵长什么样；
- 用这个矩阵去变换 `H(R)`、`S(R)`、`H(k)`、`S(k)` 后，是否和数值结果一致。

这套脚本对后续 `pyatb` 开发的价值非常高，因为它已经把“ABACUS 基底下的对称操作矩阵”这件最容易出错的事情做成了可验证原型。

### 1.2 `test-abacus-2`

这一组不是基础验证，而是“ABACUS 结果与 VASP / IRVSP 结果对接”的桥梁代码。它的目标是：

1. 用 ABACUS 的 `STRU`、`Orbital`、`HR/SR` 构造 `H(k)` 与特征子空间；
2. 用 VASP `outdir` 读取 little group、角色、irrep 标签；
3. 把 ABACUS 侧构造出的角色和 VASP / IRVSP 输出逐步比对；
4. 最终导出 `tqc.txt`、`tqc.data`、`trace.txt` 这类 IRVSP / website 兼容格式。

它说明两件事：

1. 用 ABACUS 的紧束缚矩阵完全可以复现 IRVSP 风格的特征标分析；
2. 真正决定结果正确性的，是“构造对称表示矩阵 + 在重叠度量下做子空间迹”的那一步。

### 1.3 `src_irvsp_v2_release`

这是标准 IRVSP 主程序，读取 VASP 的 `OUTCAR` 与 `WAVECAR`，在平面波基底里做：

1. 空间群初始化；
2. little group 判定；
3. 对称操作作用到波函数上的相位与重排；
4. 角色计算；
5. 依据 character table 判定不可约表示；
6. 输出 `trace.txt`、`tqc.txt`、`tqc.data` 等。

它的优点是群论部分成熟、完整，尤其是 nonsymmorphic little-group 数据。

### 1.4 `src_ir2tb_v2`

这是 IRVSP 的 TB / Wannier 版本。它不再读取 `WAVECAR`，而是读取：

1. `tbbox.in`
2. `*_hr.dat`

然后在 TB 基底中：

1. 构造 `H(k)`；
2. 对角化得到本征值与本征矢；
3. 调用 `tb_setup(...)` 构造 TB 基底下的对称矩阵；
4. 再调用 `irrep_bcs(...)` 做角色与 irrep 分析。

关键点：`src_ir2tb_v2` 本身不包含完整的 irrep 分析核心，它把最后那部分委托给外部库 `irrep_bcs`。因此这套代码最值得借鉴的是“TB 侧输入组织和对称矩阵准备流程”，不是最终识别算法本身。

## 2. ABACUS 侧最基础的公共模块

文件：`/home/zjdai/file-test/d-matrix/abacus_angular_momentum.py`

这个文件非常重要，因为后面几乎所有 ABACUS 对称脚本都依赖它来生成角动量矩阵。

### 2.1 `_complex_basis_operators(l)`

作用：

- 在复球谐基 `|l,m>` 中构造标准的 `Lx, Ly, Lz`。

意义：

- 这是最标准的角动量表示，不依赖 ABACUS 的特殊基底顺序。

### 2.2 `_abacus_basis_transform(l)`

作用：

- 构造从复球谐基到 ABACUS 实轨道基底的变换矩阵。

意义：

- 这里编码了 ABACUS 的实轨道顺序约定。
- 后面如果要在 ABACUS 原子轨道基底中构造旋转矩阵，必须先经过这一步。

### 2.3 `angular_momentum_matrices_abacus(l)`

作用：

- 输出 ABACUS 实轨道基底下的 `Lx, Ly, Lz`。

意义：

- 后面所有 `exp(-i angle * n·L)` 的轨道转动矩阵，最终都建立在这里。
- 未来 `pyatb/symmetry` 若要支持 `s/p/d/f` 轨道对称表示，这个思想必须保留。

## 3. `test-abacus` 详细笔记

## 3.1 结构与用途

这一目录主要包含：

- `generate_feo_hs_rotations.py`
- `run_feo_hs_rotations.py`
- `setup_feo_soc_rotations.py`
- `run_feo_soc_rotations.py`
- `verify_feo_hs2_rotation.py`
- `verify_feo_soc_hs2_rotation.py`
- `Bi2Se3-periodic/extract_symmetry_operations.py`
- `Bi2Se3-periodic/verify_nsoc_symmetry_covariance.py`
- `Bi2Se3-periodic/verify_soc_symmetry_covariance.py`
- `Bi2Se3-periodic/verify_bi2se3_kspace_symmetry.py`

它们可分成三类：

1. 生成测试数据
2. 真实空间协变性验证
3. 倒空间协变性验证

## 3.2 FeO 旋转数据集脚本

### 3.2.1 `generate_feo_hs_rotations.py`

目标：

- 从一个 relaxed 的 FeO 结构出发，生成多个整体刚体旋转后的 ABACUS 输入目录。

主要类与函数：

- `AtomSite`, `StruData`
  - 对 ABACUS `STRU` 的轻量封装。
- `read_stru(...)`
  - 读取 ABACUS 的 `STRU` 文件。
- 坐标变换与 Rodrigues 旋转相关函数
  - 用于把结构绕给定轴旋转。
- `write_input(...)`, `write_kpt(...)`, `write_stru(...)`
  - 写出新的 ABACUS 输入文件。
- `build_rotations(...)`
  - 生成一组旋转参数。
- `main()`
  - 组织 `stru-init` 与 `rotate1..rotate10` 目录。

输出特点：

- 每个旋转目录都会保留 `rotation_params.json`。
- `INPUT` 中开启 `out_mat_hs2=1`，即要求 ABACUS 输出 `HR/SR` 类矩阵数据，供后续验证使用。

### 3.2.2 `run_feo_hs_rotations.py` 与 `run_feo_soc_rotations.py`

目标：

- 批量运行上一步生成的多个 ABACUS 目录。

作用：

- 它们不是理论核心，只是数据驱动器。

### 3.2.3 `setup_feo_soc_rotations.py`

目标：

- 把非 SOC 旋转数据集改造成 SOC 版本。

核心处理：

- 改写 `INPUT`，设置 `nspin=4`, `lspinorb=1`, `noncolin=0`。

意义：

- 它确保非 SOC 与 SOC 旋转测试在同一套结构与旋转参数下可直接对比。

## 3.3 FeO 真实空间协变性验证

### 3.3.1 `verify_feo_hs2_rotation.py`

目标：

- 检查非 SOC 下，整体刚体旋转后的 ABACUS `H(R)` 是否满足轨道旋转协变性。

关键对象：

- `XRMatrix`
  - 读取 ABACUS 输出的 CSR 稀疏矩阵，重建每个 `R` 对应的稠密块。
- `parse_orbital_shells(...)`
  - 从 `Orbital` 文件恢复每个原子的 `(l, zeta)` 轨道壳结构。
- `_finalize_shell(...)`
  - 校验壳内 `m` 顺序是否与预期一致。
- `build_shell_rotation_matrix(...)`
  - 对单个 `l` 壳构造旋转矩阵。
- `build_total_rotation_matrix(...)`
  - 拼出全体系轨道旋转矩阵。
- `compare_xr_matrices(...)`
  - 比较 `H_target(R)` 与 `U H_init(R) U^\dagger` 是否一致。
- `verify_rotation_dataset(...)`
  - 遍历所有旋转目录，比较总能和矩阵协变性。

核心思想：

- 这是最纯粹的“轨道表示正确性”验证。
- 它不涉及空间群原子置换，只涉及整个体系作为刚体旋转时单中心轨道如何变换。

### 3.3.2 `verify_feo_soc_hs2_rotation.py`

目标：

- 做与上面同样的事情，但在 SOC 基底中验证。

新增关键函数：

- `pauli_matrices()`
- `build_spin_rotation_matrix(...)`
- `build_soc_rotation_matrix(...)`

关键结论：

- SOC 下本地旋转矩阵是 `kron(D_orbital, U_spin)`。
- 脚本显式比较了几种自旋角度约定，最终确认应使用“Pauli half-angle”约定。

这对后续 `pyatb` 极重要，因为如果自旋转动角度取错一倍，角色会整体错误。

## 3.4 Bi2Se3 真实空间与倒空间对称验证

### 3.4.1 `Bi2Se3-periodic/extract_symmetry_operations.py`

目标：

- 从 ABACUS `STRU` 结构出发，经 `spglib` 得到空间群操作，并写出一个便于后续使用的 JSON 文件。

主要工作：

- 解析 `STRU`；
- 用 `spglib.get_symmetry_dataset(...)` / `spglib.get_symmetry(...)` 获取操作；
- 记录分数坐标旋转、平移、轴角信息、proper/improper 类型等。

输出：

- `symmetry_operations_axis_angle.json`

这相当于后续验证脚本的“空间群输入文件”。

### 3.4.2 `Bi2Se3-periodic/verify_nsoc_symmetry_covariance.py`

目标：

- 在非 SOC 情况下，直接验证 `H(R)` 在空间群操作下是否协变。

核心思路：

- 对每个空间群操作 `{W|t}`，对每个原子对 `(a,b)` 与格矢 `R`，计算变换后的
  `R' = W R + L_b - L_a`
- 再用原子局域轨道旋转块 `D_a`, `D_b` 构造
  `H'(R') = D_a H(R) D_b^\dagger`
- 和 ABACUS 实际输出的目标块比较。

重点：

- 这里已经显式处理了原子重排与晶格平移补偿。
- improper 旋转时包含 `(-1)^l` 的宇称因子。

### 3.4.3 `Bi2Se3-periodic/verify_soc_symmetry_covariance.py`

目标：

- 在 SOC 基底里做上面同样的 `H(R)` 协变性验证。

关键假设：

- ABACUS SOC 基底顺序是
  `orbital_1 up, orbital_1 down, orbital_2 up, orbital_2 down, ...`

关键意义：

- 这条基底顺序假设一旦错，构造出的局域对称矩阵就会整体错位。
- 脚本通过数值对比验证了这个顺序与自旋旋转约定。

### 3.4.4 `Bi2Se3-periodic/verify_bi2se3_kspace_symmetry.py`

这是后续最值得移植到 `pyatb` 的 ABACUS 原型。

目标：

- 在给定 `k` 点上构造空间群操作的基底表示矩阵 `D_g(k)`，
- 并检查
  `D_g(k)^\dagger H(gk) D_g(k) = H(k)`
  与
  `D_g(k)^\dagger S(gk) D_g(k) = S(k)`。

关键结构：

- `Structure`, `OrbitalShell`, `SystemData`
  - 组织结构、轨道壳、矩阵与模式信息。
- `XRMatrix`
  - 通过对 `H(R)` / `S(R)` 做 Fourier 求和得到 `H(k)` / `S(k)`。
- `compute_k_star(...)`
  - 计算 `k` 的星。
- `shell_rotation(...)`
  - 给定 `l` 与空间旋转，构造轨道块表示。
- `spin_rotation(...)`
  - 构造自旋 1/2 表示。
- `atom_orbital_rotation(...)` / `atom_local_rotation(...)`
  - 在单原子局域子空间中构造完整局域表示。
- `find_atom_mapping(...)`
  - 找到空间群操作下每个原子映到哪个原子，并记录跨胞平移。
- `build_symmetry_representation(...)`
  - 最关键函数。把原子置换、局域轨道/自旋转动、Bloch 相位
    `exp(-i 2π k·cell_shift)`
    拼成全局矩阵 `D_g(k)`。
- `verify_operation_at_k(...)`
  - 用 `D_g(k)` 检验 `H(k)` 与 `S(k)` 的协变性。

结论：

- 这份脚本给出了 ABACUS 基底下 `D_g(k)` 的完整构造模板。
- 将来 `pyatb` 若只求角色，不一定要重写一整套群表，但必须基本保留这里的构造方式。

## 4. `test-abacus-2` 详细笔记

## 4.1 总体定位

这一组代码把 ABACUS 结果和 VASP / IRVSP 的 irrep 结果连了起来。它不是简单做验证，而是已经接近一个“原型版 ABACUS-irrep 工作流”。

## 4.2 `validate_bi2se3_abacus_vasp.py`

### 4.2.1 输入与数据源

ABACUS 侧输入：

- `soc/STRU`
- `soc/OUT.ABACUS/Orbital`
- `soc/OUT.ABACUS/data-HR-sparse_SPIN0.csr`
- `soc/OUT.ABACUS/data-SR-sparse_SPIN0.csr`

VASP / IRVSP 侧输入：

- `test-vasp/outdir`

### 4.2.2 关键数据结构

- `Structure`
- `OrbitalShell`
- `AbacusOperation`
- `MatchedOperation`
- `VaspBandRow`
- `VaspKPointSection`
- `XRMatrix`

这些结构已经把“ABACUS 结构/轨道/矩阵”和“VASP outdir 的 little-group / band-row / character”统一到一个 Python 层的数据模型里。

### 4.2.3 核心函数及意义

#### 基础解析

- `parse_stru(...)`
  - 读取 ABACUS `STRU`。
- `parse_orbital_shells(...)`
  - 从 `Orbital` 里恢复每个原子的局域轨道壳结构。
- `XRMatrix.read_file()`
  - 读取 CSR 稀疏矩阵。
- `XRMatrix.xk_matrix(k)`
  - 做 Fourier 求和得到 `H(k)` 或 `S(k)`。
- `parse_vasp_outdir(...)`
  - 解析 VASP / IRVSP 风格 `outdir`，得到操作信息、little group 与 band-row 角色。

#### 对称操作与 little group

- `build_abacus_operations(...)`
  - 用 `spglib` 从 ABACUS 结构生成操作，附带旋转轴、角度、proper rotation、自旋矩阵等信息。
- `match_operations_to_outdir(...)`
  - 将 ABACUS 生成的操作与 outdir 中的操作逐一匹配。
- `compute_little_group_indices(...)`
  - 对给定 `k` 判断 little group。

#### 表示矩阵构造

- `shell_rotation(...)`
- `atom_orbital_rotation(...)`
- `find_atom_mapping(...)`
- `build_symmetry_representation(...)`

这一套和 `verify_bi2se3_kspace_symmetry.py` 是同一思想，只是这里进一步对齐到了 outdir 的操作编号。

#### 本征问题与角色计算

- `solve_generalized_eigenproblem(...)`
  - 对 `H(k) c = E S(k) c` 做广义本征求解。
- `build_kramers_pairs(...)`
  - 把 SOC 下近简并能带配对成 Kramers pair。
- `select_top_occupied_pairs_abacus(...)`
  - 取 ABACUS 侧顶部占据 Kramers pair。
- `select_top_occupied_rows_vasp(...)`
  - 取 VASP 侧顶部占据 band rows。

关键公式：

- 角色不是直接拿单个波函数的重叠，而是在重叠度量下对子空间求迹：
  `Tr(C^\dagger S(k) D_g(k) C)`

这正是将来 `pyatb` 应该采用的做法，因为 ABACUS / pyatb 的基底通常不是正交规范基。

### 4.2.4 `run_validation()` 的四步流程

#### Step 1. 对称操作匹配

做什么：

- 匹配 ABACUS `spglib` 操作与 VASP outdir 的操作；
- 比较旋转矩阵、自旋矩阵、平移。

输出：

- `step1_symmetry_match_report.json`
- `step1_symmetry_match_summary.txt`

#### Step 2. Little group 比较

做什么：

- 在 `GM/T/F/L` 上分别计算 ABACUS little group；
- 与 outdir 给出的 little group 对比。

输出：

- `step2_little_group_report.json`
- `step2_little_group_summary.txt`

#### Step 3. 能带对齐

做什么：

- 解 ABACUS 的广义本征问题；
- 用 `Gamma` 点 HOMO 的位置定义一个整体能量平移；
- 与 VASP 的 band rows 对齐比较。

输出：

- `step3_band_alignment_report.json`
- `step3_band_alignment_summary.txt`

#### Step 4. 角色比较

做什么：

- 在 little-group 每个操作上构造 `D_g(k)`；
- 在 Kramers 子空间内计算 ABACUS 角色；
- 与 VASP outdir 的角色逐项比较；
- 检查 `representation @ subspace` 是否仍留在该简并子空间内。

输出：

- `step4_character_report.json`
- `step4_character_summary.txt`

### 4.2.5 这份脚本对未来 `pyatb` 的直接启发

最值得直接照搬的，不是 VASP 对齐部分，而是：

1. `XRMatrix.xk_matrix(k)` 的思路
2. `build_symmetry_representation(...)`
3. `solve_generalized_eigenproblem(...)`
4. 在 `S(k)` 度量下对子空间求角色的公式

## 4.3 `export_abacus_tqc.py`

目标：

- 在上一个验证脚本的基础上，真正导出 ABACUS 版本的 `tqc.txt`、`tqc.data`、`trace.txt`、`trace_website.txt`。

核心思路：

1. 先复用 `validate_bi2se3_abacus_vasp.py` 的基础能力；
2. 从 `outdir` 读取 little-group 双值 irreps；
3. 对 ABACUS 的占据流形按能量近简并分组；
4. 对每个流形计算 characters；
5. 把这些 characters 分解成一组 outdir irrep 的和；
6. 写成 TQC / trace 格式。

关键函数：

- `parse_outdir_irrep_sections(...)`
  - 读取 outdir 中每个高对称点的 irrep 表。
- `validate_numeric_indices(...)`
  - 检查解析出的 irrep 序号是否能复现 VASP 的 `tqc.data`。
- `group_occupied_manifolds(...)`
  - 依能量近简并关系分组。
- `decompose_characters(...)`
  - 在候选 irrep 中寻找一组角色之和等于目标流形角色。
- `compute_abacus_manifolds(...)`
  - 对每个流形计算角色并做 irrep 分解。
- `write_trace_file(...)`
  - 输出 IRVSP / website 风格 trace 文件。

这份脚本说明：一旦已经能稳定计算 characters，后面做 irrep 分解与 trace 输出就是纯后处理问题。

## 5. IRVSP 主程序 `src_irvsp_v2_release`

## 5.1 总体运行流程

主程序文件：`irrep.f90`

顶层流程可以概括成：

1. 解析命令行 `-sg`, `-v`, `-nb`
2. `SSYM(sgn, ver_n, nmax)`
   - 读取 `OUTCAR`
   - 初始化空间群与 little-group 表
3. `setarray()`
   - 分配波函数与角色数组
4. 对 `KKK = 1 .. NSPIN*NKPTS`
   - `KPTIN(KKK, nmax)`
   - 从 `WAVECAR` 读取一个 k 点的平面波系数
   - 调 `KSYM(...)` 做 little-group 分析
5. 输出 `trace.txt`, `tqc.txt`, `tqc.data`

## 5.2 关键模块与函数职责

### 5.2.1 `init.f90` / `module struct_data`

核心职责：

- 从 `OUTCAR` 中读出“对称性 + 晶格 + 计算维度”。

重要输出：

- `IORD`
  - 空间群操作数
- `IZ`
  - 以倒格矢坐标表示的整数旋转矩阵
- `TAU`
  - 分数平移
- `DZ2`
  - 三维实空间旋转矩阵
- `SU2`
  - 自旋 1/2 旋转矩阵
- `NMAT`, `NUME`, `NKPTS`, `NSPIN`
  - 平面波数、能带数、k 点数、自旋数

`Dmatrix(...)`：

- 给定旋转轴与转角，返回对应 `SU(2)` 或更高 `J` 表示矩阵。
- 在 IRVSP 里既用于生成自旋旋转，也用于一些高角动量表示。

### 5.2.2 `wave_data.f90`

核心职责：

- 从 `WAVECAR` 读取每个 k 点的平面波系数。

关键流程：

- `kptin(KKK, nmax)`
  1. 打开 `WAVECAR`
  2. 读取头信息，检查 `NSPIN/NKPTS/NUME`
  3. 读取指定 k 点的本征值
  4. 根据截止能与晶格重建平面波 `G` 向量列表
  5. 读入每条带的系数到 `coeffa`, `coeffb`
  6. 调用 `KSYM(...)`

意义：

- IRVSP 的波函数表示是平面波系数；
- `KV`, `L`, `PH` 等数组都是为后续对称操作在平面波基底中的重排与相位服务。

### 5.2.3 `kgroup.f90`

`kgroup(...)` 的作用很单纯：

- 找出所有满足
  `k * inv(R_i) = k + G`
  的操作，形成 little group。

这一步是后面全部 irrep 分析的入口。

### 5.2.4 `rotkv.f`

作用：

- 对于 little-group 操作 `R_i`，找到每个平面波 `(k+K)` 经 `inv(R_i)` 作用后映到哪个 `(k+K')`；
- 同时计算平移部分带来的相位 `PH`。

这本质上是“对称操作如何作用到平面波基底”的重排表。

### 5.2.5 `chrct.f90`

这是 IRVSP 里最关键的数值角色计算模块之一。

做的事情：

1. 规范化波函数；
2. 若需要，先做时间反演或自旋变换；
3. 对每个简并子空间、每个 little-group 元素，构造该元素在该子空间中的表示矩阵；
4. 取迹得到 `XM`。

关键点：

- 这里不是只看单带，而是按能量阈值 `TOLDG` 自动聚成简并多重态；
- 对 SOC 情形，使用 `SU2` 处理自旋；
- 最终输出的是“每个对称操作、每个简并能级”的 character。

### 5.2.6 `wrtir.f`

作用：

- 根据 `chrct` 输出的角色，与当前 point group 的 character table 比较；
- 把一个简并多重态表示成若干 irrep 之和；
- 写出 band plotting / trace 所需格式。

局限：

- 这个文件假设已经有现成的 `CTIR`, `TTIR`, `ZTIR` 等 character table 数据。

### 5.2.7 `rmprop.f`

作用：

- 对每个空间群操作求其旋转类型、轴、角度、自旋表示、共轭关系等元信息。

它完成的关键任务包括：

1. 计算 `JIJ(i,j)=R_j R_i R_j^{-1}` 的类内映射；
2. 判定操作属于 `E/C2/C3/C4/C6/I/ICn` 中哪类；
3. 调 `RAXIS` 求旋转轴；
4. 调 `SU2OP` 构造自旋表示；
5. 修正 SU(2) 符号约定。

它相当于 IRVSP 中“空间群元素预处理”的核心。

### 5.2.8 `pntgrp.f`

作用：

- 对 symmorphic 情况，根据 little-group 中各类旋转元素数量与共轭类数，识别点群类型；
- 建立该点群的 character table 描述。

特点：

- 这部分是硬编码判别逻辑；
- 一旦识别出点群，就把 `CTIR`, `TTIR`, `ZTIR` 等表填好。

### 5.2.9 `pntgrpb.f`

作用：

- 对 nonsymmorphic little groups 做类似工作，但不再按普通共轭类分类；
- 主要负责输出 little-group 基本信息，并把真正的 irrep 数据交给 `nonsymm.f90`。

### 5.2.10 `nonsymm.f90`

这是 IRVSP 支持 nonsymmorphic 空间群的关键。

它做的事情：

1. 从环境变量 `IRVSPDATA` 指向的数据目录读取 `kLittleGroups/kLG_<sgn>.data`；
2. 读入该空间群所有 little-group 的标准点、标准 irreps、Herring rule、角色表等；
3. 把 VASP / IRVSP 当前单胞约定和标准表的坐标系做转换；
4. 提供 `getkid`, `wrtirb`, `dumptableofIrs` 等接口。

它的重要意义：

- IRVSP 之所以能稳妥处理 nonsymmorphic 双群，不是现场推导出来，而是依赖预制的 little-group 数据库。

### 5.2.11 `addsign.f90`

作用：

- 给输入的 `SU2` 旋转矩阵确定正确的整体符号分支。

原因：

- 对同一个 `SO(3)` 旋转，`SU(2)` 有正负两个 lift；
- 若多元素间的乘法表不一致，整个双群结构就会错。

做法：

- 构造参考双群乘法表；
- 构造输入 `SU2` 的乘法表；
- 通过生成元和符号假设搜索，使两者一致。

这在自旋表示处理中非常关键，但如果后续 `pyatb` 直接用数值验证过的 `spin_rotation` 构造，并且只需要 characters，不一定需要把这套完整搜索都搬过去。

### 5.2.12 `symm.f90`

这是 IRVSP 的总调度中心。

`SSYM(...)`：

1. `init(...)`
2. 计算 `IIZ`
3. `MPNTGRP(...)`
4. `RMPROP(...)`
5. 打开输出文件、写头信息

`KSYM(...)`：

1. `KGROUP(...)` 求 little group
2. `CRWCND(...)` 检查 Cornwell condition
3. symmorphic 分支：
   - `PNTGRP`
   - `CLASSE`
   - `ROTKV`
   - `TRSYMA`
   - `CHRCT`
   - `WRTIR`
4. nonsymmorphic 分支：
   - `getkid`
   - `KGROUP`
   - `ROTKV`
   - `PNTGRPB`
   - `dumptableofIrs`
   - `TRSYMB`
   - `CHRCT`
   - `WRTIRB`

从程序结构上看，`symm.f90` 就是 IRVSP 的主流程图。

## 6. IR2TB 主程序 `src_ir2tb_v2`

## 6.1 定位

IR2TB 是把 IRVSP 的思想搬到 Wannier / TB 基底。它不读 `WAVECAR`，而读 `tbbox.in + hr.dat`。

## 6.2 关键模块

### 6.2.1 `comms.f90`

作用：

- 保存整个程序共享状态。

主要内容：

- `casename`, `hrfile`
- 晶格、中心、轨道数信息
- `HmnR`, `coeffa`, `coeffb`, `EE`
- 对称输入 `det_read`, `angle_read`, `axis_read`, `tau_read`
- k 路径 `k`, `len_k`

### 6.2.2 `file_util.f90`

作用：

- 提供 `tbbox.in` 的轻量级关键词解析器。

主要接口：

- `get_key_para_int`
- `get_key_para_cht`
- `get_key_para_rel`
- `get_key_para_vec`
- `get_key_para_loc`
- `get_key_para_intct`
- `get_key_para_nvec`

这部分只是输入读取工具，无群论内容。

### 6.2.3 `init.f90`

#### `read_tbbox()`

作用：

- 解析 `tbbox.in`，建立 TB 侧计算环境。

读取内容包括：

- `case`
  - 确定 `casename`, `hrfile`
- `proj/orbt`
  - 轨道约定
- `ntau`
  - 中心数、每个中心的轨道数、中心位置
- `unit_cell`
  - 实空间与倒空间基矢
- 后续紧跟的空间群操作输入
  - `det_read`, `angle_read`, `axis_read`, `tau_read`
- `kpoint`
  - k 路径节点与插值

同时它会设置：

- `isInv`
- `isSymmorphic`
- `isSpinor`
- `isComplexWF`

#### `read_HmnR()`

作用：

- 读取 Wannier90 风格 `*_hr.dat`。

结果：

- `vec_block(:,ir)`：每个 `R`
- `deg_block(ir)`：简并权重
- `HmnR(:,:,ir)`：每个 `R` 的 hopping block

#### `setarray()/downarray()`

作用：

- 分配与释放能量、本征矢、对称输入数组。

### 6.2.4 `wave_data.f90`

#### `get_Hk()`

作用：

- 用
  `H(k) = sum_R exp(i k·R) H(R)`
  构造 `H(k)`。

#### `myzheev()`

作用：

- 调 LAPACK `ZHEEV` 对角化 Hermitian 矩阵。

#### `get_WF()`

作用：

1. 调 `get_Hk()`
2. 对角化得到本征值、本征矢
3. 根据 `nspin` 把本征矢拆到 `coeffa`, `coeffb`

### 6.2.5 `main.f90`

顶层流程：

1. 解析 `-sg` 与 `-nb`
2. `read_tbbox()`
3. `read_HmnR()`
4. `setarray()`
5. 遍历所有 k 点
   - `get_WF()`
   - `tb_setup(...)`
   - `irrep_bcs(...)`
6. `downarray()`

这里的关键是：

- `tb_setup(...)`
  - 构造 TB 基底中的对称操作矩阵、k 相位等；
- `irrep_bcs(...)`
  - 真正做 band character / irrep 分析。

注意：

- `irrep_bcs` 不在这份源码中实现，属于外部库。
- 因此 `src_ir2tb_v2` 更像“把 TB 数据接入 IRVSP 分析内核的适配层”。

## 7. 四套代码之间的对应关系

可以把它们理解成四层：

1. `test-abacus`
   - 回答“ABACUS 基底里对称操作矩阵怎么写才对”
2. `test-abacus-2`
   - 回答“ABACUS 的角色怎样与 VASP / IRVSP 输出一一对应”
3. `IRVSP`
   - 回答“成熟的平面波 irrep 分析程序整体怎么组织”
4. `IR2TB`
   - 回答“如果输入变成 TB/Wannier，整个流程应如何重组”

所以从后续开发角度看：

- ABACUS 本地对称矩阵构造逻辑，优先看 `test-abacus`
- ABACUS character 计算逻辑，优先看 `test-abacus-2`
- little-group 表与 irrep 输出组织，优先看 `IRVSP`
- TB 风格输入与运行框架，优先看 `IR2TB`

## 8. 对未来 `pyatb/symmetry` 的直接建议

## 8.1 最小可行目标

如果近期目标只是：

- 任意 `k` 点
- 任意指定能带或简并子空间
- 计算各个给定对称操作下的 character

那么不应该一上来就搬 IRVSP 全套群表和 irrep 标签系统。

更合理的最小路径是：

1. 复用 `pyatb` 现有初始化与对角化
2. 新增 ABACUS 结构 / 轨道壳 / 空间群操作读取
3. 新增 `build_symmetry_representation(...)`
4. 在 `S(k)` 度量下计算
   `C^\dagger S(k) D_g(k) C`
5. 输出每个 band / manifold 的 character

## 8.2 最应该直接借鉴的函数思想

建议优先抽象并移植以下思想：

1. `abacus_angular_momentum.py`
   - ABACUS 实轨道基底的角动量矩阵
2. `verify_bi2se3_kspace_symmetry.py::build_symmetry_representation`
   - 全局 `D_g(k)` 构造
3. `validate_bi2se3_abacus_vasp.py::solve_generalized_eigenproblem`
   - `H(k), S(k)` 广义本征问题
4. `validate_bi2se3_abacus_vasp.py` / `export_abacus_tqc.py`
   - 子空间 character 的计算方式与流形分组方式

## 8.3 当前不建议直接照搬的部分

1. IRVSP 的 `WAVECAR` 读取部分
   - `pyatb` 完全不需要
2. IRVSP 的整套 `CTIR/TTIR/ZTIR` 硬编码 character table
   - 如果第一阶段只求 character，可先不搬
3. `addsign.f90` 的完整双群符号搜索
   - 先用 ABACUS 数值验证过的自旋旋转构造更稳

## 8.4 后续若要做 irrep 标签

如果第二阶段要进一步输出“不可约表示标签”，那就需要增加：

1. little-group 数据库存取
2. 对应空间群标准 k 点与当前 `k` 的识别
3. character decomposition
4. Herring rule / 双值表示处理

这一步可以优先参考：

- `IRVSP/nonsymm.f90`
- `test-abacus-2/export_abacus_tqc.py`

## 9. 当前最可靠的技术判断

1. `pyatb` 做特征标的数值核心，应该建立在 ABACUS 基底表示矩阵 `D_g(k)` 上，而不是建立在平面波映射 `ROTKV` 那套逻辑上。
2. 因为 `pyatb` 当前本来就能给出 `H(k)`、`S(k)`、本征值与本征矢，所以实现 character 功能的关键增量是“对称操作矩阵构造”，不是“再造一个求解器”。
3. ABACUS / pyatb 的本征矢不应当默认视作正交归一基下的列向量，后续 character 计算应保留 `S(k)`。
4. SOC 下必须非常谨慎处理基底顺序与 half-angle 自旋旋转约定；这点已经在 `test-abacus` 里有现成数值验证依据。

## 10. 作为后续工作的直接入口

如果下一步开始在 `pyatb-main/src` 中增加 `symmetry/` 目录，最自然的顺序应是：

1. 在 `pyatb` 内复现 ABACUS `Orbital` 文件解析，得到 shell 列表
2. 在 `pyatb` 内复现 ABACUS 结构 + `spglib` 空间群操作读取
3. 实现局域轨道 / 自旋旋转块
4. 实现原子映射与 Bloch 相位
5. 实现 `D_g(k)`
6. 接到 `pyatb` 已有 `H/S` 求解流程上做单 k 点 character 输出
7. 最后再考虑 high-symmetry k 点识别、irrep 标签和 trace / tqc 导出

