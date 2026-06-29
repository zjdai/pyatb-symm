# 速度矩阵协变关系公式整理

本文整理当前 `pyatb-6.27` 速度矩阵协变性测试中使用的公式。重点是区分三件事：

1. 本征矢系数 \(C(k)\) 在对称操作下如何变换。
2. 轨道基组下速度矩阵如何协变。
3. 本征态/能带基组下速度矩阵如何协变。

所有公式都同时写出 NSOC 与 SOC 情况下应使用的协变矩阵。

## 0. 统一记号

直接坐标下 k 点采用行向量约定。空间操作写为 \(g=\{R|t\}\)，其中 \(R\) 是直接坐标下的整数旋转矩阵。当前测试脚本使用

\[
k_g =
\begin{cases}
k R^{-1}, & \text{unitary},\\
-k R^{-1}, & \text{antiunitary}.
\end{cases}
\]

把 \(k_g\) 折回第一布里渊区后记为 \(k'\)。如果

\[
k_g = k' + G,
\]

则 \(G\) 是一个整数倒格矢。小群操作是特殊情况：

\[
k' = k \quad (\mathrm{mod}\;G).
\]

轨道 Bloch 基写为

\[
|\psi_n(k)\rangle
=
\sum_\mu |\chi_\mu(k)\rangle C_{\mu n}(k),
\qquad
|\psi(k)\rangle = |\chi(k)\rangle C(k).
\]

非正交轨道基满足

\[
C^\dagger(k)S(k)C(k)=I.
\]

本文中的 \(V^{\mathrm{orb}}_\alpha(k)\) 指当前 pyatb 速度测试里可直接按

\[
V^{\mathrm{band}}_\alpha(k)
=
C^\dagger(k)V^{\mathrm{orb}}_\alpha(k)C(k)
\]

投影到能带基的速度矩阵。也就是说这里不额外写 \(S\)。若某个程序保存的是 mixed-index 对象
\(V_{\mathrm{map}}\)，并且 \(V^{\mathrm{orb}}=S V_{\mathrm{map}}\)，才应写成
\(C^\dagger S V_{\mathrm{map}}C\)。

## 1. NSOC 与 SOC 的协变矩阵

为避免反幺正操作中的复共轭和线性矩阵混在一起，定义线性协变矩阵
\(\mathcal D_g(k')\)。真正的反幺正操作还包含复共轭 \(K\)。

### 1.1 NSOC, `nspin=1`

NSOC 轨道基没有显式自旋自由度。unitary 操作的线性矩阵记为

\[
\mathcal D_g^{\mathrm{nsoc}}(k')=D_g^{\mathrm{nsoc}}(k').
\]

这里 \(D_g^{\mathrm{nsoc}}\) 包含原子映射、cell-shift Bloch 相位和轨道角动量旋转矩阵。
当前 `build_dk_matrix` 中 cell-shift 相位采用

\[
\exp[-2\pi i\, k'\cdot L].
\]

NSOC 的时间反演矩阵是单位矩阵：

\[
U_T^{\mathrm{nsoc}}=I.
\]

因此 antiunitary 操作的线性部分仍可写为

\[
\mathcal D_{gT}^{\mathrm{nsoc}}(k')
=D_g^{\mathrm{nsoc}}(k')U_T^{\mathrm{nsoc}}
=D_g^{\mathrm{nsoc}}(k'),
\]

但完整作用是

\[
\mathcal D_{gT}^{\mathrm{nsoc}}(k')K.
\]

### 1.2 SOC, `nspin=4`

SOC 轨道基带有 spinor 块。unitary 操作的线性矩阵记为

\[
\mathcal D_g^{\mathrm{soc}}(k')=D_g^{\mathrm{soc}}(k').
\]

\(D_g^{\mathrm{soc}}\) 包含原子映射、Bloch 相位、轨道旋转以及 spin-\(1/2\) 旋转：

\[
D_g^{\mathrm{soc}}
\sim
D_{\mathrm{orbital}}(g)\otimes U_{1/2}(O_g).
\]

这里 \(O_g\) 是 Cartesian 旋转矩阵。对 improper operation，当前代码中 spinor 只取对应的 proper-rotation 部分来构造 \(U_{1/2}\)。

SOC 时间反演矩阵按当前代码约定为

\[
U_T^{\mathrm{soc}}
=
I_{N_{\mathrm{orb}}}\otimes \sigma_y,
\qquad
\sigma_y =
\begin{pmatrix}
0 & -i\\
i & 0
\end{pmatrix}.
\]

这里 \(N_{\mathrm{orb}}\) 是不含自旋的轨道基维度。SOC antiunitary 操作的线性部分为

\[
\mathcal D_{gT}^{\mathrm{soc}}(k')
=D_g^{\mathrm{soc}}(k')U_T^{\mathrm{soc}},
\]

完整作用为

\[
\mathcal D_{gT}^{\mathrm{soc}}(k')K.
\]

后续公式统一使用

\[
\mathcal D(k') =
\begin{cases}
D_g(k'), & \text{unitary},\\
D_g(k')U_T, & \text{antiunitary},
\end{cases}
\]

其中 \(U_T=I\) 用于 NSOC，\(U_T=I_{N_{\mathrm{orb}}}\otimes\sigma_y\) 用于 SOC。

## 2. \(C\) 系数之间的变换关系

### 2.1 transported gauge

如果 \(g\) 把 \(k\) 连到 \(k'\)，则可以直接由 \(C(k)\) 构造 \(k'\) 点的一套 transported gauge。

unitary:

\[
C_{\mathrm{tr}}(k')
=
\mathcal D_g(k') C(k).
\]

antiunitary:

\[
C_{\mathrm{tr}}(k')
=
\mathcal D_{gT}(k') C^*(k).
\]

这就是“通过对称操作搬运过来的” \(k'\) 点本征矢。在精确对称和完整能带空间中，

\[
C_{\mathrm{tr}}^\dagger(k')S(k')C_{\mathrm{tr}}(k')=I.
\]

注意：\(C_{\mathrm{tr}}(k')\) 不一定等于程序直接在 \(k'\) 点对角化得到的
\(C_{\mathrm{num}}(k')\)。两者可以差一个能带子空间内的 unitary 矩阵。

### 2.2 与直接对角化 gauge 的 B-switch

定义

\[
C_{\mathrm{tr}}(k') = C_{\mathrm{num}}(k')B.
\]

由于非正交归一化为 \(C^\dagger S C=I\)，所以

\[
B
=
C_{\mathrm{num}}^\dagger(k')S(k')C_{\mathrm{tr}}(k').
\]

因此 unitary 情况：

\[
B
=
C_{\mathrm{num}}^\dagger(k')S(k')
\mathcal D_g(k')C(k).
\]

antiunitary 情况：

\[
B
=
C_{\mathrm{num}}^\dagger(k')S(k')
\mathcal D_{gT}(k')C^*(k).
\]

这里 \(B\) 的方向与当前 band-basis 速度测试报告一致：

\[
C_{\mathrm{tr}}=C_{\mathrm{num}}B.
\]

所以从 transported gauge 换到直接对角化 gauge 时，速度矩阵使用

\[
V^{\mathrm{num}}=B V^{\mathrm{tr}} B^\dagger.
\]

如果另一个笔记或程序把 switch 矩阵定义成 \(B' = B^\dagger\)，则同一个关系会写成
\(V^{\mathrm{num}}=B'^\dagger V^{\mathrm{tr}}B'\)。

### 2.3 小群下的能带表示矩阵

当 \(k'=k\) 时，\(B\) 就是当前 gauge 下的小群能带表示矩阵，记为 \(\Gamma_g(k)\)。

unitary:

\[
\Gamma_g(k)
=
C^\dagger(k)S(k)\mathcal D_g(k)C(k).
\]

antiunitary:

\[
\Gamma_{gT}(k)
=
C^\dagger(k)S(k)\mathcal D_{gT}(k)C^*(k).
\]

NSOC/SOC 的差异全部在 \(\mathcal D_g\) 和 \(\mathcal D_{gT}\) 中：

\[
\mathcal D_{gT}^{\mathrm{nsoc}}=D_g^{\mathrm{nsoc}},
\qquad
\mathcal D_{gT}^{\mathrm{soc}}=D_g^{\mathrm{soc}}
\left(I_{N_{\mathrm{orb}}}\otimes\sigma_y\right).
\]

## 3. 轨道基组下速度矩阵协变关系

速度是 Cartesian polar vector，同时在时间反演下变号。先定义分量旋转后的源点速度矩阵：

\[
V^{\mathrm{mix}}_\alpha(k)
=
\sum_{\beta=x,y,z}
O_{\alpha\beta}(g)V^{\mathrm{orb}}_\beta(k),
\]

其中 \(O(g)\) 是 Cartesian 旋转矩阵。当前测试脚本对 unitary 和 antiunitary 都用同一个
\(O_{\alpha\beta}\) 做空间分量旋转；antiunitary 的时间反演效应单独体现在负号和复共轭中。

### 3.1 Unitary 操作

unitary 操作下，轨道基速度矩阵满足

\[
\boxed{
V^{\mathrm{orb}}_\alpha(k')
=
\mathcal D_g(k')
V^{\mathrm{mix}}_\alpha(k)
\mathcal D_g^\dagger(k')
}.
\]

展开为

\[
V^{\mathrm{orb}}_\alpha(k')
=
\mathcal D_g(k')
\left[
\sum_\beta O_{\alpha\beta}(g)V^{\mathrm{orb}}_\beta(k)
\right]
\mathcal D_g^\dagger(k').
\]

NSOC 时

\[
\mathcal D_g(k')=D_g^{\mathrm{nsoc}}(k').
\]

SOC 时

\[
\mathcal D_g(k')=D_g^{\mathrm{soc}}(k'),
\]

其中 \(D_g^{\mathrm{soc}}\) 已包含 spinor 旋转。

### 3.2 Antiunitary 操作

antiunitary 操作下要加上反线性复共轭，并且速度在时间反演下变号：

\[
\boxed{
V^{\mathrm{orb}}_\alpha(k')
=
\mathcal D_{gT}(k')
\left[
-\operatorname{conj}
\left(
V^{\mathrm{mix}}_\alpha(k)
\right)
\right]
\mathcal D_{gT}^\dagger(k')
}.
\]

也就是

\[
V^{\mathrm{orb}}_\alpha(k')
=
\mathcal D_{gT}(k')
\left[
-\operatorname{conj}
\left(
\sum_\beta O_{\alpha\beta}(g)V^{\mathrm{orb}}_\beta(k)
\right)
\right]
\mathcal D_{gT}^\dagger(k').
\]

这里 \(\operatorname{conj}\) 是逐元素复共轭，不是 Hermitian conjugate。

NSOC:

\[
\mathcal D_{gT}(k')=D_g^{\mathrm{nsoc}}(k').
\]

SOC:

\[
\mathcal D_{gT}(k')
=
D_g^{\mathrm{soc}}(k')
\left(I_{N_{\mathrm{orb}}}\otimes\sigma_y\right).
\]

## 4. 本征态/能带基组下速度矩阵协变关系

能带基速度矩阵定义为

\[
\left[V^{\mathrm{band}}_\alpha(k)\right]_{nm}
=
\langle \psi_n(k)|\hat v_\alpha|\psi_m(k)\rangle
=
\left[
C^\dagger(k)V^{\mathrm{orb}}_\alpha(k)C(k)
\right]_{nm}.
\]

### 4.1 transported gauge 下的非小群关系

如果使用由对称操作搬运得到的 \(C_{\mathrm{tr}}(k')\)，则速度矩阵关系最简单。

unitary:

\[
\boxed{
V^{\mathrm{tr}}_\alpha(k')
=
\sum_\beta O_{\alpha\beta}(g)V^{\mathrm{band}}_\beta(k)
}.
\]

antiunitary:

\[
\boxed{
V^{\mathrm{tr}}_\alpha(k')
=
-\operatorname{conj}
\left[
\sum_\beta O_{\alpha\beta}(g)V^{\mathrm{band}}_\beta(k)
\right]
}.
\]

这两个公式已经是能带指标 \(n,m\) 下的矩阵关系。这里的 \(n,m\) 是通过对称操作从 \(k\) 点搬运到
\(k'\) 点的 band label。

这一层公式不显式出现 \(\mathcal D_g\)，因为 \(\mathcal D_g\) 已经用于定义
\(C_{\mathrm{tr}}(k')\)。NSOC/SOC 的差别体现在第 2 节的

\[
C_{\mathrm{tr}}=\mathcal D C
\quad \text{或} \quad
C_{\mathrm{tr}}=\mathcal D C^*.
\]

### 4.2 换到直接对角化 gauge 后的非小群关系

如果需要和直接对角化得到的 \(C_{\mathrm{num}}(k')\) 比较，则必须做 B-switch。

unitary:

\[
\boxed{
V^{\mathrm{num}}_\alpha(k')
=
B
\left[
\sum_\beta O_{\alpha\beta}(g)V^{\mathrm{band}}_\beta(k)
\right]
B^\dagger
}.
\]

antiunitary:

\[
\boxed{
V^{\mathrm{num}}_\alpha(k')
=
B
\left[
-\operatorname{conj}
\left(
\sum_\beta O_{\alpha\beta}(g)V^{\mathrm{band}}_\beta(k)
\right)
\right]
B^\dagger
}.
\]

其中

\[
B
=
C_{\mathrm{num}}^\dagger(k')S(k')\mathcal D_g(k')C(k)
\]

用于 unitary 操作，而

\[
B
=
C_{\mathrm{num}}^\dagger(k')S(k')\mathcal D_{gT}(k')C^*(k)
\]

用于 antiunitary 操作。

### 4.3 小群操作

小群是 \(k'=k\) 的特殊情况。如果目标 gauge 仍取 \(C(k)\)，则第 4.2 节的 \(B\) 变成
\(\Gamma_g(k)\) 或 \(\Gamma_{gT}(k)\)。

unitary:

\[
\boxed{
V^{\mathrm{band}}_\alpha(k)
=
\Gamma_g(k)
\left[
\sum_\beta O_{\alpha\beta}(g)V^{\mathrm{band}}_\beta(k)
\right]
\Gamma_g^\dagger(k)
}.
\]

antiunitary:

\[
\boxed{
V^{\mathrm{band}}_\alpha(k)
=
\Gamma_{gT}(k)
\left[
-\operatorname{conj}
\left(
\sum_\beta O_{\alpha\beta}(g)V^{\mathrm{band}}_\beta(k)
\right)
\right]
\Gamma_{gT}^\dagger(k)
}.
\]

其中

\[
\Gamma_g(k)
=
C^\dagger(k)S(k)\mathcal D_g(k)C(k),
\]

\[
\Gamma_{gT}(k)
=
C^\dagger(k)S(k)\mathcal D_{gT}(k)C^*(k).
\]

## 5. 测试输出应如何理解

当前 band-basis 测试里有三类关键量：

1. `transported_diagonal_rotation_error` 和 `transported_matrix_rotation_error`

   检查

   \[
   C^\dagger\left[\sum_\beta O_{\alpha\beta}V^{\mathrm{orb}}_\beta\right]C
   \]

   与

   \[
   \sum_\beta O_{\alpha\beta}
   \left[C^\dagger V^{\mathrm{orb}}_\beta C\right]
   \]

   是否一致。这是投影和 Cartesian 分量旋转的内部一致性检查。

2. `direct_diagonal_error_before_U` 和 `block_trace_error_before_U`

   比较 \(V^{\mathrm{tr}}\) 与直接 \(C_{\mathrm{num}}(k')\) gauge 下的结果，但还没有做
   B-switch。非简并带的对角元通常可直接比；简并块内单个对角元会受 gauge 混合影响，
   block trace 更可靠。

3. `basis_switch_velocity_error`

   检查完整矩阵关系

   \[
   V^{\mathrm{num}}_\alpha(k')
   =
   B V^{\mathrm{tr}}_\alpha(k')B^\dagger.
   \]

   这是和直接对角化 gauge 对齐后的最终矩阵元检查。若 band window 包含完整的对称相关子空间，
   同时 \(B^\dagger B\approx I\)，这个误差应接近数值精度。

## 6. 实际使用时的最短流程

从 \(k\) 点能带速度矩阵得到 \(k'\) 点结果：

1. 用 \(k_g=kR^{-1}\) 或 \(k_g=-kR^{-1}\) 找到目标点 \(k'\)。
2. 若只需要 transported gauge：

   \[
   V^{\mathrm{tr}}_\alpha(k')
   =
   \sum_\beta O_{\alpha\beta}V_\beta(k)
   \]

   或 antiunitary 的

   \[
   V^{\mathrm{tr}}_\alpha(k')
   =
   -\operatorname{conj}
   \left[
   \sum_\beta O_{\alpha\beta}V_\beta(k)
   \right].
   \]

3. 若需要和直接对角化的 \(k'\) 结果逐矩阵元比较，先构造

   \[
   B=C_{\mathrm{num}}^\dagger(k')S(k')C_{\mathrm{tr}}(k'),
   \]

   再比较

   \[
   V^{\mathrm{num}}_\alpha(k')=B V^{\mathrm{tr}}_\alpha(k')B^\dagger.
   \]

4. 构造 \(C_{\mathrm{tr}}\) 时，必须使用对应的 NSOC/SOC 协变矩阵：

   \[
   C_{\mathrm{tr}}^{\mathrm{nsoc}}
   =
   D_g^{\mathrm{nsoc}}C
   \quad \text{or} \quad
   D_g^{\mathrm{nsoc}}C^*,
   \]

   \[
   C_{\mathrm{tr}}^{\mathrm{soc}}
   =
   D_g^{\mathrm{soc}}C
   \quad \text{or} \quad
   D_g^{\mathrm{soc}}
   \left(I_{N_{\mathrm{orb}}}\otimes\sigma_y\right)
   C^*.
   \]

这里前一个公式对应 unitary，后一个公式对应 antiunitary。
