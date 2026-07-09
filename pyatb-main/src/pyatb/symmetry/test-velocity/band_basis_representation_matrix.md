# 能带基下对称表示与速度矩阵协变关系

本文档整理从轨道基到能带基的表示矩阵，以及如何用对称操作把
\(k\) 点的能带速度矩阵搬运到非小群目标点 \(k'\)。

核心结论先写在前面：

1. \(D_{\mathrm{orb}}\) 是轨道 Bloch 基之间的系数映射，因此转到能带基时使用
   \(C^\dagger S D_{\mathrm{orb}} C\)。
2. 速度矩阵 \(V_{\mathrm{orb}}\) 若已经是轨道基算符矩阵元，则转到能带基时使用
   \(C^\dagger V_{\mathrm{orb}} C\)，不要额外乘 \(S\)。
3. 对非小群操作 \(g:k\rightarrow k'\)，如果接受由对称操作定义的 transported gauge，
   那么 \(k'\) 点能带速度矩阵可直接由 \(k\) 点速度矩阵得到：

   \[
   V^{\mathrm{tr}}_\alpha(k')
   =
   \sum_\beta O_{\alpha\beta}(g) V_\beta(k)
   \qquad \text{unitary},
   \]

   \[
   V^{\mathrm{tr}}_\alpha(k')
   =
   -\operatorname{conj}
   \left[
     \sum_\beta O_{\alpha\beta}(g) V_\beta(k)
   \right]
   \qquad \text{antiunitary}.
   \]

   这里不需要在 \(k'\) 点重新对角化。若要换到 pyatb 在 \(k'\) 点直接对角化得到的
   数值 gauge，才需要 \(C(k')\)。

## 1. k 点映射和记号

直接坐标下的 k 点采用当前 pyatb 速度协变性脚本中的行向量约定：

\[
k_g = k R^{-1}
\qquad \text{unitary},
\]

\[
k_g = -k R^{-1}
\qquad \text{antiunitary}.
\]

这里 \(R\) 是空间操作在直接坐标下的整数旋转矩阵，代码中通常写作
`k @ inv(R)`。如果 \(k_g\) 不在第一布里渊区，把它写成

\[
k_g = k' + G,
\]

其中 \(k'\) 是 wrap 后的目标点，\(G\) 是整数倒格矢。小群是特殊情况：

\[
k_g = k + G.
\]

轨道 Bloch 基组记为

\[
\{|\chi_\mu(k)\rangle\}.
\]

能带态写作

\[
|\psi_n(k)\rangle
= \sum_\mu |\chi_\mu(k)\rangle C_{\mu n}(k),
\]

矩阵形式为

\[
|\psi(k)\rangle = |\chi(k)\rangle C(k).
\]

ABACUS/pyatb 使用非正交轨道基组，因此本征矢满足

\[
C^\dagger(k) S(k) C(k) = I,
\]

其中

\[
S_{\mu\nu}(k)=\langle \chi_\mu(k)|\chi_\nu(k)\rangle.
\]

## 2. 非正交基下必须区分两类对象

非正交基中最容易混淆的是“轨道系数向量或系数映射”和“算符矩阵元”。

### 2.1 轨道系数向量

设某个态在轨道基下的系数为 \(a_{\mathrm{orb}}\)：

\[
|\phi\rangle=|\chi(k)\rangle a_{\mathrm{orb}}.
\]

它在能带基下的系数是

\[
a_{\mathrm{band}}
= C^\dagger(k) S(k) a_{\mathrm{orb}}.
\]

这是因为

\[
a_{\mathrm{band},n}
= \langle \psi_n(k)|\phi\rangle
= C_n^\dagger(k) S(k) a_{\mathrm{orb}}.
\]

因此 \(C^\dagger S\) 是把轨道系数投影到能带系数的左投影。

### 2.2 轨道系数映射

如果一个矩阵 \(A_{\mathrm{orb}}\) 是轨道系数映射，即

\[
|\chi_{\mathrm{target}}\rangle A_{\mathrm{orb}} a
\]

表示某个操作作用在源点轨道系数 \(a\) 之后得到的目标点轨道系数，那么它在能带基中的矩阵为

\[
A_{\mathrm{band}}
= C^\dagger_{\mathrm{target}} S_{\mathrm{target}}
  A_{\mathrm{orb}}
  C_{\mathrm{source}}.
\]

轨道对称表示矩阵 \(D_{\mathrm{orb}}\) 属于这一类。所以 \(D_{\mathrm{band}}\) 公式里会出现
\(S\)。

### 2.3 算符矩阵元

如果 \(O_{\mathrm{orb}}\) 已经是轨道基下的算符矩阵元：

\[
(O_{\mathrm{orb}})_{\mu\nu}
= \langle \chi_\mu(k)|\hat O|\chi_\nu(k)\rangle,
\]

那么能带基矩阵元是

\[
O_{\mathrm{band}}
= C^\dagger(k) O_{\mathrm{orb}}(k) C(k).
\]

这里不额外乘 \(S\)。如果某个代码存储的是 mixed-index 系数映射
\(O_{\mathrm{map}}\)，满足 \(O_{\mathrm{orb}}=S O_{\mathrm{map}}\)，那么才写成

\[
O_{\mathrm{band}}
= C^\dagger S O_{\mathrm{map}} C.
\]

本文后续把 \(V_{\mathrm{orb}}\) 记为当前 pyatb 速度测试中可直接按
\(C^\dagger V_{\mathrm{orb}} C\) 投影的速度矩阵。

## 3. 轨道基下的对称操作

设 \(g=\{R|t\}\) 是一个 unitary 空间群操作。轨道 Bloch 基之间有

\[
g|\chi(k)\rangle
= |\chi(k_g)\rangle D_{\mathrm{orb}}(g,k_g).
\]

如果 \(k_g=k'+G\)，定义折回矩阵 \(F_G\)：

\[
|\chi(k_g)\rangle = |\chi(k')\rangle F_G.
\]

于是

\[
g|\chi(k)\rangle
= |\chi(k')\rangle A_g(k\rightarrow k'),
\]

其中

\[
A_g(k\rightarrow k')
= F_G D_{\mathrm{orb}}(g,k_g).
\]

当前 pyatb 的 `build_dk_matrix` 约定中，Bloch phase 来自 atom mapping 的整数 cell shift，
相位形式为

\[
\exp(-2\pi i\, k_g\cdot L).
\]

在这种 gauge 下通常 \(F_G=I\)。如果 Bloch sum 把原子位置也放入相位，则可能需要

\[
(F_G)_{\mu\nu}
= \delta_{\mu\nu} e^{2\pi i G\cdot\tau_\mu},
\]

具体正负号必须和 \(D_{\mathrm{orb}}\) 的相位约定一致。

反幺正操作写作 \(gT\)。nsoc 情况下 \(U_T=I\)；SOC 情况下 \(U_T\) 是自旋时间反演矩阵。
对任意源点轨道系数 \(c\)，有

\[
gT\,|\chi(k)\rangle c
= |\chi(k')\rangle
   A_g(k\rightarrow k') U_T c^*.
\]

## 4. 一般非小群 \(k\rightarrow k'\)：transported gauge

如果 \(g\) 是体系的对称操作，且 \(k_g=k'+G\)，则可以用源点本征矢直接定义目标点的一套
transported gauge。

unitary 情况：

\[
C_{\mathrm{tr}}(k')
= A_g(k\rightarrow k') C(k).
\]

antiunitary 情况：

\[
C_{\mathrm{tr}}(k')
= A_g(k\rightarrow k') U_T C^*(k).
\]

这组 \(C_{\mathrm{tr}}(k')\) 是从 \(k\) 点波函数经对称操作搬运到 \(k'\) 点得到的本征矢。
在精确对称和完整能带空间内，它满足

\[
C_{\mathrm{tr}}^\dagger(k') S(k') C_{\mathrm{tr}}(k') = I.
\]

它与直接在 \(k'\) 点数值对角化得到的本征矢 \(C_{\mathrm{num}}(k')\) 一般不同：

\[
C_{\mathrm{tr}}(k') = C_{\mathrm{num}}(k') B.
\]

非简并时 \(B\) 至少包含任意相位；简并时 \(B\) 可以是简并子空间中的任意 unitary 混合。
如果最终物理量是规范无关的，可以直接使用 transported gauge，不必追求和
\(C_{\mathrm{num}}(k')\) 完全一致。

## 5. 非小群速度矩阵的基本关系

设源点能带速度矩阵为

\[
[V_\beta(k)]_{nm}
= \langle \psi_n(k)|\hat v_\beta|\psi_m(k)\rangle.
\]

目标点 transported gauge 下的速度矩阵定义为

\[
[V^{\mathrm{tr}}_\alpha(k')]_{nm}
=
\langle \psi^{\mathrm{tr}}_n(k')|
\hat v_\alpha
|\psi^{\mathrm{tr}}_m(k')\rangle.
\]

速度是 Cartesian 矢量。按当前脚本约定，Cartesian 分量用矩阵
\(O_{\alpha\beta}(g)\) 混合：

\[
V_{\mathrm{mix},\alpha}(k)
= \sum_\beta O_{\alpha\beta}(g) V_\beta(k).
\]

unitary 操作下，非小群 \(k\rightarrow k'\) 的 transported-gauge 速度矩阵为

\[
\boxed{
V^{\mathrm{tr}}_\alpha(k')
=
\sum_\beta O_{\alpha\beta}(g) V_\beta(k)
}
\qquad \text{unitary}.
\]

反幺正操作下，速度是时间反演奇的，并且矩阵元出现逐元素复共轭：

\[
\boxed{
V^{\mathrm{tr}}_\alpha(k')
=
-\operatorname{conj}
\left[
  \sum_\beta O_{\alpha\beta}(g) V_\beta(k)
\right]
}
\qquad \text{antiunitary}.
\]

这里的 \(\operatorname{conj}\) 是逐元素复共轭，不是 Hermitian conjugate。这个公式已经是在
能带指标下的关系；指标 \(n,m\) 是从 \(k\) 点通过对称操作搬运到 \(k'\) 点的 band label。

因此，如果你的目标不是复现 pyatb 直接对角化 \(k'\) 的数值 gauge，而是得到一个合法的
\(k'\) 点能带 gauge 下的速度矩阵，那么无需 \(C(k')\)。只要知道 \(g\) 的
Cartesian 旋转矩阵 \(O\)，就能从 \(V(k)\) 得到 \(V^{\mathrm{tr}}(k')\)。

## 6. 换到任意目标点能带 gauge

如果仍希望把 transported gauge 下的结果换到另一套目标点能带 gauge
\(C_a(k')\)，例如 pyatb 直接对角化得到的 gauge，则定义

\[
B_{a\leftarrow \mathrm{tr}}
= C_a^\dagger(k') S(k') C_{\mathrm{tr}}(k').
\]

unitary 情况：

\[
B_{a\leftarrow \mathrm{tr}}
=
C_a^\dagger(k') S(k')
A_g(k\rightarrow k') C(k).
\]

antiunitary 情况：

\[
B_{a\leftarrow \mathrm{tr}}
=
C_a^\dagger(k') S(k')
A_g(k\rightarrow k') U_T C^*(k).
\]

它满足

\[
C_{\mathrm{tr}}(k') = C_a(k') B_{a\leftarrow \mathrm{tr}}.
\]

因此速度矩阵换 gauge 为

\[
\boxed{
V^a_\alpha(k')
=
B_{a\leftarrow \mathrm{tr}}
V^{\mathrm{tr}}_\alpha(k')
B_{a\leftarrow \mathrm{tr}}^\dagger
}.
\]

代入第 5 节可得 unitary 情况：

\[
V^a_\alpha(k')
=
B
\left[
  \sum_\beta O_{\alpha\beta}(g) V_\beta(k)
\right]
B^\dagger.
\]

antiunitary 情况：

\[
V^a_\alpha(k')
=
B
\left[
  -\operatorname{conj}
  \left(
    \sum_\beta O_{\alpha\beta}(g) V_\beta(k)
  \right)
\right]
B^\dagger.
\]

这里 \(B=B_{a\leftarrow \mathrm{tr}}\)。当 \(C_a=C_{\mathrm{tr}}\) 时，\(B=I\)，回到
transported gauge 的简单公式。

## 7. 直接用轨道基速度矩阵构造目标点速度矩阵

如果你不是从 \(V_{\mathrm{band}}(k)\) 出发，而是有目标点的轨道基速度矩阵
\(V_{\mathrm{orb},\alpha}(k')\)，则在 transported gauge 下也可以直接投影：

\[
V^{\mathrm{tr}}_\alpha(k')
=
C_{\mathrm{tr}}^\dagger(k')
V_{\mathrm{orb},\alpha}(k')
C_{\mathrm{tr}}(k').
\]

按对称性，这个结果应与第 5 节的

\[
\sum_\beta O_{\alpha\beta} V_\beta(k)
\]

在数值误差内一致。反幺正时应与

\[
-\operatorname{conj}
\left[
  \sum_\beta O_{\alpha\beta} V_\beta(k)
\right]
\]

一致。

注意这里仍然是 \(C^\dagger V_{\mathrm{orb}} C\)，不是
\(C^\dagger S V_{\mathrm{orb}} C\)，前提是 \(V_{\mathrm{orb}}\) 是算符矩阵元形式。

## 8. 小群是一般公式的特殊情况

小群条件是

\[
k_g = k + G.
\]

此时目标物理点就是源点。若选择目标 gauge 仍为原来的 \(C(k)\)，则第 6 节中的
\(B\) 变为小群能带表示矩阵。

unitary：

\[
D_{\mathrm{band}}(g,k)
=
C^\dagger(k) S(k)
F_G D_{\mathrm{orb}}(g,k_g)
C(k).
\]

antiunitary：

\[
D_{\mathrm{band}}(gT,k)
=
C^\dagger(k) S(k)
F_G D_{\mathrm{orb}}(g,k_g)
U_T C^*(k).
\]

若当前 pyatb gauge 下 \(F_G=I\)，则可简化为

\[
D_{\mathrm{band}}(g,k)
=
C^\dagger(k) S(k)
D_{\mathrm{orb}}(g,k_g)
C(k).
\]

小群速度协变性就是第 6 节换 gauge 公式在 \(k'=k\)、\(C_a=C(k)\) 下的结果。

unitary：

\[
V_\alpha(k)
=
D_{\mathrm{band}}(g,k)
\left[
  \sum_\beta O_{\alpha\beta}(g) V_\beta(k)
\right]
D_{\mathrm{band}}^\dagger(g,k).
\]

antiunitary：

\[
V_\alpha(k)
=
D_{\mathrm{band}}(gT,k)
\left[
  -\operatorname{conj}
  \left(
    \sum_\beta O_{\alpha\beta}(g) V_\beta(k)
  \right)
\right]
D_{\mathrm{band}}^\dagger(gT,k).
\]

如果选择的是 transported gauge，而不是原来的 \(C(k)\) gauge，则小群情况下也可以让
\(D_{\mathrm{band}}\) 不出现；此时比较的是另一套合法 gauge 下的速度矩阵。

## 9. 简并子空间和 band window

实际计算时必须注意完整子空间。

设一个能带子空间 \(I\) 包含 band index

\[
n=n_{\mathrm{start}},\ldots,n_{\mathrm{stop}}.
\]

取该块本征矢

\[
C_I(k)=C(k)[:,n_{\mathrm{start}}:n_{\mathrm{stop}}].
\]

如果该窗口包含完整的对称相关子空间，则 \(B_I\) 或 \(D^I_{\mathrm{band}}\) 应接近 unitary：

\[
(B_I)^\dagger B_I - I \approx 0.
\]

如果简并块被截断，或者 \(k\) 与 \(k'\) 的 band window 不对应，则 \(B_I\) 一般不会保持
unitary，速度矩阵协变性也会看起来失败。

非简并带之间通常只差相位；简并子空间内可以差任意 unitary 混合。规范无关物理量应当对完整子空间
取 trace、sum 或其他 gauge-covariant 组合，而不是依赖单个矩阵元。

## 10. 实际使用流程

### 10.1 只需要 transported gauge 下的 \(k'\) 速度矩阵

对每个源点 \(k\) 和对称操作 \(g\)：

1. 计算

   \[
   k_g =
   \begin{cases}
   kR^{-1}, & \text{unitary},\\
   -kR^{-1}, & \text{antiunitary}.
   \end{cases}
   \]

2. wrap 得到 \(k'\)，记录 \(G=k_g-k'\)。
3. 取源点能带速度矩阵 \(V_\beta(k)\)。
4. 用 Cartesian rotation 混合分量：

   \[
   V_{\mathrm{mix},\alpha}
   =
   \sum_\beta O_{\alpha\beta}V_\beta(k).
   \]

5. unitary 时：

   \[
   V^{\mathrm{tr}}_\alpha(k') = V_{\mathrm{mix},\alpha}.
   \]

6. antiunitary 时：

   \[
   V^{\mathrm{tr}}_\alpha(k')
   =
   -\operatorname{conj}(V_{\mathrm{mix},\alpha}).
   \]

这个流程不需要 \(C(k')\)，也不需要 \(D_{\mathrm{band}}\)。如果要显式保存目标点波函数 gauge，
则使用

\[
C_{\mathrm{tr}}(k')
=F_GD_{\mathrm{orb}}(g,k_g)C(k)
\]

或反幺正的

\[
C_{\mathrm{tr}}(k')
=F_GD_{\mathrm{orb}}(g,k_g)U_TC^*(k).
\]

### 10.2 需要和直接对角化的 \(k'\) gauge 对齐

如果要和 pyatb 直接对角化 \(k'\) 后的 band gauge 对齐：

1. 在 \(k'\) 点对角化，得到 \(C_a(k')\)、\(S(k')\)。
2. 构造

   \[
   B=C_a^\dagger(k')S(k')F_GD_{\mathrm{orb}}(g,k_g)C(k)
   \]

   或反幺正的

   \[
   B=C_a^\dagger(k')S(k')F_GD_{\mathrm{orb}}(g,k_g)U_TC^*(k).
   \]

3. 把 transported-gauge 速度矩阵换 gauge：

   \[
   V^a_\alpha(k')=B V^{\mathrm{tr}}_\alpha(k')B^\dagger.
   \]

这一步会受到数值对角化相位、简并混合和 band 排序的影响，但物理量应保持 gauge 不变。

### 10.3 小群检查

当 \(k_g=k+G\) 时，可以选择目标 gauge 为源点原始 gauge \(C(k)\)，构造

\[
D_{\mathrm{band}}^I
=
C_I^\dagger(k) S(k) F_GD_{\mathrm{orb}}(g,k_g) C_I(k),
\]

反幺正时：

\[
D_{\mathrm{band}}^I
=
C_I^\dagger(k) S(k) F_GD_{\mathrm{orb}}(g,k_g) U_T C_I^*(k).
\]

然后按第 8 节公式检查速度矩阵协变性。

## 11. 常见错误

1. 把轨道系数向量投影公式

   \[
   a_{\mathrm{band}}=C^\dagger S a_{\mathrm{orb}}
   \]

   误用到速度算符矩阵上。若 \(V_{\mathrm{orb}}\) 是算符矩阵元，应使用

   \[
   V_{\mathrm{band}}=C^\dagger V_{\mathrm{orb}} C.
   \]

2. 把 \(D_{\mathrm{orb}}\) 当作普通算符矩阵元。它是轨道 Bloch 基之间的系数映射，所以
   \(D_{\mathrm{band}}\) 需要 \(C^\dagger S D_{\mathrm{orb}}C\)。

3. 非小群 \(k\rightarrow k'\) 时，误以为一定要对角化 \(k'\)。如果接受 transported gauge，
   速度矩阵直接按第 5 节变换即可；只有要换到直接对角化 gauge 时才需要 \(C(k')\)。

4. 对反幺正操作漏掉速度的时间反演奇号：

   \[
   V \rightarrow -\operatorname{conj}(V_{\mathrm{mix}}).
   \]

5. 对 \(k_g\) wrap 后忘记 \(F_G\)。当前 pyatb gauge 下通常 \(F_G=I\)，但如果 Bloch gauge 改变，
   必须重新检查。

6. 在被截断的简并块里检查 \(B^\dagger B=I\) 或速度协变性。必须使用完整的对称相关子空间。

## 12. 推荐公式汇总

非小群、transported gauge、unitary：

\[
\boxed{
V^{\mathrm{tr}}_\alpha(k')
=
\sum_\beta O_{\alpha\beta}(g) V_\beta(k)
}
\]

非小群、transported gauge、antiunitary：

\[
\boxed{
V^{\mathrm{tr}}_\alpha(k')
=
-\operatorname{conj}
\left[
  \sum_\beta O_{\alpha\beta}(g) V_\beta(k)
\right]
}
\]

换到任意 \(k'\) gauge：

\[
\boxed{
V^a_\alpha(k')
=
B_{a\leftarrow \mathrm{tr}}
V^{\mathrm{tr}}_\alpha(k')
B_{a\leftarrow \mathrm{tr}}^\dagger
}
\]

其中

\[
B_{a\leftarrow \mathrm{tr}}
=
C_a^\dagger(k')S(k')F_GD_{\mathrm{orb}}(g,k_g)C(k)
\]

或反幺正的

\[
B_{a\leftarrow \mathrm{tr}}
=
C_a^\dagger(k')S(k')F_GD_{\mathrm{orb}}(g,k_g)U_TC^*(k).
\]

小群、源点原始 gauge、unitary：

\[
\boxed{
V_\alpha(k)
=
D_{\mathrm{band}}(g,k)
\left[
  \sum_\beta O_{\alpha\beta}(g)V_\beta(k)
\right]
D_{\mathrm{band}}^\dagger(g,k)
}
\]

其中

\[
D_{\mathrm{band}}(g,k)
=
C^\dagger(k)S(k)F_GD_{\mathrm{orb}}(g,k_g)C(k).
\]

速度矩阵投影：

\[
\boxed{
V_{\mathrm{band},\alpha}(k)
=
C^\dagger(k)V_{\mathrm{orb},\alpha}(k)C(k)
}
\]

系数映射投影：

\[
\boxed{
A_{\mathrm{band}}
=
C^\dagger_{\mathrm{target}}S_{\mathrm{target}}
A_{\mathrm{orb}}
C_{\mathrm{source}}
}
\]
