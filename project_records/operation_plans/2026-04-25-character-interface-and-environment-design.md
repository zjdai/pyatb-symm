# CHARACTER 接口与环境搭建设计

日期：2026-04-25  
工作区：`/home/zjdai/file-test/pyatb_symm`

## 1. 目标

本设计仅覆盖第一阶段工作：

1. 在 `pyatb` 的 `Input` 中新增 `CHARACTER` 功能块接口。
2. 把 `CHARACTER` 接到 `read_input()` 和 `main.py` 的功能分发流程中。
3. 新增 `CHARACTER` 模块骨架，但不在本阶段实现真正的特征标算法。
4. 新建 `symm` conda 环境。
5. 在 `symm` 环境内安装 `pyatb`。
6. 在 `symm` 环境内编译 `irvsp` 和 `ir2tb`。

本阶段不做以下内容：

1. 不实现空间群自动识别逻辑。
2. 不实现 little group 构造。
3. 不实现 `H(k)` 空间旋转矩阵构造。
4. 不实现 character 计算。
5. 不实现 irrep 判定。
6. 不输出真实 `trace.txt`、`tqc.txt`、`tqc.data`。

本阶段的核心目标是把“输入接口 + 运行入口 + 模块骨架 + 独立环境”先稳定下来。

## 2. 输入接口设计

### 2.1 块名

新增一个新的大写功能块：

`CHARACTER`

### 2.2 参数名风格

块内参数保持 `pyatb` 现有风格，使用小写：

- `nspin`
- `group`
- `symm_prec`
- `occ_band`
- `band`
- `mag_tag`
- `mag`

这样做的原因是：

1. `pyatb` 现有输入系统全部按小写参数名组织。
2. 若块内参数也改成大写，会扩大输入解析层的改动范围。
3. 当前阶段应优先最小化接口接入改动。

### 2.3 参数语义

#### `nspin`

含义：

- 该模块内部使用的自旋模式。

允许值：

- `1`
- `2`
- `4`

默认值：

- 无默认值，必须在 `CHARACTER` 块中显式设置。

说明：

- 这里先允许与 `INPUT_PARAMETERS.nspin` 并存。
- 第一阶段仅检查合法值与一致性，不在这里重新定义求解器行为。

#### `group`

含义：

- 空间群来源。

允许值：

- `auto`
- 正整数空间群编号

默认值：

- `auto`

确定规则：

- `auto` 表示后续阶段从结构自动判断空间群编号。
- 手动模式仅接受空间群编号整数，不接受 Hermann-Mauguin 符号。

#### `symm_prec`

含义：

- 调用 `spglib` 时的对称性判定精度。

允许值：

- 正浮点数

默认值：

- 建议默认 `1e-5`

说明：

- 第一阶段仅保存和检查该参数，不执行 `spglib`。

#### `occ_band`

含义：

- 占据带数。

允许值：

- 正整数

默认值：

- 无默认值，必须显式设置。

#### `band`

含义：

- 需要分析的能带范围，两个正整数分别表示起始带和终止带。

允许值：

- 两个正整数

默认值：

- 无默认值，必须显式设置。

约束：

- `band[0] <= band[1]`

#### `mag_tag`

含义：

- 是否考虑磁性输入。

允许值：

- `0`
- `1`

默认值：

- `0`

#### `mag`

含义：

- 磁矩来源。

允许值：

- `auto`
- 手动输入的扁平浮点列表

默认值：

- `auto`

手动输入格式：

例如：

```text
mag 0 0 5  0 0 -5
```

即按每个磁性原子的三个分量顺序依次写入。

第一阶段约束：

- 若为手动列表，则长度必须是 `3N`。
- 第一阶段不去判断 `N` 是否已经与结构中磁性原子数严格匹配，后续阶段再补。

## 3. 解析层设计

## 3.1 目标

让现有 `pyatb` 输入系统在不大改框架的前提下支持：

1. 固定参数
2. `auto / int` 混合类型参数 `group`
3. `auto / 可变长列表` 混合类型参数 `mag`

## 3.2 文件落点

需要修改：

- `pyatb-main/src/pyatb/io/default_input.py`
- `pyatb-main/src/pyatb/io/input.py`

## 3.3 `default_input.py` 改动

### 3.3.1 新增功能开关

在 `function_switch` 中新增：

```python
'CHARACTER': False
```

### 3.3.2 新增输入块定义

在 `INPUT` 字典中新增：

```python
'CHARACTER':
{
    'nspin'      : [int, 1, None],
    'group'      : [str, 1, 'auto'],
    'symm_prec'  : [float, 1, 1e-5],
    'occ_band'   : [int, 1, None],
    'band'       : [int, 2, None],
    'mag_tag'    : [int, 1, 0],
    'mag'        : [str, 1, 'auto']
}
```

这里先将 `group` 和 `mag` 以字符串入口形式挂入默认输入系统，再由后处理做类型转换。

原因：

1. `pyatb` 当前 `get_general_parameter()` 适合固定长度、固定类型字段。
2. `group` 和 `mag` 都是混合模式参数，直接放进通用解析器会使框架改动变大。
3. 第一阶段以局部特判方式接入更安全。

## 3.4 `input.py` 改动

### 3.4.1 基本策略

保留现有通用解析流程不动，只对 `CHARACTER` 做局部附加解析。

### 3.4.2 `group` 解析

规则：

1. 如果读到 `auto`，则内部保持为字符串 `'auto'`
2. 如果不是 `auto`，则尝试转成整数
3. 若转整数失败，则报错
4. 若整数不为正，则报错

### 3.4.3 `mag` 解析

规则：

1. 若读到 `auto`，则内部保持为字符串 `'auto'`
2. 若不是 `auto`，则从 `mag` 所在位置开始，把直到下一个已知参数名之前的所有 token 都读成浮点数
3. 最终转成一维 `numpy.ndarray`
4. 若长度不是 `3N`，则报错

### 3.4.4 附加一致性检查

在 `check()` 或 `parameter_require_additional_operations()` 后增加对 `CHARACTER` 的检查：

- `nspin in {1, 2, 4}`
- `symm_prec > 0`
- `occ_band > 0`
- `band` 两个值都为正
- `band[0] <= band[1]`
- `mag_tag in {0, 1}`
- 若 `group != 'auto'`，则为正整数
- 若 `mag != 'auto'`，则为一维浮点数组且长度可被 3 整除

建议同时检查：

- `CHARACTER.nspin == INPUT_PARAMETERS.nspin`

第一阶段若不一致，直接报错，而不是做隐式覆盖。

## 4. 主流程接入设计

## 4.1 文件落点

需要修改：

- `pyatb-main/src/pyatb/main.py`

需要新增：

- `pyatb-main/src/pyatb/symmetry/__init__.py`
- `pyatb-main/src/pyatb/symmetry/character.py`

## 4.2 `main.py` 改动

新增分支：

```python
if function_switch['CHARACTER']:
    character_parameters = INPUT['CHARACTER']
    cal_CHARACTER = Character(m_tb)
    cal_CHARACTER.calculate_character(**character_parameters, **input_parameters)
```

说明：

1. 仍沿用现有模块化风格。
2. `Character` 类与现有 `Band_Structure`、`PDOS`、`Spin_Texture` 等模块并列。
3. 第一阶段不要求它真正完成物理计算，只要求链路能完整走通。

## 4.3 `symmetry` 子包结构

第一阶段仅新增两个文件：

### `src/pyatb/symmetry/__init__.py`

职责：

- 导出 `Character` 类。

### `src/pyatb/symmetry/character.py`

职责：

1. 建立 `Character` 骨架类。
2. 创建输出目录。
3. 打印参数和模块标题。
4. 做参数校验。
5. 给出明确的“未实现”提示。

第一阶段不在这个文件中加入：

1. `spglib` 调用
2. `STRU` 自动读取逻辑
3. 小群构造逻辑
4. `H(k)` 旋转矩阵逻辑
5. character 计算逻辑
6. irrep / trace / tqc 输出逻辑

## 5. 骨架模块行为设计

## 5.1 输出目录

建议输出目录：

`OUTPUT/CHARACTER`

行为与现有模块保持一致：

- 若目录已存在，则重建
- 若不存在，则新建

## 5.2 日志输出

在 `RUNNING_LOG` 中写入：

1. 模块标题
2. 参数摘要
3. 当前阶段提示

需要至少记录：

- `nspin`
- `group`
- `symm_prec`
- `occ_band`
- `band`
- `mag_tag`
- `mag` 模式（`auto` 或手动）

## 5.3 参数检查

骨架模块内部再做一次本地检查，避免未来该模块被独立调用时绕过解析层检查。

## 5.4 占位退出行为

第一阶段不允许“成功退出但什么都没做”，也不允许“由于缺少算法在某个深层函数里偶然崩溃”。

建议行为：

- 在所有前置准备都完成后，抛出一个明确异常：

```text
NotImplementedError: CHARACTER calculation kernel is not implemented yet.
```

这样能证明：

1. `Input` 已经解析成功
2. `main.py` 已经进入 `CHARACTER`
3. 输出目录和日志逻辑已生效
4. 当前失败点就是算法尚未实现，而不是入口没接通

## 6. 环境搭建设计

## 6.1 conda 环境

新增环境名：

`symm`

目标：

- 作为本项目独立的对称性开发和测试环境
- 不污染当前 `abacus_env`

## 6.2 Python 依赖

环境中需要安装：

- `python`
- `numpy`
- `scipy`
- `matplotlib`
- `mpi4py`
- `pybind11`
- `spglib`
- `openblas`
- `lapacke`

特殊要求：

- 不安装官方 `ase`
- 必须安装 `ase-abacus`

原因：

- 用户明确指出官方 `ase` 不能直接读取 ABACUS `STRU`
- 后续对称模块一定会涉及 `STRU` 读取与结构解析

## 6.3 `pyatb` 安装方式

建议在 `symm` 环境中使用 editable 安装：

```bash
pip install -e /home/zjdai/file-test/pyatb_symm/pyatb-main
```

原因：

1. 后续还会持续修改源码
2. editable 安装便于边改边测
3. 不需要反复重新打包

## 6.4 `irvsp` 编译

编译目录：

`/home/zjdai/file-test/pyatb_symm/IRVSP-master/src_irvsp_v2_release`

已确认本地存在：

- `Makefile`
- `configure.sh`
- `kLittleGroups/`

注意事项：

1. `Makefile` 使用 `ifort`
2. 编译前需要确保 oneAPI 环境已经加载
3. `IRVSPDATA` 是运行时必须环境变量

处理策略：

- 不直接执行仓库里的 `configure.sh` 去写入 `~/.bashrc`
- 改为在 `symm` 环境的 `etc/conda/activate.d/` 中写一个激活脚本，导出：

```bash
export IRVSPDATA=/home/zjdai/file-test/pyatb_symm/IRVSP-master/src_irvsp_v2_release
```

这样：

1. 只在 `symm` 环境激活时生效
2. 不污染用户全局 shell
3. 更容易复现

## 6.5 `ir2tb` 编译

编译目录：

`/home/zjdai/file-test/pyatb_symm/IRVSP-master/src_ir2tb_v2`

已确认本地存在：

- `Makefile`
- `irrep_bcs.a`

说明：

- 当前仓库中已经有 `src_ir2tb_v2/irrep_bcs.a`
- 因此本阶段不必额外解压或重建该库，直接利用现有静态库即可

注意事项：

1. 其 `Makefile` 使用 `ifort`
2. 链接时依赖 `MKLROOT`
3. 因此同样需要 oneAPI 环境加载

## 7. 验证设计

## 7.1 输入解析验证

目标：

- 验证 `CHARACTER` 块已经接入输入解析系统

测试场景至少包括：

1. `group auto`, `mag auto`
2. `group 166`, `mag auto`
3. `group 166`, 手动 `mag` 列表

检查点：

1. `input.json` 中出现 `CHARACTER`
2. `group` 被正确解析为 `'auto'` 或整数
3. `mag` 被正确解析为 `'auto'` 或一维数组

## 7.2 骨架运行验证

目标：

- 验证 `pyatb.main -> CHARACTER` 的链路已经接通

检查点：

1. 能进入 `Character` 模块
2. 能创建 `OUTPUT/CHARACTER`
3. `running.log` 中有 `CHARACTER` 标题和参数
4. 最终在占位异常处明确退出

## 7.3 环境验证

目标：

- 验证 `symm` 环境可用于后续开发

检查点：

1. `python -c "import pyatb"` 通过
2. `python -c "import ase"` 指向 `ase-abacus` 提供的安装位置
3. `python -c "import spglib"` 通过
4. `irvsp` 可执行
5. `ir2tb` 可执行
6. 激活 `symm` 后存在 `IRVSPDATA`

## 8. 文件清单

第一阶段预计涉及的文件如下。

### 修改

- `pyatb-main/src/pyatb/io/default_input.py`
- `pyatb-main/src/pyatb/io/input.py`
- `pyatb-main/src/pyatb/main.py`

### 新增

- `pyatb-main/src/pyatb/symmetry/__init__.py`
- `pyatb-main/src/pyatb/symmetry/character.py`
- `project_records/operation_plans/2026-04-25-character-interface-and-environment-design.md`

### 可能新增

- `project_records/change_logs/2026-04-25-character-interface-and-environment-implementation-log.md`
- `test_workspace/...` 下的最小接口验证样例
- `symm` 环境的 `activate.d` 脚本

## 9. 风险与控制

## 9.1 输入框架对混合类型参数支持差

风险：

- `group` / `mag` 不能直接复用现有固定类型解析。

控制：

- 只在 `CHARACTER` 上做局部附加解析，不动整套输入框架。

## 9.2 全局 shell 污染

风险：

- 直接执行 `IRVSP` 自带 `configure.sh` 会写 `~/.bashrc`

控制：

- 使用 `symm` 环境自己的激活脚本设置 `IRVSPDATA`

## 9.3 `ase` 包选错

风险：

- 安装官方 `ase` 后续无法直接按用户预期处理 ABACUS `STRU`

控制：

- 明确用 `ase-abacus`，并在验证中检查安装来源

## 9.4 第一阶段“骨架模块”行为不清晰

风险：

- 接口看似接通，但运行时失败点不明确

控制：

- 明确规定：
  - 先完成输出目录与参数检查
  - 再抛 `NotImplementedError`

## 10. 本阶段完成判据

当且仅当以下条件全部满足时，本阶段视为完成：

1. `CHARACTER` 输入块已被 `pyatb` 正确识别。
2. `CHARACTER` 参数可写入并通过解析。
3. `main.py` 可把控制流分发到 `Character` 骨架模块。
4. `Character` 模块可以创建输出目录、写日志、做参数检查。
5. `Character` 模块当前以明确占位异常结束。
6. `symm` conda 环境已创建。
7. `symm` 环境中已安装 `pyatb` 与 `ase-abacus`。
8. `irvsp` 和 `ir2tb` 已编译完成。
9. 上述内容已记录到 git 历史中。
