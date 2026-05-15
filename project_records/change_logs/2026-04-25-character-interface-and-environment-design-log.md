# 修改记录：CHARACTER 接口与环境搭建设计

日期：2026-04-25

## 本次完成内容

1. 明确第一阶段只做 `CHARACTER` 接口接入、模块骨架和独立环境搭建。
2. 固定 `CHARACTER` 输入块的参数集合与语义：
   - `nspin`
   - `group`
   - `symm_prec`
   - `occ_band`
   - `band`
   - `mag_tag`
   - `mag`
3. 固定手动 `group` 仅接受空间群编号整数。
4. 固定 `mag` 手动输入采用扁平三分量列表格式。
5. 确定 `CHARACTER` 第一阶段为“可进入、可校验、明确占位退出”的骨架模块，而不是空接线。
6. 确定独立 conda 环境名为 `symm`。
7. 确定环境中必须使用 `ase-abacus`，而不是官方 `ase`。
8. 确定对称性环境在第一阶段就安装 `spglib`。
9. 确定 `IRVSPDATA` 不写入全局 `~/.bashrc`，而写入 `symm` 环境激活脚本。

## 本次输出文件

- `project_records/operation_plans/2026-04-25-character-interface-and-environment-design.md`

## 对后续实现的作用

这份设计文档已经把第一阶段接口、骨架和环境的边界固定。后续实现阶段可以直接按文档逐项执行，不必再次重新定义输入协议和环境策略。
