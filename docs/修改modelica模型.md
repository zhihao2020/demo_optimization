# Modelica 模型 Git 修改对比说明

## 结论

这两次提交的职责不同：`4b083d8` 将完整的 Modelica 典型场景模型导入仓库；紧随其后的 `ef25cdc` 没有重写设备物理模型，而是将其调整为面向 Python/RL 的物理仿真 FMU——Modelica 负责状态演化和物理量输出，经济结算、奖励和约束罚分由 Python 侧计算。

本文对比范围为 `git diff 4b083d8..ef25cdc`，即两个提交之间的 Modelica 改动。

## 提交概览

| 提交 | 时间（+08:00） | 变更规模 | 结论 |
| --- | --- | --- | --- |
| `4b083d8` — `modelida模型` | 2026-07-13 23:24:32 | 新增 14 个 Modelica 源/包文件；整个提交共 15 个文件、55,257 行新增 | 导入完整模型库及其包结构，建立 FMU 的物理侧基线。 |
| `ef25cdc` — `修改modelica模型` | 2026-07-13 23:25:15 | 仅修改 2 个 Modelica 文件：94 行新增、67 行删除 | 重构 FMU 的输出接口和经济计算职责，以支撑 Python/RL。 |

第二次提交只涉及：

- [`PowerSystem_8760h.mo`](../resources/Example/TypicalScene/PowerSystem_8760h.mo)：78 行新增、33 行删除；
- [`TypicalScenarios.mo`](../resources/TypicalScenarios.mo)：16 行新增、34 行删除。

## `PowerSystem_8760h.mo`：从聚合目标输出改为物理观测输出

### 删除的顶层 FMU 输出

以下输出不再导出：

- `OPT_goal`：全年现金流目标；
- `P_res`：同时表示弃电和缺供的带符号功率残差；
- `battery_penalty`、`thermal_penalty`、`caes_gas_penalty`、`caes_hot_penalty`、`caes_cold_penalty`：设备或储罐罚函数。

这意味着新 FMU 不再向 Python 提供内置的优化目标和惩罚项。

### 新增的物理量输出

| 类别 | 输出 |
| --- | --- |
| 电力平衡 | `p_curtailment = max(-bus.P_res, 0)`（弃电）、`p_unserved = max(bus.P_res, 0)`（缺供） |
| 储能状态 | `battery_soc`、`caes_gas_soc`、`caes_hot_soc`、`caes_cold_soc` |
| 设备实际功率 | `p_thermal`、`p_battery`、`p_caes`、`p_grid` |
| 风光与负荷 | `p_wind_available`、`p_wind_actual`、`p_pv_available`、`p_pv_actual`、`p_load_actual` |
| CAES 热力状态 | `caes_gas_pressure`、`caes_gas_temperature`、`caes_hot_temperature`、`caes_cold_temperature` |

新接口把原来混合在 `P_res` 中的两种运行结果拆成两个非负量，也暴露了策略构造 observation、计算约束违规和诊断 CAES 边界所需的原始状态。

### 输入与符号约定

输入仍为 `u_tp`、`u_battery`、`u_caes`，未改动控制主链路。提交补充并明确了以下约定：

- 功率单位为 W，压力为 Pa，温度为 K，SOC 为 0～1；
- 发电为负；用电和充电为正；
- `p_battery`、`p_caes` 正值表示充电、负值表示放电；
- `p_grid` 正值表示购电、负值表示售电；
- `p_curtailment` 与 `p_unserved` 均为非负量。

## `TypicalScenarios.mo`：经济罚函数迁出 Modelica

`ThermalPower`、`Battery`、`CompressedAirEnergyStorage` 和 `Bus` 都移除了 `k`、`C_penality` 或各储罐 `C_*_penality` 变量及其计算式。设备的 `C` 仅保留按实际功率和买/售价计算的现金流；母线 `Income` 仅累加各设备现金流。

移除的主要内容包括：

- 火电出力边界的指数型罚函数；
- 电池 SOC 上下界的指数型罚函数；
- CAES 气罐、热罐、冷罐 SOC 的指数型罚函数；
- 母线功率残差的 `(k * P_res)^2` 罚函数。

`Bus.OPT_goal` 仍作为内部累计现金流保留，但顶层模型不再导出它。

## 为什么这样修改

1. **数值稳定性**：旧罚函数包含 `e^x`。特别是火电公式直接以 W 量级的功率参与指数运算，容易产生 `Inf`/`NaN`，不适合作为 FMU 仿真和训练的稳定奖励来源。
2. **职责分离**：物理模型应提供真实状态和实际功率；电价、惩罚权重、奖励形式和约束处理属于优化/RL 实验逻辑，应由 Python 灵活定义。
3. **观测可用性**：Python 需要原始 SOC、功率平衡、设备实际功率及 CAES 热力状态，才能构造马尔可夫观测、奖励和越界诊断，而不是只能消费一个已经混合了固定权重的目标值。

## 兼容性与验证边界

- 这是 **FMU 输出接口的破坏性变更**：Python 必须停止读取 `OPT_goal`、`P_res` 与 `*_penalty`，并改用新的物理输出名构造结算和奖励。
- 输入 `u_tp`、`u_battery`、`u_caes` 及设备间的物理连接未在本次差异中改变；改动重点是输出接口和经济计算位置，而非设备物理过程。
- 本说明基于 Git 源码对比。未在此对比范围内重新导出 FMU 或运行仿真，因此不能将其视为新 FMU 的运行验证结果；导出后需同步核对 Python 的输出名映射和相关测试。

## 逐行改动记录

以下行号来自 `git diff --unified=0 4b083d8..ef25cdc`。`—` 表示该侧没有对应源码行；记录覆盖两个文件的全部 Git 变更块，不展开未改动的长时序表或查表数据。

### `PowerSystem_8760h.mo`

| 基线行 → 修改后行 | 删除 / 新增内容 | 行为影响 |
| --- | --- | --- |
| `2` → `3–5` | 新增 FMU 职责边界、Python 结算边界及功率/物理量单位注释。 | 明确模型只输出物理状态，避免将 reward/罚分误认为 Modelica 的职责。 |
| `6` → `9` | `u_tp` 注释补充“无量纲，约 `[0,1]`”。 | 不改变输入值或连接，仅明确火电调度输入的含义。 |
| `8` → `11` | `u_battery` 注释补充“正充电、负放电”。 | 固化电池功率方向约定。 |
| `10` → `13` | `u_caes` 注释补充“正充电、负放电”。 | 固化 CAES 功率方向约定。 |
| `13–29` → `16–59` | 删除 `OPT_goal`、`P_res`、5 个 `*_penalty` 顶层输出；新增弃电/缺供、四个 SOC、设备实际功率、风光/负荷功率和 CAES 热力状态输出。 | FMU 输出由“固定优化目标”改为“Python 可自由组合的原始物理观测”。 |
| `227–228` → `257` | 火电实例删除 `k = 1`。 | 对应 `ThermalPower.k` 已删除，实例不再传入火电罚函数权重。 |
| `321` → `—` | 电池实例删除 `k = 0.0001`。 | 对应 `Battery.k` 已删除，SOC 罚分不再由 Modelica 计算。 |
| `328` → `356` | 母线实例从 `Bus bus(k = 1e6)` 改为无参数实例。 | 对应 `Bus.k` 已删除，功率残差惩罚权重迁出。 |
| `671` → `699` | CAES 长参数行删除 `k = 1`，其余时序表和储罐参数不变。 | 对应 CAES 的 SOC 罚函数权重删除；不改变该行携带的查表数据。 |
| `744–751` → `772–777` | 移除旧聚合输出赋值；新增 `p_curtailment`、`p_unserved` 及 `battery_soc` 赋值。 | 将带符号 `bus.P_res` 拆为两个非负指标，保留电池 SOC 但改用新的导出接口。 |
| `—` → `778–797` | 新增 CAES SOC、设备实际功率、风光可用/实际、负荷实际功率和 CAES 压力/温度赋值。 | 使 Python 可直接获得状态、真实执行功率和边界诊断所需量。 |

### `TypicalScenarios.mo`

| 基线行 → 修改后行 | 删除 / 新增内容 | 行为影响 |
| --- | --- | --- |
| `358–360` → `358` | `ThermalPower` 删除 `k` 与 `C_penality`，改为迁出说明注释。 | 火电模型不再持有经济罚函数状态。 |
| `396–399` → `394–395` | 删除火电出力边界罚函数及 `C_penality` 方程；`positivePlug.C` 仅保留燃料/电价现金流。 | 避免指数罚函数造成 `Inf/NaN`；火电约束惩罚交给 Python。 |
| `587` → `583` | `Battery.k` 改为说明注释。 | 电池 SOC 罚函数权重从 Modelica 接口移除。 |
| `590` → `—` | 删除 `Battery.C_penality`。 | 电池不再保存独立罚函数值。 |
| `—` → `603` | 新增电池 SOC 指数罚函数已迁出的注释。 | 说明后续充/放电分支只计算现金流。 |
| `609–612` → `605` | 充电分支删除 SOC 罚函数和 `C_penality` 累加；保留 `PBS.C = -PBS.P_act * PBS.c1 / 3.6e6`。 | 正功率（充电）仅按购电成本结算。 |
| `615–617` → `608` | 放电分支删除 SOC 罚函数和 `C_penality` 累加；保留 `PBS.C = -PBS.P_act * PBS.c2 / 3.6e6`。 | 负功率（放电）仅按售电价格结算。 |
| `677` → `668` | `Bus.k` 改为说明注释。 | 母线不再接受功率失衡罚分系数。 |
| `680` → `671` | 补充 `P_res` 的符号语义：负为弃电、正为缺供，并指出顶层拆分输出。 | `P_res` 仍是内部功率平衡量，但其对外语义更明确。 |
| `683–684` → `674` | 删除 `Bus.C_penality`；将 `OPT_goal` 说明改为“内部累计现金流”。 | `OPT_goal` 保留内部记账，不再代表可直接导出的优化目标。 |
| `753–754` → `743–744` | 删除 `der(Income)` 中的 `- C_penality` 和 `(k * P_res)^2` 方程；仅累加设备现金流。 | 失衡成本不再硬编码在 Modelica，需由 Python 基于弃电/缺供输出计算。 |
| `766` → `756` | CAES 的 `k` 改为说明注释。 | 移除气、热、冷罐 SOC 罚函数权重。 |
| `776–778` → `—` | 删除 `C_GasTank_penality`、`C_HotTank_penality`、`C_ColdTank_penality`。 | CAES 不再对外或内部累加三类储罐罚函数。 |
| `—` → `1020` | 新增 CAES 罚函数迁出的注释。 | 明确下方充/放电分支的现金流计算边界。 |
| `1034–1037` → `1022` | CAES 正功率分支删除三类 SOC 指数罚函数；保留买电现金流方程。 | CAES 充电成本不再附加储罐 SOC 罚分。 |
| `1039–1042` → `1024` | CAES 负功率分支删除三类 SOC 指数罚函数；保留售电现金流方程。 | CAES 放电收益不再附加储罐 SOC 罚分。 |
