# Modelica 模型 Git 修改对比说明

## 结论

这两次提交的职责不同：`4b083d8` 将完整的 Modelica 典型场景模型导入仓库；紧随其后的 `ef25cdc` 没有重写设备物理模型，而是将其调整为面向 Python/RL 的物理仿真 FMU——Modelica 负责状态演化和物理量输出，经济结算、奖励和约束罚分由 Python 侧计算。

本文对比范围为 `git diff 4b083d8..ef25cdc`，即两个提交之间的 Modelica 改动。

## 提交概览

| 提交 | 时间（+08:00） | 变更规模 | 结论 |
| --- | --- | --- | --- |
| `4b083d8` — `modelida模型` | 2026-07-13 23:24:32 | 新增 14 个 Modelica 源/包文件；整个提交共 15 个文件、55,257 行新增 | 导入完整模型库及其包结构，建立 FMU 的物理侧基线。 |
| `ef25cdc` — `修改modelica模型` | 2026-07-13 23:25:15 | 仅修改 2 个 Modelica 文件：94 行新增、67 行删除 | 重构 FMU 的输出接口和经济计算职责，以支撑 Python/RL。 |

第三次提交只涉及：

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

以下是 `ef25cdc` 新增到顶层 FMU 的全部 19 个输出。右列给出其在 `PowerSystem_8760h.mo` 中的直接赋值来源；功率的符号统一为“发电负、用电/充电正”。

| 类别 | FMU 变量 | 代码来源 / 定义 | 含义与单位 |
| --- | --- | --- | --- |
| 电力平衡 | `p_curtailment` | `max(-bus.P_res, 0)` | 未被消纳的富余发电功率（弃电），W，非负。 |
| 电力平衡 | `p_unserved` | `max(bus.P_res, 0)` | 经可下调负荷后仍未满足的负荷缺口，W，非负。 |
| 储能状态 | `battery_soc` | `battery.SOC` | 电池荷电状态，0～1，无量纲。 |
| 储能状态 | `caes_gas_soc` | `gastank.SOC = gastank.p / p_norm` | CAES 储气罐压力相对于额定压力的状态，0～1，无量纲。 |
| 储能状态 | `caes_hot_soc` | `hottank.SOC = level / (V0 / A)` | CAES 热罐液位相对于额定液位的状态，0～1，无量纲。 |
| 储能状态 | `caes_cold_soc` | `coldtank.SOC = level / (V0 / A)` | CAES 冷罐液位相对于额定液位的状态，0～1，无量纲。 |
| 设备实际功率 | `p_thermal` | `thermalPower.positivePlug.P_act` | 火电实际出力，W；负值表示发电。当前模型中 `P_act = P_plan = -u_tp * P_cap`。 |
| 设备实际功率 | `p_battery` | `battery.PBS.P_act` | 电池实际功率，W；正值充电、负值放电。当前模型中 `P_act = P_plan = u_battery * P_cap`。 |
| 设备实际功率 | `p_caes` | `compressedAirEnergyStorage.PBS.P_act` | CAES 实际电功率，W；正值压缩/充电、负值膨胀/放电。当前模型中 `P_act = P_plan = u_caes * P_cap`。 |
| 设备实际功率 | `p_grid` | `grid.Power.P_act` | 与外部电网的实际交换功率，W；正值购电、负值售电。它是按联络线限额截断后的值。 |
| 风资源 | `p_wind_available` | `wind.P_WT.P_plan` | 给定风速和风机功率曲线计算的可用风电功率，W；负值表示可发电。 |
| 风资源 | `p_wind_actual` | `wind.P_WT.P_act` | 母线弃电策略之后实际并网的风电功率，W；负值表示实际发电。 |
| 光伏资源 | `p_pv_available` | `pV_e.P_PV.P_plan` | 由辐照、环境温度、风速及组件参数计算的可用光伏功率，W；负值表示可发电。 |
| 光伏资源 | `p_pv_actual` | `pV_e.P_PV.P_act` | 母线弃电策略之后实际并网的光伏功率，W；负值表示实际发电。 |
| 负荷 | `p_load_actual` | `eLoad.ELoad.P_act` | 实际被供给的负荷功率，W；正值表示用电。发生缺供时它可低于计划负荷。 |
| CAES 热力状态 | `caes_gas_pressure` | `gastank.p` | CAES 储气罐气体压力，Pa。 |
| CAES 热力状态 | `caes_gas_temperature` | `gastank.T` | CAES 储气罐气体温度，K。 |
| CAES 热力状态 | `caes_hot_temperature` | `hottank.T` | CAES 热罐温度，K。 |
| CAES 热力状态 | `caes_cold_temperature` | `coldtank.T` | CAES 冷罐温度，K。 |

新接口把原来混合在 `P_res` 中的两种运行结果拆成两个非负量，也暴露了策略构造 observation、计算约束违规和诊断 CAES 边界所需的原始状态。

### 可用功率、实际功率与弃电的关系

是的，二者的联系不是 Python 的推断，而是 `TypicalScenarios.mo` 的 `Bus` 方程直接规定的。`P_plan` 是资源或调度计划值，`P_act` 是母线平衡和弃电/降负荷逻辑完成后真正进入系统的值；顶层仅将这两个内部量分别导出为 `*_available` 和 `*_actual`。

光伏可用功率首先由 `PV_e` 模型计算：

```modelica
P_PV.P_plan = -max(Pn * G_in / Gstc
  * (1 - KT * (T_pv - T_stc)) * eta, 0);
```

因此 `p_pv_available` 是在当前辐照 `G_in`、组件表面温度 `T_pv`、装机 `Pn`、温度系数 `KT` 和光电效率 `eta` 下本可发出的功率。负号只来自本模型“发电为负”的符号约定，并不表示负的发电量。

随后 `Bus` 处理功率平衡。在富余发电分支 `P_res1 < 0` 中，先弃风、再弃光伏：

```modelica
Power_WT.P_act = max(Power_WT.P_plan + P_res1, 0);
Power_PV.P_act = max(Power_PV.P_plan + P_res2, 0);
P_res = -Power_PV.P_act + P_res2 + Power_PV.P_plan;
```

这里 `P_res2` 是风电已处理后的剩余富余量。于是弃电时 `p_pv_actual` 的数值会从负的 `p_pv_available` 向 0 靠近，`0` 表示该时刻光伏已全部弃掉；不需要弃电时，代码直接令 `Power_PV.P_act = Power_PV.P_plan`，两者相等。风电变量同理，但由于策略“先弃风电”，风电实际功率通常先于光伏被削减。缺供分支不弃风光，而是令二者实际功率等于计划/可用功率，再下调负荷。

`p_curtailment` 不是“某一台光伏的弃电量”，而是风、光依次削减后仍残留的总富余功率；`p_unserved` 也不是计划负荷，而是最多下调至计划负荷 20% 后仍无法满足的剩余缺口。这两个量共同替代了原来含义混合的带符号 `P_res`。

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

### 原惩罚项的作用，以及为何去掉

旧代码的目的不是改变设备的物理状态方程，而是在经济现金流中叠加一个随越界或功率失衡增大的软成本，以驱动优化器远离不可行工况：

| 位置 | 原罚函数（按旧代码转写） | 原本想约束的对象 | 删除后的变化 |
| --- | --- | --- | --- |
| `ThermalPower` | `C_penality = k * P_cap * (e ^ max(P_act + P_min, 0) - 1) + k * (e ^ (-min(P_max + P_act, 0)) - 1)` | 火电实际功率越过最小/最大出力边界。 | `positivePlug.C` 只保留燃料/电价现金流；火电出力约束的评价迁到 Python。 |
| `Battery` | `k * E_cap * (e ^ (-min(SOC - SOC_min, 0)) - 1) + k * (e ^ (-min(SOC_max - SOC, 0)) - 1)` | 电池 SOC 低于下限或高于上限。 | 充、放电分支仅按实际功率结算买/售价；Python 由 `battery_soc` 判断并施加策略所需的约束或罚分。 |
| `CompressedAirEnergyStorage` | 对 `gastank.SOC`、`hottank.SOC`、`coldtank.SOC` 分别使用与电池相同形式的指数罚函数，三项相加。 | CAES 气、热、冷三类储罐的 SOC 上下界。 | CAES 仅保留买/售电现金流；Python 使用三项 CAES SOC 及压力/温度输出诊断和评价。 |
| `Bus` | `C_penality = (k * P_res)^2`，且 `der(Income) = ΣC - C_penality`。 | 弃电（`P_res < 0`）和缺供（`P_res > 0`）。 | `der(Income) = ΣC`；Python 从互不混淆的 `p_curtailment` 和 `p_unserved` 单独定价或惩罚。 |

去掉它们的原因有三层。第一，罚函数是优化目标的一部分而非设备物理规律，放在 FMU 内会把奖励权重和约束表达式固化，无法按实验调整。第二，火电公式把以 W 表示的 `P_act`、`P_min`、`P_max` 直接放入指数的自变量；只要出现正的 W 量级越界，`e^x` 就很容易溢出为 `Inf`，继而使现金流、奖励或仿真出现 `NaN`。第三，旧的 `P_res` 同时承载弃电和缺供，单个平方罚项无法让 Python 对两种后果采用不同系数；拆分输出后，可以显式采用例如 `λ_curtail * p_curtailment + λ_unserved * p_unserved`，并使缺供的权重远大于弃电。

这并不表示物理边界被删除：例如 CAES 气罐仍保留 `assert(SOC <= SOC_max)` 与 `assert(SOC > SOC_min)`，储罐模型的状态演化也未因本提交而改写。被移除的是进入现金流/目标函数的软经济罚项及其顶层导出；Python 侧须重新、明确地定义奖励和违规处理。

## 为什么这样修改

1. **数值稳定性**：旧罚函数包含 `e^x`。特别是火电公式直接以 W 量级的功率参与指数运算，容易产生 `Inf`/`NaN`，不适合作为 FMU 仿真和训练的稳定奖励来源。
2. **职责分离**：物理模型应提供真实状态和实际功率；电价、惩罚权重、奖励形式和约束处理属于优化/RL 实验逻辑，应由 Python 灵活定义。
3. **观测可用性**：Python 需要原始 SOC、功率平衡、设备实际功率及 CAES 热力状态，才能构造马尔可夫观测、奖励和越界诊断，而不是只能消费一个已经混合了固定权重的目标值。

## 兼容性与验证边界

- 这是 **FMU 输出接口的破坏性变更**：Python 必须停止读取 `OPT_goal`、`P_res` 与 `*_penalty`，并改用新的物理输出名构造结算和奖励。
- 输入 `u_tp`、`u_battery`、`u_caes` 及设备间的物理连接未在本次差异中改变；改动重点是输出接口和经济计算位置，而非设备物理过程。
- 本说明基于 Git 源码对比。未在此对比范围内重新导出 FMU 或运行仿真，因此不能将其视为新 FMU 的运行验证结果；导出后需同步核对 Python 的输出名映射和相关测试。

