# Modelica 模型修改记录

文档更新：2026-09-01 12:43 (+08:00)

## 设计原则

- Modelica 负责状态演化和物理量输出
- Python侧负责经济结算、奖励和约束罚分。

## `PowerSystem_8760h.mo`：输出改为物理观测输出 

### 新增的物理量输出

> 功率的符号统一为：发电负、用电/充电正。

| 类别 | FMU 变量 | 代码定义 | 含义与单位 |
| --- | --- | --- | --- |
| 电力平衡 | `p_curtailment` | `max(-bus.P_res, 0)` | 未被消纳的富余发电功率（弃电），W，非负。 |
| 电力平衡 | `p_unserved` | `max(bus.P_res, 0)` | 经可下调负荷后仍未满足的负荷缺口，W，非负。 |
| 电池储能状态 | `battery_soc` | `battery.SOC` | 电池荷电状态，0～1，无量纲。 |
| CAES气罐储能状态 | `caes_gas_soc` | `gastank.SOC = gastank.p / p_norm` | CAES 储气罐压力相对于额定压力的状态，0～1，无量纲。 |
| CAES热罐储能状态 | `caes_hot_soc` | `hottank.SOC = level / (V0 / A)` | CAES 热罐液位相对于额定液位的状态，0～1，无量纲。 |
| CAES冷罐储能状态 | `caes_cold_soc` | `coldtank.SOC = level / (V0 / A)` | CAES 冷罐液位相对于额定液位的状态，0～1，无量纲。 |
| 火力设备实际功率 | `p_thermal` | `thermalPower.positivePlug.P_act` | 火电实际出力，W；负值表示发电。当前模型中 `P_act = P_plan = -u_tp * P_cap`。 |
| 电池设备实际功率 | `p_battery` | `battery.PBS.P_act` | 电池实际功率，W；正值充电、负值放电。当前模型中 `P_act = P_plan = u_battery * P_cap`。 |
| CAES设备实际功率 | `p_caes` | `compressedAirEnergyStorage.PBS.P_act` | CAES 实际电功率，W；正值压缩/充电、负值膨胀/放电。当前模型中 `P_act = P_plan = u_caes * P_cap`。 |
| 电网设备实际功率 | `p_grid` | `grid.Power.P_act` | 与外部电网的实际交换功率，W；正值购电、负值售电。它是按联络线限额截断后的值。 |
| 风电可用功率 | `p_wind_available` | `wind.P_WT.P_plan` | 给定风速和风机功率曲线计算的可用风电功率，W；负值表示可发电。 |
| 风电实际功率 | `p_wind_actual` | `wind.P_WT.P_act` | 母线弃电策略之后实际并网的风电功率，W；负值表示实际发电。 |
| 光伏可用功率 | `p_pv_available` | `pV_e.P_PV.P_plan` | 由辐照、环境温度、风速及组件参数计算的可用光伏功率，W；负值表示可发电。 |
| 光伏实际功率 | `p_pv_actual` | `pV_e.P_PV.P_act` | 母线弃电策略之后实际并网的光伏功率，W；负值表示实际发电。 |
| 负荷实际功率 | `p_load_actual` | `eLoad.ELoad.P_act` | 实际被供给的负荷功率，W；正值表示用电。发生缺供时它可低于计划负荷。 |
| CAES气罐压力 | `caes_gas_pressure` | `gastank.p` | CAES 储气罐气体压力，Pa。 |
| CAES气罐温度 | `caes_gas_temperature` | `gastank.T` | CAES 储气罐气体温度，K。 |
| CAES热罐温度 | `caes_hot_temperature` | `hottank.T` | CAES 热罐温度，K。 |
| CAES冷罐温度 | `caes_cold_temperature` | `coldtank.T` | CAES 冷罐温度，K。 |


### 可用功率、实际功率与弃电的关系

`*_plan` 是资源或调度计划值，`*_act` 是母线平衡和弃电/降负荷逻辑完成后真正进入系统的值；
顶层仅将这两个内部量分别导出为 `*_available` 和 `*_actual`。

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

输入为 `u_tp`、`u_battery`、`u_caes`，符号约定：
> 观测对象都是输入电网为负，输出电网为正。

- 功率单位为 W，压力为 Pa，温度为 K，SOC 为 0～1。
- 发电为负；用电和充电为正。
- `p_battery`、`p_caes` 正值表示充电、负值表示放电。
- `p_grid` 正值表示购电、负值表示售电。
- `p_curtailment` 与 `p_unserved` 均为非负量。

## 经济罚函数 Modelica
`ThermalPower`、`Battery`、`CompressedAirEnergyStorage` 和 `Bus` 都移除了 `k`、`C_penality` 或各储罐 `C_*_penality` 变量及其计算式。

设备的 `C` 仅保留按实际功率和买/售价计算的现金流；母线 `Income` 仅累加各设备现金流。

移除的主要内容包括：
- 火电出力边界的指数型罚函数；
- 电池 SOC 上下界的指数型罚函数；
- CAES 气罐、热罐、冷罐 SOC 的指数型罚函数；
- 母线功率残差的 `(k * P_res)^2` 罚函数。

`Bus.OPT_goal` 仍作为内部累计现金流保留，但顶层模型不再导出它。



| 位置 | 原罚函数 | 原本想约束的对象 |  原罚函数会造成的问题|
| --- | --- | --- | --- |
| `ThermalPower` | `C_penality = k * P_cap * (e ^ max(P_act + P_min, 0) - 1) + k * (e ^ (-min(P_max + P_act, 0)) - 1)` | 火电实际功率越过最小/最大出力边界。 | 火力设备实际功率 `P_act` 可能为正的 W 量级越界，`e^x` 就很容易溢出为 `Inf`，继而使现金流、奖励或仿真出现 `NaN`。|
| `Battery` | `k * E_cap * (e ^ (-min(SOC - SOC_min, 0)) - 1) + k * (e ^ (-min(SOC_max - SOC, 0)) - 1)` | 电池 SOC 低于下限或高于上限。 | 电池设备实际功率 `P_act` 可能为正的 W 量级越界，`e^x` 就很容易溢出为 `Inf`，继而使现金流、奖励或仿真出现 `NaN`。|
| `CompressedAirEnergyStorage` | 对 `gastank.SOC`、`hottank.SOC`、`coldtank.SOC` 分别使用与电池相同形式的指数罚函数，三项相加。 | CAES 气、热、冷三类储罐的 SOC 上下界。 | CAES 气罐、热罐、冷罐 SOC 可能为负的 W 量级越界，`e^x` 就很容易溢出为 `Inf`，继而使现金流、奖励或仿真出现 `NaN`。|
| `Bus` | `C_penality = (k * P_res)^2`，且 `der(Income) = ΣC - C_penality`。 | 弃电（`P_res < 0`）和缺供（`P_res > 0`）。 | 弃电（`P_res < 0`）和缺供（`P_res > 0`）同时发生时，单个平方罚项无法让 Python 对两种后果采用不同系数；拆分输出后，可以显式采用例如 `λ_curtail * p_curtailment + λ_unserved * p_unserved`，并使缺供的权重远大于弃电。|

改动的理由：
- 第一，职责分离，FMU保证物理过程正确，python侧控制优化目标和约束。
- 第二，模型中保留必要的assert,保证物理过程正确
- 第三，强化学习需要构成马尔科夫观测，需要原始SOC、功率平衡、设备实际功率及CAES热力状态等物理量，而不是只能消费一个已经混合了固定权重的目标值。

## Tank 质量守恒与 Battery 模式切换（待导出 0903）

活源码在 `D:\Code\0622\m_resources\`，不是 `docs/TypicalScensrio/` 那份快照。

热罐/冷罐 `Tank` 不再用 `if noEvent(SOC > SOC_min and SOC < SOC_max)` 在越界后把 `der(m)`、`der(h)` 置零。方程为：

```modelica
der(m) = port_a.m_flow + port_b.m_flow;
m * der(h) = port_a.m_flow * inStream(port_a.h_outflow)
           + port_b.m_flow * port_b.h_outflow + Q_gen;
```

SOC 上下限由 Python Oracle / GiveSafe 保证；越界仍 `assert`。`GasTank` 本来就始终积分，`SOC=p/p_norm`。

电池：

```modelica
if noEvent(PBS.P_act >= 0) then
  der(SOC) = PBS.P_act * eta_charge / E_cap;
else
  der(SOC) = PBS.P_act / E_cap / eta;
end if;
```

CAES 工况选择保持 `noEvent(u_dispatch > 0 / < 0)`；冷罐表 −152.876 保持。`data/0903PowerSystem_8760h.fmu` 已按此源码导出并锁入 `env_config`。

