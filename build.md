# FMU 接口与数值稳定性修改计划

## 一、修改目标

本轮修改针对 `PowerSystem_8760h.mo` 及其关联模型，目标是：

1. python侧约束外部控制输入进入物理模型前均满足设备允许上下限，如果超出上下限就报错；
2. 将功率不平衡量拆分为弃电和缺供两个明确的非负输出；
3. 删除不必要的经济目标和 penalty 类型输出；
4. 补充强化学习所需的真实物理状态、设备实际功率和可再生能源信息；
5. 移除可能产生 `Inf`、`NaN` 或 FMU 求解失败的指数罚函数；
6. 明确职责边界：
   * FMU 负责物理状态演化和真实物理量输出；
   * Python 环境负责市场结算、reward、经济成本和约束惩罚。

---

# 二、修改范围

主要修改文件：

```text
PowerSystem_8760h.mo
TypicalScenarios.mo
```

必要时还需要修改以下内部组件：

```text
Battery
ThermalPower
CompressedAirEnergyStorage
GasTank
HotTank
ColdTank
Bus / EnergyBus
```

实际文件名根据工程中的类定义位置确定。


# 三、拆分 `P_res`

## 3.1 删除或停用原单一输出

原来的：

```modelica
RealOutput P_res;
```

同时表达弃电和缺供，语义不清晰，应改为两个非负量。

## 3.2 新增输出

```modelica
RealOutput p_curtailment;
RealOutput p_unserved;
```

计算方式：

```modelica
p_curtailment = max(-bus.P_res, 0);
p_unserved    = max( bus.P_res, 0);
```

物理意义：

[
p_{\text{curtailment}}\geq0
]

表示系统存在过剩功率，无法被负荷、储能或电网吸收。

[
p_{\text{unserved}}\geq0
]

表示系统供电不足或存在未满足负荷。

## 3.3 符号核验

实施前必须通过一个人工构造工况确认 `bus.P_res` 的符号定义：

* 发电大于负荷时，是否为负值；
* 发电小于负荷时，是否为正值。

若当前模型的符号方向相反，需要交换两个公式，不能仅凭变量名判断。

## 3.4 验收标准

始终满足：

```text
p_curtailment >= 0
p_unserved >= 0
```

正常平衡状态下：

```text
p_curtailment = 0
p_unserved = 0
```

两者原则上不应同时为正。

---

# 四、重构 FMU 输出接口

## 4.1 储能状态输出

新增：

```modelica
RealOutput battery_soc;

RealOutput caes_gas_soc;
RealOutput caes_hot_soc;
RealOutput caes_cold_soc;
```

赋值：

```modelica
battery_soc = battery.SOC;

caes_gas_soc =
    compressedAirEnergyStorage.gastank.SOC;

caes_hot_soc =
    compressedAirEnergyStorage.hottank.SOC;

caes_cold_soc =
    compressedAirEnergyStorage.coldtank.SOC;
```

用途：

* 强化学习 observation；
* 动态动作限幅；
* 终端库存约束；
* 设备运行状态诊断。

---

## 4.2 设备实际功率输出

新增：

```modelica
RealOutput p_thermal;
RealOutput p_battery;
RealOutput p_caes;
RealOutput p_grid;
```

赋值：

```modelica
p_thermal =
    thermalPower.positivePlug.P_act;

p_battery =
    battery.PBS.P_act;

p_caes =
    compressedAirEnergyStorage.PBS.P_act;

p_grid =
    grid.Power.P_act;
```

这些输出应使用设备的实际执行功率，而不是计划功率或控制命令。

用途：

* 实际运行成本；
* 购售电结算；
* 火电爬坡计算；
* 储能寿命和循环成本；
* 动作执行偏差分析。

---

## 4.3 风光可用功率与实际功率

新增：

```modelica
RealOutput p_wind_available;
RealOutput p_wind_actual;

RealOutput p_pv_available;
RealOutput p_pv_actual;
```

赋值：

```modelica
p_wind_available =
    wind.P_WT.P_plan;

p_wind_actual =
    wind.P_WT.P_act;

p_pv_available =
    pV_e.P_PV.P_plan;

p_pv_actual =
    pV_e.P_PV.P_act;
```

需要确认 `P_plan` 是否确实代表自然条件下的最大可用功率。

如果 `P_plan` 只是调度计划，则不应命名为 `available`，而应单独从风速、辐照度模型中输出：

```text
p_wind_available
p_pv_available
```

真实可用功率。

可再生能源弃电量可在 Python 中交叉计算：

[
P_{\text{curt}}
===============

\max(P_{\text{wind,available}}-P_{\text{wind,actual}},0)
+
\max(P_{\text{pv,available}}-P_{\text{pv,actual}},0)
]

并与 `p_curtailment` 对比校验。

---

## 4.4 负荷输出

新增：

```modelica
RealOutput p_load_actual;
```

赋值：

```modelica
p_load_actual =
    eLoad.ELoad.P_act;
```

若负荷模型同时存在需求值和实际供给值，建议后续分别输出：

```modelica
RealOutput p_load_demand;
RealOutput p_load_served;
```

这样：

[
p_{\text{unserved}}
===================

\max(p_{\text{load,demand}}-p_{\text{load,served}},0)
]

物理含义会更明确。

---

# 五、补充 CAES 关键热力状态

新增：

```modelica
RealOutput caes_gas_pressure;
RealOutput caes_gas_temperature;
RealOutput caes_hot_temperature;
RealOutput caes_cold_temperature;
```

赋值：

```modelica
caes_gas_pressure =
    compressedAirEnergyStorage.gastank.p;

caes_gas_temperature =
    compressedAirEnergyStorage.gastank.T;

caes_hot_temperature =
    compressedAirEnergyStorage.hottank.T;

caes_cold_temperature =
    compressedAirEnergyStorage.coldtank.T;
```

这些量主要用于：

* 保证 observation 具备足够的马尔可夫性；
* 判断 CAES 是否接近压力或温度边界；
* 分析 FMU 数值异常；
* 构造动态动作可行域。

## 单位要求

所有输出必须明确并统一单位：

```text
压力：Pa 或 MPa，不能混用
温度：K
功率：W 或 MW，不能混用
SOC：0～1
```

建议 FMU 内部保持 SI 单位，Python 侧再做归一化。

---

# 六、删除不必要的顶层输出

建议从顶层 FMU 接口删除：

```modelica
RealOutput battery_penalty;
RealOutput thermal_penalty;
RealOutput caes_gas_penalty;
RealOutput caes_hot_penalty;
RealOutput caes_cold_penalty;
```

`OPT_goal` 也不应作为强化学习即时 reward 使用。

处理方案：

```text
方案 A：从 FMU 顶层完全删除 OPT_goal
方案 B：保留为调试输出，但不纳入 observation 和 reward
```

如果保留，应改名为：

```modelica
RealOutput cumulative_income;
```

明确它是累计量，而不是即时奖励。

---

# 七、移除内部经济 penalty 计算

不仅要删除顶层 `RealOutput`，还应检查并移除内部实际计算：

```text
battery.C_penality
thermalPower.C_penality
C_GasTank_penality
C_HotTank_penality
C_ColdTank_penality
bus.C_penality
```

原因是即使不输出，只要这些方程仍在 FMU 内部，求解器仍会计算它们，仍可能产生：

```text
Inf
NaN
overflow
event iteration failure
nonlinear solver failure
```

## 处理原则

FMU 内保留：

* 物理方程；
* 设备边界；
* 饱和逻辑；
* 状态变量；
* 真实能量流；
* 必要的保护状态。

FMU 外部 Python 负责：

* 市场收益；
* 燃料成本；
* 碳成本；
* 启停成本；
* 爬坡成本；
* 储能退化；
* 弃电惩罚；
* 缺供惩罚；
* reward 汇总。

---

# 八、修复火电罚函数数值溢出

## 9.1 当前风险公式

当前类似：

```modelica
C_penality =
    k * P_cap *
    (e ^ max(positivePlug.P_act + P_min, 0) - 1)
  + k *
    (e ^ (-min(P_max + positivePlug.P_act, 0)) - 1);
```

问题在于指数项直接使用瓦级功率，可能出现：

[
e^{10^6},\quad e^{10^8}
]

必然导致浮点溢出。

## 9.2 推荐方案

若经济 penalty 完全迁移到 Python，则直接删除该公式。

若 FMU 内仍需保留约束违反量，仅输出无量纲违反程度：

[
v_{\min}
========

\max\left(
\frac{P_{\min}-P_{\text{gen}}}
{P_{\text{cap}}},
0
\right)
]

[
v_{\max}
========

\max\left(
\frac{P_{\text{gen}}-P_{\max}}
{P_{\text{cap}}},
0
\right)
]

可选诊断量：

```modelica
Real thermal_lower_violation;
Real thermal_upper_violation;
```

计算：

```modelica
thermal_lower_violation =
    max((P_min - p_thermal) / P_cap, 0);

thermal_upper_violation =
    max((p_thermal - P_max) / P_cap, 0);
```

若确实需要内部标量：

[
C_{\text{pen}}
==============

k(v_{\min}^2+v_{\max}^2)
]

但该量不应继续作为主要经济成本来源。

---

# 十、Python reward 迁移计划

Python 环境基于 FMU 实际输出计算即时奖励：

[
r_t =
R^{market}_t
------------

## C^{thermal}_t

## C^{grid}_t

## C^{storage}_t

## C^{curtailment}_t

## C^{unserved}_t

## C^{ramp}_t

C^{degradation}_t
]

推荐至少包含：

[
C^{unserved}_t
==============

\lambda_{\text{shed}}
p_{\text{unserved},t}\Delta t
]

[
C^{curtailment}_t
=================

\lambda_{\text{curt}}
p_{\text{curtailment},t}\Delta t
]

[
C^{thermal}_t
=============

f(P_{\text{thermal},t})
]

[
C^{grid}_t
==========

\lambda_t^{buy}
\max(P_{\text{grid},t},0)\Delta t
---------------------------------

\lambda_t^{sell}
\max(-P_{\text{grid},t},0)\Delta t
]

缺供惩罚权重通常应显著高于弃电惩罚：

[
\lambda_{\text{shed}}
\gg
\lambda_{\text{curt}}
]

以后可在此基础上加入：

* 日前市场申报；
* 实时市场偏差；
* 备用容量；
* 调频辅助服务；
* 碳交易；
* 启停成本；
* 火电最小开停机时间；
* 储能寿命退化。

---

# 十、修改后的推荐 FMU 顶层接口

## 输入

```modelica
RealInput u_tp;
RealInput u_battery;
RealInput u_caes;
```

可选外部场景输入：

```modelica
RealInput market_price;
RealInput load_profile;
RealInput wind_profile;
RealInput pv_profile;
```

取决于当前数据是否已内置在 Modelica 模型中。

## 输出

```modelica
RealOutput p_curtailment;
RealOutput p_unserved;

RealOutput battery_soc;

RealOutput caes_gas_soc;
RealOutput caes_hot_soc;
RealOutput caes_cold_soc;

RealOutput p_thermal;
RealOutput p_battery;
RealOutput p_caes;
RealOutput p_grid;

RealOutput p_wind_available;
RealOutput p_wind_actual;

RealOutput p_pv_available;
RealOutput p_pv_actual;

RealOutput p_load_actual;

RealOutput caes_gas_pressure;
RealOutput caes_gas_temperature;
RealOutput caes_hot_temperature;
RealOutput caes_cold_temperature;
```

可选诊断输出：

```modelica
RealOutput cumulative_income;
RealOutput constraint_violation_flag;
RealOutput numerical_warning_flag;
```

---

# 十一、实施顺序

## 阶段 1：接口整理

将 `P_res` 拆为 `p_curtailment` 和 `p_unserved`；
增加储能、设备功率、风光和负荷输出；
增加 CAES 压力和温度输出；
删除顶层 penalty 输出。

## 阶段 2：数值风险清理

1. 搜索全部 `C_penality`、`penalty`、`exp`、`e^`；
2. 删除经济 penalty 方程；
3. 将必须保留的约束量改为无量纲二次违反量；
4. 检查所有 `assert`；

