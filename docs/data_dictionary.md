# 数据字典

## 边界 CSV

CSV 两列：`time(s), value`（可带 `time,value` 表头）。

### winds.csv

- **time**: 仿真时间（秒），步长 3600s
- **value**: 风速 (m/s)

### Gstc.csv

- **time**: 仿真时间（秒）
- **value**: 辐照强度 (W/m²)

### environment.csv

- **time**: 仿真时间（秒）
- **value**: 环境温度 (K)

### load.csv

- **time**: 仿真时间（秒），步长 3600s
- **value**: 用电负荷功率幅值 (W，正数)

> 由 FMU 内嵌 `eLoad.table` 导出（`scripts/export_load_csv.py`）。仿真仍读 FMU 内嵌表；CSV 用于边界校验、可视化与 RL 负荷预报观测。母线侧 `bus.Power_Eload.P_act` 为负号（用电约定），与 CSV 正数幅值相差一个符号。

> Demo 阶段风光环境数据亦内嵌于 FMU；CSV 用于校验与可视化对比。

## FMU 变量映射

权威配置见 [`configs/fmu_variables.yaml`](../configs/fmu_variables.yaml)。

### 调度输入（逻辑名 -> FMU 变量）

权威上下限与推导见 [`fmu_input_bounds.md`](fmu_input_bounds.md)（按 `PowerSystem_8760h` 实例参数）。

| 逻辑名 | FMU 输入 | 说明 | 范围 |
|--------|----------|------|------|
| u_tp | u_tp | 火电负荷率调度指令 | [1/3, 1] ≈ [0.333, 1] |
| u_battery | u_battery | 电池调度指令（正充负放） | [-1, 1] |
| u_caes | u_caes | 压空储能调度指令（正充负放） | [-1, -0.33] ∪ {0} ∪ [0.86, 1] |

### 仿真输出（请求列表）

| FMU 变量 | 逻辑指标 |
|----------|----------|
| OPT_goal | opt_goal（全年现金流） |
| P_res | curtailment（弃电/缺口） |
| battery_penalty | penalties[0] |
| thermal_penalty | penalties[1] |
| caes_gas_penalty | penalties[2] |
| caes_hot_penalty | penalties[3] |
| caes_cold_penalty | penalties[4] |
| battery.SOC | state.battery_soc |

配置中每个逻辑指标支持多个别名（如 `OPT_goal` / `bus.OPT_goal`），按顺序匹配。

## 调度输出 plan.csv

| 列 | 说明 |
|----|------|
| time | 时间 (s) |
| u_tp | 火电指令 |
| u_battery | 电池指令 |
| u_caes | 压空指令 |

## 测评指标 metrics.json

| 字段 | 说明 |
|------|------|
| opt_goal | 现金流终值（有 FMU 输出时） |
| penalty_count | 惩罚变量触发数 |
| constraint_violations | 约束违背数 |
| feasible | 是否可行 |
| relative_replay_error | 模型执行与 CSV 独立回放的收益相对误差 |
