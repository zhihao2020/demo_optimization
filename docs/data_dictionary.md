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

> 由 FMU 内嵌 `eLoad.table` 导出（`scripts/export_load_csv.py`）。仿真仍读 FMU 内嵌表；CSV 用于边界校验、可视化与 RL 前瞻观测。母线侧 `bus.Power_Eload.P_act` 为负号（用电约定），与 CSV 正数幅值相差一个符号。

> 风光环境数据亦内嵌于 FMU。Python 会从四份同源 CSV 读取 `t+1..t+24` 的已知日前值，按每小时 `wind, irradiance, ambient_temperature, planned_load` 的顺序附加到 19 维物理 observation 后；FMU 仍是唯一物理边界和真值来源。年末越界预测使用年度末最后一个值填充。

> `scripts/train_hybrid_td3.py --no-forecast` 仅用于与前瞻模型进行同种子基线对比，恢复原 19 维 observation；它不修改 FMU 边界、奖励或 GiveSafe。

> 在比较训练完成后加 `--annual-eval`，会按 52 个完整周窗口和 1 个 24 小时尾窗覆盖全年，输出全年原始成本、缺供/弃电、储能吞吐、失败数与有效步数；它不把跨周重置的储能状态伪装成连续年度状态。

## FMU 变量映射

权威配置见 [`src/config/env_config.yaml`](../src/config/env_config.yaml)。

### 调度输入（逻辑名 -> FMU 变量）

权威上下限与推导见 [`fmu_input_bounds.md`](fmu_input_bounds.md)（按 `PowerSystem_8760h` 实例参数）。

| 逻辑名 | FMU 输入 | 说明 | 范围 |
|--------|----------|------|------|
| u_tp | u_tp | 火电负荷率调度指令 | [1/3, 1] ≈ [0.333, 1] |
| u_battery | u_battery | 电池调度指令（正充负放） | [-1, 1] |
| u_caes | u_caes | 压空储能调度指令（正充负放） | [-1, -0.33] ∪ {0} ∪ [0.86, 1] |

### 仿真输出（当前物理接口）

| FMU 变量 | 逻辑指标 |
|----------|----------|
| p_curtailment / p_unserved | 弃电 / 缺供功率（W，非负） |
| battery_soc / caes_*_soc | 储能 SOC |
| p_thermal / p_battery / p_caes / p_grid | 实际设备与联络线功率（W） |
| p_wind_* / p_pv_* / p_load_actual | 风光可用/实际与负荷功率（W） |
| caes_*_pressure / caes_*_temperature | CAES 热力状态（Pa/K） |

### 累计经济输出（不进入 observation）

`economic_cashflow_total` 与 wind/pv/thermal/battery/caes/load/grid 七个分项均为累计现金流（CNY，正=收益、负=成本）。Python 只做相邻步差分、训练归一化和报告汇总，详见 [`economic_fmu_interface.md`](economic_fmu_interface.md)。

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
