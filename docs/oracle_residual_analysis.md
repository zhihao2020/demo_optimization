# Oracle Prediction Residual Analysis

日期：2026-07-14  
数据：`runs/feasibility_probe/train/safety_dataset.json`（800 safe transitions）  
定义：`residual = actual − predicted`

## 总体

| key | n | mean | median | P90 | P95 | P99 | max | min |
|-----|--:|------|--------|-----|-----|-----|-----|-----|
| battery_soc | 800 | 1.1e-6 | — | — | 0.031 | 0.038 | 0.042 | -0.043 |
| caes_gas_soc | 800 | -0.0083 | — | — | 0.067 | 0.073 | 0.077 | -0.103 |
| caes_hot_soc | 800 | -0.0030 | — | — | 0.020 | 0.022 | 0.023 | -0.027 |
| caes_cold_soc | 800 | -0.0022 | — | — | 0.044 | 0.047 | 0.049 | -0.054 |
| caes_gas_pressure | 800 | -3.5e4 | — | — | 6.6e4 | 9.8e4 | 1.16e5 | -3.3e5 |
| caes_gas_temperature | 800 | -0.31 | — | — | 2.18 | 3.12 | 3.37 | -8.02 |
| p_thermal | 800 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| p_grid | 800 | -6.7e4 | — | — | 4.7e7 | 5.9e7 | 8.1e7 | -6.7e7 |

## 方向感知（危险残差）

| 条件 | 统计 | 用于裕度 |
|------|------|----------|
| u_battery > 0.05（充电，推向 SOC_max） | P99(residual) ≈ **0.013** | `battery.residual_p99_charge_high` |
| u_battery < -0.05（放电，推向 SOC_min） | P01(residual) ≈ **-0.018** | `battery.residual_p99_discharge_low` |
| CAES charge/discharge | gas/hot/cold/pressure P99 | `caes.charge/discharge.*` |
| CAES idle | pressure P95≈6.5e4 | `caes.idle.residual_p99_pressure` |

## 按模式（摘要）

- **DISCHARGE**：hot/cold 残差偏正（模型低估放电路径罐态变化）
- **CHARGE**：hot/cold 残差偏负（对称偏置）
- **IDLE**：gas/pressure 仍有漂移；与 probe `caes_pressure_low@idle` 一致
- **p_thermal**：各模式 residual=0（映射确定）

## 危险残差定义

对充电电池：`residual>0` 为 dangerous_high；放电：`residual<0` 为 dangerous_low。  
CAES 充/放类推至 gas/hot/cold/pressure。

写入裕度时取危险方向 P99 的绝对值，叠加工程 `margin_*`，**禁止**统一 +0.02。
