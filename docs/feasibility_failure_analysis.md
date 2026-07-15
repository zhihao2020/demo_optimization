# Feasibility Failure Analysis (Phase D.5)

日期：2026-07-14  
Oracle 版本：`d5.2-probe-calibrated`

## 数据来源

| 来源 | 有效步 | 后验硬失败 | 失败率 | 细粒度 FailureRecord |
|------|--------|------------|--------|----------------------|
| `runs/hybrid_td3_smoke` | 5000 | 190 | 3.80% | **否**（旧 step_log 仅有训练指标） |
| `runs/hybrid_td3_short` | 20000 | 1053 | 5.27% | **否** |
| `runs/feasibility_probe`（新仪表） | 800 | 1 | 0.125% | **是** |

旧 smoke/short 日志无法回填细类型；以 probe FailureRecord 为主证据，smoke/short 仅作失败率对照。

## Probe 失败细类型

| fine_failure_type | count | ratio |
|-------------------|------:|------:|
| caes_pressure_low | 1 | 100% of probe fails |

预检拒绝（未进 FMU）：collector `precheck_rejections=4`，细类型计数含 `caes_gas_soc_low` / `caes_pressure_low`（Oracle 预测拒斥）。

### CAES 模式（有效步）

| mode | count | share |
|------|------:|------:|
| DISCHARGE (0) | 278 | 34.8% |
| IDLE (1) | 261 | 32.6% |
| CHARGE (2) | 261 | 32.6% |

Probe 唯一后验失败发生在 **IDLE**（压力漂移越下界），说明仅靠 gas-SOC 阈值不足以覆盖压力维。

### Smoke vs Short 失败率

- smoke：190/5000 = **3.80%**
- short：1053/20000 = **5.27%**（更高：探索更久、更近界）
- probe（收紧 Oracle 后）：1/800 = **0.125%**

## Top-3 原因（综合）

1. **CAES 压力/联合 SOC 边界**（probe 实证 `caes_pressure_low`；历史失败大概率同类 SOC/压力越 assert）
2. **旧 Oracle 统一 0.02 safe 裕度 + 一阶 gas-only mask**：近界充电/放电预测不足
3. **电池 SOC 近界方向残差**（非主导于 probe，但 charge/discharge P99 残差用于收紧）

## Residual P99（SafetyDataset，residual = actual − predicted）

| 量 | mean | P95 | P99 | 备注 |
|----|------|-----|-----|------|
| battery_soc | ≈0 | 0.031 | 0.038 | 充电子集危险 P99 ≈ **0.013**；放电危险 |P01| ≈ **0.018** |
| caes_gas_soc | -0.008 | 0.067 | 0.073 | 能量模型偏置 |
| caes_hot_soc | -0.003 | 0.020 | 0.022 | |
| caes_cold_soc | -0.002 | 0.044 | 0.047 | |
| caes_gas_pressure | -3.5e4 | 6.6e4 | **9.8e4** | idle 漂移风险 |
| caes_gas_temperature | -0.31 | 2.18 | 3.12 | |
| p_thermal | 0 | 0 | 0 | 精确（指令映射） |
| p_grid | 大偏置 | 4.7e7 | 5.9e7 | 弃风/不平衡动态；Oracle 另用电网裕度预检 |

完整分模式表见 `runs/feasibility_probe/residual_summary.yaml` 与 `docs/oracle_residual_analysis.md`。

## 幅度 / SOC 箱 / 步序

Probe 单次失败样本量=1，无法稳健做幅度箱；历史 smoke/short 缺 FailureRecord。  
后续门控要求持续写入 `failure_records.json`，再刷新分箱。

## 重复状态-动作

当前 probe 未观测到完全重复 (state, action) 失败簇；smoke/short 无足够字段可复现。

## Oracle / 安全改造响应

- 逐设备方向裕度写入 `src/config/feasibility_margins.yaml`
- CAES 模式联合约束（SOC+P+T）
- Thermal 仅用 **实际** `previous p_thermal`
- SafetyClassifier + SafeActionGenerator 在动作生成端过滤
- EconomicReplayBuffer vs SafetyDataset 分离

## Phase E

仍 **阻断**：需零后验失败累计证据、BoundaryStress ≥20000 通过、足够 fail 样本上 false-safe 门控。
