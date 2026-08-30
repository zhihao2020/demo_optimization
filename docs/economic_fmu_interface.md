# FMU 累计经济接口

文档更新：2026-08-29 12:30 (+08:00)

modelica侧保留 Modelica 的设备现金流核算，并新增以下只读、累计 FMU 输出（单位 CNY）：

| 输出 | 来源 |
| --- | --- |
| `economic_cashflow_total` | `bus.OPT_goal` |
| `economic_cashflow_wind` / `pv` / `thermal` | 风电、光伏、火电现金流积分 |
| `economic_cashflow_battery` / `caes` | 两类储能现金流积分（诊断；默认不进厂级 \(J\)） |
| `economic_cashflow_load` / `grid` | 负荷、电网现金流积分 |

正值是现金流收益，负值是成本。Python 在 reset 记录所有累计值，并且只使用相邻样本差分。厂级 \(\Delta J^{\mathrm{cash}}\) = FMU total Δ − 电网 FMU Δ + 分时结算 − 电池/CAES 设备 Δ。`Income_CAES` / `Income_BT` 留作诊断：设备侧按 0.2/0.4 元/kWh 记账，但 `Electrical.C` 是 flow，接到母线后符号相反，且与联络线 TOU 重复。火电/负荷/风光 FMU 现金流仍进入 \(J\)。离线报告的正成本为该增量的相反数。

这些输出不属于 agent observation；物理观测和动态可行域仍完全由 Python 侧处理。CAES 无最短运行锁（`min_run_steps=1`；崔 2024 只用启停费）。
