# FMU 累计经济接口

modelica侧保留 Modelica 的设备现金流核算，并新增以下只读、累计 FMU 输出（单位 CNY）：

| 输出 | 来源 |
| --- | --- |
| `economic_cashflow_total` | `bus.OPT_goal` |
| `economic_cashflow_wind` / `pv` / `thermal` | 风电、光伏、火电现金流积分 |
| `economic_cashflow_battery` / `caes` | 两类储能现金流积分 |
| `economic_cashflow_load` / `grid` | 负荷、电网现金流积分 |

正值是现金流收益，负值是成本。Python 在 reset 记录所有累计值，并且只使用相邻样本差分：训练经济 reward 为 `delta(economic_cashflow_total) / C_ref`；离线报告的正成本为该增量的相反数。Python 不再按价格或火电参数重算同一笔经济量。

这些输出不属于 agent observation；物理观测、动态可行域和 CAES 最短运行约束仍完全由 Python 侧处理。
