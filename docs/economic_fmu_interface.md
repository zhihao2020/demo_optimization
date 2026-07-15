# FMU 累计经济接口

`PowerSystem_8760h` 现保留 Modelica 的设备现金流核算，并新增以下只读、累计 FMU 输出（单位 CNY）：

| 输出 | 来源 |
| --- | --- |
| `economic_cashflow_total` | `bus.OPT_goal` |
| `economic_cashflow_wind` / `pv` / `thermal` | 风电、光伏、火电现金流积分 |
| `economic_cashflow_battery` / `caes` | 两类储能现金流积分 |
| `economic_cashflow_load` / `grid` | 负荷、电网现金流积分 |

正值是现金流收益，负值是成本。Python 在 reset 记录所有累计值，并且只使用相邻样本差分：训练经济 reward 为 `delta(economic_cashflow_total) / C_ref`；离线报告的正成本为该增量的相反数。Python 不再按价格或火电参数重算同一笔经济量。

这些输出不属于 agent observation；物理观测、动态可行域和 CAES 最短运行约束仍完全由 Python 侧处理。

## 重新导出要求

当前 `data/TypicalScensrio_Example_TypicalScene_PowerSystem_8760h.fmu` 是修改前的 FMI 3 二进制，不含上述八个输出。必须使用生成该现有 FMI 3 FMU 的 Modelica 工具重新导出，并替换该文件后再运行真实 FMU 校验。

本机 OpenModelica 1.26.3 只能导出 FMI 1/2，且仓库 `resources/` 目录名与根包 `TypicalScensrio` 不一致，不能作为该 FMI 3 重新导出的替代工具。Python 的真实 FMU 注册测试会在检测到旧二进制时跳过，避免将未更新的二进制误报为通过。
