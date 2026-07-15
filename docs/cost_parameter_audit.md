# 成本参数审计

来源：`resources/Example/TypicalScene/PowerSystem_8760h.mo`、`resources/TypicalScenarios.mo`。
审计日期：2026-07-14。

原则：不使用 `bus.Income` / `OPT_goal` / 全部组件 `C` 简单求和；储能 `c_buy/c_sale` 与电网结算存在双计风险。

| Parameter | Modelica path | Unit | Meaning | Use in reward | Double-counting risk | Decision |
|-----------|---------------|------|---------|---------------|----------------------|----------|
| grid.c_buy | `PowerSystem_8760h.grid` (`c_buy=0.6`) | 元/kWh | 购电价 | `buy_price_yuan_per_mwh=600` | 低（系统边界结算） | **采用** |
| grid.c_sale | `PowerSystem_8760h.grid` (`c_sale=0.1`) | 元/kWh | 售电价 | `sell_price_yuan_per_mwh=100` | 低 | **采用** |
| thermal.c | `thermalPower` 实例 `c=0.4` | 元/kWh | 火电边际燃料 | `thermal_b=400`，`a=c=0` | 低 | **采用**线性 |
| battery.c_buy / c_sale | `battery` 0.2 / 0.4 | 元/kWh | 设备内部结算 | 否 | **高**（与电网购售叠算） | **不采用** |
| caes.c_buy / c_sale | CAES 默认/实例 | 元/kWh | 设备内部结算 | 否 | **高** | **不采用** |
| wind/PV.c | 风电/光伏组件 | 元/kWh | 内部收益系数 | 否 | 高（非系统对外成本） | **不采用** |
| eLoad.c | 负荷 `c=0.5` | 元/kWh | 负荷侧支付 | 否 | 高 | **不采用** |
| bus.Income | `Bus.Income` | 元（累计） | 全体设备 C 积分 | **禁止** | 极高 | **禁用** |
| OPT_goal | `OPT_goal=Income` | — | 优化别名 | **禁止** | 极高 | **禁用** |
| C_penality | 已删除 | — | 旧惩罚 | 否 | — | 不存在 |
| battery/caes throughput O&M | 无独立 Modelica 参数 | 元/MWh | 退化/吞吐代理 | 置 0 | — | **暂 0**，有实测后再填 |
| curtailment price | 无 VOLL 参数 | 元/MWh | 弃电外部性 | 置 0 | — | 宽裕电网下≈0 |
| unserved / VOLL | 无 | 元/MWh | 缺供 | 置 0 | — | 同上 |
| ramp_cost | 无独立价格；`rate_max` 仅物理 | 元/MW | 合法爬坡磨损 | 置 0 | — | 硬爬坡由 Oracle；成本暂 0 |
| grid P1/P2 | `P1=5e8`,`P2=-5e8` | W | 联络线容量 | 硬约束非 reward | — | **硬约束实现** |

## p_grid 符号

已核验：`p_grid>0` 购电，`p_grid<0` 售电。

## 电网容量

**非无限电网**：±500 MW。实践中常足以吸收残差，故 `p_curtailment`/`p_unserved` 常为 ~0；不得因此宣称缺供训练信号已建立。

## 火电爬坡

`rate_max=0.0025/60` 存在，但 Modelica 中爬坡饱和方程已注释；Python `FeasibilityOracle` 强制硬爬坡。
