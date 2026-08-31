# 贡献口径：PC-HybridTD3 多能源安全经济协同调度

文档更新：2026-08-31 12:30 (+08:00)

主方法：**PC-HybridTD3**（Physics-Constrained Hybrid TD3）。研究对象是多能源系统安全经济协同调度，不是「把 CAES 调起来」。CAES 模式–幅值是异构动作的代表实例。投影连续 TD3 与分量支撑 hybrid TD3 为仅有的两组消融。FS-HSAC / HMSD 不以正文身份出现。

## 三条贡献（不超过三条）

1. **异构设备混合动作** — 火电、电池连续调节；CAES 用离散模式 + 连续幅值。避免连续投影在非连续运行域上的策略退化。
2. **系统级联合动态可行域** — \(\mathcal A_f(s)\) 嵌入爬坡、库存、CAES 热力限制与联络线耦合；GiveSafe 只作最终物理验证。
3. **预测感知 FMU 闭环经济调度** — 24 h 风光荷/电价前瞻；物理-only Bellman；36/8/8 TEST 与 8760 h 部署；对照 rule / rolling MILP / projection TD3。碳价、磨损、启停、GiveSafe、Gumbel、TD3 本身不单独当贡献。

## 不写

- “仿真很少用 FMI”；“option 可自由选”；“PC-HybridTD3 天然优于双层”
- HMSD / c-step / FS-HSAC 作为正文身份；「RL 会预判所以优于 MILP」
- 把断开合法集当发现；把动态区间说成纯 FMU 物理
- 用 \(R^F\) / storage_use 把 CAES 跑起来再当贡献
- 结果未完成 Stage D 前的优越性断言；禁止把 `fs_hsac_*` 或 `seasonal_v1` 现金填进正文表
