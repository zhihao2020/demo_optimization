# 贡献口径：PC-HybridTD3 厂级多模态压空调度

文档更新：2026-08-31 12:00 (+08:00)

主方法：**PC-HybridTD3**（物理约束参数化混合 TD3：Actor 在 \(\mathcal A_f(s)\) 上解码 mode+magnitude；Critic 看 \((e_m,z)\)；经济 Bellman 只用 FMU 真转移）。投影连续 TD3 与静态带宽 hybrid TD3 为仅有的两组消融。FS-HSAC / HMSD 不以正文身份出现。

## 三条贡献（不超过三条）

1. **非凸 CAES 混合动作** — 模式–幅值 \((m,z)\) 表示断开的充/闲/放区间，避免连续 TD3 硬投影死区。
2. **联合状态相关可行域** — \(\mathcal A_f(s)=(\mathcal A_T\times\mathcal A_B\times\mathcal A_C)\cap\mathcal A_{\mathrm{grid}}\)。解析 decoder：CAES ∩ 联络线 → \(u_C\) → 收紧火电 → \(u_T\) → 条件电池。GiveSafe 是最后屏障，greedy 只尝试 1 次。不是分量盒再 64 次蒙 GiveSafe。
3. **FMU 闭环验证** — 经济 Bellman 只用真转移；36/8/8 TEST；rule / rolling MILP（代理优化、同一 FMU 评估）/ projection TD3；noisy forecast 与 8760 h 部署。碳价、磨损、启停、24 h 预测、终端 SOC、GiveSafe、Gumbel、TD3 本身不单独当贡献。

## 不写

- “仿真很少用 FMI”；“option 可自由选”；“PC-HybridTD3 天然优于双层”
- HMSD / c-step / FS-HSAC 作为正文身份；「RL 会预判所以优于 MILP」
- 把断开合法集当发现；把动态区间说成纯 FMU 物理
- 用 \(R^F\) / storage_use 把 CAES 跑起来再当贡献
- 结果未完成 Stage D 前的优越性断言；禁止把 `fs_hsac_*` 或 `seasonal_v1` 现金填进正文表
