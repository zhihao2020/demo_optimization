# 贡献口径：FS-HSAC 厂级多模态压空调度

日期：2026-08-25（因果链：DAE 对象 → \(\mathcal A(s)\) → 孪生可执行；购电改为月度代理购电）

主方法：**FS-HSAC**（状态相关可行支撑集上的同时间尺度混合 SAC）。旧 Hybrid SAC / 投影 SAC 为消融。不以 HMSD/HRL 为正文身份。

## 四条贡献

1. **系统与机制** — 火电 + 电池 + 绝热多罐压空 DAE；山东**月度**代理购电分时 + ETS；Sysplorer Modelica → FMI FMU 闭环（FMI 是交换标准，不是创新）。因果链：高保真对象 → 状态相关混合支撑 \(\mathcal A(s)\) → 孪生上可执行的 FS-HSAC。
2. **形式化** — \(\mathcal A(s)\) 由 \(\mathcal K(s)\)（当前可选模式）与 \(\mathcal M_k(s)\)（模式条件幅值区间）构成。
3. **算法** — 同时间尺度 FS-HSAC：模式头 + 条件幅值头、精确枚举、双温度、\(C_\psi\)、GiveSafe 采用；不声称天然优于 HRL。
4. **验证** — 三季周；消融与 PSO/LP/MILP；分项成本。

## 不写

- “仿真很少用 FMI”；“option 可自由选”；“FS-HSAC 天然优于双层”
- HMSD / c-step 作为正文身份；「RL 会预判所以优于 MILP」
- 把断开合法集当发现；把动态区间说成纯 FMU 物理
- 结果未过 gate 前的优越性断言
