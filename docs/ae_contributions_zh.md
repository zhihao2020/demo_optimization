# 贡献口径：FS-HSAC 支撑一致改写（锁定）

日期：2026-08-21

主方法（live）：**FS-HSAC-support** = Hybrid SAC 的同小时支撑一致改写。对照：院内固定带 Hybrid SAC（`sac_param`，潜变量密度再 clamp）。不以 HMSD / GHTD3 / 库存 HRL 为正文身份。完整残余 $C_\psi$ FS-HSAC 仅附录。

## 一条贡献

同小时 Hybrid SAC 把**采样**与 **$\log\pi$** 绑到同一个状态相关支撑
\[
\mathcal A(s)=\mathcal K(s)\times\mathcal M_k(s)
\]
（模式掩码 × 库存区间盒）。

自洽只表示 sample 与 $\log\pi$ 使用同一个 $\mathcal A(s)$。$\mathcal A(s)$ **不是** 电厂 / FMU 可行性（不含最短运行、SoC 否决、FMU 残差）。对 $\mathcal K(s)$ 的离散求和可以精确；**不写** exact hybrid entropy。

## Highlights（只能两条）

1. Same-hour Hybrid SAC ties sampling and $\log\pi$ to one support $\mathcal A(s)=\mathcal K(s)\times\mathcal M_k(s)$.
2. Contrast is in-house fixed-band Hybrid SAC (latent density then clamp).

「电仅出售」「`fs_hsac_support` vs `sac_param`」放 System / Setup，不放 Highlights。

## 设定（不是贡献）

- 厂级风光–火电–BESS–绝热 CAES 周调度；电是唯一出售载体
- FMI / TOU / ETS / 断开 CAES 包线
- GiveSafe 采用；soft_shell OFF
- 热–气耦合是孪生物理，不是出售产品

## 主实验（过门后才填数）

季节 seed-0：`fs_hsac_support`（`--method fs_hsac --support` 或 `FS_HSAC_NO_FEAS=1`）vs `sac_param`（`--method sac`）。同一 GiveSafe，soft_shell OFF。指标：reject rate、`valid_steps=168`、comprehensive cost。结果门仍为 `false`，不写优越性。

训练目标 $\max\mathbb E\sum\gamma^t r_t$；评测 168 h $J^{\mathrm{gen}}$。不写 $\min J^{\mathrm{gen}}$。

## 不写

- 四条贡献包装（对象 / 形式化 / 求解器 / 验证）
- 「同一支撑孪生都能接受」
- FMI/TOU/ETS 写成 C1 或 Intro gap
- option/HRL 写回贡献；「FS-HSAC 天然优于双层」
- 「soft value enumerated exactly」/ exact hybrid entropy
- 把 $C_\psi$ 写进 live claim
- 旧冬 PSO $14.36\times10^6$ vs linprog $10.19\times10^6$
- 结果未过 gate 前的优越性断言或假数
