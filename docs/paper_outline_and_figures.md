# Paper outline and figure checklist

**Working title:** *Same-hour support-consistent hybrid SAC for weekly wind–PV–thermal–BESS–CAES dispatch*

**短标题:** *Support-consistent hybrid SAC for CAES plant dispatch*

**主方法（live）:** **FS-HSAC-support** — Hybrid SAC 的同小时支撑一致改写：采样与 $\log\pi$ 共用 $\mathcal A(s)=\mathcal K(s)\times\mathcal M_k(s)$（模式掩码 × 库存区间盒）。**不是**新的 actor–critic 家族，**不是** HMSD / GHTD3 / 库存 HRL。

**对照（in-house）:** 固定带 Hybrid SAC（`sac_param`：潜变量密度再 clamp）。`--method sac`，`parameterized_caes=True`。

**附录变体:** 带残余 $C_\psi$ 的完整 FS-HSAC（`--method fs_hsac` 且不带 `--support`）。

**体裁:** Applied Energy 能源调度 + DRL。OCTD3 / GHTD3 只作相关工作（SMDP+initiation / $c$ 步库存目标），不写回贡献。

**单一源（数字，过门后）:** 主表取 `runs/seasonal_v1/**/fs_hsac_support_s0` vs `sac_param_s0`。完整 `fs_hsac_s0` 仅附录。`docs/fs_hsac_results_gate.md` 现为 `gate_passed: false`。

---

## 0. Status snapshot

| 项 | 现状 |
|----|------|
| 代码 `src/training/fs_hsac/` | 已落地；**不改** actor / algorithm / action_support 数学 |
| 论文主线入口 | `scripts/train_seasonal.py --method fs_hsac --support`（或 `FS_HSAC_NO_FEAS=1`） |
| 对照入口 | `scripts/train_seasonal.py --method sac`（`parameterized_caes=True`） |
| GiveSafe / soft_shell | 采用 GiveSafe；soft_shell **OFF** |
| `Paper/main.tex` | 已按锁定身份改写；结果表留空 |
| 结果门 | `gate_passed: false`；禁止填优越性数字 |

**写作门槛:** 过门前 **§5 表图不填假数、不写优越性**。旧冬 PSO $14.36\times10^6$ vs linprog $10.19\times10^6$ **撤回**（held-out fit，不是本文主张）。

---

## 0.5 一条贡献（锁定）

**唯一贡献:** 同小时 Hybrid SAC 把采样与 $\log\pi$ 绑到同一个状态相关支撑 $\mathcal A(s)=\mathcal K(s)\times\mathcal M_k(s)$。对照是院内固定带 Hybrid SAC（潜变量密度再 clamp）。

自洽只表示：sample 与 $\log\pi$ 使用同一个 $\mathcal A(s)$。$\mathcal A(s)$ 是模式掩码 × 库存区间盒，**不是** 电厂 / FMU 可行性（不含最短运行、SoC 否决、FMU 残差）。对 $\mathcal K(s)$ 的离散求和可以是精确的；**不写** exact hybrid entropy。

**设定（不是贡献）**

- 厂级风光–火电–BESS–绝热 CAES 周调度；**电是唯一出售载体**
- FMI / TOU / ETS / 断开 CAES 包线
- GiveSafe 采用
- 热–气耦合是孪生物理，不是出售产品

**勿写**

- HMSD / c-step / option HRL 作为正文身份或贡献
- 「RL 会预判所以优于 MILP」；OCTD3 行禁止写 RL>精确优化
- FMI/TOU/ETS 写成 C1/gap
- 「同一支撑孪生都能接受」（过声称）
- 「soft value enumerated exactly」/ exact hybrid entropy
- 把残余 $C_\psi$ 写进 live claim 或 Highlights
- 「电仅出售」或 `fs_hsac_support vs sac_param` 写进 Highlights（放 System / Setup）
- 未过门的优越性、假数、四条贡献包装

---

## 0.6 Highlights（只能两条）

1. Same-hour Hybrid SAC ties sampling and $\log\pi$ to one support $\mathcal A(s)=\mathcal K(s)\times\mathcal M_k(s)$.
2. Contrast is in-house fixed-band Hybrid SAC (latent density then clamp).

---

## 0.7 推荐期刊

| 档 | 期刊 | 条件 |
|----|------|------|
| **主攻** | **Applied Energy** | 一条贡献写干净；设定与方法分开 |
| **同等** | **Energy** | 不把 GHTD3 同刊当成要赢的分层身份 |
| **不优先** | IEEE TPWRS；NeurIPS/ICML | 不是电网出清，也不是新 RL 家族论文 |

---

## 1. Paper outline

### 1. Introduction（崔文结构，不借崔文主张）

1. price-taker + TOU/ETS **作为设定**；电是唯一出售载体。
2. 文献三类：凸/启发式；盒动作或投影 DRL；option/库存 HRL（OCTD3=SMDP+initiation，GHTD3=$c$ 步库存目标）。
3. **一个 gap:** 同小时密度绑在 $\mathcal A(s)$ vs latent-then-clamp。

方法优化 $\max\mathbb E\sum\gamma^t r_t$；评测是 168 h $J^{\mathrm{gen}}$。**不写** $\min J^{\mathrm{gen}}$。

**表 `tab:lit` 列:** timescale / density-contains-support / main contrast / what they did not compare。

- OCTD3 行：无 MILP，不写 RL>exact opt。
- GHTD3 行：凸 QP 常给出最低标量 CC，用维度解释。
- Constrained RL / safe exploration：density-contains-support = no。
- **删除** FMI/Safety checkbox 表。
- 不把 option/HRL 写回贡献。

### 2. System description

- 拓扑：风光、火电、BESS、绝热 CAES、母线、电网。电是唯一出售载体。
- Sysplorer → FMI FMU，通信间隔 1 h（设定）。
- 三季边界与 TOU、ETS（设定）。
- **图:** Fig.1–3；**表:** `tab:params`。

### 3. Problem formulation

- 3.1–3.3 物理（设定）
- 3.4 运行约束：最短运行、模式锁写在约束里，不当发现
- 3.5 评测分 $J^{\mathrm{gen}}$（168 h 求和）。训练是 $\max\mathbb E\sum\gamma^t r_t$
- 3.6 混合 MDP：$\mathcal A(s)$ = 模式掩码 × 库存区间盒；**不是** 孪生可接受集

### 4. Solution methodology

- 同小时支撑一致 Hybrid SAC；离散求和可精确；不写 exact hybrid entropy
- 对照：固定带 Hybrid SAC（latent then clamp）
- $C_\psi$、完整 FS-HSAC → 附录
- GiveSafe 采用；soft_shell OFF
- **图 / 算法框:** `fig_algorithm`；`fig_action_rep`；Alg. 主线无 $C_\psi$

### 5. Simulation results（门仍关）

- 主实验（若存在）: 季节 seed-0 `fs_hsac_support` vs `sac_param`
- 指标稍后: reject rate, `valid_steps=168`, comprehensive cost
- 表留空；不填优越性

### 6. Conclusions

- 重申一条贡献与对照
- 不写未过门数字、不写 PSO>linprog

---

## 1.5 reference_papers 证据边界

| 来源 | 借鉴 | 禁止写法 |
|------|------|----------|
| **OCTD3** | 机制组 × 算法组对照思路 | 无 MILP 对照；勿写 RL>精确优化 |
| **GHTD3** | 三季周、综合成本分项 | 凸 QP 常最低 CC → **按维度解释**，禁止 RL 全面优于求解器 |
| **GiveSafe / 约束 RL** | 安全层采用 | density-contains-support = no；不是本文贡献 |

---

## 1.6 KPI（过门后）

| 组 | KPI | JSON / kpi 字段 |
|----|-----|-----------------|
| **执行** | reject rate, `valid_steps` | GiveSafe 拒绝、`valid_steps` |
| **经济** | $J^{\mathrm{gen}}$, $CC$ | `sum_delta_j_gen`, `comprehensive_cost_cny` |
| | 分项 | cash / CO₂ / CUT / deg / su / grid |

只在 `valid_steps=168` 时进经济表。

---

## 2. Figures

Basename 均在 `Paper/figures/`。Fig.1–4 为设定；Fig.5 为投影 vs 固定带 vs 支撑一致；Fig.6 为 live 闭环（无 $C_\psi$）；§5 图待数据。

---

## 3. Code ↔ paper mapping

| 论文对象 | 代码 |
|----------|------|
| $\mathcal A(s),\mathcal M_k(s)$ | `fs_hsac/action_support.py`（数学锁定，本任务不改） |
| 支撑一致 actor / critic | `fs_hsac/actor.py`, `critic.py`（不改数学） |
| 离散求和 + $\alpha_d/\alpha_c$ | `fs_hsac/algorithm.py`（不改数学） |
| 论文主线训练 | `--method fs_hsac --support` 或 `FS_HSAC_NO_FEAS=1` |
| 对照 | `--method sac`（`parameterized_caes=True`） |
| 附录 $C_\psi$ | `fs_hsac/feasibility.py`；裸 `--method fs_hsac` |
| $J^{\mathrm{gen}}$ / $CC$ | `envs/reward_calculator.py` |
| 末段不改写 $u_{\mathrm{caes}}$ | `market.soc_recovery_horizon: 0`（`horizon<=0` 不改写） |

**奖励锁定:** 不改正在远程训练的奖励语义。
