# Paper outline and figure checklist

文档更新：2026-08-31 08:30 (+08:00)

**Working title:** *Feasible-Support Hybrid TD3 for Multi-Energy Scheduling with Nonconvex Compressed Air Energy Storage Actions*

**短标题:** *Feasible-support hybrid TD3 for nonconvex CAES actions*

**主方法:** **PC-HybridTD3**（物理约束参数化混合 TD3：Actor 在 \(\mathcal A_f(s)\) 上解码 mode+magnitude；Critic 看 \((e_m,z)\)；经济 Bellman 只用 FMU 真转移）。投影连续 TD3 与静态带宽 hybrid TD3 为仅有的两组消融。FS-HSAC 不再作为论文身份。

**体裁:** Applied Energy 能源调度 + DRL（对标 OCTD3 / GHTD3）。

**单一源（数字）:** Stage D 之后主表取 `runs/seasonal_tou2026/**/pc_hybrid_td3_s*`。**禁止**把 year-constant `seasonal_v1` 或归档 `fs_hsac_*` 现金混入月度代理购电正文。投影消融 = `--ablation projection`；静态支撑消融 = `--ablation static-support`。

---

## 0. Status snapshot（2026-08-31）

| 项 | 现状 |
|----|------|
| 代码 `src/training/hybrid_td3/` | **P0 已落地**：动态 \(\mathcal A_f(s)\) 解码、6-D critic、target \(\arg\max m'\)+只噪 \(z\)、physical-only replay |
| 训练入口 | `scripts/train_seasonal.py --method td3 --season all`；36/8/8 |
| `Paper/main.tex` | 非实验正文已齐：联合 \(\mathcal A_f=\mathcal A_{\mathrm{dev}}\cap\mathcal A_{\mathrm{grid}}\)、解析 decoder 窗、无最短运行锁、graphical abstract、利益声明。§5 表图仍空 |
| 购电 | 2026 月度 110 kV 两部制；`tab:tou-monthly` + `tab:tou-windows`；09–12 顺延 8 月（S） |
| PAMDP 形式化 | `docs/pamdp_formalization.md` |
| 主数字源 | Stage D TEST weeks（尚未跑） |
| 消融对照 | 投影 TD3；静态支撑 hybrid TD3；rule / rolling MILP |

**写作门槛:** Stage D 前 **§5 表图占位不填假数**；不声称 RL 现金优于 MILP。FS-HSAC 不再是论文身份。

文档更新：2026-08-27 22:10 (+08:00)。`Paper/main.tex` 摘要、highlights、引言分段、`tab:lit`（+Jendoubi/Ochoa/Fan 2019）、§5 实验协议与 §6 已改成期刊口径；内部 “results gate” 字样已从正文去掉。空表保留。`pdflatex`+`bibtex` 通过（24 页）。

---

## 0.5 三条贡献（不超过三条）

1. **非凸 CAES 混合动作** — \((m,z)\) 保拓扑，避免盒投影死区。
2. **联合 \(\mathcal A_f(s)=(\mathcal A_T\times\mathcal A_B\times\mathcal A_C)\cap\mathcal A_{\mathrm{grid}}\)** — 解析 decoder，不是分量盒再 GiveSafe 蒙。
3. **FMU 闭环验证** — 物理-only Bellman；36/8/8 TEST；rule / rolling MILP（同一 FMU）/ projection TD3；noisy + 8760 h 部署。碳价/磨损/启停/预测/GiveSafe/Gumbel/TD3 不当贡献。

**勿写**

- HMSD / c-step 分层作为正文身份（可作为相关工作对照）
- “RL 会预判所以优于 MILP”
- 把断开合法集当主要创新或“发现”
- 把动态幅值收缩说成纯 FMU 物理（含 oracle 余量）
- “仿真很少用 FMI”“option 可自由选”“PC-HybridTD3 天然优于 HRL”
- 把投影静音 / clamp / Bellman 自环写成 Intro gap
- first hierarchical CAES；new TD3 家族；\(r^{\mathrm{lo}}=r^{\mathrm{ext}}\)

**相对文献的差**

| 参照 | 本工作差在哪 |
|------|----------------|
| Cui *Applied Energy* OCTD3 | option/HRL 以多时间尺度与 initiation 为主；这里强调同小时参数化混合动作 + \(\mathcal K(s),\mathcal M_k(s)\) 密度；闭环在 Sysplorer FMU |
| Cui *Energy* GHTD3 | 不以库存目标分层为身份；直接优化模式—幅值混合支撑 |
| CHPO NeurIPS 2025 | 会议方法名；这里是厂级 FMU 调度应用 |
| GiveSafe / Ceusters | 安全层「采用」，不是提出 |
| 凸 IES / 盒动作 DRL | 少写状态相关混合支撑，或不用多罐 CAES 多物理闭环验证 |

---

## 0.6 推荐期刊

| 档 | 期刊 | 条件 |
|----|------|------|
| **主攻** | **Applied Energy** | 系统+支撑形式化+PC-HybridTD3+TEST 分项成本表写干净 |
| **同等** | **Energy** | 更挤（GHTD3 同刊）；强调多模态压空 + FMU |
| **备选** | ECM；IEEE TII | 孪生与工业可执行性作辅线 |
| **不优先** | IEEE TPWRS；NeurIPS/ICML | 不是电网出清，也不是新 RL 家族论文 |

---

## 1. Paper outline（六章，照 GHTD3）

### 1. Introduction（已写入 `main.tex`）

- 厂级 price-taker：山东 TOU + 国家 ETS；多模态压空与火电—电池协同。
- 文献三类：凸/启发式优化 / 盒动作或投影 DRL / option 或库存分层 DRL（OCTD3、GHTD3）。
- 三个 gap（可辩护口径）：
  1. **模型与验证**：多能 DRL 调度多用代数/简化环境；针对多罐 CAES 热—气耦合、并在 FMI 封装多物理模型上做闭环策略验证的工作有限（不写“仿真很少用 FMI”）。
  2. **形式化**：需把 \(\mathcal K(s)\)（当前可选模式）与 \(\mathcal M_k(s)\)（模式条件幅值区间）统一为状态相关混合支撑 \(\mathcal A(s)\)；option initiation set 也可限制模式，但不能替代本对象的模式—幅值参数化写进策略密度。
  3. **方法需求**：在每小时同时决定模式与幅值、且仅有三个模式时可精确枚举时，需要同时间尺度最大熵混合策略；不宣称 HRL 无法处理。
- 四条贡献见 §0.5。
- **表：** `tab:lit`。

### 2. System description（物理章保留）

- 拓扑：风光、火电、电池、多罐 CAES、母线、电网。
- Sysplorer → FMI FMU，通信间隔 1 h。
- 三季边界与 **月度** 110 kV TOU、ETS。Held-out 周对应 2 月 / 5 月 / 8 月。
- **图：** Fig.1 `fig_topology`；Fig.2 `fig_price_tou`（冬评日 + 1 月 vs 8 月）；Fig.3 `fig_seasonal_boundary`（周 5/18/31 真实月度价，非 tile 1 月）。
- **表：** `tab:params`；`tab:tou-monthly`；`tab:tou-windows`。

### 3. Problem formulation

- 3.1–3.3 发电 / 转换 / 储能物理（保留）
- 3.4 系统运行约束：min-load 带、模式锁——**写在约束里，不当发现**；断开合法集作设备包线；无最短运行锁（崔 2024 启停费）
- 3.5 优化目标 \(J^{\mathrm{gen}}\) 分项 + 周末库存软加分
- 3.6 **状态相关混合 MDP**：\(\mathcal A(s)=\mathcal A_{\mathrm{tp}}\times\mathcal A_{\mathrm{bat}}\times\bigcup_k\{k\}\times\mathcal M_k(s)\)；解码进动态区间；奖励 \(r^{\mathrm{ext}}\)
- **图：** Fig.4 `fig_caes_legal`。

### 4. Solution methodology（已按 PC-HybridTD3 重写）

- 4.0 **为何同时间尺度、而非库存 HRL**（已成小节）
- 4.1 孪生闭环：Actor 解码已在 \(\mathcal A_f(s)\)；拒绝不进 \(\mathcal D_B\)；季节栈 shadow off
- 4.2 动作表示对照：投影 vs 静态带 vs 动态支撑；`fig_action_rep`
- 4.3 PC-HybridTD3 actor：掩码分类、单一幅值头、动态区间仿射
- 4.4 6 维 hybrid critic + TD3 target（\(\arg\max m'\)，只噪 \(z'\)）
- 4.5 网络：256-ReLU 双层；lr \(10^{-4}\)；batch 64
- 4.6 物理-only 经济 replay + 采用 GiveSafe；拒绝进安全审计
- 4.7 两组消融：连续 vs 混合；静态 vs 动态。主方法：rule / rolling MILP / projection TD3 / PC-HybridTD3
- **图 / 算法框 / 表：** `fig_algorithm`；`fig_action_rep`；Alg. PC-HybridTD3；`tab:hyper`

### 5. Simulation results（数字过门后再填）

- 5.1 设置：硬件、超参、三季、基线；**统一结算**见 `docs/comprehensive_cost_terms.md`；**参数出处**见 `docs/parameter_evidence.md`（profile `official-2024-ets-sd-grid-v1`）
- **5.2a 经济：** `tab:main` / `tab:econ` — 仅 `valid_steps=168`；主列 \(CC=-J^{\mathrm{gen}}\) 与分项（cash / CO₂ / CUT / deg / su / grid）
- **5.2b 消纳与灵活性：** 弃电 MWh/率、可再生利用率；合同越限 MWh/小时、\(|P_{\mathrm{grid}}|_{\max}\)、峰谷差/爬坡（1 h，不写快速瞬态）
- **5.2c 机制消融（多能协同）：** 完整 thermal+BESS+CAES vs **lock-CAES**（及已有储能受限对照）；报告 \(\Delta CC\)、\(\Delta E_{\mathrm{curt}}\)、越限/峰谷、BESS/CAES 分时功率与启停——用轨迹+分项说明互补，**不**造单一“协同指数”
- **5.2d 可靠与在线计算：** unserved、有效步、FMU 失败、GiveSafe 拒绝、\(E_{\mathrm{terminal}}\)；PC-HybridTD3 推理时延 vs rolling MILP 每步求解（mean/p95/max/超时率）
- 5.3 灵敏度：官方碳价带、β/η、可行性裕度、压空容量、弃电/缺供、合同价、**启停缩放模式**、TOU 构造基价（见 `docs/sensitivity_section.md`）
- **声明规则：** 最低 CC 只称经济最优；更低弃电/越限只称对应维；仅当 CC+弃电+可靠性均不差时才写“总体更优”，否则 Pareto；价格参数分 O/M/L/S 四级，禁止把情景价写成监管价
- **辅助：** `tab:run` 作脚注，不是主经济 KPI

### 6. Conclusions

- 编号四点：DAE 对象；\(\mathcal A_f(s)\)；同时间尺度 PC-HybridTD3；月度 110 kV 结算。无现金排名。
- 限制：Stage D 前无数；动态区间含 oracle 余量；周重置；09–12 为 S 级顺延
- 若 MILP 更便宜：按价格/约束权衡与代理模型近似解释（文献路径），不改论文身份，不写“RL 预判更聪明”

---

## 1.5 reference_papers 证据边界（写作约束）

每条论文声明须能回答「比较了什么、没有比较什么」。借鉴评价框架，**不**照搬过强结论。

| 来源 | 借鉴 | 适用边界 / 禁止写法 |
|------|------|---------------------|
| **OCTD3 / CAES–BESS（AE 2024）** | 机制组 × 算法组两套对照；经济—稳定—协同多维（PFI/CCI/SRSI 思路） | 本项目 1 h 分辨率 → **不直接照搬 SRSI**；改用合同越限、网交换波动、设备机制消融。OCTD3 **没有** MILP 对照，勿借其结论写“RL 优于精确优化” |
| **GHTD3** | 三季典型周、综合成本分项、QP 对照 | **保留关键事实**：凸化 QP 常给出最低标量 CC，但弃电/多能利用可能较差 → **按维解释**，禁止写成 RL 全面优于求解器 |
| **A3C vs CPLEX** | “成本接近最优 + 在线策略”定位 | 用作 PC-HybridTD3 **不必击败**精确优化器的论据；其成本约高 0.33%，不是 RL 胜出 |
| **多智能体 HRL vs CPLEX/MPC** | 全知优化器作下界；RL 作非完美预测反馈策略 | 本项目**主矩阵共享 perfect forecast** → **不得**用不确定性优势替经济结果辩护 |
| **MADRL 多时间尺度竞价** | 同时报告利润、失衡量、跟踪率、每步计算时间 | 对应报告 CC、合同越限/网交换、可执行性、推理/求解时间 |

**Intro / baseline / Discussion 引用口径：** 只写上表允许的比较；若某方法在 CC 优而弃电差，写 Pareto + 灵敏度，不写算法智商叙事。

---

## 1.6 四组可复现 KPI ↔ `train_result.json` 字段

权威累加：`src/training/evaluate_td3.py` → `metrics` / `cost_terms`；抽取：`src/optimization.metrics.extract_kpi_from_eval`（`scripts/train_seasonal.py` 的 `kpi_from_eval` 同构）。

| 组 | KPI | JSON / kpi 字段 |
|----|-----|-----------------|
| **经济** | \(J^{\mathrm{gen}}\), \(CC\) | `sum_delta_j_gen`, `comprehensive_cost_cny` |
| | 分项 | `net_cashflow_j`, `carbon_cost_cny`, `cut_cost_cny` / `curtailment_cost_cny`+`unserved_cost_cny`, `battery_deg_cost_cny`, `caes_startup_cost_cny`, `grid_contract_cost_cny` |
| **消纳** | 弃电、可用、利用率 | `curtailment_mwh`, `renewable_available_mwh`, `curtailment_rate`, `renewable_utilization` |
| **灵活性** | 合同越限、峰谷、爬坡 | `grid_contract_excess_mwh`, `grid_contract_violation_hours`, `grid_abs_max_mw`, `grid_peak_valley_mw`, `max_grid_ramp_mw` |
| **可靠** | 缺供、执行 | `unserved_mwh`, `valid_steps`, `fmu_failure_count`, `forbidden_action_count`, `terminal_soc_*` |
| **计算** | 时延 / 超时 | `decision_time_{mean,p95,max,sum}_s`, `solver_timeout_count`, `solver_timeout_rate` |

`Paper/main.tex` 表头与公式符号须与上表一致。PC-HybridTD3 训练代码只消费统一评测结果，不另造加分指标。

---

## 1.7 系统机制消融（证明多能协同）

在同一季节、同一预测（perfect）、同一 \(J^{\mathrm{gen}}\) 下：

| 变体 | 含义 |
|------|------|
| Full | thermal + BESS + CAES（主矩阵） |
| lock-CAES | 压空强制 idle / 不可调度（机制关断） |
| storage-limited（若已有） | 储能功率/能量受限对照 |

报告：\(\Delta CC\)、\(\Delta E_{\mathrm{curt}}\)、合同越限与峰谷差、BESS/CAES 分时段功率、模式小时、启停次数。结论用分项成本 + 轨迹叙述互补，禁止未校准的单一协同指数。

---

## 2. Figures（体裁标配）

Basename 均在 `Paper/figures/`。

### A. 主文

| ID | File | 作用 | 状态 |
|----|------|------|------|
| Fig.1 | `fig_topology` | 厂级拓扑 | 有 |
| Fig.2 | `fig_price_tou` | 月度 TOU：冬评日 + 1 月 vs 8 月 | 有（`gen_fig_price_tou.py`） |
| Fig.3 | `fig_seasonal_boundary` | 三季 held-out 周风光荷 + 月度价 | 有（`gen_fig_seasonal_boundary.py`；周 5/18/31） |
| Fig.4 | `fig_caes_legal` | CAES 合法包线 + 模式锁 | 有 |
| Fig.5 | `fig_action_rep` | 投影 vs 固定带 vs \(\mathcal M_k(s)\) | 有（三栏 matplotlib） |
| Fig.6 | `fig_algorithm` | PC-HybridTD3 闭环（\(\mathcal D_B\) 物理 / \(\mathcal D_S\) 审计） | 有（`gen_fig_algorithm.py`） |
| Fig.7+ | 功率平衡 / SoC / 成本条 | §5 | 待数据 |

### B. 附录 / 补充

| File | 作用 |
|------|------|
| `fig_aux_obs` | 观测栈示意 |
| `fig_givesafe_reject` | 拒绝不进 Bellman |
| `fig_caes_feasible_set` | \(\mathcal M_k(s)\) 动态收缩示意 |

---

## 3. Code ↔ paper mapping

| 论文对象 | 代码 |
|----------|------|
| \(\mathcal A_f(s),\mathcal M_k(s)\) | `actions/caes_u.py` + FeasibilityOracle |
| PC-HybridTD3 actor / critic | `hybrid_td3/actor.py`, `critic.py` |
| TD3 target + ST-Gumbel | `hybrid_td3/algorithm.py` |
| 物理-only replay | `replay/hybrid_replay_buffer.py`, `hybrid_td3/givesafe_collector.py` |
| 训练入口 | `hybrid_td3/train.py`；`--method td3` |
| \(J^{\mathrm{gen}}\) / \(CC\) 真源 | `envs/reward_calculator.py` + `config/reward_config.yaml` |
| 参数出处台账 | `docs/parameter_evidence.md`（`parameter_profile_id`） |
| 统一评测 KPI | `training/evaluate_td3.py`；`optimization/metrics.py` |
| 季节入口 KPI | `scripts/train_seasonal.py` → `kpi_from_eval` |
| MILP/linprog 时延 | `rolling_*.py` → `last_solve_s` / timeout flags |
| fixed-band 消融 | 旧 `hybrid_sac` / `parameterized_caes` |
| 投影消融 | 旧连续 SAC/TD3 + `clamp` |

**已知未写进论文声明的工程细节：** 执行侧仍以 GiveSafe 为硬门；\(C_\psi\) 目前作 actor 风险惩罚与分类器训练，**未**另设 \(C_\psi\ge 1-\varepsilon\) 硬门控。消融 `FS_HSAC_NO_FEAS=1` 对应 support-only 变体。

**奖励锁定：** 不改正在远程训练的奖励语义；KPI 扩展只增加评测/汇总字段，新评测阶段统一重算。
