# Paper outline and figure checklist

文档更新：2026-08-31 18:30 (+08:00)

**Working title:** *Economic Scheduling Optimization of Multi-Energy Systems via Physics-Constrained Hybrid-Action Reinforcement Learning*

**短标题:** *Physics-constrained hybrid TD3 for multi-energy scheduling*

**主方法:** **PC-HybridTD3**（预测感知状态 → 异构 Actor → 联合 \(\widehat{\mathcal A}_f(s)\) → GiveSafe → FMU；经济 Bellman 只用真转移）。\(\widehat{\mathcal A}_f\) 是可计算 feasible-support 近似，\(\mathcal A_{\mathrm{FMU}}\) 才是孪生可执行集。研究对象是多能源安全经济协同调度。投影连续 TD3 与分量支撑 hybrid TD3 为仅有的两组消融。FS-HSAC 不再作为论文身份。

**体裁:** IEEE conference（唯一正式稿 `Paper/main.tex`）。Elsevier CAS / highlights / graphical abstract / 大 nomenclature 已退出正文。FS-HSAC 会议稿 `Paper/icpre2026/` 已删除。

**单一源（数字）:** Stage D 之后主表取 `runs/seasonal_tou2026/**/pc_hybrid_td3_s*`。**禁止**把 year-constant `seasonal_v1` 或归档 `fs_hsac_*` 现金混入月度代理购电正文。投影消融 = `--ablation projection`；静态支撑消融 = `--ablation static-support`。

---

## 0. Status snapshot（2026-08-31）

| 项 | 现状 |
|----|------|
| 代码 `src/training/hybrid_td3/` | **P0 已落地**：动态 \(\mathcal A_f(s)\) 解码、**3-D critic** \(Q(s,u_T,u_B,u_C)\)、target \(\arg\max m'\)+只噪 \(z\)、physical-only replay；Oracle `d5.3-idle-robust-endpoint` |
| Stage C 过门 | `compute_stage_c_gates`：NaN/Inf、FMU hard、held-out NoSafeAction、缺供、成本优于 random。CAES 使用不是过门 |
| 队列 | `logs/pc_hybrid_queue.py`：A→B→C；`stage_c_passed` 前 SKIP Stage D |
| 训练入口 | `scripts/train_seasonal.py --method td3 --season all`；36/8/8 |
| `Paper/main.tex` | **IEEE conference 重置完成**：I Intro → II Problem → III Method → IV Case study（四陈述小节）→ V Conclusion。图冻结 Fig.1–5。弃用图在 `Paper/figures/_dropped/`。月度 TOU 全表在 `Paper/supplementary_tou.tex`。数字仍空 |
| 购电 | 正文一句 + Table I TOU range；完整月度表在仓库，不进会议正文 |
| PAMDP 形式化 | `docs/pamdp_formalization.md` |
| 主数字源 | Stage D TEST weeks（C 过门前不启动） |
| 消融对照 | 投影 TD3；静态支撑 hybrid TD3；rule / rolling MILP |

**写作门槛:** Stage D 前 **§5 表图占位不填假数**；不声称 RL 现金优于 MILP。FS-HSAC 不再是论文身份。

文档更新：2026-09-02 18:00 (+08:00)。标题改为 Economic Scheduling；GiveSafe 为 residual filter，不是 adopted。

---

## 0.5 三条贡献（不超过三条）

1. **异构设备混合动作** — 火电/电池连续；CAES 模式–幅值。避免连续投影死区。
2. **系统级联合 \(\widehat{\mathcal A}_f(s)\)** — 可计算动态 feasible-support；GiveSafe 最后验证。不把 \(\widehat{\mathcal A}_f\) 写成 \(\mathcal A_{\mathrm{FMU}}\)。
3. **FMU 闭环学习** — 24 h 前瞻作为输入（不进标题）；物理-only Bellman；36/8/8 TEST；rule / rolling MILP / projection TD3；8760 h 部署。

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
| GiveSafe / Ceusters | 安全层是本工作实现的 residual filter；Garcia 等是 safety-filter 背景，不是 GiveSafe 来源 |
| 凸 IES / 盒动作 DRL | 少写状态相关混合支撑，或不用多罐 CAES 多物理闭环验证 |

---

## 0.6 体裁

**当前唯一正式稿：IEEE conference**（`Paper/main.tex`）。期刊 Applied Energy / Energy 路线已冻结，不并行维护第二份 CAS 稿。

---

## 1. Paper outline（五节，IEEE conference）

### I. Introduction（约 1 页）

只讲四件事：多能源为何协同调度；MILP/DRL 各自问题；continuous / hierarchical DRL 的 gap；PC-HybridTD3 与 3 条贡献。Cui 放 related-work 段落。山东 110 kV 电价不在引言展开。

### II. Multi-Energy Scheduling Problem（约 1.5 页）

- A. Topology：`fig_topology`。
- B. Dispatch-level constraints：功率平衡、火电箱与爬坡、电池 SoC、CAES 包线与状态、电网限。PV 温度方程、风机 cubic、冷热罐 ODE **不**进会议正文。
- C. \(C_t\) 七项 + MDP：\(s_t=[x_t,f_{t:t+23},\lambda_{t:t+23},z_t]\)，\(\widehat{\mathcal A}_f=\mathcal A_{\mathrm{dev}}\cap\mathcal A_{\mathrm{grid}}\)。
- **表：** `tab:params`（含 TOU range）。月度电价全表、碳账户图退出正文。

### III. PC-HybridTD3（约 2 页，全文重点）

- 联合 \(\widehat{\mathcal A}_f\) 顺序解码；GiveSafe 残差；physical-only TD3。
- 消融：Continuous-projection TD3 → Component-support Hybrid TD3 → PC-HybridTD3。
- **图 / 算法 / 表：** `fig_algorithm`；`fig_action_rep`；Alg. PC-HybridTD3；`tab:hyper`。

### IV. Case study（约 2.5–3 页；数字 Stage D 后填）

四小节（标题用陈述句，不用疑问句）：

1. Training convergence → Fig.~4 training（Stage D 生成）。
2. System-level scheduling performance → `tab:kpi`（cost / grid / carbon / curt / uns / viol / runtime）。
3. Coordinated weekly dispatch → Fig.~5 四联 dispatch+SOC+price（Stage D 生成）。
4. Ablation and robustness → 消融链 + noisy forecast + 8760 h 部署检查。

不写 cold-tank 调试、碳账户、annual-reset debug、GiveSafe reject 主文图。KPI bar 与 Table 重复则删图留表。

### V. Conclusion（约 0.3 页）

协同调度 + \(\widehat{\mathcal A}_f\) + physical-only TD3。限制：oracle 余量、周重置、09–12 顺延。不声称 RL 优于 MILP。

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

## 2. Figures（冻结 5 张）

Basename 均在 `Paper/figures/`。会议正文只保留下列主文图。

| ID | File | 作用 | 状态 |
|----|------|------|------|
| Fig.1 | `fig_topology` | 多能源拓扑 | 有 |
| Fig.2 | `fig_algorithm` | PC-HybridTD3 闭环 | 有；caption 已按 \(\widehat{\mathcal A}_f\) 重写 |
| Fig.3 | `fig_action_rep` | 消融用 CAES 子动作表示 | 有；面板标题对齐 baseline 名 |
| Fig.4 | `fig_training` | 三方法验证成本收敛 | Stage D 生成；正文 comment，不渲染 placeholder |
| Fig.5 | `fig_dispatch_week` | 典型周：外生 / 调度 / 电价+净负荷 / 库存 | Stage D 生成；正文 comment |

**删除（不进正式稿）：** `fig_placeholder`、`fig_caes_legal`、`fig_caes_feasible_set`（二者 SHA 相同）、`fig_aux_obs`、`fig_seasonal_boundary`、KPI bar、cold-tank guard、annual-reset、carbon-position / settlement、GiveSafe reject。完整月度 TOU 表移出正文。

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
