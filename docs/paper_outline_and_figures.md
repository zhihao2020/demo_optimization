# Paper outline and figure checklist

文档更新：2026-08-28 16:40 (+08:00)

**Working title:** *Plant-level weekly dispatch of a thermal–BESS–CAES system with state-dependent multi-mode commands: a feasible-support hybrid soft actor–critic approach on a high-fidelity Modelica twin*

**短标题:** *Feasible-support hybrid SAC for multi-mode CAES plant dispatch*

**主方法:** **FS-HSAC v2**（状态相关可行支撑集上的混合 SAC：动态 \(\mathcal M_k(s)\)、精确模式枚举、双温度、残余可行性分类器）。旧 Hybrid SAC（固定设备带 + clamp）与投影 SAC 作消融。

**体裁:** Applied Energy 能源调度 + DRL（对标 OCTD3 / GHTD3）。

**单一源（数字）:** 主表取 `runs/seasonal_tou2026/**/fs_hsac_s0`（过 `docs/fs_hsac_results_gate.md` 后）。**禁止**把 year-constant 五档的 `seasonal_v1` 现金混入月度代理购电正文。`sac_param_s0` = fixed-band 消融；旧 `sac_s0` = 投影消融。

---

## 0. Status snapshot（2026-08-26）

| 项 | 现状 |
|----|------|
| 代码 `src/training/fs_hsac/` | **已落地**；推理只采当前模式；精确三模式枚举留在 update |
| 训练入口 | `scripts/train_seasonal.py --method fs_hsac`；远程 seed-0 在 `runs/seasonal_tou2026`（CPU 三季重开中） |
| `Paper/main.tex` | **可投骨架**（cas-sc，24 页）：摘要/引言/`tab:lit`/C1–C4/§5 协议已去门控黑话；经济表仍空 |
| 购电 | 2026 月度 110 kV 两部制；`tab:tou-monthly` + `tab:tou-windows`；09–12 顺延 8 月（S） |
| PAMDP 形式化 | `docs/pamdp_formalization.md` |
| 主数字源 | `seasonal_tou2026` FS-HSAC seed-0（待三季满 168 h 过门） |
| 消融对照 | 投影 SAC/TD3；fixed-band hybrid SAC；FS-HSAC-support；PSO / linprog / MILP |

**写作门槛（results gate）:** FS-HSAC 须三季满 168 h，且综合成本优于投影 SAC/TD3、fixed-band Hybrid SAC 与 PSO。数字未回之前 **§5 表图占位不填假数**；§6 只收束因果链，不写优越性百分比。

文档更新：2026-08-27 22:10 (+08:00)。`Paper/main.tex` 摘要、highlights、引言分段、`tab:lit`（+Jendoubi/Ochoa/Fan 2019）、§5 实验协议与 §6 已改成期刊口径；内部 “results gate” 字样已从正文去掉。空表保留。`pdflatex`+`bibtex` 通过（24 页）。

---

## 0.5 四条贡献（对齐 OCTD3 / GHTD3）

1. **系统与机制**  
   厂级火电 + 电池 + 绝热多罐压空 DAE，三模态（充/闲/放），山东**月度**代理购电分时 + 国家 ETS；Sysplorer Modelica 经 FMI 导出为 FMU 闭环（FMI 是交换标准，不是创新）。因果链：对象硬 → \(\mathcal A(s)\) → 孪生上可执行。

2. **形式化**  
   状态相关混合动作支撑 \(\mathcal A(s)\)：\(\mathcal K(s)=\) 当前可选模式；\(\mathcal M_k(s)=\) 选定模式 \(k\) 下允许的幅值区间。统一写进策略支撑，而不是仅事后投影/屏蔽。

3. **算法（FS-HSAC）**  
   **同时间尺度**参数化混合 SAC（离散模式头 + 条件连续幅值头）：Jacobian 校正密度、精确模式枚举、双温度 \(\alpha_d/\alpha_c\)、残余 \(C_\psi\)、GiveSafe 采用。不是高层每 \(c\) 步的库存 HRL。

4. **验证**  
   三季典型周；算法消融 + **系统机制消融（lock-CAES）**；对照 PSO、LP、MILP；分项综合成本 + 消纳/灵活性/可靠性/时延 KPI + 灵敏度（含弃电价与合同价）。

**勿写**

- HMSD / c-step 分层作为正文身份（可作为相关工作对照）
- “RL 会预判所以优于 MILP”
- 把断开合法集当主要创新或“发现”
- 把动态幅值收缩说成纯 FMU 物理（含 oracle 余量）
- “仿真很少用 FMI”“option 可自由选”“FS-HSAC 天然优于 HRL”
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
| **主攻** | **Applied Energy** | 系统+支撑形式化+FS-HSAC+三季分项成本表写干净 |
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

### 4. Solution methodology（已按 FS-HSAC 重写）

- 4.0 **为何同时间尺度、而非库存 HRL**（已成小节）
- 4.1 孪生闭环：推理只采当前模式；拒绝不进 \(\mathcal D_B\)；季节栈 shadow off
- 4.2 动作表示对照：投影 vs 固定带 vs 可行支撑；`fig_action_rep`
- 4.3 FS-HSAC actor：掩码分类、每模式幅值头、区间仿射 + Jacobian
- 4.4 混合 critic + 精确 \(V(s)\) + \(L_Q\) / \(L_\pi\)
- 4.5 网络：256-ReLU 双层；\(\log\sigma\in[-5,2]\)
- 4.6 残余 \(C_\psi\)（惩罚非第二硬门）+ 采用 GiveSafe；拆分 replay
- 4.7 消融与基线：投影 / fixed-band / FS-HSAC-support；PSO、LP、MILP
- **图 / 算法框 / 表：** `fig_algorithm`；`fig_action_rep`；Alg. FS-HSAC；`tab:hyper`

### 5. Simulation results（数字过门后再填）

- 5.1 设置：硬件、超参、三季、基线；**统一结算**见 `docs/comprehensive_cost_terms.md`；**参数出处**见 `docs/parameter_evidence.md`（profile `official-2024-ets-sd-grid-v1`）
- **5.2a 经济：** `tab:main` / `tab:econ` — 仅 `valid_steps=168`；主列 \(CC=-J^{\mathrm{gen}}\) 与分项（cash / CO₂ / CUT / deg / su / grid）
- **5.2b 消纳与灵活性：** 弃电 MWh/率、可再生利用率；合同越限 MWh/小时、\(|P_{\mathrm{grid}}|_{\max}\)、峰谷差/爬坡（1 h，不写快速瞬态）
- **5.2c 机制消融（多能协同）：** 完整 thermal+BESS+CAES vs **lock-CAES**（及已有储能受限对照）；报告 \(\Delta CC\)、\(\Delta E_{\mathrm{curt}}\)、越限/峰谷、BESS/CAES 分时功率与启停——用轨迹+分项说明互补，**不**造单一“协同指数”
- **5.2d 可靠与在线计算：** unserved、有效步、FMU 失败、GiveSafe 拒绝、末端库存；FS-HSAC 推理时延 vs rolling MILP/linprog 每步求解（mean/p95/max/超时率）
- 5.3 灵敏度：官方碳价带、β/η、可行性裕度、压空容量、弃电/缺供、合同价、**启停缩放模式**、TOU 构造基价（见 `docs/sensitivity_section.md`）
- **声明规则：** 最低 CC 只称经济最优；更低弃电/越限只称对应维；仅当 CC+弃电+可靠性均不差时才写“总体更优”，否则 Pareto；价格参数分 O/M/L/S 四级，禁止把情景价写成监管价
- **辅助：** `tab:run` 作脚注，不是主经济 KPI

### 6. Conclusions

- 编号四点：DAE 对象；\(\mathcal A(s)\)；同时间尺度 FS-HSAC；月度 110 kV 结算。无现金排名。
- 限制：单 seed；动态区间含 oracle 余量；周重置；\(C_\psi\) 非第二硬门；09–12 为 S 级顺延
- 若 MILP 更便宜：按价格/约束权衡与代理模型近似解释（文献路径），不改论文身份，不写“RL 预判更聪明”

---

## 1.5 reference_papers 证据边界（写作约束）

每条论文声明须能回答「比较了什么、没有比较什么」。借鉴评价框架，**不**照搬过强结论。

| 来源 | 借鉴 | 适用边界 / 禁止写法 |
|------|------|---------------------|
| **OCTD3 / CAES–BESS（AE 2024）** | 机制组 × 算法组两套对照；经济—稳定—协同多维（PFI/CCI/SRSI 思路） | 本项目 1 h 分辨率 → **不直接照搬 SRSI**；改用合同越限、网交换波动、设备机制消融。OCTD3 **没有** MILP 对照，勿借其结论写“RL 优于精确优化” |
| **GHTD3** | 三季典型周、综合成本分项、QP 对照 | **保留关键事实**：凸化 QP 常给出最低标量 CC，但弃电/多能利用可能较差 → **按维解释**，禁止写成 RL 全面优于求解器 |
| **A3C vs CPLEX** | “成本接近最优 + 在线策略”定位 | 用作 FS-HSAC **不必击败**精确优化器的论据；其成本约高 0.33%，不是 RL 胜出 |
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

`Paper/main.tex` 表头与公式符号须与上表一致。FS-HSAC 训练代码只消费统一评测结果，不另造加分指标。

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
| Fig.6 | `fig_algorithm` | FS-HSAC 闭环（拆分 \(\mathcal D_B/\mathcal D_F\)） | 有（`gen_fig_algorithm.py`；Image 草稿 `fig_algorithm_imgen.jpg`） |
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
| \(\mathcal A(s),\mathcal M_k(s)\) | `fs_hsac/action_support.py` |
| FS-HSAC actor / critic | `fs_hsac/actor.py`, `critic.py` |
| 精确枚举 + \(\alpha_d/\alpha_c\) | `fs_hsac/algorithm.py` (`fs_hsac_v2`) |
| 拆分 replay | `replay/fs_hsac_replay.py`, `fs_hsac/collector.py` |
| \(C_\psi\) | `fs_hsac/feasibility.py` |
| 训练入口 | `fs_hsac/train.py`；`--method fs_hsac` |
| \(J^{\mathrm{gen}}\) / \(CC\) 真源 | `envs/reward_calculator.py` + `config/reward_config.yaml` |
| 参数出处台账 | `docs/parameter_evidence.md`（`parameter_profile_id`） |
| 统一评测 KPI | `training/evaluate_td3.py`；`optimization/metrics.py` |
| 季节入口 KPI | `scripts/train_seasonal.py` → `kpi_from_eval` |
| MILP/linprog 时延 | `rolling_*.py` → `last_solve_s` / timeout flags |
| fixed-band 消融 | 旧 `hybrid_sac` / `parameterized_caes` |
| 投影消融 | 旧连续 SAC/TD3 + `clamp` |

**已知未写进论文声明的工程细节：** 执行侧仍以 GiveSafe 为硬门；\(C_\psi\) 目前作 actor 风险惩罚与分类器训练，**未**另设 \(C_\psi\ge 1-\varepsilon\) 硬门控。消融 `FS_HSAC_NO_FEAS=1` 对应 support-only 变体。

**奖励锁定：** 不改正在远程训练的奖励语义；KPI 扩展只增加评测/汇总字段，新评测阶段统一重算。
