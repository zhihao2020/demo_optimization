<!-- ARCHIVED: superseded by FS-HSAC; see docs/paper_outline_and_figures.md -->

# 原理层创新：不止“教师残差 / packaging Cui”

文档更新：2026-08-10 21:15 (+08:00)

> **定位**：投稿叙事 / 创新主张底稿，**不是**运行配置说明书。  
> 代码主线默认见 `src/config/ghtd3_config.yaml` 与 `docs/GHTD3分层实现说明.md`：`goal_dim=2`、`low_reward=ext`、`market_goal_prior=false`、`f_mle_pretrain=false`、**无** Hybrid residual teacher。  
> 下文 MSGP / F-MLE / λ-SoC 等为 **可辩护可选机制**（消融或论文模块），启用时须与 yaml 一致。

> 目标：把投稿叙事从 *GHTD3 + 工程插件* 抬到 **market–inventory–feasibility (MIF) 约束下的分层决策理论 + 可证伪机制**。  
> 主方法名：**HMSD**（Hierarchical Market-Safe Dispatch）。  
> Backbone 工具：goal-conditioned hierarchical TD3（Cui 等，Energy 2025）——**承认借用，不假装新 actor–critic**。

---

## 0. 审稿人会怎么打

| 攻击 | 若我们只写 | 正确反击 |
|------|-----------|----------|
| “只是 Cui 换场景” | 列表式 MSGP/HER/BC | **决策问题不同**：周尺度 price-taker **现金流 \(J\)** + 能量 SoC 门 + 混合 CAES 可行域 \(\mathcal{F}(s)\) + FMU 孪生；Cui 是 H-IES 综合成本 |
| “组件是工程凑的” | prior / BC / shield 堆叠 | **统一目标**：在投影 MDP 上同时优化市场套利、库存闭环、可执行性 |
| “15k 消融 full 不是最优” | 辩解短训 | **35k 同预算消融**（主表预算）闭合归因 |
| “离最优多远” | 无 | 松弛 LP / 简化周 MILP 上界 gap |
| “安全是口号” | 名义关 GiveSafe 无差 | **扰动下** \(J:0\to1.24\times10^7\) + 投影足迹表 |

---

## 1. 科学主张（一句话）

**HMSD 不是新 TD3，而是把“峰谷套利–周库存闭环–动态可行域”写成可学习的双时间尺度决策，并给出在投影动力学下可训练的分层信用分配与目标空间结构。**

形式化为带安全投影的混合动作 MDP：

\[
\max_{\pi}\ \mathbb{E}\Big[\sum_t \gamma^t r_t^{\mathrm{ext}}(J)\Big]
\quad\text{s.t.}\quad
a_t=\Pi_{\mathcal{F}(s_t)}(\tilde a_t),\quad
c_{\mathrm{soc}}(s_T)\le\varepsilon.
\]

高层在 **市场–库存目标空间** \(\mathcal{G}\) 上选意图；底层在 \(\mathcal{F}(s)\) 内跟踪；GiveSafe 定义投影执行。

---

## 2. 五条可辩护机制（按“原理含量”排序）

### M1. 投影执行 MDP + 可行性足迹（GiveSafe 理论位）

- **原理**：学习发生在 \(\Pi_{\mathcal{F}}\) 诱导的 **投影动力学** 上，而非开环无约束 MDP。  
- **可测**：投影率 / \(\ell_2\) / 无效转移 / 扰动 stress（正文 Table givesafe + \(\sigma=0.4\)）。  
- **与工程 shield 的区别**：我们 **同时** 报告 (a) 名义策略已内化可行、(b) 分布偏移下 shield 是生存条件——这是 safe RL 标准叙事，不是“挂了个投影器”。

### M2. 市场–库存目标空间（MSGP）≠ 动作教师残差

- **原理**：结构先验进 **goal** \(g\in\mathcal{G}\)，不进底层 residual 教师。  
  \[
  g\leftarrow(1-w_m-w_r)\mu^{\mathrm{hi}}+w_m g^{\mathrm{mkt}}+w_r g^{\mathrm{rec}}.
  \]
- **为何不是 packaging**：Cui 的 goal 是通用 residual inventory；本文 goal 显式编码 **TOU 相位 + 周末回收课程**，与 \(J\) 与 \(c_{\mathrm{soc}}\) 同构。  
- **禁忌**：正文禁止把 F-MLE / 旧 Hybrid residual 写成“强教师主贡献”。

### M3. 市场结构 hindsight（MS-HER）

- **原理**：高层 off-policy 信用在 **可达成 + 市场加权** 候选上重标，而不是均匀 HER。  
- 候选集含 \(g^{\mathrm{orig}},g^{\mathrm{ach}},g^{\mathrm{fut}},g^{\mathrm{mkt}}\)，权重跟存储活跃度 / 窗内 \(\Delta J\) 相关。  
- **可消融**：no-HER → 种子方差爆炸（15k 已见）；35k 验证是否转为 mean 增益。

### M4. 意图–调度因果通路（obs_norm + \(\kappa\) + IDD）

- **原理**：MW 量级 raw obs 会淹没 goal 通道 → \(\partial a/\partial g\approx0\) → 分层退化为单层。  
- 归一化 + goal scale 恢复 **意图可微性**；IDD（冻高层/冻底层）是 **机制实证**，不是调参附录。  
- 这是相对 Cui 默认实现的 **可学习性条件**，可写成命题级 claim。

### M5. 库存对偶 / 约束分层（λ-SoC，可选升格为主机制）

- **原理**：硬 SoC 门化为高层约束：  
  \[
  R^{\mathrm{hi}}=\bar r^{\mathrm{ext}}-\lambda[c_{\mathrm{soc}}]_+,\quad
  \lambda\leftarrow[\lambda+\eta c_{\mathrm{soc}}]_+.
  \]
- seed2 夏季失败 → λ 从初始化修复 SoC（已有 Table lambda）。  
- **升格路径**：若 35k full 在 SoC 上仍弱于某 dropout，把 λ-SoC 写进 **主栈**（非仅 sensitivity），叙事变为 **constrained hierarchical market RL**。

---

## 3. 下一档“更大创新”（实现优先级）

仅当审稿或自检仍嫌增量时，按 **改动量 / 重训成本 / 新颖性** 选 1 条做深，不要五条并行：

| ID | 机制 | 新颖性 | 成本 | 建议 |
|----|------|--------|------|------|
| **A** | **Tariff-aligned SMDP (TAS)**：\(c_t\) 对齐峰谷边界，折扣 \(\gamma^{c_t}\) | 高（时间抽象真变） | 中（改交互 + 重训） | P1 首选算法创新 |
| **B** | **Projection-aware actor**：\(\nabla_\phi Q(s,\Pi_{\mathcal{F}}(\mu_\phi))\) 或 executed-action DPG | 高（safe RL 理论位） | 中 | 与 GiveSafe 叙事合一 |
| **C** | **Realizability-aware high level**：高层惩罚“\(c\) 步后不可达 goal” | 中高 | 低–中 | 易塞进 MSGP/MS-HER |
| **D** | **简化周 MILP / 凸上界 gap** | 实验硬通货 | 中（建模） | P0 实验，不算法 |
| **E** | 真新 backbone（SAC-HRL / 新 critic） | 易被说堆方法 | 高 | **不建议**作主创新 |

**推荐组合（稳进 1 区证据链）**：  
实验 P0（35k 消融 + gap + stress 数字） + 叙事 MIF + **可选一条 A 或 B 作为“机制深化”**。

---

## 4. 与 15k 消融尴尬结果的逻辑闭环

| 现象 | 机制解释 | 35k 期望 |
|------|----------|----------|
| w/o F-MLE 均值更高 | 冷启动结构挤占早期探索 | full ≥ nofmle 或 SoC 更稳 |
| w/o MSGP 均值更高 | prior 早期偏置 | full 峰谷/回收更稳 |
| w/o MS-HER 方差大 | 信用噪声 | full 方差显著更低 |
| full 15k 非最优 | **短预算诊断**，非主表 | **主表预算下 full 不劣于各 dropout** |

若 35k 仍出现 “拿掉更好”：  
1) 诚实改主 claim（模块 = 可靠性/可训练性）；  
2) 启用 M5 λ-SoC 作 full+constraint；  
3) 勿伪造表。

---

## 5. 投稿定位（避免 Energy 同刊撞车）

- **宜投**：Applied Energy / Energy Conversion and Management / Applied Soft Computing（强调 FMU 闭环 + 市场库存可行域 + 安全投影证据）。  
- **慎投 Energy**：Cui 2025 同刊骨架论文，新颖性对比最刺眼。  
- 摘要/贡献 **第一句永远是问题与耦合**，第二句才是 HMSD 机制；禁止摘要以 “we propose GHTD3 variant” 起笔。

---

## 6. 执行清单（与仓库对齐）

| 项 | 状态 | 路径 |
|----|------|------|
| MIF 叙事 + 贡献三条 | ✅ 正文 | `Paper/main.tex` |
| \(J\) 主表 | ✅ | `tab:main` |
| GiveSafe 足迹表 + stress 数字 | ✅ / 强化 | `tab:givesafe` + \(1.24\times10^7\) |
| 15k×3 seed 消融 | ✅ | `runs/paper_ablation_multiseed_15k.*` |
| **35k×3 seed 同预算消融** | ⏳ 远程 | `_remote_ablation_grid.py --steps 35000` |
| MILP/凸上界 gap | ⏳ | 待脚本 |
| SAC-Hybrid 80k | ⏳ 可选 | `scripts/train_hybrid_sac.py` |
| TAS / projection-aware actor | ⏳ 可选深化 | 见 §3 A/B |

---

*本文档供写作与排期；实现以实验脚本与 `Paper/main.tex` 为准。*
