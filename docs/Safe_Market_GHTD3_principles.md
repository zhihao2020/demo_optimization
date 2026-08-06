# Safe Market-GHTD3：原理与机制（方法节底稿）

> 骨架：goal-conditioned hierarchical TD3（与 Cui 同类，**不复现** H-IES 细节）。  
> 主线：绝对 \(\pi_{\mathrm{lo}}(s,g)\) + 无 Hybrid 教师；对照：典型单层 TD3-scratch。

---

## 0. 一句话

**Safe Market-GHTD3** = 绝对目标条件分层 TD3  
+ **MSGP**（市场结构目标先验）  
+ **MS-HER**（市场结构化 hindsight 回放）  
+ **F-MLE**（可行集过滤逆动力学预热）  
+ **GiveSafe**（执行投影）  
+ **IDD**（意图–调度解耦实证）  
（可选 **TAS-SMDP**、**λ-SoC**）

---

## 1. 与 Cui 同构对照

| Cui | 本文升级 |
|-----|----------|
| 经验回放 / HER | **MS-HER** |
| MLE / 行为复制 | **F-MLE**（规则+GiveSafe 可行 demo，非 Hybrid） |
| 高低层敏感性 | **IDD** 套件 |

---

## 2. 形式化

### 2.1 分层与执行

\[
g\in\mathcal{G}\subset\mathbb{R}^5,\quad
a=\pi_{\mathrm{lo}}(s,g),\quad
\tilde a=\Pi_{\mathcal{F}(s)}(a),\quad
s'=f_{\mathrm{FMU}}(s,\tilde a).
\]

### 2.2 MSGP（目标空间，非动作教师残差）

\[
g=\mathrm{clip}_{\mathcal{G}}\!\big((1-\lambda_t)g^{\mathrm{prior}}+\lambda_t\mu^{\mathrm{hi}}(s)\big),
\quad\lambda_t:0.55\to0.12.
\]

\(g^{\mathrm{prior}}\) = TOU 峰谷 prior ⊔ 周末 SoC 回收 prior。

### 2.3 MS-HER

候选 \(\{g^{\mathrm{orig}},g^{\mathrm{ach}},g^{\mathrm{fut}},g^{\mathrm{mkt}}\}\)，  
\(P(g^{\mathrm{ach}})\propto w_{\mathrm{tou}}\)，\(P(g^{\mathrm{fut}})\propto w_{\mathrm{econ}}\)（窗内 \(\Delta\mathrm{CF}\)/外在回报）。

### 2.4 F-MLE

\[
\max_\theta\mathbb{E}_{(s,g^{\mathrm{hind}},a)\sim\mathcal{D}_{\mathrm{feas}}}
\log\pi_{\mathrm{lo},\theta}(a\mid s,g^{\mathrm{hind}}),\quad a\in\mathcal{F}(s).
\]

\(\mathcal{D}_{\mathrm{feas}}\)：峰谷/规则轨迹中 **物理有效** 转移（`transition_valid`）。

### 2.5 GiveSafe

\[
a^{\mathrm{exec}}=\Pi_{\mathcal{F}(s)}(\pi_{\mathrm{lo}}(s,g)).
\]

### 2.6 可选 TAS / λ-SoC

- TAS：高层换 goal 对齐 TOU 边界，\(\gamma^{c_t}\)。  
- λ-SoC：\(r^{\mathrm{hi}}=\bar r^{\mathrm{ext}}-\lambda[c_{\mathrm{soc}}]_+\)，仅更新高层与 \(\lambda\)。

---

## 3. 配置开关

| 机制 | yaml / 代码 |
|------|-------------|
| abs 主线 | `src/config/ghtd3_config_abs.yaml` |
| MS-HER | `goal_relabel_mode: ms_her` 或 `her_mix`+`ms_her_weighting: true` |
| F-MLE | `f_mle_pretrain: true`（`feasible_mle.py`） |
| MSGP | `market_goal_prior: true` + 退火权重 |
| λ-SoC | `high_lambda_soc: true` |
| TAS | `tariff_aligned_c: true`（后续） |

---

## 4. 消融表（建议）

plain-HRL → +MSGP → +MS-HER → +F-MLE → Full；另 no-GiveSafe（噪声）。

## 5. IDD

`scripts/diagnose_ghtd3_goal_sensitivity.py` + `scripts/eval_idd_decoupling.py`  
G→A 灵敏度、冻高层、冻底层、峰谷条件灵敏度。
