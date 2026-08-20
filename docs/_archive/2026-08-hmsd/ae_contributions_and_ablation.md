<!-- ARCHIVED: superseded by FS-HSAC; see docs/paper_outline_and_figures.md -->

# Applied Energy — Contributions, Ablations, and Writing Spine

Last update: 2026-08-12

Use this file as the **single source of truth** for Introduction / Highlights / ablation design.  
Mainline config: `src/config/ghtd3_config.yaml` (`low_reward: ext`, `market_goal_prior: false`).

---

## 1. Working title (options)

1. **Preferred:** Hierarchical goal-conditioned reinforcement learning for multi-energy economic dispatch under hard safety and nonconvex CAES operating sets  
2. Safe hierarchical dispatch of wind–PV–thermal–battery–CAES systems with continuous CAES parameterization and reject-aware learning  
3. Fair seasonal evaluation of hierarchical vs flat deep RL for secure multi-energy scheduling

Avoid “Market-aware” in the title unless market goal prior is **on** in main results.

---

## 2. Highlights (3–5 bullets, AE style)

1. Formulates hour-ahead multi-energy economic dispatch under a **nonconvex CAES command set** and **hard safety filtering**, with a generalized cash objective (cash, carbon, curtailment/unserved, battery degradation).  
2. Proposes **HMSD**: inventory goal-conditioned hierarchical TD3 with continuous CAES representation, GiveSafe execution, and **reject-aware** low-level learning.  
3. Ensures **fair comparison** with flat TD3 by using the **same external economic reward** at the control layer and a multi-week train / held-out week eval protocol per season.  
4. Across winter–transition–summer held-out weeks (multi-seed), HMSD improves **mean economics and stability** versus flat TD3; ablations isolate hierarchy, hindsight goal relabeling, and reject learning.  
5. Analyzes safety–learning coupling (rejection rates) and residual failure modes (e.g., high-variance flat TD3 seeds; max-entropy baseline divergence).

---

## 3. Three contributions (English, Introduction-ready)

**Contribution 1 (Problem).**  
We cast multi-energy economic dispatch of a wind–PV–thermal–battery–CAES plant as a constrained sequential decision problem in which (i) CAES commands lie in a **disconnected legal set**, (ii) unsafe actions are **hard-filtered** before plant simulation, and (iii) the learning objective is a **generalized economic reward** aligned with operation KPIs. We further specify a **seasonal fair-evaluation protocol**: multi-week training pools and held-out evaluation weeks under identical environment, prices, and safety stack for all algorithms.

**Contribution 2 (Method).**  
We develop **HMSD (Hierarchical Market-safe Dispatch / Hierarchical multi-energy safe dispatch)**†: a goal-conditioned hierarchy in which a high level issues 2-D inventory increments every \(c\) hours and a low level outputs device commands \((u_{\mathrm{tp}},u_{\mathrm{bat}},u_{\mathrm{caes}})\). Continuous CAES parameterization respects the nonconvex legal bands; GiveSafe blocks illegal/unsafe proposals; **reject transitions** are stored as self-loops with shaped constraint costs so the low-level policy learns to avoid repeated rejections; hindsight goal relabeling improves sample use for inventory goals. For the fair study, the low level optimizes the **same external reward** as flat TD3.

**Contribution 3 (Evidence).**  
On a high-fidelity co-simulation plant, multi-seed held-out weekly evaluations in winter, transition, and summer show that HMSD attains **higher and more stable** generalized cash performance than flat hybrid TD3 in winter and transition seasons, while remaining competitive and more consistent in summer where flat TD3 exhibits large seed variance. Ablations (no hierarchy / no HER / filter-only safety without reject learning) quantify each mechanism. We report safety statistics (rejection rates) alongside economic KPIs.

† Naming note for authors: if “Market” is not a main-result module, use **Hierarchical Multi-energy Safe Dispatch** in the paper body and keep “HMSD” as the acronym.

---

## 4. What we explicitly do **not** claim (avoid reviewer traps)

| Do not claim | Why |
|--------------|-----|
| First hierarchical RL for energy | Literature (e.g., GHTD3) exists |
| Market prior is essential in main results | Mainline `market_goal_prior: false` |
| Dominates all seeds of all baselines | Summer TD3 can win individual seeds |
| SAC is a fully fair complete baseline | Multiple seeds diverged (re-runs ongoing with temperature clamp) |
| Rolling LP is a global optimality upper bound | Surrogate + receding horizon only |

---

## 5. Ablation table design (camera-ready)

### 5.1 Design matrix

| ID | Variant | Hierarchy | HER | Reject learning | GiveSafe filter | Config / command |
|----|---------|-----------|-----|-----------------|-----------------|------------------|
| A0 | **HMSD (full)** | Yes | Yes | Yes | Yes | `ghtd3_config.yaml` |
| A1 | No HER | Yes | **No** | Yes | Yes | `ablation/ghtd3_no_her.yaml` |
| A2 | Filter-only safety | Yes | Yes | **No** | Yes | `ablation/ghtd3_no_reject_learn.yaml` |
| A3 | Flat TD3 | **No** | — | (buffer reject OK) | Yes | `--method td3` |

Optional later (not blocking first submission):

| ID | Variant | Note |
|----|---------|------|
| A4 | Intrinsic low reward | `low_reward: intrinsic` — different objective; only as sensitivity |
| A5 | No GiveSafe | Risky; simulation-only, ethics/safety narrative |

### 5.2 Protocol (must match main fair study)

- Season for formal ablation: **winter** (primary); optional transition replicate.  
- Episodes: **5000** (\(\approx 840\mathrm{k}\) steps), seed **0** minimum; seeds **0–2** if budget allows.  
- Eval: held-out week of that season (winter week index 5).  
- Report both **economic KPI** and **safety KPI**.

### 5.3 Metrics for ablation Table X

| Column | Definition |
|--------|------------|
| \(R\) | Held-out weekly episode reward |
| \(J_{\mathrm{gen}}\) | Generalized cash increment over the week (CNY) |
| SOC-ok | Terminal battery+gas inventory within tolerance |
| Reject rate | `#rejected / (#physical + #rejected)` over training (from `safety_learning`) |
| Valid steps | Should be \(\approx 840000\) if completed |

### 5.4 Expected pattern (hypothesis for writing)

| Comparison | Expected if story holds |
|------------|-------------------------|
| A0 vs A3 | A0 higher \(J_{\mathrm{gen}}\) / \(R\), lower variance across seeds |
| A0 vs A1 | A0 better sample efficiency / final \(R\) or \(J_{\mathrm{gen}}\) |
| A0 vs A2 | A0 lower late-training reject rate and/or better economics |

If A2 \(\approx\) A0 economically but reject rate stays high, still claim **learning efficiency / safety coupling**, not only cash.

### 5.5 LaTeX-ready stub

```latex
\begin{table}[t]
\centering
\caption{Ablation on winter held-out week (seed 0 unless noted).}
\label{tab:ablation}
\begin{tabular}{lcccc}
\hline
Variant & $R$ & $J_{\mathrm{gen}}$ (CNY) & SOC-ok & Reject rate \\
\hline
HMSD (full) &  &  &  &  \\
w/o HER &  &  &  &  \\
Filter-only (no reject learning) &  &  &  &  \\
Flat TD3 &  &  &  &  \\
\hline
\end{tabular}
\end{table}
```

---

## 6. Main results table design (three seasons)

### 6.1 Per-season multi-seed (Table 1 skeleton)

For each season \(\in\{\mathrm{winter},\mathrm{transition},\mathrm{summer}\}\):

| Method | \(R\) mean±std | \(J_{\mathrm{gen}}\) mean±std | SOC-ok rate | Notes |
|--------|----------------|--------------------------------|-------------|-------|
| HMSD | | | | 3 seeds complete |
| Flat TD3 | | | | 3 seeds complete |
| Hybrid SAC | | | | partial; mark incomplete/failed seeds |
| PSO | | | | 1 seed |
| Rolling LP | | | | 1 seed |

### 6.2 Provisional numbers from completed remote fair suite (fill after `aggregate_fair_results.py`)

**Winter (already measured, illustrative):**

| Method | \(R\) (seeds) | Comment |
|--------|---------------|---------|
| HMSD | 113.6 / 99.7 / 100.9 | Strong, stable |
| TD3 | −1.3 / 44.7 / −1.3 | High variance |
| SAC | 102.9 / 45.9 / 44.7 | One strong seed |
| LP | 21.9 | Weak |
| PSO | 7.9 | Weak on \(R\) |

**Transition:** HMSD ~41–50; TD3 mostly near 0–16; LP negative.  
**Summer:** HMSD ~61–83 all SOC-ok; TD3 98 / −8.8 / 98 — report mean **and** worst seed.

---

## 7. Introduction paragraph (English draft)

Multi-energy systems that integrate wind, photovoltaics, thermal generation, batteries, and compressed-air energy storage (CAES) must co-optimize hourly energy arbitrage with inventory continuity under plant-level safety limits. Unlike box-constrained continuous control, CAES commands are restricted to a **nonconvex** legal set induced by minimum operating bands, while feasibility oracles may **reject** large fractions of policy proposals before physics advances. Deep reinforcement learning (DRL) methods such as TD3 are attractive for nonlinear plants, yet flat policies struggle with sparse effective samples under hard filtering and with week-long credit assignment for storage. Hierarchical goal-conditioned RL can structure inventory coordination, but existing energy applications rarely enforce **identical external economic objectives** between hierarchical and flat agents, and seldom quantify how reject handling interacts with learning.

This paper studies **secure multi-energy economic dispatch** under hard safety and nonconvex CAES sets. We propose HMSD, a hierarchical goal-conditioned TD3 agent with continuous CAES parameterization, GiveSafe execution, and reject-aware replay, trained against the **same** external reward as flat TD3. Using a seasonal multi-week training and held-out evaluation protocol, we show improved mean economics and stability relative to flat TD3, and ablate hierarchy, hindsight relabeling, and reject learning.

---

## 8. Method claims checklist (align text ↔ code)

| Claim in paper | Code truth |
|----------------|------------|
| Low-level optimizes external economic reward | `low_reward: ext` |
| 2-D goals \(\Delta\mathrm{bat},\Delta\mathrm{gas}\) | `goal_dim: 2` |
| \(c=8\) | `subgoal_interval: 8` |
| HER mix | `goal_relabel_mode: her_mix` |
| Continuous CAES | `continuous_caes: true` / `caes_u` |
| Reject learning on | `learn_from_reject: true` |
| Market prior off in mainline | `market_goal_prior: false` |
| Safety metrics in results | `result["safety_learning"]` after code sync |

---

## 9. Ablation run commands (server)

```bash
# Already queued via scripts/remote_continue_ae_experiments.py:
#   ablation_full_winter_s0
#   ablation_no_her_winter_s0
#   ablation_no_reject_winter_s0
#   ablation_flat_td3_winter_s0
# plus SAC re-runs with alpha clamp

# After finish, aggregate:
python scripts/aggregate_fair_results.py --root runs/seasonal_v1 --out docs/ae_results_table.md
# For ablations, point root at runs/ablation or extend aggregator.
```

---

## 10. Suggested paper section map

| Section | Content source |
|---------|----------------|
| Abstract | Highlights §2 condensed |
| Introduction | §7 + Contributions §3 |
| Problem | `docs/ae_problem_depth.md` + reward formulas in `当前训练方案报告.md` |
| Method | HMSD stack; Fig. hierarchy + GiveSafe + CAES map |
| Experiments | Protocol + Table main + Table ablation §5 |
| Discussion | Summer TD3 seeds; SAC divergence; reject rates |
| Conclusion | C1–C3 restated + limits |

---

## 11. One-page “depth” self-check before submission

- [ ] One sharp problem sentence (nonconvex + hard safety + same \(r^{\mathrm{ext}}\))  
- [ ] Three contributions match mainline flags  
- [ ] Ablation table filled (A0–A3)  
- [ ] Three-season main table with mean±std  
- [ ] Worst-seed / failure discussion  
- [ ] At least one behavior or safety figure  
- [ ] No oversell of market prior / global optimality of LP  
