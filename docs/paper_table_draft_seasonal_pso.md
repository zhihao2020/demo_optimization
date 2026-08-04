# Table draft: Seasonal closed-loop FMU comparison

**Setting.** Price-taker TOU (Shandong proxy); weekly horizon $T=168$ h; same FMU and start times.
B0 is the *original model operation* baseline (high thermal, storage idle).
All metrics are measured on the high-fidelity FMU trajectory.

## Table I. Weekly net cash flow, energy SOC, storage and thermal generation

| Season | Method | Net cash flow $J$ (CNY) | $\Delta J$ vs B0 | Episode reward | Energy SOC pass | $E^{\mathrm{th}}$ (MWh) | Bat thr. (MWh) | CAES thr. (MWh) | Curt. (MWh) |
|--------|--------|--------------------------:|------------------:|---------------:|:---------------:|------------------------:|---------------:|----------------:|------------:|
| Winter | B0 Rule (baseline) | 8.333e+06 | — | 67.5 | Y | 25200 | 0 | 275 | 0.00 |
| Winter | B1 Price-aware rule | 6.436e+06 | -1.898e+06 | 52.1 | Y | 24807 | 4395 | 11002 | 0.00 |
| Winter | M1 Rolling LP (relaxed) | 5.596e+06 | -2.737e+06 | 46.7 | Y | 24996 | 4674 | 11061 | 0.00 |
| Winter | M2 PSO (parametric) | 7.371e+05 | -7.596e+06 | 1.8 | N | 811 | 181 | 592 | 0.00 |
| Winter | M3 Hybrid-GiveSafe-TD3 | 1.851e+07 | 1.018e+07 | 128.1 | Y | 8598 | 467 | 516 | 0.00 |
| Winter | M4 Safe Market-GHTD3 (ours) | 1.831e+07 | 9.975e+06 | 126.8 | Y | 8958 | 644 | 600 | 0.00 |
| Summer | B0 Rule (baseline) | -8.415e+04 | — | 13.3 | Y | 25200 | 0 | 1881 | 0.00 |
| Summer | B1 Price-aware rule | -9.036e+05 | -8.195e+05 | 5.5 | Y | 24935 | 2337 | 5465 | 0.00 |
| Summer | M1 Rolling LP (relaxed) | -1.810e+06 | -1.726e+06 | -0.8 | Y | 25050 | 2869 | 8855 | 0.00 |
| Summer | M2 PSO (parametric) | 4.459e+06 | 4.543e+06 | 24.0 | N | 6311 | 1318 | 0 | 0.00 |
| Summer | M3 Hybrid-GiveSafe-TD3 | 1.168e+07 | 1.177e+07 | 83.7 | Y | 9143 | 467 | 1287 | 0.00 |
| Summer | M4 Safe Market-GHTD3 (ours) | 1.118e+07 | 1.126e+07 | 80.3 | Y | 9838 | 644 | 1881 | 0.00 |
| Transition | B0 Rule (baseline) | 6.883e+06 | — | 58.6 | Y | 25200 | 0 | 0 | 0.00 |
| Transition | B1 Price-aware rule | 6.615e+06 | -2.680e+05 | 53.3 | Y | 24980 | 2337 | 8059 | 0.00 |
| Transition | M1 Rolling LP (relaxed) | 6.109e+06 | -7.748e+05 | 49.6 | Y | 25073 | 2869 | 7978 | 0.00 |
| Transition | M2 PSO (parametric) | 8.924e+06 | 2.040e+06 | 69.6 | Y | 21773 | 1762 | 0 | 0.00 |
| Transition | M3 Hybrid-GiveSafe-TD3 | 1.630e+07 | 9.419e+06 | 113.6 | Y | 10255 | 467 | 2363 | 0.00 |
| Transition | M4 Safe Market-GHTD3 (ours) | 1.618e+07 | 9.295e+06 | 113.0 | Y | 10205 | 644 | 2664 | 0.00 |

## Table II. Relative improvement vs B0 (selected)

| Season | Method | $\Delta J$ / $J_{B0}$ | Thermal ratio | Storage throughput ratio |
|--------|--------|------------------------:|--------------:|-------------------------:|
| Winter | B1 Price-aware rule | -22.8% | 0.98 | 55.92 |
| Winter | M1 Rolling LP (relaxed) | -32.8% | 0.99 | 57.15 |
| Winter | M2 PSO (parametric) | -91.2% | 0.03 | 2.81 |
| Winter | M3 Hybrid-GiveSafe-TD3 | 122.1% | 0.34 | 3.57 |
| Winter | M4 Safe Market-GHTD3 (ours) | 119.7% | 0.36 | 4.52 |
| Summer | B1 Price-aware rule | 973.8% | 0.99 | 4.15 |
| Summer | M1 Rolling LP (relaxed) | 2050.6% | 0.99 | 6.23 |
| Summer | M2 PSO (parametric) | -5399.0% | 0.25 | 0.70 |
| Summer | M3 Hybrid-GiveSafe-TD3 | -13982.6% | 0.36 | 0.93 |
| Summer | M4 Safe Market-GHTD3 (ours) | -13384.0% | 0.39 | 1.34 |
| Transition | B1 Price-aware rule | -3.9% | 0.99 | 10396011151599.88 |
| Transition | M1 Rolling LP (relaxed) | -11.3% | 0.99 | 10846084446656.70 |
| Transition | M2 PSO (parametric) | 29.6% | 0.86 | 1761784567236.90 |
| Transition | M3 Hybrid-GiveSafe-TD3 | 136.8% | 0.41 | 2829821818270.40 |
| Transition | M4 Safe Market-GHTD3 (ours) | 135.0% | 0.40 | 3308385014074.30 |

## Table III. Computational effort (order of magnitude)

| Method | FMU steps per season (eval) | Notes |
|--------|----------------------------:|-------|
| B0 / B1 / LP / Hybrid / GHTD3 (eval) | 168 | Single closed-loop week |
| PSO search | $\approx 15\times 10\times 168 = 2.52\times 10^4$ | Plus final re-eval; three seasons $\times 3$ |
| Hybrid/GHTD3 training (offline) | $10^4$–$10^5$ valid steps | Not counted in weekly eval wall time |

### Discussion bullets (for paper)

1. **RL methods dominate** weekly $J$ in all seasons: Hybrid and GHTD3 improve net cash flow by about $+1.0\times 10^7$ CNY/week in winter and transition, and turn summer from near-zero/negative B0 cash flow to $+1.1\times 10^7$.
2. **Thermal generation** falls from 25200 MWh (B0 full load) to ~8600–10300 MWh under RL, indicating reduced fuel burn with storage–market coordination.
3. **PSO** (low-dimensional parametric policy, limited budget) beats B0 in summer/transition cash flow but **lags RL** substantially; winter PSO under-ran thermal and failed energy SOC — consistent with under-parameterized black-box search on hybrid non-convex actions.
4. **Relaxed LP/heuristic** is safer than early broken LP but does not match RL economic performance; full non-convex CAES MILP is left as future work / upper-bound discussion.
5. **Curtailment** is ~0 MWh for all methods in these three weeks under current FMU boundary (report physical metric; economic penalty coefficient may be zero).
6. **GHTD3 ≈ Hybrid** economically; hierarchical method retains interpretability (SoC goals + market prior) with comparable closed-loop KPI.

