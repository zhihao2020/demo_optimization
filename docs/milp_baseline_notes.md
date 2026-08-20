# MILP baseline notes (paper §5.1)

Implementation: `src/optimization/rolling_milp.py` (`RollingMilpController`).  
Entry: `python scripts/train_seasonal.py --method milp --season <season> --seed 0`.

## Same as linprog

- Rolling horizon (default 24 h), execute first step, close loop on FMU twin.
- Objective: fuel + carbon + buy/sell + curtailment/unserved + battery discharge wear + terminal SoC soft penalty.
- Battery and **gas** SoC: linear energy balance.
- Thermal ramp from `device_params.thermal.rate_max_per_s`.
- Boundary / prices from the same `forecast_provider` as the RL observation stack (`mode=perfect` in the main matrix; GHTD3-style perfect foresight for classical baselines).

## What MILP adds vs linprog

- Binary \(z^{\mathrm{chg}}_t, z^{\mathrm{dis}}_t\) with mutual exclusion.
- Min-load bands: charge power in \([0.86,1]P_{\mathrm{cap}}\) when \(z^{\mathrm{chg}}=1\); discharge in \([0.33,1]P_{\mathrm{cap}}\) when \(z^{\mathrm{dis}}=1\).
- Minimum-run / mode-lock style inequalities inside the horizon (`MIN_CAES_RUN_STEPS=4`), warm-started from the previous closed-loop mode.

## Approximations that must be stated in the paper

1. No hot-tank / cold-tank / pressure–temperature DAE (energy gas SoC only).
2. No off-design CAES efficiency map (constant power ↔ energy coefficients).
3. Perfect foresight over the MPC horizon (not stochastic programming).
4. Safety / twin rejection after decode is outside the MILP; GiveSafe may still rewrite or abort in evaluation protocols that keep fallback off.

If MILP reports a lower CC than hybrid SAC on the twin-closed-loop KPI, discuss (1)–(4) and any non-executable first-step commands — same rhetorical path as GHTD3 vs Gurobi QP.
