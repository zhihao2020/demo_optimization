# Comprehensive monetary terms (Scheme B)

文档更新：2026-08-20 (+08:00)  
**Profile:** `official-2024-ets-sd-grid-v1` — 详见 [`docs/parameter_evidence.md`](parameter_evidence.md).

**Authoritative layer: Python** (`RewardCalculator` + `src/config/reward_config.yaml`).  
FMU supplies physics and device *energy-bookkeeping* cash; TOU grid settlement, carbon, curtailment, battery cycle, CAES startup, and the Story A interchange contract are applied here.

All learning agents and classical baselines (FS-HSAC, PSO, rolling linprog/MILP, ablations) are ranked on the **same** weekly comprehensive objective when evaluated on the FMU. Do **not** invent method-specific bonus scores for paper tables.

\[
\Delta J^{\mathrm{gen}}
=
\Delta J^{\mathrm{cash}}
- C^{\mathrm{CO_2}}
- C^{CUT}
- C^{\mathrm{deg}}
- C^{\mathrm{su,caes}}
- C^{\mathrm{grid}},
\qquad
CC = -J^{\mathrm{gen}}.
\]

Primary ranking key: full-week \(CC\) (equivalently \(J^{\mathrm{gen}}\)) with `valid_steps=168`.

| Term | Source physics | Price / formula | Default (claim level) |
|------|----------------|-----------------|------------------------|
| \(\Delta J^{cash}\) | FMU Δ + optional TOU replace | CNY | constructive Shandong TOU path |
| \(C^{\mathrm{CO_2}}\) | `p_thermal`, `p_grid` | **`intensity_benchmark`:** \(A=\beta E_{\mathrm{th}}\), \(E=\eta_{\mathrm{th}}E_{\mathrm{th}}\); settle \(-\pi Q\); grid step \(\pi\eta_g E_{\mathrm{buy}}\). \(\pi=\mathbf{97.49}\) (O), \(\beta=\mathbf{0.8049}\) (O), \(\eta_g=\mathbf{0.6191}\) (O), \(\eta_{\mathrm{th}}=0.85\) (S) | on |
| \(C^{CUT}\) | curtail / unserved | \(\nu_c E_{\mathrm{curt}}+\nu_u E_{\mathrm{uns}}\) | 300 (L≈GHTD3) / 1000 (S) |
| \(C^{\mathrm{deg}}\) | battery discharge | \(\psi(\delta)=a_0\delta^{2.03}\) | L / LCOS |
| \(C^{\mathrm{su,caes}}\) | CAES mode switch | Cui \(3.42\) USD@800 kW × FX × **linear \(P\) extrapolation** | ≈4617 CNY (S) |
| \(C^{\mathrm{grid}}\) | `|p_grid|` | \(\nu\max(0,\|P\|-P_{\lim})\Delta t\), \(P_{\lim}=200\) MW | 600 (S) |

FMU hard interchange remains ±500 MW. Code identity:  
`generalized_cashflow_delta ≈ cash − carbon − cut_total − deg − caes_startup − grid_contract`.

Legacy `runs/**` snapshots with π=80 / η_g=0.5703 / β=0.82 → tag `legacy-2022-grid-factor/proxy-benchmark`; do not rewrite.

## Battery degradation calibration (convex)

\[
E_{\mathrm{life}}=N\cdot\mathrm{DoD}\cdot E_{\mathrm{cap}},\quad
\mathrm{Capex}=c_{\mathrm{kWh}}\cdot E_{\mathrm{cap}}\cdot 1000\ \mathrm{CNY},
\]
\[
a_0=\mathrm{Capex}/E_{\mathrm{life}}^{p},\quad p=2.03,\quad
C_t=\psi(\delta_0+\delta_t)-\psi(\delta_0+\delta_{t-1}).
\]
Weekly default \(\rho=0\). Legacy `linear_throughput` still available.

## Disable

```yaml
curtailment: {enabled: false}
battery_degradation: {enabled: false}
carbon: {enabled: false}
caes_startup: {enabled: false}
grid_contract: {enabled: false}
```
