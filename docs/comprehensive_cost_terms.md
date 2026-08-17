# Comprehensive monetary terms (Scheme B)

文档更新：2026-08-17 12:30 (+08:00)

**Authoritative layer: Python** (`RewardCalculator`).  
FMU supplies physics and device *energy-bookkeeping* cash; TOU grid settlement, carbon, curtailment, battery cycle, CAES startup, and the Story A interchange contract are applied here.

\[
\Delta J^{\mathrm{gen}}
=
\Delta J^{\mathrm{cash}}
- \pi_{\mathrm{CO_2}}\Delta m
- C^{CUT}
- C^{\mathrm{deg}}
- C^{\mathrm{su,caes}}
- C^{\mathrm{grid}}
\]

| Term | Source physics | Price / formula | Default |
|------|----------------|-----------------|--------|
| \(\Delta J^{cash}\) | FMU `economic_cashflow_*` Δ + optional TOU grid replace | CNY | — |
| Carbon \(\pi\Delta m\) | `p_thermal`, `p_grid` | \(\eta_{\mathrm{th}}E_{\mathrm{th}}+\eta_g E_{\mathrm{buy}}\), \(\pi=80\) CNY/t | on |
| \(C^{CUT}\) | `p_curtailment`, `p_unserved` | \(\nu_c E_{\mathrm{curt}}+\nu_u E_{\mathrm{uns}}\) | 300 / 1000 CNY/MWh |
| \(C^{\mathrm{deg}}\) | discharge from `p_battery` | Cui-style \(\psi(\delta)=a_0\delta^{2.03}\), \(a_0=\mathrm{Capex}/E_{\mathrm{life}}^{2.03}\), mid-life offset \(\rho=0.25\) | convex cumulative |
| \(C^{\mathrm{su,caes}}\) | mode from `p_caes` | Cui 2024 Table 2: \(3.42\) USD/switch at 800 kW, \(\times P/P_{\mathrm{ref}}\times\) 7.2 CNY/USD | \(\approx 4617\) CNY / event |
| \(C^{\mathrm{grid}}\) | `|p_grid|` | \(\nu\max(0,\|P\|-P_{\lim})\Delta t\), \(P_{\lim}=200\) MW | 600 CNY/MWh |

FMU hard interchange remains ±500 MW (compiled, not settable). \(C^{\mathrm{grid}}\) is the Story A lever: 150 MW CAES blocks sit inside the 200–500 MW band that used to be free.

## Battery degradation calibration (convex)

\[
E_{\mathrm{life}}=N\cdot\mathrm{DoD}\cdot E_{\mathrm{cap}},\quad
\mathrm{Capex}=c_{\mathrm{kWh}}\cdot E_{\mathrm{cap}}\cdot 1000\ \mathrm{CNY},
\]
\[
a_0=\mathrm{Capex}/E_{\mathrm{life}}^{p},\quad p=2.03,\quad
C_t=\psi(\delta_0+\delta_t)-\psi(\delta_0+\delta_{t-1}).
\]
- \(\delta\): episode cumulative **discharge** MWh only.  
- \(\delta_0=\rho E_{\mathrm{life}}\): mid-life offset (default \(\rho=0.25\)) so a fresh weekly episode is not stuck where marginal cost ≈ 0.  
- Full-life identity: \(\psi(E_{\mathrm{life}})=\mathrm{Capex}\).  
- Legacy `mode: linear_throughput` still available.

## Modelica Battery (resources)

`TypicalScenarios.Battery` cash is only `±P * c_buy/sale`.  
**No** cycle-life cash. Python \(C^{\mathrm{deg}}\) does **not** double-count cycle fees.

## Disable

```yaml
curtailment: {enabled: false}
battery_degradation: {enabled: false}
carbon: {enabled: false}
```
