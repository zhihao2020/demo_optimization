# Sensitivity analysis protocol (§5.3)

Genre slot: same position as GHTD3 §5.3. One season (default: **transition**), seed 0, `T=168`, GiveSafe fallback off. Primary policy: `fs_hsac` checkpoint after results gate; baselines: linprog + milp on the same week. Shared forecast mode = `perfect`.

**Parameter profile:** re-evaluate under `official-2024-ets-sd-grid-v1` (`docs/parameter_evidence.md`). Do not mix legacy `80/0.5703/0.82` snapshots into the main sensitivity table without an explicit profile tag.

Primary ranking remains full-week \(CC=-J^{\mathrm{gen}}\). Physical KPIs sit beside \(CC\).

---

## 1. Carbon price sweep (official band + lit cross-check)

| `carbon.price_cny_per_t` | Values |
|--------------------------|--------|
| absolute | `{69.30, 80.0, 86.4, 97.49, 105.65}` |

- 97.49 = 2024 YE official close (main).  
- 69.30 / 105.65 = 2024 band / peak.  
- 80 = legacy OP; 86.4 ≈ GHTD3 \(0.012\) USD/kg × 7.2.

Report: CC, C_CO2, cash, thermal MWh, CAES mode hours, curtailment MWh.

Optional: `beta ∈ {0.8049, 0.8155, 0.85}` and `eta_thermal ∈ {0.8049, 0.85}` to separate benchmark vs emission factor.

---

## 2. Feasibility margin scale

Scale `caes.*.margin_*` / `residual_p99_*` by \(\alpha\in\{0.0,0.5,1.0,1.5\}\).

Report: hours/168, GiveSafe reject rate, CC.

---

## 3. CAES power capacity

Scale `device_params.caes.P_cap_W` by `{0.75, 1.0, 1.25}`.

---

## 4. Curtailment / unserved prices

| Axis | Values |
|------|--------|
| `nu_curt` multiplier | `{0.0, 0.5, 1.0, 2.0, 5.0}` (nominal 300) |
| `nu_uns` | `{1000, 4000, 10000}` (VOLL-style)

Cross-check: 300 ≈ GHTD3 295; 350 from CAES curtailment papers as optional mid point.

---

## 5. Grid-contract price / band

| Axis | Values |
|------|--------|
| `nu_cny_per_mwh` multiplier | `{0.0, 0.5, 1.0, 2.0}` (nominal 600, **scenario**) |
| `p_lim_w` | `{1.5e8, 2.0e8, 3.0e8}` |

---

## 6. CAES startup scale mode

| `caes_startup.scale_mode` | Meaning |
|---------------------------|---------|
| `none` | keep 3.42 USD×FX at 800 kW (no capacity scale) |
| `linear_capacity` | default extrapolation to 150 MW |
| `sqrt_capacity` | milder scale |

---

## 7. TOU base / sell (constructive path)

| Axis | Values |
|------|--------|
| float base \(B\) multiplier | `{0.8, 1.0, 1.2}` |
| sell CNY/kWh | `{0.10, 0.1875, 0.30}` |

Do not claim these as official settlement schedules.

---

## 8. Table / figure

- `tab:sensitivity` — (axis, level, method, **profile_id**) → CC + axis KPI + hours/168  
- `fig_sensitivity` — carbon / margin / capacity / curt / contract / startup-scale / TOU-base

---

## 9. Pareto declaration rules

- Lowest CC → “economically best” only.  
- Lower curtailment / fewer contract violations → that axis only.  
- “Overall better” only under weak Pareto on (CC, curtailment, reliability).  
- Never mix truncated weeks into CC rankings.  
- Never attribute MILP↔RL gaps to forecast advantage on the perfect-forecast matrix.

---

## 10. Do not

- Sweep forecast mode in the main sensitivity  
- Claim robustness without full-week runs  
- Mix margin=0 abort weeks into cost rankings  
- Rewrite in-flight remote reward coefficients; stamp `parameter_profile_id` and re-evaluate
