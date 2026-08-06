# Modelica → Paper equation map (§3)

Source tree: `D:\Code\0622\docs\TypicalScensrio\`  
Top-level instance: `Example/TypicalScene/PowerSystem_8760h.mo`  
Library: `TypicalScenarios.mo`

Paper uses **academic signs** (gen ≥ 0, charge +, discharge −, grid import +).  
Modelica uses generation-negative; magnitudes match.

| Paper eq | Modelica source | Instance parameters |
|----------|-----------------|---------------------|
| (eq:pv) PV available power | `PV_e`: `P_PV.P_plan + max(Pn*G_in/Gstc*(1-KT*(T_pv-T_stc))*eta,0)=0` | `Pn=260 MW`, `eta=0.95`, `Gstc=1000`, `T_stc=298.15`, `KT=0.005` |
| (eq:pv-temp) module T | `T_pv = T_air + 0.0138*(1+0.031*(T_air-273.15))*(1-0.042*v_in)*G_in` | same |
| (eq:wt) wind curve | `Wind`: piecewise cubic with `vci,vr,vco,Pn` | `Pn=300 MW`, `vci=3`, defaults `vr=10`, `vco=15` |
| (eq:th) thermal | `ThermalPower`: `P_plan = -u_dispatch*P_cap`, `P_act=P_plan` | `P_cap=150 MW`, `P_min=50 MW` |
| (eq:th-ramp) ramp | `rate_max=0.0025/60` (enforced in Python oracle; Modelica ramp commented) | `0.15 pu/h` |
| (eq:caes-p)(eq:caes-legal) | `CAES`: `PBS.P_plan=u_dispatch*P_cap`; min-run bands in Python/`device_params` | `P_cap=150 MW`; `u∈[-1,-0.33]∪{0}∪[0.86,1]` |
| (eq:bat-p)(eq:bat-soc) | `Battery`: charge `der(SOC)=P_act/E_cap`; discharge `der(SOC)=P_act/(E_cap*eta)` | `P_cap=100 MW`, `E_cap=500 MWh`, `eta=0.85`, `SOC∈[0.1,0.9]` |
| (eq:gas-soc)(eq:gas-mass)(eq:gas-energy) | `GasTank`: `SOC=p/p_norm`; mass/energy DAEs | `p_norm=100 bar`, `SOC∈[0.6,1]`, `V≈4.27e5 m³` |
| (eq:thermal-soc)(eq:tank-bal) | `Tank`: `SOC=level/(V0/A)`; mass/enthalpy | `V0=15000 m³`, `A=100 m²`, `SOC∈[0.05,0.95]` |
| (eq:balance) bus | `Bus`: residual → curtail wind then PV; load down to 20%; `P_res` residual | exported as `p_curtailment`, `p_unserved` |
| (eq:grid) | `Grid`: `P1=500 MW`, `P2=-500 MW` | ±500 MW |

## Notes

- Exported FMU for experiments may have removed soft exponential penalties (`C_penality`); hard asserts / Python GiveSafe remain.
- Market TOU settlement replaces constant grid prices in Python (`src/market/settlement.py`).
- Do not invent compressor isentropic efficiency curves beyond mass-flow maps driven by `|u|*P_cap` unless extracted from `Utilities` tables used by the CAES circuit.
