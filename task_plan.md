# Task Plan: Phase D.5 Feasibility Hardening

## Goal
Audit post-step failures, rebuild per-device FeasibilityOracle with residual-aware margins, add SafetyClassifier in action generation, BoundaryStressTester, SafetyDataset, C_ref re-exam, and keep Phase E gated.

## Phases
- [x] D.5.1 Audit failures + FailureRecord + analysis doc
- [x] D.5.2 Oracle residual stats
- [x] D.5.3 Rebuild per-device FeasibilityOracle + margins config
- [x] D.5.4 SafetyClassifier / FeasibilityCalibrator
- [x] D.5.5 Safety in action generation (Actor→Mask→FeasibleSet→Classifier)
- [x] D.5.6 BoundaryStressTester
- [x] D.5.7 EconomicReplayBuffer vs SafetyDataset
- [x] D.5.8 C_ref seasonal recalibration script
- [x] D.5.9 Logging + Phase E gates + tests + report

## Status
**Complete for D.5 implementation** — Phase E remains blocked (`formal_default_blocked=true`).

## Key Findings
- smoke 3.8% / short 5.27% post-step fails (legacy logs coarse)
- probe 0.125%: fine type `caes_pressure_low` @ idle
- Residual P99 used: bat charge 0.013, discharge 0.018, caes gas 0.073, pressure 9.8e4
- SafetyClassifier false-safe=0.0 on 1 fail (insufficient for gate)
- BoundaryStress 500 pass; need ≥20000 for Phase E
- C_ref 96199 → 156540 seasonal P95 (material, updated)

## Hard principles (enforced)
1. Forbidden area = hard constraint, NOT reward penalty
2. Post-step violations do NOT enter economic replay
3. FMU numerical failure is NOT system cost
4. No silent clip/project/replace of actions in env
5. Never replace illegal CAES mode with idle
6. reward_t = -C_sys/C_ref + terminal_soc_bonus
7. Episode = 168 hourly steps
8. Phase E stays OFF until new gates pass
