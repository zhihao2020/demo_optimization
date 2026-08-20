<!-- ARCHIVED: superseded by FS-HSAC; see docs/paper_outline_and_figures.md -->

# Results gate for paper writing

**Rule:** Do not rewrite `Paper/main.tex` until this gate passes. Outline and code may advance.

Date checked: 2026-08-20.

## Pass criteria

1. `runs/seasonal_v1/{winter,transition,summer}/sac_param_s0` each have `valid_steps=168` and `status != eval_failed`.
2. Hybrid SAC comprehensive cost **CC = −J^gen** is strictly better (lower CC / higher Jgen) than same-season **proj. SAC**, **proj. TD3**, and **PSO** on full weeks only.
3. Truncated weeks are never ranked against full weeks on cash.

## Fail / contingency

| Situation | Paper handling |
|-----------|----------------|
| Hybrid SAC still aborts | Fix training / clamp; do not claim method superiority |
| Hybrid SAC beats projection + PSO but MILP has lower CC | Keep hybrid SAC as proposed method; in §5.2.3 explain MILP energy linearization, omitted hot/cold–pressure DAE, and any twin non-executability / safety rewrites (GHTD3 path: solver cheaper but not the executable schedule) |
| Hybrid SAC loses to PSO on CC | Do not submit; retune or narrow claim to action-representation ablation only |

## Current snapshot

See `docs/matrix_status_hybrid_sac.md` and `docs/tab_main_seed0.md`.

As of gate creation:

- Legacy `sac_s0` / `td3_s0`: projection aborts / incomplete weeks (ablation only).
- `sac_param_s0` / `td3_param_s0`: remote queue; **gate not yet passed**.
- `linprog`: three seasons full week (energy surrogate + continuous CAES relaxation).
- `milp`: code landed (`src/optimization/rolling_milp.py`); seasonal runs not yet in matrix.

## Commands

```text
python logs/_poll_remote_param.py
python scripts/pull_seasonal_v1.py --methods sac_param,td3_param,pso,linprog,milp
python scripts/build_tab_main_seed0.py
```

Re-check this file after pull; flip `gate_passed` below when criteria hold.

```yaml
gate_passed: false
blocker: "sac_param_s0 not yet completed on all three seasons"
```
