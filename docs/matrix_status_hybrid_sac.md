# seasonal_v1 matrix status (FS-HSAC paper)

Date checked: 2026-08-20 (restart under **official-2024-ets-sd-grid-v1**).

Protocol: fair weekly, seed 0, obs=166, T=168, GiveSafe fallback **off**, `forecast.mode=perfect`.  
**Parameter profile:** `official-2024-ets-sd-grid-v1` (π=97.49, β=0.8049, η_g=0.6191). See `docs/parameter_evidence.md`.

Paper identity: **FS-HSAC v2** (`src/training/fs_hsac/`). See `docs/paper_outline_and_figures.md`, `docs/pamdp_formalization.md`, `docs/fs_hsac_results_gate.md`, `docs/fs_hsac_ablation_matrix.md`.

## FS-HSAC mainline (remote `172.16.1.80`)

| season | FS-HSAC (`fs_hsac_s0`) | FS-HSAC-support |
|--------|------------------------|-----------------|
| winter | restarted (official profile) | restarted |
| transition | restarted (+ support smoke) | restarted |
| summer | restarted | restarted |

Entry: `python scripts/train_seasonal.py --method fs_hsac --season <season> --seed 0`.  
Support-only ablation: `FS_HSAC_NO_FEAS=1`.  
Queue: `logs/fair_queue_fs_hsac.json` / `logs/start_fair_queue_fs_hsac.bat`.  
Reset helper: `logs/_reset_official_profile_remote.py`.

Unit gates: `python logs/_smoke_fs_hsac.py` → **ALL_SMOKE_OK** (18 tests).

## Fixed-band Hybrid SAC ablation

| season | hybrid SAC (`sac_param_s0`) | hybrid TD3 (`td3_param_s0`) |
|--------|-----------------------------|-----------------------------|
| all | **stopped** (legacy carbon book); re-queue only after official profile sync if needed for paper ablation | same |

Do not mix legacy π=80 / η_g=0.5703 / β=0.82 checkpoints into the main table.

## Classical baselines

PSO / linprog / milp: re-evaluate (or re-solve) under the same official reward book; no RL weight retrain.

## Implications

- Do **not** claim FS-HSAC superiority until `docs/fs_hsac_results_gate.md` passes.
- Do **not** mention HMSD.
- Do **not** mix rejection self-loops into Bellman for FS-HSAC (split replay enforced).
- `Paper/main.tex` wait for results gate; require `parameter_profile_id=official-2024-ets-sd-grid-v1` in `train_result.json`.
