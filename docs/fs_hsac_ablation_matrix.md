# FS-HSAC ablation matrix & phased training protocol

Date: 2026-08-20. Code: `src/training/fs_hsac/`. Entry: `--method fs_hsac`.

## Methods

| Tag | Meaning | Run dir pattern |
|-----|---------|-----------------|
| proj SAC | old projected continuous (`sac_s0`) | `runs/seasonal_v1/<season>/sac_s0` |
| param SAC | fixed-band parameterized Hybrid SAC | `runs/seasonal_v1/<season>/sac_param_s0` |
| FS-HSAC-support | dynamic support + exact hybrid entropy + dual α; `use_feasibility_penalty=False` | `runs/seasonal_v1/<season>/fs_hsac_support_s0` |
| FS-HSAC | full residual feasibility classifier | `runs/seasonal_v1/<season>/fs_hsac_s0` |
| PSO / linprog / milp | classical baselines | existing |

## Phase order (do not skip)

1. **Unit / smoke** — `python logs/_smoke_fs_hsac.py` (optionally `--fmu-steps 100`)
2. **transition seed 0, small budget** — FS-HSAC-support then FS-HSAC  
   ```text
   python scripts/train_seasonal.py --method fs_hsac --season transition --episodes 20 --seed 0 --run-dir runs/seasonal_v1/transition/fs_hsac_support_s0
   ```
   For support-only, pass `use_feasibility_penalty=False` via a thin wrapper or temporary train flag (default train enables penalty; set env `FS_HSAC_NO_FEAS=1` if wired).
3. **three seasons seed 0 full budget** (`episodes=5000` or project default)
4. **≥3 seeds** after seed 0 passes full-week + beats proj SAC / PSO on CC

## Acceptance

- `valid_steps=168` on eval week
- lower CC (or higher Jgen) than projected SAC and PSO on full weeks only
- report rejection rate, idle mute rate, mode hours, dual entropy
- truncated weeks never ranked on cash

## Status

| Step | Status |
|------|--------|
| Unit gates | implemented (`tests/test_fs_hsac_*.py`) |
| Smoke script | `logs/_smoke_fs_hsac.py` |
| transition small budget | **queued / not run in this coding pass** |
| three-season full | pending |
| multi-seed | pending |

Remote `*_param_s0` continues as the fixed-band Hybrid SAC ablation arm; do not overwrite.
