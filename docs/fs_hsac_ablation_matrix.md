# FS-HSAC ablation matrix & phased training protocol

Date: 2026-08-21. Code: `src/training/fs_hsac/`.  
Paper mainline: `--method fs_hsac --support` (or `FS_HSAC_NO_FEAS=1`) vs `--method sac`.

Do **not** change the density math in `actor.py` / `algorithm.py` / `action_support.py`.

## Methods

| Tag | Role | Run dir pattern |
|-----|------|-----------------|
| FS-HSAC-support | **live** same-hour support-consistent Hybrid SAC; `use_feasibility_penalty=False` | `runs/seasonal_v1/<season>/fs_hsac_support_s0` |
| param SAC | **live contrast** fixed-band Hybrid SAC (latent density then clamp) | `runs/seasonal_v1/<season>/sac_param_s0` |
| FS-HSAC | appendix residual $C_\psi$ | `runs/seasonal_v1/<season>/fs_hsac_s0` |
| proj SAC | optional archive | `runs/seasonal_v1/<season>/sac_s0` |
| PSO / linprog / milp | setting diagnostics only; no superiority claim | existing |

## Phase order

1. **Unit / smoke** — `python logs/_smoke_fs_hsac.py` (optionally `--fmu-steps 100`)
2. **transition seed 0, small budget** — support-only first
   ```text
   python scripts/train_seasonal.py --method fs_hsac --support --season transition --episodes 20 --seed 0 --run-dir runs/seasonal_v1/transition/fs_hsac_support_s0
   python scripts/train_seasonal.py --method sac --season transition --episodes 20 --seed 0 --run-dir runs/seasonal_v1/transition/sac_param_s0
   ```
3. **three seasons seed 0 full budget** (`episodes=5000`) for the live pair
4. Appendix full FS-HSAC and extra seeds only after the live pair exists

## Acceptance (bookkeeping; not a manuscript superiority claim)

- report reject rate, `valid_steps=168`, comprehensive cost
- truncated weeks never ranked on cash
- `docs/fs_hsac_results_gate.md` stays `gate_passed: false` until the live pair is archived

## Status

| Step | Status |
|------|--------|
| Unit gates | implemented (`tests/test_fs_hsac_*.py`) |
| CLI `--support` / `--no-feas` | `scripts/train_seasonal.py` |
| three-season full live pair | pending |
| results gate | **false** |
