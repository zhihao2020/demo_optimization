# Fair seasonal comparison protocol

文档更新：2026-08-17 12:30 (+08:00)

Code truth: `scripts/train_seasonal.py`, `scripts/eval_seasonal_fair.py`,  
`src/config/ghtd3_config_seasonal_min.yaml` (same stack as `ghtd3_config.yaml`).

## Principle

Algorithm comparison requires the **same optimization problem**:

- same env / FMU / prices / GiveSafe
- same external reward \(r^{\mathrm{ext}}\) for the main value update
- train on a **set of weeks**, evaluate on **held-out weeks** in the same season

## Defaults

| Piece | Choice |
|-------|--------|
| Train weeks | winter `0–4`, transition `13–17`, summer `26–30` |
| Eval week (held-out) | winter `5`, transition `18`, summer `31` |
| HMSD low-level reward | **`low_reward: ext`** (= env \(r^{\mathrm{ext}}\), same as TD3) |
| Goal | 2D inventory (conditioning + HER only) |
| HER | `her_mix` |
| GiveSafe | on |

Legacy goal-tracking low reward: set `low_reward: intrinsic` (ablation only).

## Train

```bash
# formal fair protocol (Story A J includes ±200 MW grid contract)
python scripts/train_seasonal.py --method hmsd --season winter --episodes 5000 --seed 0
python scripts/train_seasonal.py --method td3  --season winter --episodes 5000 --seed 0

# lock-CAES counterfactual (same J, u_caes forced idle)
python scripts/train_seasonal.py --method hmsd --season winter --episodes 5000 --seed 0 --lock-caes

# debug single week (not for paper comparison)
python scripts/train_seasonal.py --method hmsd --season winter --episodes 200 --single-week
```

Env vars set by the script:

- `OPTIMAL_DEMO_TRAIN_WEEK_STARTS` — comma-separated start seconds (train pool)
- `OPTIMAL_DEMO_EVAL_EPISODE_START` — held-out eval start
- `OPTIMAL_DEMO_FORCE_EPISODE_START` — only with `--single-week`
- `OPTIMAL_DEMO_LOCK_CAES` — `1` when `--lock-caes`

## Re-eval existing checkpoints

```bash
python scripts/eval_seasonal_fair.py --method hmsd --ckpt path/to/ghtd3.pt --weeks 5,6 --out out.json
python scripts/eval_seasonal_fair.py --method td3  --ckpt path/to/hybrid_givesafe_td3.pt --weeks 5 --out out.json
```

## Primary KPI

1. `sum_delta_j_gen` = sum of `generalized_cashflow_delta`
2. `unserved_energy_mwh`
3. `terminal_soc_satisfied`

Do not use train-week-only scores as the sole comparison table.
