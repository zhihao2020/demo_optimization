# Paper ablations (HMSD)

| Config | Proposition | Diff vs mainline |
|--------|-------------|------------------|
| `../ghtd3_config.yaml` | Full HMSD | baseline |
| `ghtd3_no_her.yaml` | P3 HER useful | `goal_relabel: false` |
| `ghtd3_no_reject_learn.yaml` | P2 reject-learning useful | `learn_from_reject: false` (filter still on) |
| Flat TD3 | P1 hierarchy useful | `scripts/train_seasonal.py --method td3` |
| `ghtd3_cui_style.yaml` | GHTD3-style on this plant | `low_reward: intrinsic`; HER on; reject-learn off |
| `ghtd3_wear.yaml` | Wear quota HMSD-W | `goal_dim: 3`, `wear_budget` + `wear_enforce` |
| `ghtd3_budget.yaml` | HMSD-B dual quota | `goal_dim: 4`, wear + thermal, both enforced |
| `ghtd3_aligned.yaml` | Geometry-aligned HMSD | `hybrid_caes` + wear + caes-on quotas |

## Formal (one season, one seed)

```bash
python scripts/train_seasonal.py --method hmsd --season winter --episodes 5000 --seed 0 \
  --config src/config/ghtd3_config.yaml --run-dir runs/ablation/full_winter_s0

python scripts/train_seasonal.py --method hmsd --season winter --episodes 5000 --seed 0 \
  --config src/config/ablation/ghtd3_no_her.yaml --run-dir runs/ablation/no_her_winter_s0

python scripts/train_seasonal.py --method hmsd --season winter --episodes 5000 --seed 0 \
  --config src/config/ablation/ghtd3_no_reject_learn.yaml --run-dir runs/ablation/no_rej_winter_s0

python scripts/train_seasonal.py --method td3 --season winter --episodes 5000 --seed 0 \
  --run-dir runs/ablation/flat_td3_winter_s0
```

Smoke (200 episodes) before formal:

```bash
python scripts/run_paper_ablations.py --season winter --episodes 200 --seed 0
```
