# Fair weekly comparison protocol

文档更新：2026-08-30 22:40 (+08:00)

论文 live 方法：**PC-HybridTD3**（`scripts/train_seasonal.py --method td3`）。下文 36/8/8 周拆分是设定，不是 HMSD/FS-HSAC 身份。

Code truth: `src/training/episode_starts.py` (`TRAIN_WEEK_IDS` / `VAL_WEEK_IDS` / `TEST_WEEK_IDS`) and `scripts/seasonal_cli.py` (`SEASON_WEEKS`).

## Principle

Algorithm comparison requires the **same optimization problem**:

- same env / FMU / prices / GiveSafe
- same external reward \(r^{\mathrm{ext}}\) (no `storage_use` / \(R^F\))
- train on TRAIN weeks only; pick checkpoints on VAL; **tables on TEST**
- 8760 h rollout = deployment evaluation, not test

## Defaults (formal paper)

| Piece | Choice |
|-------|--------|
| Split | 52 weeks, 9/2/2 per quarter → **36 / 8 / 8** |
| Winter | train 0–8, val 9–10, test 11–12 |
| Transition | train 13–21, val 22–23, test 24–25 |
| Summer | train 26–34, val 35–36, test 37–38 |
| Autumn | train 39–47, val 48–49, test 50–51 |
| `--season all` | the union above |
| GiveSafe | on, fallback off |
| Forecast | 24 h perfect (noisy robustness: 10/10/8 %) |

Legacy 5-train / 1-eval windows are withdrawn.

## Train

```bash
# paper mainline
python scripts/train_seasonal.py --method td3 --season all --episodes 2000 --seed 0

# continuous-projection ablation
python scripts/train_seasonal.py --method td3 --ablation projection --season all --seed 0

# static-support ablation
python scripts/train_seasonal.py --method td3 --ablation static-support --season all --seed 0

# rolling MILP (surrogate opt, FMU eval)
python scripts/train_seasonal.py --method milp --season winter --seed 0

# debug single week (not for paper tables)
python scripts/train_seasonal.py --method td3 --season winter --episodes 20 --single-week
```

Env vars set by the script:

- `OPTIMAL_DEMO_TRAIN_WEEK_STARTS` — training pool
- `OPTIMAL_DEMO_VAL_WEEK_STARTS` — validation
- `OPTIMAL_DEMO_TEST_WEEK_STARTS` — TEST (tables)
- `OPTIMAL_DEMO_EVAL_EPISODE_START` — first TEST week
- `OPTIMAL_DEMO_FORCE_EPISODE_START` — only with `--single-week`
- `OPTIMAL_DEMO_LOCK_CAES` — `1` when `--lock-caes`

## Four methods

| Method | CLI |
|--------|-----|
| Price-aware rule | evaluated alongside each RL run |
| Rolling MILP | `--method milp` |
| Continuous-projection TD3 | `--method td3 --ablation projection` |
| **PC-HybridTD3** | `--method td3` |

Budget: 300k–500k **physical** steps × seeds 0,1,2, same for projection TD3.
