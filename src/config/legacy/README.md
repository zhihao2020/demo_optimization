# Legacy GHTD3 configs (not mainline)

## Mainline (use these)

| File | Role |
|------|------|
| `src/config/ghtd3_config.yaml` | **Default** HMSD-min: 2D goals, plain HER, continuous CAES, no MSGP/F-MLE |
| `src/config/ghtd3_config_seasonal_min.yaml` | Same content; seasonal CLI may pin this path |
| `src/config/ghtd3_config_abs.yaml` | Compatibility alias of mainline |

Train: `scripts/train_seasonal.py` or `scripts/train_ghtd3.py`.

## Archived here

### Teacher / residual / TEA / HSAC (YAML only; runtime code removed)
- `ghtd3_config_tea*.yaml`, `ghtd3_config_hsac*.yaml`, `ghtd3_config_ltar.yaml`, …
- Matching Python paths were deleted from mainline; these YAMLs are historical reference.

### Old 5D + MS-HER + F-MLE pack
- `ghtd3_config_abs_5d_msher.yaml` — previous “Safe Market” pack (goal_dim=5, MS-HER weights, MSGP, F-MLE)
- `ghtd3_config_abs_no{prior,her,fmle}.yaml`, `ghtd3_config_abs_sens_*.yaml`, `ghtd3_config_abs_lambda.yaml`, …

These remain for reference/ablation only. Mainline code **no longer implements MS-HER market weighting**;
`goal_relabel_mode: ms_her` is accepted as an alias of plain `her_mix`.
Higher `goal_dim` still works if set in YAML, but is not the default.

Main `src/training/ghtd3` rejects non-`goal_conditioned` `execution_mode`.
