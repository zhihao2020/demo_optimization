# 论文结果清单（PC-HybridTD3）

文档更新：2026-08-30 22:40 (+08:00)

**提纲：** [`docs/paper_outline_and_figures.md`](paper_outline_and_figures.md)

**书账：** [`docs/fs_hsac_results_gate.md`](fs_hsac_results_gate.md) — Stage D 前不在 `Paper/main.tex` 填表、不写优越性。归档 `fs_hsac_*` / `seasonal_v1` 现金禁止混入正文。

**主文：** `Paper/main.tex`

**主表工作稿：** [`docs/tab_main_seed0.md`](tab_main_seed0.md)

**参数口径：** [`docs/parameter_evidence.md`](parameter_evidence.md)  
- 新主表 / 新评测：`parameter_profile_id = official-2024-ets-sd-grid-v1`（π=97.49，β=0.8049，η_g=0.6191）  
- 已有 `runs/**/config/reward_config.yaml` 快照若仍为 80 / 0.5703 / 0.82 → 标签 **`legacy-2022-grid-factor/proxy-benchmark`**，**勿改写**；主表只收录新口径重评（必要时重训）结果。

## 数字源（过门后）

| 角色 | 路径 |
|------|------|
| 主方法（live） | `runs/seasonal_tou2026/{winter,transition,summer,autumn,all}/pc_hybrid_td3_s*`（Stage D 后） |
| 对照 | 同树 `td3_proj_s*` / `td3_static_s*` / `milp_s0` / rule eval |
| 附录完整 FS-HSAC | `runs/seasonal_v1/.../fs_hsac_s0` |
| 投影消融 | `runs/seasonal_v1/.../sac_s0`（及 proj. TD3） |
| 经典对照 | `pso_s0` / `linprog_s0` / `milp_s0` |

旧 HMSD 结果表与 `aggregate_fair_results.py → ae_results_table.md` 叙事已归档（见 `_archive/2026-08-hmsd/`），**不得再写入正文主表**。

## 拉取 / 汇总（过门后）

```bash
python scripts/pull_seasonal_v1.py
python scripts/build_tab_main_seed0.py
python scripts/plot_paper_plan_figures.py
```

## 文件约定

| 路径 | 用途 |
|------|------|
| `train_result.json` / `summary.json` | 评测 KPI + `parameter_profile_id` + 碳价/β/η_g 来源字段；`valid_steps=168` 才进经济表 |
| `checkpoints/*.pt` | `algorithm_version=fs_hsac_v2` 权重 |
| `trajectories/eval.csv` | held-out 周轨迹（评测完成时） |
| `config/reward_config.yaml` | 运行快照；历史 legacy ≠ 新主口径 |

`eval_failed` / 截断周只进可执行性附表，不与满周现金混排。
