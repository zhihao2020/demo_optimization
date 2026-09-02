# Figure set (genre checklist)

文档更新：2026-09-02 22:00 (+08:00)

IEEE conference live paper `Paper/main.tex`. Frozen to **five** main-text figures. Result panels are LaTeX comments until Stage D; `fig_placeholder` is not rendered.

## Main text

| Fig | Basename | Content | Script | Data gate |
|-----|----------|---------|--------|-----------|
| 1 | `fig_topology` | Wind/PV/Thermal/BESS/CAES/Grid/Load | labeled schematic | none |
| 2 | `PC-HybridTD3_Architecture` | Hybrid actor, joint support, critic | hand-drawn | none |
| 3 | `fig_action_rep` | (a) Continuous-projection TD3; (b) Component-support Hybrid TD3; (c) PC-HybridTD3 | `Paper/figures/gen_fig_action_rep.py` | none |
| 4 | `fig_training` | Validation operating cost, 3 seeds | — | Stage D |
| 5 | `fig_dispatch_week` | 4-panel week: exogenous / dispatch / price+net load / inventories | — | full-week TEST traj |

## Dropped from the live manuscript (deleted)

- `fig_placeholder`, `fig_caes_legal`, `fig_caes_feasible_set`, `fig_aux_obs`, `fig_seasonal_boundary`
- KPI bar, cold-tank guard, annual-reset, carbon-position / settlement, GiveSafe reject
- Graphical abstract, highlights, nomenclature (CAS leftovers)
