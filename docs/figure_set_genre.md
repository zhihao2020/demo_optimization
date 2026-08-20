# Figure set (genre checklist)

Aligned with GHTD3 Fig.4–9 and OCTD3 style. Full paths under `Paper/figures/`.

## Main text

| Fig | Basename | Content | Script | Data gate |
|-----|----------|---------|--------|-----------|
| 1 | `fig_topology` | Plant + FMU export | schematic / existing | none |
| 2 | `fig_price_tou` | Shandong TOU | `plot_paper_plan_figures.py` | none |
| 3 | `fig_seasonal_boundary` | Wind / PV / load / price × 3 seasons | same | none |
| 4 | `fig_caes_legal` | Operating envelope (constraints §3.3) | `plot_paper_v2_figures.py` | none |
| 5 | `fig_action_rep` | Projection mute vs (mode, mag) | same | none |
| 6 | `fig_algorithm` | Hybrid SAC + mask + GiveSafe | same | none |
| 7 | `fig_training` | Reward / cost curves: hybrid SAC vs proj. | `plot_paper_figures_v2.py` | `sac_param` progress |
| 8–10 | `fig_balance_{winter,transition,summer}` | 4 panels: hybrid SAC / hybrid TD3 / PSO / MILP | `plot_paper_figures_cui_style.py` | full-week traj |
| 11 | `fig_storage_strategies` | Battery SoC + CAES mode footprint, 3×4 | new / cui-style | full-week traj |
| 12 | `fig_cost_components` | Stacked CC breakdown | `plot_paper_plan_figures.py` | `tab:main` |
| 13 | `fig_sensitivity` | Carbon / margin / capacity | TBD | sensitivity runs |

## Auxiliary (footnote / appendix)

- `fig_horizon`, `fig_givesafe_reject` — executability only
- `fig_project_vs_hybrid` — action-representation ablation

## Dropped from narrative

- `fig_cstep`, HMSD algorithm figures, hierarchical mechanism bars as identity plots
