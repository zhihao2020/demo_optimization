# Figure set (genre checklist)

文档更新：2026-08-31 10:45 (+08:00)

Aligned with 检查.txt Fig.1–6 (system scheduling, not CAES-only). Full paths under `Paper/figures/`. Result panels are empty placeholders until the PC-HybridTD3 TEST campaign; archived GHTD3/HMSD/FS-HSAC plots have been deleted.

## Main text

| Fig | Basename | Content | Script | Data gate |
|-----|----------|---------|--------|-----------|
| 1 | `fig_topology` | Wind/PV/Thermal/BESS/CAES/Grid/Load | labeled schematic | none |
| 2 | `fig_algorithm` | Forecast → hybrid actor → joint support → GiveSafe → FMU | `Paper/figures/gen_fig_algorithm.py` | none |
| 3 | `fig_placeholder` | Training curves (empty) | — | Stage D |
| 4 | `fig_placeholder` | Typical TEST week dispatch (empty) | — | full-week TEST traj |
| 5 | `fig_placeholder` | Battery SoC + CAES SoC + TOU (empty) | — | full-week traj |
| 6 | `fig_placeholder` | System KPI (empty) | — | `tab:kpi` |

## Auxiliary (footnote / appendix)

- TOU clocks / seasonal boundaries / CAES legal envelope / action-rep: schematics, kept
- Cold-tank, annual SoC, carbon settlement, GiveSafe reject bars: **empty placeholders** (old numeric plots deleted)

## Dropped from narrative

- `fig_cstep`, HMSD algorithm figures, hierarchical mechanism bars, GHTD3 weekly balance, HMSD training curves, B0/linprog/PSO/HMSD carbon bars
