# PC-HybridTD3 ablation matrix & staged training

文档更新：2026-08-30 22:40 (+08:00)

Code: `src/training/hybrid_td3/`.  
Paper mainline: `--method td3` (dynamic \(\mathcal A_f(s)\)).

Two ablations only (会议论文不再堆矩阵):

| Tag | Role | CLI / flag |
|-----|------|------------|
| PC-HybridTD3 | **live** parameterized hybrid TD3 on \(\mathcal A_f(s)\) | `--method td3` |
| projection TD3 | Ablation 1: continuous box + \(\Pi_{\mathrm{bands}}\) | `--method td3 --ablation projection` |
| static-support hybrid TD3 | Ablation 2: mode+mag on static bands | `--method td3 --ablation static-support` |
| price-aware rule | engineering baseline | eval companion |
| rolling MILP | model-based baseline (surrogate opt, FMU eval) | `--method milp` |

FS-HSAC / param SAC / HMSD are archive, not this matrix.

## Stages (重构.txt)

| Stage | Budget | Gate |
|-------|--------|------|
| A support-only | 10k decoded actions, no train | illegal mode = 0, bound violation = 0, NaN = 0 |
| B FMU smoke | ~5k physical steps | charge and discharge appear; \(\Delta SOC^{gas}>0.05\); FMU fail = 0 |
| C learnability | 20–50k | critic stable; mode not collapsed idle |
| D formal | 300–500k physical / seed × {0,1,2} | TEST tables; 840k only if 500k still climbing |

## Status

| Step | Status |
|------|--------|
| P0 actor/critic/replay | implemented (`tests/test_pc_hybrid_td3.py`) |
| CLI `--ablation` + 36/8/8 | implemented |
| Stage A–D remote | not started |
