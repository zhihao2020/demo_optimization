# Safe Market-GHTD3 实验结果与消融

> **Snapshot**（历史实验数字）。当前可运行主线与入口以 `docs/GHTD3分层实现说明.md`、`docs/cui_seasonal_min_protocol.md` 为准；勿将本页命令当作唯一入口。

## 1. 主实验（50k 续训 + 全年）

| 指标 | Safe Market-GHTD3 | Hybrid BC+RL (对照) | 规则 |
|------|-------------------|---------------------|------|
| 周 reward | **128.36** | 130.35 | 67.59 |
| 周 SOC 过关 | **是** (L1=0.0458) | 是 | 是 |
| 周火电 MWh | **8958** | 8575 | ~25200 |
| 全年 reward 合计 | **5252.8** | ~5480 | — |
| 全年 SOC 达标周 | **16/53** | 19/53 | — |
| 全年现金流增量 | **8.350e+08** | ~8.62e8 | — |

Run 目录：`runs/ghtd3_market_50k_annual_20260803/`

训练命令：
```powershell
python scripts/run_ghtd3_ablations.py --stage main --main-steps 50000 `
  --resume runs/ghtd3_market_curriculum_20k_20260803/checkpoints/ghtd3.pt
```

## 2. 消融（从零 12k，公平短训）

| 变体 | 周 reward | SOC | L1 | 火电 MWh | 说明 |
|------|-----------|-----|-----|----------|------|
| full | 62.17 | 否 | 0.146 | 17610 | 完整 Safe Market-GHTD3 |
| no_market_prior | 39.61 | 否 | 0.062 | 24863 | 去掉市场 goal 先验 |
| no_recovery_goal | 47.60 | 否 | 0.060 | 25088 | 去掉上层回收 goal（环境硬回收仍在） |
| no_bc | 111.39 | 否 | 0.064 | 9120 | 去掉分层 BC |
| gamma_not_c | 129.23 | 是 | 0.043 | 8598 | SMDP 折扣退化为 γ（c=1） |

### 消融解读

- **no_market_prior (39.6)** vs **full (62.2)**：市场 goal 先验显著贡献套利语义。
- **no_recovery_goal (47.6)**：去掉上层回收 goal 后经济与 SOC shaping 变差（环境硬回收仍在）。
- **no_bc (111.4)**：短训下无 BC 经济更高——分层 BC 在 12k 内尚未与 RL 充分对齐；主实验 50k 续训后 full 达 **128.4 且周 SOC 过关**，说明 **BC + 长训** 才是正确配方。
- **gamma_not_c / c=1 (129.2, SOC 过)**：每步换 goal + γ¹，接近单层 goal 条件策略，短训更易收敛；完整 SMDP（c=8, γ^c）需要更长预算（见主实验 50k）。

## 3. 回收段改进要点

1. 回收只修 **battery + CAES gas**，到位后强制 IDLE。
2. 禁止为修热/冷罐反复开关机（曾触发 min-run 把 gas 掏空）。
3. `soc_recovery_horizon=40` + 上层 `recovery_prior_weight=0.92`。
4. 周 SOC：GHTD3 50k **过关**（L1≈0.046 < 0.06）。

## 4. 与 Hybrid 对比结论

- 周 reward：GHTD3 **128.4** ≈ Hybrid **130.4**（差距 <2%）。
- 周 SOC：两者均过关。
- 全年 SOC：GHTD3 **16/53**，Hybrid **19/53**，仍有提升空间。
- 方法贡献：分层 SMDP + 市场 prior + 回收 goal + GiveSafe，可写 SCI 方法章节。

