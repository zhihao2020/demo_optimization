# 论文对照：弱单层 SAC（对齐 Cui vs TD3）

## 角色

| 方法 | 角色 |
|------|------|
| SAC-80k | **弱单层**（≈ Cui 文中 TD3 / inaction） |
| Hybrid-TD3 | **强单层**教师 |
| ares-35k Safe Market-GHTD3 | **主方法** |

## 周 reward（主表）

| 季节 | SAC | Hybrid | ares | Δ vs SAC (ares) | Δ vs Hybrid (ares) |
|------|-----|--------|------|-----------------|---------------------|
| 冬 | 63.4 | 128.10 | **129.18** | **+104%** | +1.08 |
| 夏 | 8.2 | 91.70 | **92.28** | **~+1025%** | +0.58 |
| 过渡 | 53.3 | 110.37 | **111.02** | **+108%** | +0.65 |

数据：`runs/benchmark_extended_linprog_sac_80k...`、`runs/ghtd3_ares_35k/vs_hybrid.json`。

## 弱单层签名（SAC 冬）

- thermal MWh = **25200**（与 B0 同档满火电）
- reward 接近 B0，远低于 Hybrid

## 正文位置

`Paper/main.tex` Results：先 vs SAC/B0，再 vs Hybrid；主表 `tab:main` 已用 ares，去掉 TEA-s0 虚高。
