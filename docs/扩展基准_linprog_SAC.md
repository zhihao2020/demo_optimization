# 扩展基准：真滚动 linprog + SAC-Hybrid

## 方法说明

| 代号 | 实现 | 备注 |
|------|------|------|
| **M1b linprog** | `src/optimization/rolling_linprog.py` | scipy HiGHS；闭环 FMU |
| **M5 SAC** | `src/training/hybrid_sac/` | Hybrid-GiveSafe-SAC **80k**（15k 续训 +65k） |

ckpt: `runs/givesafe_sac_80k_20260804/checkpoints/hybrid_givesafe_sac.pt`

## 周窗口 reset 依据

见 `docs/周窗口Reset依据.md`。

## 表：三季闭环 FMU 指标

| Season | Method | Net cash flow J | ΔJ vs B0 | Reward | SOC | Thermal MWh | Bat thr. | CAES thr. |
|--------|--------|----------------:|---------:|-------:|:---:|------------:|---------:|----------:|
| Winter | B0 Rule (original) | 8.333e+06 | — | 67.5 | Y | 25200 | 0 | 275.3 |
| Winter | B1 Price rule | 6.436e+06 | -1.898e+06 | 52.1 | Y | 24807.2 | 4395 | 11002 |
| Winter | M1 Heuristic rolling | 5.596e+06 | -2.737e+06 | 46.7 | Y | 24995.8 | 4673.9 | 11061.1 |
| Winter | M1b True linprog MPC | 7.053e+06 | -1.280e+06 | 54.7 | Y | 24175.6 | 6007.8 | 10539.9 |
| Winter | M2 PSO parametric | 7.371e+05 | -7.596e+06 | 1.8 | N | 810.6 | 181.1 | 591.6 |
| Winter | M5 Hybrid-SAC (80k) | 8.376e+06 | 4.327e+04 | 63.4 | Y | 25200 | 466.8 | 450 |
| Winter | M3 Hybrid-TD3 | 1.851e+07 | 1.018e+07 | 128.1 | Y | 8597.5 | 466.8 | 516 |
| Winter | M4 Safe Market-GHTD3 | 1.831e+07 | 9.975e+06 | 126.8 | Y | 8957.5 | 643.9 | 600 |
| Summer | B0 Rule (original) | -8.415e+04 | — | 13.3 | Y | 25200 | 0 | 1881.5 |
| Summer | B1 Price rule | -9.036e+05 | -8.195e+05 | 5.5 | Y | 24935.4 | 2336.8 | 5464.5 |
| Summer | M1 Heuristic rolling | -1.810e+06 | -1.726e+06 | -0.8 | Y | 25050.3 | 2868.5 | 8855.2 |
| Summer | M1b True linprog MPC | -5.916e+05 | -5.075e+05 | 5.9 | Y | 25200 | 3495.8 | 9092.7 |
| Summer | M2 PSO parametric | 4.459e+06 | 4.543e+06 | 24.0 | N | 6310.7 | 1317.5 | 0 |
| Summer | M5 Hybrid-SAC (80k) | -1.397e+05 | -5.559e+04 | 8.2 | Y | 25200 | 466.8 | 1287.1 |
| Summer | M3 Hybrid-TD3 | 1.168e+07 | 1.177e+07 | 83.7 | Y | 9142.5 | 466.8 | 1287.1 |
| Summer | M4 Safe Market-GHTD3 | 1.118e+07 | 1.126e+07 | 80.3 | Y | 9837.5 | 643.9 | 1881.5 |
| Transition | B0 Rule (original) | 6.883e+06 | — | 58.6 | Y | 25200 | 0 | 0 |
| Transition | B1 Price rule | 6.615e+06 | -2.680e+05 | 53.3 | Y | 24979.5 | 2336.8 | 8059.2 |
| Transition | M1 Heuristic rolling | 6.109e+06 | -7.748e+05 | 49.6 | Y | 25072.7 | 2868.5 | 7977.6 |
| Transition | M1b True linprog MPC | 6.579e+06 | -3.046e+05 | 51.4 | Y | 24398.7 | 3526.1 | 10790.3 |
| Transition | M2 PSO parametric | 8.924e+06 | 2.040e+06 | 69.6 | Y | 21773.4 | 1761.8 | 0 |
| Transition | M5 Hybrid-SAC (80k) | 6.815e+06 | -6.874e+04 | 53.3 | Y | 25200 | 466.8 | 2058.5 |
| Transition | M3 Hybrid-TD3 | 1.630e+07 | 9.419e+06 | 113.6 | Y | 10255 | 466.8 | 2363 |
| Transition | M4 Safe Market-GHTD3 | 1.618e+07 | 9.295e+06 | 113.0 | Y | 10205 | 643.9 | 2664.4 |

## SAC 长训

| 项 | 内容 |
|----|------|
| 起点 | `runs/givesafe_sac_15k_20260804` |
| 续训 | +65k → 约 80k |
| 三季 | 本表 M5；三季 SOC=Y |

## 附录：连续年 SOC

见 `docs/连续年SOC附录协议.md`。
