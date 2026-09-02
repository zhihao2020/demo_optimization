# Paper-min final holdout KPIs (0903 FMU, weeks 12/25/38/51)

Source: `172.16.1.80:D:\xuzh\demo_optimization\runs\paper_min\*\summary.json`
Plant SHA `e28c6753…c05cbb`, oracle `d5.7-0903-charge-headroom`.
All rows: `eval_status=ok`, `valid_steps=168`, unserved=0, fmu_failure=0, action_violation=0.

`weekly_raw_total_cost` CNY/week; lower (more negative) is better.

## Rule (deterministic)

| week | CC |
|------|-----|
| 12 | -6465684.722763436 |
| 25 | -1906795.8284767207 |
| 38 | -7188619.012575479 |
| 51 | -8866480.647114152 |
| mean | -6106895.052732447 |

## Rolling MILP (deterministic)

| week | CC | dt_s |
|------|-----|------|
| 12 | -8970866.400686603 | 0.2004354678522629 |
| 25 | -6469233.954067575 | 0.1208659440439771 |
| 38 | -9978353.150782837 | 0.33836981845074443 |
| 51 | -12209583.664946828 | 0.238866534522025 |
| mean | -9407009.29262096 | 0.22463444121725236 |

## PC-HybridTD3 seeds

| week | s0 | s1 | s2 | mean |
|------|-----|-----|-----|------|
| 12 | -17254478.51032868 | -17364991.921770517 | -7174700.00532642 | -13931390.14580854 |
| 25 | -13639763.879393257 | -13895905.836444547 | -2416355.8473544093 | -9984008.52106407 |
| 38 | -17737047.192904018 | -17869696.98838518 | -7423191.483000642 | -14343311.888096613 |
| 51 | -18918832.629830908 | -19175571.05656753 | -9017735.191111576 | -15704046.292503337 |
| four-week mean of seed-means | | | | -13490689.21186814 |

Seed 2 failed Stage D C5 (worse than random on the default eval week) and is kept in the mean.

PC mean decision time: 4.01 ms/step.

## Seed-0 utilization (paper Table `tab:util`)

Source: `runs/paper_min/{rule,milp,pso,pc_s0}_w{12,25,38,51}/summary.json` pulled 2026-09-02.

PC-HybridTD3 seed 0 on **all four** weeks: thermal mean \(51.04\) MW (\(P_{\min}\) to numerical precision), BESS throughput \(168.9\) MWh, terminal battery SoC \(0.156\). CAES charge/idle/discharge hours and settled buy/sell vary by week; settled sell is \(20\)--\(26\) GWh vs buy \(0.8\)--\(4.3\) GWh. Live paper title is **Economic** scheduling (not Coordinated).

文档更新：2026-09-02 18:00 (+08:00)
