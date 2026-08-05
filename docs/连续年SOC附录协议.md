# 附录协议：连续年 SOC（Continuous-Year SoC Carry）

> 主表仍用 **周窗口 reset**（见 `docs/周窗口Reset依据.md`）。  
> 本文件定义 **附录可选实验**：储能状态跨周连续传递，回答审稿人关于「是否只是 53 次独立周」的质疑。

---

## 1. 两种协议对照

| 项目 | 主表：`weekly_reset` | 附录：`continuous_soc` |
|------|----------------------|-------------------------|
| API | `evaluate_annual_policy(..., continuous_soc=False)` | `evaluate_annual_policy(..., continuous_soc=True)` 或 `evaluate_continuous_annual_policy` |
| FMU | 每 168 h **reset** 一次 | **仅**在年起点 `reset` 一次 |
| SOC | 每窗回到 FMU 标称初值 | 物理状态跨周传递 |
| terminal SoC | 每窗期末门控 / 奖励 | **仅年终**门控（`episode_steps` 临时=8760） |
| 用途 | 周运营 + 公平算法对比 | 季节性库存、跨周价值、鲁棒性 |
| 代码 | `src/training/evaluate_td3.py` | 同文件 `evaluate_continuous_annual_policy` |

---

## 2. 为何不能「reset 后写入上一窗 SOC」

当前 PowerSystem_8760h FMU 中 `battery_soc` / `caes_*_soc` 为 **output**，  
`FmuSession.reset` 会重新实例化并从标称初值初始化。  
在 Python 层改写 `last_outputs` **不会**改变 FMU 内部连续状态，下一步 `doStep` 会回到真实内部轨迹。

因此唯一正确的跨周 SOC 传递是：

> **单次实例化 → 连续 8760 个决策步 → 不中途 re-instantiate。**

---

## 3. 运行步骤（复现）

```powershell
. .\scripts\with_e_cache.ps1
$env:PYTHONPATH = "src"

python scripts/eval_continuous_annual.py `
  --methods b0,hybrid,ghtd3,sac `
  --horizon-hours 8760 `
  --sac-ckpt runs/givesafe_sac_80k_20260804/checkpoints/hybrid_givesafe_sac.pt `
  --out-dir runs/appendix_continuous_soc_8760_20260804
```

常用参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--horizon-hours` | 8760 | 年长 |
| `--methods` | 见脚本 | 逗号分隔 |
| `--sac-ckpt` | 最新长训 SAC | Hybrid-GiveSafe-SAC 权重 |
| `--hybrid-ckpt` / `--ghtd3-ckpt` | 仓库主结果 | 与主表一致 |
| `--smoke-hours` | 无 | 冒烟：如 336（两周连续） |

输出：

- `continuous_summary.json`：各方法年汇总  
- 每方法子目录：`continuous_year.csv`、`window_snapshots.json`（168 h 切片 KPI，**非**独立 reset）

---

## 3.1 已跑结果（2026-08-04，山东 TOU）

数据：`runs/appendix_continuous_soc_8760_20260804/continuous_summary.json`（E: 缓存同路径）

| Method | steps | valid | Net J（至终止） | year-end SOC | invalid | fmu_fail | 说明 |
|--------|------:|------:|----------------:|:------------:|--------:|---------:|------|
| **B0 Rule** | **8760** | **8760** | **3.121e+08** | **Y** | 0 | 0 | 完整连续年 |
| Hybrid-TD3 | 1276 | 1275 | 1.298e+08 | N | 1 | 1 | ~第 53 天失败中止 |
| GHTD3 | 1276 | 1275 | 1.296e+08 | N | 1 | 1 | 同上 |
| SAC-80k | 1277 | 1276 | 5.072e+07 | N | 1 | 1 | 同上 |

**解读（写附录 Discussion）：**

1. **协议已跑通**：单次 reset、SOC 跨周传递；B0 证明 FMU 可完整 8760h 连续。  
2. **主表周 reset 的必要性**：在 **连续年** 下，主 RL（周 episode 训练）约在 **1276 h** 触发 FMU/约束失败，**不能**完成年终 SOC——与「周运营期末回收」设定一致。  
3. **J 不可与主表 53 窗求和直接比**：RL 行为 **截断轨迹**；B0 的年 J 是完整年。  
4. **主对比仍用 weekly_reset**；本表支撑「连续年更难 / 周协议非偷懒」。

---

## 4. 报告指标（附录表建议列）

| 列 | 含义 |
|----|------|
| Annual net cash flow J | `-annual_raw_total_cost` 或 `annual_economic_cashflow` |
| Annual reward sum | `annual_episode_reward`（含年终 terminal 项） |
| Year-end terminal SOC | `terminal_soc_satisfied_year_end` + 四分量 SOC |
| Valid / invalid steps | 运行完整性 |
| Thermal / Bat / CAES thr. | `metrics` |
| Week-slice J series | `window_snapshots[*].net_cashflow_j`（可选图） |

**禁止**将附录 `continuous_soc` 的 J 与主表 `weekly_reset` 的「53 窗求和 J」直接并排宣称同一协议，须标注 protocol。

---

## 5. 论文表述模板

**主实验（正文）**

> Weekly-horizon evaluation with storage re-initialization at each 168 h window enforces terminal SoC recovery consistent with weekly market operation.

**附录**

> As a sensitivity study, we further evaluate a continuous-year protocol in which the FMU is reset only once at the beginning of the 8760 h horizon, so storage SoC carries across week boundaries. Terminal SoC is assessed only at year end. This protocol isolates seasonal inventory effects and is not used for the primary algorithm ranking.

---

## 6. 与训练协议的关系

- **训练**仍用 168 h episode + 周起点循环（可学性）。  
- **附录评估**只改评估轨迹长度，不要求重新训练。  
- 若连续年下策略年末 SOC 崩溃，可讨论「周运营目标 vs 跨季节持仓」的目标错配，而非否定主表结论。

---

## 7. 验收清单

- [x] `evaluate_continuous_annual_policy` + `continuous_soc` 开关  
- [x] 文档与 CLI `scripts/eval_continuous_annual.py`  
- [x] B0 **完整 8760**；Hybrid / GHTD3 / SAC-80k **连续年评估已跑**（RL 于 ~1276h 失败中止——结果有效）  
- [x] 附录结果表见 §3.1  
- [x] SAC-80k 纳入对照  
- [ ] 可选：附录图（年 SOC 轨迹 / 失败时刻）
