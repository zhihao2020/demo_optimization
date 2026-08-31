# PC-HybridTD3 检查清单工程报告

文档更新：2026-08-31 10:36 (+08:00)

对照 `检查.txt` 会议论文 v1.0。研究对象：**多能源系统安全经济协同调度**（Thermal + BESS + CAES + Grid）。CAES 模式–幅值只是异构动作的代表实现，不是全文主 KPI。

方法名：**PC-HybridTD3**。题目：*Physics-Constrained Hybrid TD3 for Forecast-Aware Economic Scheduling of Multi-Energy Systems*。

目标 \(\min_\pi\mathbb{E}[\sum_t C_t]\)，\(C_t=C^{grid}+C^{thermal}+C^{carbon}+C^{curt}+C^{uns}+C^{deg}+C^{su}\)。

**禁止进入主线（已遵守）：** 双层 GHTD3、高层 SOC goal、wear/carbon budget、SAC/TD3/GHTD3 切换、storage-use 奖励、把 CAES 使用率当主优化目标。Cui GHTD3 只作 Related Work。Critic P0 保持 \(Q(s,u_T,u_B,u_C)\)。

---

## 1. 验收清单 1–20

| # | 要求 | 状态 | 证据 |
|---|------|------|------|
| 1 | 单层 TD3，不启用 GHTD3 主线 | 完成 | `scripts/train_seasonal.py --method td3` |
| 2 | 24 h 风光/负荷预测 + 24 h 电价前瞻 | 完成 | `forecast.horizon_hours=24`；`PriceProfile` |
| 3 | Oracle 扩成系统级联合动态可行域 | 完成 | `joint_caes_support` / `conditional_thermal_bounds` / `conditional_battery_bounds` |
| 4 | Sequential joint decoder | 完成 | `src/actions/joint_support.py` |
| 5 | Actor 在联合区间解码，禁止静态 clamp 当策略 | 完成 | `u_from_mode_onehot_dynamic`；`physical_dict(project=False)` |
| 6 | Target：mode \(\arg\max\)，只噪 magnitude | 完成 | `hybrid_td3/algorithm.py` |
| 7 | Deterministic GiveSafe = 1 | 完成 | `n_try = 1 if deterministic` |
| 8 | Eval 失败落盘、不 fallback、不退出进程 | 完成 | `failed_no_safe_action` / `failed_fmu`；partial traj + JSON |
| 9 | `summary.json` 无条件写盘 | 完成 | `finally` + `training_failed` |
| 10 | 经济 replay 1.0 / 0.0 | 完成 | paper yaml + givesafe yaml |
| 11 | 拒绝进 SafetyDataset | 完成 | |
| 12 | `storage_use=false` | 完成 | Stage B `storage_use_enabled=0` |
| 13 | 36/8/8，禁止 eval 回退 train 周 | 完成 | `OPTIMAL_DEMO_FORMAL_SPLIT` |
| 14 | RandomFeasiblePolicy 走同一 decoder | 完成 | |
| 15 | 四条单元测试 | 完成 | 本地 **21 passed**（本轮相关文件）；此前全套 **44 passed** |
| 16 | 重跑 Stage B 5k | **完成** | 远程 3-D critic；见 §4 |
| 17 | Stage C 20k–50k | **进行中** | 30k；10:28 启动，约 12%（3535/30000）时仍在训 |
| 18 | Stage C 五条 formal gate | 代码完成；B 实测未过 | 见 §4。C 未结束，**不以 B 代替 C 过门** |
| 19 | Gate 通过后才允许 Stage D | 队列已卡住 | `stage_c_ok=false` → SKIP D |
| 20 | 中文工程报告 | 本文件 | 只用新 3-D 跑的数 |

---

## 2. 单元测试

```text
PYTHONPATH=src python -m pytest tests/test_pc_hybrid_td3.py tests/test_joint_support.py tests/test_givesafe_deterministic.py tests/test_caes_decode.py tests/test_hybrid_parameterized_actor.py tests/test_phase_d5_feasibility.py tests/test_invalid_replay.py -q
```

覆盖检查.txt §40.15：10k 联合支撑无 device/grid/NaN；GiveSafe 确定性 1 次；`givesafe_fraction=0` 时经济 batch 无拒绝样本；magnitude 噪声不改 mode。另：Stage C 过门在缺周/FMU failure 时不得用残缺轨迹和 random 比成本。

---

## 3. 代码修改摘要

- 联合 \(\mathcal A_f(s)\) 解析 decoder；actor 动态区间；target 只噪 \(z\)。
- GiveSafe 评估 1 次；NoSafeAction / FMU failure 记 `eval_failed`，写 `eval_failure.json`，进程不退出。
- 经济 replay 只用物理转移；拒绝进 SafetyDataset。
- `compute_stage_c_gates`：NaN/Inf、FMU hard、NoSafeAction、缺供、**完整 168 h** 成本优于 random。CAES 是否启动不是过门。
- 队列：C 训练崩溃才停；过门失败 SKIP Stage D，rule/MILP 仍跑。
- `physical_from_dict` 允许并存 `caes_magnitude` 诊断字段（Stage B 曾因此在第 1119 步崩过，已修并重跑）。
- 论文 Fig.1–6 与结果顺序已按检查.txt 改；`Paper/main.pdf` 已编译。

---

## 4. Stage A / B 实测 KPI（3-D critic，2026-08-31）

机器：`172.16.1.80` `D:\xuzh\demo_optimization`。**禁止**把 `seasonal_v1` / `fs_hsac_*` / 旧 6-D 权重填进主表。

### Stage A（联合支撑，无 FMU 训练）

| 项 | 值 |
|----|----|
| n | 10 000 |
| status | completed |
| illegal_caes_mode | 0 |
| dynamic_bound_violation | 0 |
| grid_violation | 0 |
| nan | 0 |

### Stage B（5 000 physical steps）

| 项 | 值 |
|----|----|
| status | completed |
| training_status | completed |
| valid_steps | 5000 |
| stage_b_interaction | passed |
| warmup D / I / C | 331 / 348 / 345 |
| 训练期 CAES 计数 D / I / C | 1239 / 2613 / 1148 |
| 训练 \(\Delta\)SOC\(_\mathrm{gas}\) | 0.271 |
| physical replay | 5000 |
| givesafe replay（审计） | 66 |
| proposal rejection rate | 1.03% |
| 训练 post-step hard / unsafe | **14 / 14** |
| main FMU safety rate | 99.72% |
| storage_use | 0（关闭） |
| last critic_loss / actor_loss | \(2.78\times10^9\) / \(-8980\)（finite，未发散成 NaN） |

**Held-out greedy eval（TEST 第一周，start 6 652 800 s）：**

| 项 | Policy | RandomFeasible | Price-aware rule |
|----|--------|----------------|------------------|
| eval_status | ok（旧口径） | ok | ok |
| valid_steps | **48**（未满 168） | 168 | 168 |
| fmu_failure_count | **1** | 0 | 0 |
| unserved MWh | 0 | 0 | 0 |
| curtailment MWh | 0 | 0 | 0 |
| weekly_raw_total_cost | \(-1.239\times10^6\) | \(-1.122\times10^7\) | \(-7.097\times10^6\) |
| CAES h (D/I/C) | 7 / 42 / 0 | — | — |

旧口径把 greedy 标成 `passed`，但 **48 步 + 1 次 FMU failure 不是可部署周**。成本列也不可与 168 h 的 random/rule 直接比。代码已改为：FMU failure → `eval_status=failed_fmu`；`valid_steps<168` 不得判 greedy 通过、不得做成本过门。该修复**未注入正在跑的 Stage C 进程**（避免打断 30k）；C 结束后的下一轮评估会带上。

**Stage B 上的五条 C 门（诊断，不是正式 C 过门）：**

| 门 | 结果 | 说明 |
|----|------|------|
| c1 NaN/Inf = 0 | 通过 | Q/loss finite |
| c2 FMU hard = 0 | **失败** | 训练 14 次 hard + eval 1 次 FMU failure |
| c3 held-out NoSafeAction = 0 | 通过 | 无 NoSafeAction |
| c4 缺供 ≈ 0 | 通过（残缺周） | 48 步内 unserved=0；完整周才能算正式 |
| c5 \(C_{policy}<C_{random}\) | **失败** | 残缺周成本不可比；完整口径下也不应过 |
| **passed** | **false** | 不得进入 Stage D |

Failure taxonomy：本周 greedy 不是 NoSafeAction，而是 **后验硬约束截断**。`eval.csv` 第 48 步：

```text
u_tp=1, u_battery=0, u_caes=0 (idle)
fmu_status=failure
failure_type=PostStepHardConstraintViolation
fmu_error=caes_gas_pressure=6496310.259 越界 [6500000, 9500000]
```

前一步压力 6.514 MPa，idle 一小时后 6.496 MPa。训练 14 条 false-safe 样本：13×`caes_pressure_low` + 1×`caes_temperature_high`，同样是 `u_caes=0`。Oracle `predict_next_state` 在 idle 下 \(\Delta SOC=0\)，把压力当成不变；`_caes_mode_mask` 写明「当前态在物理界内则始终允许 idle，残差裕度不得禁止待机」。`feasibility_margins.yaml` 已有 idle 压力裕度 120 kPa，但未用于禁止待机。这是 **GiveSafe false-safe**，不是 TD3 没学够。贪心策略本周 idle 42/48 h，加训只会更常贴着下界待机，**C2 不能靠 Stage C/D 训掉**。下一轮开训前应：贴下界时禁 idle、保留 charge；charge 也被挡时记 C3，不要把 idle 送进主 FMU。在那之前 **不得进入 Stage D**。

### Stage C（30k，诊断跑，不过门不算正式）

队列 10:28:51 `START stageC_s0`。进程用的是启动时内存中的旧门控（无 complete_week）。C2 根因在 Oracle，与远程是否多一份源码无关。**正式五条 gate 只认修 idle 预检之后重跑的 Stage C summary。**

系统级指标顺序（检查.txt §三十）：operating cost → grid trading → curtailment → unserved → carbon → grid contract excess → hard violations → FMU failures → decision time。设备吞吐只作机制。CAES 与 MILP 同为接近 idle 时不判学习失败。

---

## 5. 协议与超参

- 36/8/8；eval = 第一 TEST 周
- `paper_pc_hybrid_td3.yaml`：\(\gamma=0.99\)，\(\tau=0.005\)，delay=2，lr \(3\times10^{-4}\)，replay 1.0/0.0，`storage_use=false`，C 30k，D 400k
- GiveSafe 训练 64 次 / 评估 1 次；fallback off；soft_shell off
- Terminal SOC tolerance 0.06；报告 \(E_T\)，不只 bonus

---

## 6. 论文

- `Paper/main.tex` + 已编译 `Paper/main.pdf`（34 页）
- Fig.1 拓扑，Fig.2 算法，Fig.3–6 为 TEST 占位（**不是** HMSD/GHTD3 旧图）
- 结果顺序：经济 → 消纳/可靠性 → 安全 → 机制
- 主表以 \(CC=\sum C_t\) 为首；消融两组；baseline：rule / rolling MILP / projection TD3 / PC-HybridTD3
- **不填 Stage D 数字**

---

## 7. Stage D

`pc_hybrid_queue_state.json`：`stage_c_ok=false`，`greedy_ok=true`。队列逻辑：C 过门前 **SKIP** `stageD_s0/s1/s2` 和 `proj_D`。B 未过门、C 未结束 ⇒ **现在不得启动 3×400k**。

C 结束后若 `stage_c_passed=true` 才允许 D；若仍失败，先根据 `eval_failure.json` 和 14 次训练 hard 做诊断，而不是加 storage-use 或把 CAES 使用率当 KPI。
