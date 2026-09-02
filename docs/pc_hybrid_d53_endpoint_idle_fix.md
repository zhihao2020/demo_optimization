# PC-HybridTD3 d5.3：endpoint snap 与 idle robust guard

文档更新：2026-09-01 14:10 (+08:00)

按 `检查.txt` 只修 **safety / action interface**。未改 TD3 \(\gamma,\tau,\) lr、网络宽度、reward、Gumbel、GiveSafe `n_try=1`、Shadow（主线仍 `enable_shadow=False`），未开 `storage_use`，**未启动 Stage D**。

Oracle version：`d5.2-probe-calibrated` → **`d5.3-idle-robust-endpoint`**。

## 三种事件（不得混记）

| 事件 | 含义 | 是否拒绝 |
|------|------|----------|
| **numerical endpoint snap** | float32 端点 vs float64 区间，\|Δ\|≤`DYNAMIC_ENDPOINT_ATOL=1e-7` 时钉到端点 | 否。审计计数 `numerical_endpoint_snap_count` / `max_endpoint_snap_abs` |
| **Oracle rejection** | 动作不在 \(\mathcal A_f(s)\)（含 idle robust envelope、动态区间、预测下一状态） | 是。GiveSafe 拒绝，主 FMU 不步进 |
| **FMU post-step false-safe** | Oracle 放过但主 FMU 一步后硬约束失败 | 是。计 `givesafe_false_safe` / `main_fmu_unsafe_execution` |

endpoint snap **不算**安全拒绝。被 Oracle 正确提前拦住的动作 **不算** false-safe。

## C3 评估根因（已修）

同一类 **float32 动态端点** 在充电上沿和放电 `-0.33` 都复现过，不要把其中一次写成「唯一 C3 根因」：

- 充电：`u_caes=0.93000000715` 对上界 `0.9299999999999999`。
- 放电：`mag=1` 打在 `hi=-0.33`，float32 插值得到 `-0.32999998`，曾被扫成 idle。

修复：`mag∈{0,1}` 时直接取区间端点；`project_u_caes(_torch)` 对端点 ULP 做 snap。真空隙（如 0.2）仍投影到 idle。

Stage C 总表是 **7 次** unsafe transition（7/7），不要和某一 taxonomy 子集的 6 混写成两套 C2 数字。修完后需新目录重跑 Stage C 才能再过门。

## 修了什么

1. `src/actions/caes_u.py`：`snap_to_interval_endpoint`。只修 ULP，不把 `hi+1e-4` 投影回去。静态合法集 `[-1,-0.33]∪{0}∪[0.86,1]` 未动。
2. `HybridActor.act_numpy`：按当前 mode 的动态区间 snap，再 `physical_dict`。audit 字段不进 FMU 三元组。
3. `FeasibilityOracle.check_action_executable`：同一 helper，去掉硬编码 `1e-9`。**先**按当前 mode 做 endpoint canonicalization，**再** `predict_p_grid` / grid check / next-state；env 与 GiveSafe checker 执行同一 canonical 动作。**IDLE 被 mask 禁止**时直接拒绝；即使传入过期 `idle=True` 的 feasible，也再用 `_caes_idle_step_ok` 拦一层，**不用** `predict_next_state(u_caes=0)` 放行 idle。
4. `legalize_mode_mask`：全 False 行 **raise** `EmptyModeMaskError`，不再静默改成全 True。在线路径仍是 empty → `FeasibleSetEmpty`、不调 actor。TD3 target 对 empty next-state 记 `done=1`，forward 只用 idle-only dummy，不开放三个模式。
5. 端点审计拆成 `caes_raw_endpoint_miss_abs` 与 `caes_numerical_snap_abs`。raw miss `> ENDPOINT_SNAP_HARD_FAIL` 视为 decoder bug，不再要求先 `snapped=True`。
6. `_caes_idle_step_ok`：不用 `predict_next_state(u_caes=0)`。idle 用 residual P99 + margin 的双边 envelope（气/热/冷 SOC、压力、有限温度侧）。`6.685 MPa` 由 `gas_pressure_min_Pa + residual_p99_pressure + margin_pressure_Pa` 算出，不硬编码。
7. idle 被禁时不回退成 idle。无合法 mode → `feasible_set_empty`。
8. 文档 critic 与代码对齐：3-D \(Q(s,u^{\mathrm{th}},u^{\mathrm{bat}},u_{\mathrm{caes}})\)，lr \(3\times10^{-4}\)。未把 critic 改回 6-D。

## 测试

本地 `pytest`：endpoint / idle / mode_mask / illegal_no_fmu / pc_hybrid / joint / givesafe_deterministic / phase_d5 / caes_decode **70 passed**。新增：grid check 使用 snap 后的 `u_caes`；all-false mask raise；raw miss `>1e-3` 不依赖 `snapped=True` 也会 hard-fail。

## 修复前后（代码层，非 Stage C 数字）

| 项 | 修前 | 修后 |
|----|------|------|
| `0.93000000715` vs `[0.86, 0.9299999999999999]` | Oracle `1e-9` 误拒；grid check 在 snap 前 | 先 snap 再 grid / next-state |
| all-false mode mask | 静默改成全 True | raise；replay 记 done |
| idle @ 6.60 MPa（物理下界 6.50） | `post_step_hard_ok` 允许 idle | idle mask False |
| idle 近 SOC/温度硬界 | 只看当前是否越界 | 双边 δ=P99+margin |
| Stage D | 仍阻断 | **仍阻断**，直到新 Stage C 的 C1–C5 全过 |

## 尚未做（检查.txt 20–23）

- 用现有 30k Stage C checkpoint 做 168 h greedy regression（需远端 `pc_hybrid_td3_stageC_s0`）。
- `boundary_stress_min_actions: 20000`，要求 \(N_{\text{false-safe}}=0\)。
- **新目录**重跑 Stage C 30k；旧 run 只作 diagnosis。
- 新 C 五门全过之前 **禁止 Stage D**。

C 过门仍是：C1∧C2∧C3∧C4∧C5。`complete_week` **只**看评估 `eval_status=ok` 且 `valid_steps=168`，**不含 C2**。训练 false-safe 只打 C2，不再把 C4/C5 连带置 False。

## Idle 冷罐 0.13→0.045：direct FMU replay（未改 margin）

Modelica `TypicalScenarios.mo` idle 分支把六路 `Mdot_*.mflow_in=0`；`Tank` 为 `der(m)=port_a.m_flow+port_b.m_flow`，`caes_cold_soc=compressedAirEnergyStorage.coldtank.SOC`。`p_caes=PBS.P_act=u_dispatch*P_cap`，因此日志 `p_caes=0` **只证明指令为 0，不是实测液压功率**。

仓库默认路径现已换成 **0831 新导出**（2026-08-31T15:05:31Z，guid `9b0edb50-…`，sha256 `31c8fec7…`，有边界口）。它与服务器 8-17 训练 FMU（guid `433bca45-…`）仍不是同一份。0831 上：`u_caes` read-back=0 时 idle 5 h 从 SOC=0.5 **冷罐不变**；ep13 气候下充/idle 前缀后 idle 一步 Δcold **= 0**。不是 0.08–0.12。

因此 **不能**把 14 条 false-safe 写成「FMU idle 天然掉 0.1」。在证实那 14 小时的前缀动作可复现之前，禁止用 0.12 裕度掩盖。下一步需要 episode 13 / step 125 的动作前缀 replay（当前 run 只有 eval 轨迹，没有训练逐步 `u_caes`）。summary 现写入 `fmu_sha256` / `guid` / `model_description_hash` / `git_commit`。

## 2026-09-01：hash lock、read-back、完整 prefix（未改 Oracle）

未改 idle residual、未加 previous-mode guard、未改 TD3/reward。D 三 seed 已确认同一 0831 binary（`31c8fec7…` / `{9b0edb50-…}`）。

| 项 | 现状 |
|----|------|
| FMU cache | `resolve_fmu_path` 按 SHA256 复制；`env_config.expected_sha256` / `expected_guid` fail-fast |
| `git_commit` | 可读 `OPTIMAL_DEMO_GIT_COMMIT`；另写 `source_manifest` |
| input read-back | `FmuSession.set_inputs` 后 `getFloat64`；`last_input_readback` |
| 内部 Mdot | **0831 无独立 VR**。`Mdot_c1/c2`、`coldtank.port_*.m_flow` 不在 95 个变量里。`mflow_*.x3` 与 `t_air_in` 同 VR（读到的是 262.4 K，不是质量流） |
| prefix 日志 | `trajectories/executed_actions.jsonl`：GiveSafe 通过后送入主 FMU 的 physical action + read-back + cold/p_caes before/after。失败记录 `extra.action_prefix` 含整集 |
| C2 130 条 | 128 idle `caes_cold_soc_low` + 2 `caes_temperature_high`（s0 idle 一条、s2 放电一条）。无 unknown |
| 低 SOC 矩阵 | 从 SOC=0.5 充放 400 h 冷罐下限约 **0.40**，到不了 0.10–0.25。该下限再充 8 h 后 idle：`u_rb=0`，Δcold **+0.003**，Mdot=null。0.08–0.17 依赖更长 episode 前缀 |

下一步：用新 prefix 日志 exact replay `s1 ep10/step153`；需要内部流量时必须 **只加只读诊断输出、不改方程** 的 diagnostic FMU。C5 独立，不在修 C2 时动 reward。C1–C5 全过前不开 3×400k。

## 2026-09-01 Modelica（`D:\Code\0622\m_resources`）

- `mflow_coldtank` 在 T=280.65 K、P=140 MW、p=8.58 MPa 有一处 `+152.876`，热罐对应点为 `+152.876`，其余 44/45 点满足冷=−热。库默认表已改为 `-152.876`。
- **8760h 实例修饰仍保留 +152.876**，并新增 `diag_*` 只读口（Mdot command、tank port flow、mass/SOC、`u_dispatch`）。这是 0831-equivalent diagnostic 导出源。
- `Tank` 越界冻结 **未改**。8760h 冷罐 `V0=60000`，150 kg/s×1 h 只对应 ΔSOC≈0.009，不能把 C2 写成充电流量延迟一小时。
- 未改 Oracle。未开训。导出 diagnostic FMU 后用 `DIAGNOSTIC_OUTPUTS` 读内部流。

## 2026-09-01 0901 diagnostic FMU

`data/0901PowerSystem_8760h.fmu`（2026-09-01T09:53:26Z，guid `{12476586-6213-4d49-9d02-b1e50c4abfc3}`，sha256 `ccfbd76d…`，131 vars / 11 `diag_*` 独立 VR，不与 `t_air_in` 共用）。0831 本地副本 `data/0831TypicalScensrio_Example_TypicalScene_PowerSystem_8760h.fmu` 仍是 `31c8fec7…` / `{9b0edb50-…}` / 95 vars。

`DEFAULT_OUTPUTS` 与 0831 逐步对齐（idle 5 h、t=0/t=6652800 充 8 h+idle 3 h、以及 T=280.65 K 的 u=0.933/0.93 充 4 h+idle 3 h）：**0 个 mismatch**。0901 是 0831-equivalent diagnostic 导出。

四量判定（通信点读数）：

| 场景 | `u` | Mdot c1,c2 | net | Δcold | 判定 |
|------|-----|-----------|-----|-------|------|
| 初值 idle 5 h | 0 | 0,0 | 0 | 0 | idle 方程成立 |
| 默认 T=262.4 充 8 h | 0.93 | 0, −114→−147 | =c2 | −0.006～−0.0085 | 充电抽冷罐，符号正确 |
| 随后 idle 第 1 h | 0 | 0,0 | 0 | **−0.0045**（质量同步掉） | 通信点流量已 0，积分仍走了约半小时旧流 |
| 随后 idle 第 2–3 h | 0 | 0,0 | 0 | 0 | 稳定 |
| T=280.65、u=140/150 第 1 h | 0.933 | 0, **+109.6** | +109.6 | **+0.0072** | `+152.876` 表错误活着：充电往冷罐打质量 |
| 该前缀后 idle 第 1 h | 0 | 0,0 | 0 | **−0.0454**（Δm≈−2.72e6 kg） | 仍是「u=0 且采样流=0 但 SOC/质量在动」；量级大于默认 T 的 0.0045，仍小于 D 中 0.08–0.17 |

冷罐 `mass/SOC=6e7` kg、`level/SOC=60` m → 与实例 `V0=60000`、`A=1000` 一致。

D s1 目录没有 `executed_actions.jsonl`，`safety_dataset.json` 也没有 `action_prefix`。还不能对 C2 失败步做 exact-prefix replay。未改 Oracle，未开训。

## 2026-09-01 通信步敏感性（最小测例）

0901/0831 的 `CoSimulation` 相同：`hasEventMode=false`，`canHandleVariableCommunicationStepSize=false`，`fixedInternalStepSize=360`，`canInterpolateInputs=false`。`FmuSession` 调用 `FMU3Slave.instantiate()` 时 `eventModeUsed` 默认 **False**（且 FMU 不允许开 Event Mode）。

doStep 通信步必须被 360 整除。检查表里的 900/300/72/60 **全部被 FMU 拒绝**（`communication step size N was not divisible by 360`）。

同一 4 h 充电 → 1 h idle 前缀：

| comm | T=280.65 u=0.933 idle ΔSOC | 小时内 max\|net\| | 小时末 net |
|------|----------------------------|-------------------|------------|
| 3600 | **−0.045392** | 0（只看到终点） | 0 |
| 1800 / 720 / 360 | **−0.045392（完全相同）** | 837 / 1093 / **1174** | 0 |
| 900, 300, 72, 60 | 无法运行 | — | — |

T=262.4、u=0.93 同样：3600 与 360 的小时 ΔSOC 都是 **−0.031927**。缩短通信步**不能**消掉小时质量损失。

360 s 分辨率下第一小时 idle 的内部过程（T=280.65）：

- 全程 `u_dispatch=0`、`p_caes=0`
- 前 9 个内部步 `diag_cold_mflow_c2` **精确等于** 3D 表在 **P=0 外推** 的值（与 `+152.876` 造成的陡坡一致，−462→−1174 kg/s）
- 第 10 步代数流量才跳到 0（else 分支终于生效）
- 质量积分相对代数流量 **滞后一个 360 s**：`Δm_k = net_{k-1}·360`
- 从未充过电的 idle-from-init：net 全程 0，表外推不启用

因此：charge→idle 后掉 SOC **不是**「3600 s 步长太大、改成 300 s×12 就会消失」。本 FMU 内部本来就是 360 s；3600 s 只是把内部非零流藏在通信点。根因是 **`if u_dispatch>0` 在无 Event Mode 下延迟约一个决策小时才切到 idle**，idle 期间仍走充电支路并在 P=0 外推流量表；`+152.876` 把外推放大到 10³ kg/s。`300 s×12` 在这份 FMU 上既不合法，也不能修小时 ΔSOC。

未改 Tank、未改 Oracle、未开训。本机没有 Sysplorer/omc CLI，原生 Modelica 对照仍缺。

## 2026-09-01 源码修复（待导出 0902）

- `TypicalScenarios.mo`：CAES `if u_dispatch>0` 改为 `noEvent(...)`，按当前 u 选支路。
- `PowerSystem_8760h.mo`：实例表 `+152.876` → `-152.876`。`diag_*` 保留。Tank 未改。
- `FmuSession`：`hasEventMode=true` 时 `eventModeUsed=True` 并在输入后做 Event Mode 迭代；通信步必须整除 `fixedInternalStepSize`。
- `env_config` 已锁 0902：sha256 `ecb1ecf2…c8f536`，guid `{61354d2d-25bb-44cd-a7e8-4e5525f8eec7}`。`hasEventMode` 仍为 false；`noEvent` 源码修复已生效。M1–M4 全过。未改 Oracle，未开训。

## 2026-09-01 0902 验收

`data/0902PowerSystem_8760h.fmu`（2026-09-01T11:26:13Z，131 vars，11 `diag_*` 独立 VR）。

| 门 | 结果 |
|----|------|
| M1 模式切换 | idle 第一拍 net=0；小时 ΔSOC=**−0.000906**（0901 为 −0.0454）。3600 s 与 360 s 相同 |
| M2 表符号 | T=280.65 充电 c2=**−153.2** kg/s（0901 为 +109） |
| M3 idle-from-init | 5 h ΔSOC=0、net=0 |
| M4 步长 | 300/60/900 拒绝 |

下一步是 **Oracle 按 0902 重标定**，然后 ≥20k stress、Stage C。旧 0831 的 C/D 作废。不开 3×400k。

## 2026-09-01 0902 Oracle 重标定

开环（T=280.65）：充电 c2=−153 kg/s，Δcold=−Δhot；idle 第一小时 net=0、Δcold=−0.0009；放电冷罐回灌。2000 有效步拟合 α（充 0.189/0.091/−0.091，放 0.270/0.131/−0.131，R²≥0.95），冷热精确反号。

`feasibility_margins.yaml` → `d5.4-0902-recalibrated`。开环补测后：气罐 idle 第一小时在 T_env=262.4 时 ΔSOC=−0.0237（SOC=p/p_norm，壁面换热，**无质量流**），不能用 0.008 丢掉尾巴，已改为 0.027。电池第一小时残差是同样的 360 s 斜率滞后（充电 0.18 vs 0.20），不是 E_cap 错误。电网开环 |resid| 64–86 MW。未开训。

## 2026-09-01 0902 20k stress

`runs/boundary_stress/`：20000 attempted，18479 Oracle 合法，**post-step fail = 0**，FMU fail = 0。`passed: true`。这只证明 **0902 + d5.4 Oracle** 在近界应力下无 false-safe，**不是**开训许可：源码 Tank 冻结仍在 0902 二进制里。

## 2026-09-01 源码（待导出 0903）

- `Tank`：去掉 `if noEvent(SOC∈(min,max))` 冻结。始终 `der(m)=port_a.m_flow+port_b.m_flow`，能量方程保持原 `m*der(h)=…`。越界走 assert，不把罐冻死。
- `Battery`：`if noEvent(PBS.P_act>=0)` 选充/放效率，避免 CS 无 Event Mode 时效率支路滞后。
- CAES `noEvent(u_dispatch)` 与冷罐表 −152.876 **保持**。`diag_*` 保持。
- **不要覆盖 0902**。导出到例如 `data/0903PowerSystem_8760h.fmu` 后再锁 SHA/GUID、重跑 M1–M4、视需要重标定 Oracle、再 20k、再 Stage C。未开 3×400k。

## 2026-09-01 0903 验收

`data/0903PowerSystem_8760h.fmu`（2026-09-01T12:59:52Z，sha256 `e28c6753…c05cbb`，guid `{6b3357ac-171b-413e-85d5-7318a860a18d}`，131 vars，11 `diag_*`）。`hasEventMode` 仍为 false，内部步 360 s。

| 门 | 结果 |
|----|------|
| M1 模式切换 | idle 第一拍 net=0；小时 ΔSOC=**−0.000906**（与 0902 同量级） |
| M2 表符号 | T=280.65 充电 c2=**−153.2** kg/s |
| M3 idle-from-init | 5 h ΔSOC=0、net=0 |
| M4 步长 | 300/60/900 拒绝 |

开环：电池第一小时仍 0.18 vs 0.20（欧拉 360 s 混斜率；`noEvent` 没有消掉这条滞后）。idle 气罐 T=262.4 第一小时 ΔSOC=−0.0237，质量流=0。冷热精确反号。

`env_config` 已锁 0903。`feasibility_margins.yaml` → `d5.5-0903-recalibrated`（α 充 0.191/0.091/−0.091，放 0.274/0.131/−0.131）。idle 气罐残差仍用开环 0.027，不用随机策略 P99=0.013。

20k 近界应力：20000 attempted，18444 Oracle 合法，**post-step fail = 0**，FMU fail = 0。`passed: true`。未开 3×400k。

## 2026-09-01 Stage C 0903（C2 两条 false-safe）

30k 训完：C1/C3/C4/C5 **过**（greedy 成本 −1.81e7 < random −1.34e7，168 h 完整周）。C2 失败：训练 2 次 `caes_pressure_high`。都是 **充电（u≈1）切最弱放电（u=−0.33）**，当时 p=9.48–9.49 MPa；Oracle 预测降到 9.39，FMU 因 360 s 滞后 + 气罐升温升到 9.51。评估周无 FMU 失败。

`d5.6` 把 idle 高压侧放到硬界后 **更差**：943 次待机顶压 + 评估第 13 步 9.53 MPa。正确修法是 **别冲进 9.11–9.50 死区**：`d5.7-0903-charge-headroom` 在 ≈8.89 MPa 停充（idle envelope 0.39 + 一步充电 0.22），放电仍在 >9.38 禁止，idle 高压侧恢复 270 kPa 冷天尾巴。

`d5.7` Stage C **全过**：C1–C5 true，false-safe=0，评估 168 h，greedy −1.77e7 < random −1.28e7。Stage D 3×400k 已排队（s0 进行中，随后 s1、s2）。
