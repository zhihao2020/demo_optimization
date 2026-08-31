# PC-HybridTD3 d5.3：endpoint snap 与 idle robust guard

文档更新：2026-08-31 18:40 (+08:00)

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

C 过门仍是：C1 NaN/Inf=0；C2 训练 hard=0 / main-FMU unsafe=0 / eval FMU fail=0；C3 held-out NoSafeAction=0；C4 168 h unserved≈0；C5 成本优于 random feasible。CAES 利用率不是门。
