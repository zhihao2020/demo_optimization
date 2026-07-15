# FMU + Gymnasium + RL 实现报告（总览）

> **2026-07-15**：正式训练算法已切换为 **Hybrid-GiveSafe-TD3**。见 `reports/hybrid_rl_implementation_report.md` 与 `docs/givesafe_design.md`。普通 Box TD3 为 legacy；无 fallback。

## 1. 状态与结论（第一阶段遗留摘要）

已实现并测试：FMI 生命周期适配、变量注册、严格动作检查、物理量 observation、Python 单步 reward、Gymnasium 环境、规则策略、以及后续 Hybrid 动作 / GiveSafe。

未实现：GHTD3、分层控制、目标重标记、任何状态/奖励 `VecNormalize`。没有修改 Modelica，也没有在 Python 侧裁剪或投影动作。

关键接口事实与任务描述中的“输入为真实 W”假设不同：当前 FMU 的三个输入是 **FMU 原生无量纲调度指令**（单位 `1`），而不是 W；它们原样进入 FMU，Python 没有做 W/标幺转换或归一化。输出才是 W、Pa、K 和 SOC 等物理量。

## 2. FMU 与现有工程审查

- FMU：`data/TypicalScensrio_Example_TypicalScene_PowerSystem_8760h.fmu`
- 模型：`TypicalScensrio.Example.TypicalScene.PowerSystem_8760h`
- FMI：3.0 Co-Simulation；默认 `start=0`、`stop=31,536,000 s`、`step=3600 s`。
- 原有执行入口：`src/fmu/session.py` 的 `FmuSession`，负责 extract/instantiate/initialize/set/doStep/read/terminate/freeInstance；`src/fmu/validate.py` 已有输入和输出安全检查。
- 模型接口来源：`resources/Example/TypicalScene/PowerSystem_8760h.mo`。FMU metadata 没有给 input min/max，边界从同一实例 Modelica 参数与既有 `docs/fmu_input_bounds.md` 显式注册，未使用任意默认边界。

### FMU 输入表

| Action | FMU variable | Unit | Minimum | Maximum | Bound source |
|---|---|---:|---:|---:|---|
| u_tp | u_tp | 1 | 1/3 | 1 | thermal `P_min=50 MW` / `P_cap=150 MW` |
| u_battery | u_battery | 1 | -1 | 1 | 既有电池额定调度约束 |
| u_caes | u_caes | 1 | -1 | 1 | CAES 启机约束；合法集合 `[-1,-0.33] ∪ {0} ∪ [0.86,1]` |

`u_battery/u_caes` 正值为充电、负值为放电。`u_caes` 的中间区间虽落在 Box 外包络内，但会由 `ActionValidator` 立即抛 `ActionConstraintError`；不执行 FMU step。

### FMU 输出表

| Output | Unit | 说明 |
|---|---:|---|
| p_curtailment, p_unserved | W | 非负弃电、缺供 |
| battery_soc, caes_gas_soc, caes_hot_soc, caes_cold_soc | 1 | SOC |
| p_thermal, p_battery, p_caes, p_grid | W | 实际设备/联络线功率 |
| p_wind_available, p_wind_actual, p_pv_available, p_pv_actual, p_load_actual | W | 风光/负荷 |
| caes_gas_pressure | Pa | 气罐绝对压力 |
| caes_gas_temperature, caes_hot_temperature, caes_cold_temperature | K | CAES 热力状态 |

## 3. action_space 与 observation_space

`action_space` 是原生指令的 `LegalActionBox`（`gym.spaces.Box` 子类），下/上界为 `[1/3,-1,-1]` / `[1,1,1]`。它的 `sample()` 只采样 CAES 合法带，以便 Gymnasium 检查和随机基线不产生**已知**非法点；这不会重写 TD3 或调用者给出的动作。`step()` 记录的 `requested_action` 和 `applied_action` 在成功时完全相等。

固定 observation 顺序如下；未确认硬边界的功率/热力量在 Box 中明确使用 `±inf`，运行时仍拒绝 NaN/Inf。没有市场时序，故 `market_price` 没有伪造为 0，也不纳入 observation。

| Index | State | Source | Unit | Low | High |
|---:|---|---|---:|---:|---:|
| 0-3 | battery_soc, caes_gas_soc, caes_hot_soc, caes_cold_soc | FMU | 1 | 0 | 1 |
| 4 | caes_gas_pressure | FMU | Pa | 0 | inf |
| 5-7 | caes_gas_temperature, caes_hot_temperature, caes_cold_temperature | FMU | K | 0 | inf |
| 8-16 | p_thermal, p_battery, p_caes, p_grid, p_wind_available, p_wind_actual, p_pv_available, p_pv_actual, p_load_actual | FMU | W | -inf | inf |
| 17-18 | p_curtailment, p_unserved | FMU | W | 0 | inf |

## 4. reward

每个决策周期（当前 3600 s）在 Python 侧计算：

`r_t = -C_t`

`C_t = C_grid + C_thermal + C_battery + C_caes + C_curtailment + C_unserved + C_ramp + C_terminal + C_solver_failure`。

所有 W 到 MWh 的结算均乘 `1e-6 * dt_hours`；火电发电功率在成本中取绝对值，`p_grid>0` 购电、`p_grid<0` 售电。每步 `info["reward_terms"]` 含全部分项、`total_cost` 和 `reward`，且测试保证 `reward == -total_cost`。

`src/config/reward_config.yaml` 的市场、火电、吞吐、爬坡和终端 SOC 参数目前均为 `null`，代码显式告警并按 0 占位；仅 `solver_failure_penalty=1e9` 用于训练安全截断。这不是可用于经济比较的参数集。

## 5. 符号与功率不平衡核验

真实 FMU 单步工况确认：

- 大量净出力尝试 `[1,1,1]`：`p_grid=-303.66 MW`（售电）。
- 大量不足尝试 `[1/3,-1,-1]`：`p_grid=+96.34 MW`（购电）。
- 因此 `p_grid` 符号与模型注释一致；`p_thermal` 为负发电，电池/CAES 正充负放。
- `p_curtailment=max(-bus.P_res,0)`、`p_unserved=max(bus.P_res,0)` 在 Modelica 源中存在，两个输出运行时均非负。

上述 FMU 的电网接口会自动吸收残差，三个极端尝试中 `p_curtailment/p_unserved` 都为零（仅出现约 `3e-8 W` 的数值噪声）。因此无法在这个 FMU 配置下构造“只有非零缺供”与“只有非零弃电”的独立验收工况；这不是通过零填充绕过，而是当前无限电网结构的限制。对应脚本为 `scripts/verify_balance_signs.py`。

## 6. reset、测试与规则闭环

- 实际 FMU 连续两次 `reset()` 的 19 项输出最大绝对差为 **0**。
- `pytest -q tests`：**26 passed**。覆盖动作边界/NaN/Inf/形状/CAES 禁带、观测 dtype/不归一化、reward 审计、生命周期 mock、真实 FMU registry、rollout 与 `gymnasium.utils.env_checker.check_env`。
- `check_env` 通过；它对非对称原始 Box 和无穷 observation 边界给出 warning，这两点是本实现明确保留的真实接口信息。
- 规则策略 `[u_tp,u_battery,u_caes]=[1,0,0]` 完成 **168** 实际 FMU 步，FMU 失败 0、动作越界 0、NaN/Inf 0。轨迹：`runs/rule_controller/trajectories/rollout.csv`。

该规则轨迹的 reward/成本均为 0，原因是上节所列经济参数缺失，不是零成本的物理或经济结论。

## 7. TD3 与评估状态

训练入口：`src/training/train_td3.py`，配置采用题设 TD3 默认值并提供 checkpoint、TensorBoard、配置快照和 `summary.json`。当前补齐了 Python 环境的 `tensorboard` 依赖后，**32-step TD3 smoke 已完成**，模型在 `runs/td3_smoke/model.zip`。

这 32 步小于 `learning_starts=5000`，因此没有有效 TD3 更新，绝不能称为正式训练或训练结果；该 smoke 期间记录到 2 次 FMU 物理输出失败，已写入 `runs/td3_smoke/train/step_log.csv`。

SB3 `TD3Policy.squash_output=True`：库内部将 actor 的有界输出映射到 Box 的原生范围；`BasePolicy.predict()` 随后把动作返回为 action_space 单位。实测环境收到并写 FMU 的预检值为 `[0.33333334, 1.0, 1.0]`，即 FMU 原始无量纲指令，而不是 `[-1,1]` 的内部值。环境本身未做缩放、clip 或投影。不过 CAES 合法集非凸而 TD3 actor 连续，且该库的内部缩放与“完全禁止 Actor 重新缩放”的严格要求存在框架层冲突；正式训练前必须选择一种经用户确认的合法动作参数化，而不能以环境静默修正来掩盖它。

确定性评估与随机合法基线结果：

| Policy | 已执行步数 | FMU failures | 动作越界 | 结果 |
|---|---:|---:|---:|---|
| rule | 168 | 0 | 0 | 完成；无经济参数，reward 0 |
| random legal | 16 | 1 | 0 | SOC 到 1.013… 后截断 |
| TD3 smoke model | 3 | 1 | 0 | SOC 到 1.08 后截断 |

随机和 TD3 的失败均写入 trajectory 的 `fmu_error`、最后有效状态和 `solver_failure_cost`，未崩溃、未裁剪。汇总及 CSV 在 `runs/baseline_comparison/`。由于随机合法动作已出现物理失败、TD3 smoke 也失败，且经济参数缺失，**没有启动正式长时间 TD3 训练**。

## 8. 当前限制与后续 GHTD3 接口建议

1. 需要提供有来源的买卖电价、燃料/火电曲线、储能退化/吞吐、弃电/缺供和终端 SOC 参数，才能形成经济 reward。
2. 当前无限电网使非零 `p_unserved/p_curtailment` 分支不可观测；如需检验，应导出/配置受限电网工况。
3. 非凸 CAES 合法集不能由普通连续 TD3 的单个 Box 直接自然表示。下一阶段应在**不改变 FMU 动作语义、不做静默投影**前提下，显式设计离散模式 + 连续幅值的混合动作接口，并先取得对此接口变化的确认。
4. GHTD3 可复用 `VariableRegistry`、`ActionValidator`、`ObservationBuilder`、`RewardCalculator`、轨迹格式与 FMU 故障边界；高/低层切换不应绕过 validator。
