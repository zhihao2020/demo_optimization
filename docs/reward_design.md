# Reward 设计说明

## 公式

合法**物理**步：

\[
r_t = -\frac{C^{\mathrm{sys}}_t}{C_{\mathrm{ref}}} + \mathbf{1}[t=T-1]\, b\, \mathbf{1}[e_{\mathrm{SOC}}\le \tau]
\]

其中 \(T=168\) **真实主 FMU 小时步**，\(C^{\mathrm{sys}}\) 为审计后的七项综合成本（见 `docs/cost_parameter_audit.md`）。

- 硬约束（CAES 禁区、SOC 运行界、火电硬爬坡、电网容量）**不**进入经济 reward。
- FMU/FMI 失败**不**进入经济 reward。
- GiveSafe 拒绝动作使用独立 **constraint_reward**（见 `docs/givesafe_design.md`），**不得**计入货币成本或触发终端 SOC。

## GiveSafe 约束奖励

\[
r^{\mathrm{constraint}} = -\!\Bigl(\mathrm{base} + \sum_j w_j v_j^2\Bigr)
\]

禁止 `constraint_reward = -1e9`。自环样本：`economic_reward = terminal_soc_bonus = 0`。

## Critic 动作表达

锁定：`observation + u_tp + u_battery + caes_mode one-hot(3) + caes_magnitude`。

## 终端 SOC

- 默认 `binary_bonus`；仅 **physical_transition_count == 168** 后可能非零。
- GiveSafe 候选拒绝次数**不**计入 168。

## C_ref

当前季节标定值见 `src/config/reward_config.yaml`（约 156539.84 元/步）；smoke 可用，正式训练前确认代表性周覆盖。
