# GiveSafe Replay 语义

## Physical

- `transition_type = physical`
- `physically_valid = True`
- `next_observation = s_{t+1}`（主 FMU 成功后）
- `reward = economic_reward`
- `constraint_reward = 0`
- 计入 `episode_valid_steps` / `simulation_time`

## GiveSafe rejection

- `transition_type = givesafe_rejection`
- `next_observation = observation`（自环）
- `terminated = truncated = False`（拒绝本身不截断；达上限时由 episode 外层截断）
- `reward = constraint_reward < 0`
- `economic_reward = terminal_soc_bonus = 0`
- **不**调用主 FMU；**不**增加 `physical_transition_count`

## False-safe（主 FMU 后验失败）

- 不进 PhysicalReplay
- 以执行前状态记入 GiveSafeReplay（自环 + constraint_reward）
- 写入 SafetyDataset
- 若无法证明主 FMU 已恢复 → episode 截断，实例不得继续用于该 episode

## 混合采样

`replay_sampling.physical_fraction` / `givesafe_fraction`（默认 0.7 / 0.3）。敏感性建议：0.8/0.2、0.7/0.3、0.5/0.5。
