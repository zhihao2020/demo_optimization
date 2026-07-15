# Shadow FMU 验证

## 能力探测

对当前 FMU（`TypicalScensrio_...PowerSystem_8760h.fmu`）：

| 能力 | 值 |
|------|----|
| `canGetAndSetFMUstate` | **False** |
| `canSerializeFMUstate` | **False** |

因此 **不能** 使用 get/set FMU state 的快照回滚。

## 采用策略：同步 Shadow + `reset_and_replay` 恢复

1. 每个 episode 新建独立 Shadow `FmuAdapter` 并 reset；
2. 安全候选先在 Shadow 推进一步；主 FMU 确认成功后，该 Shadow 状态即与主 FMU同步；
3. Shadow 拒绝、异常或主 FMU未确认候选时，丢弃该实例；下次验证才 reset 到 episode 起点并重放已确认历史；
4. 主 FMU从不由 Shadow 推进或回滚。

正常路径每候选 O(1)；只有失步恢复才是 O(episode_step)。Oracle 校准改善后可将 `shadow_validation.mode` 改为 `near_boundary`。

## 模式

- `always`（默认）：一级通过后一律 Shadow
- `near_boundary`：仅当到安全边界距离小于阈值
- `disabled`：仅一级 Oracle（不推荐用于当前 false-safe 阶段）

## 不是 Fallback

Shadow 失败 → GiveSafe 拒绝样本 + 重采样；**绝不**用 Shadow 结果改写策略动作或推进主仿真。
