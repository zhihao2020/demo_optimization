# 硬约束设计（GiveSafe 更新）

## 原则

1. Forbidden area / SOC / 爬坡 / 电网容量 = **硬约束**，不进经济 reward。  
2. 主 FMU 仅在 GiveSafe 两级检查通过后执行。  
3. 禁止环境 clip / 投影 / 替换 / 规则 fallback。  
4. 后验硬约束违反 → 不进 PhysicalReplay；记 GiveSafe / SafetyDataset。

## 两级检查

1. **Oracle 解析**：静态与动态可行域、模式 mask、预测下一状态。  
2. **Shadow FMU**：`reset_and_replay`（当前无 FMI state）；见 `docs/shadow_fmu_validation.md`。

## Episode 时间

仅成功主 FMU 步推进 3600 s 并计入 `episode_valid_steps`。拒绝尝试不计 168。

## Phase E

见 `docs/givesafe_phase_e_gate.md`。默认 `formal_default_blocked=true`。
