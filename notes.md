# Notes: FMU 原始物理量 RL 第一阶段

## 审查记录

### FMU 元数据（2026-07-14 实测）
- `data/TypicalScensrio_Example_TypicalScene_PowerSystem_8760h.fmu`：FMI 3.0、Co-Simulation、默认步长 3600 s、停止时间 31,536,000 s。
- 3 个输入为 `u_tp/u_battery/u_caes`，单位均为 `1`，即 FMU 原生无量纲调度指令；不是 W。
- 19 个顶层输出可用，功率 W、压力 Pa、温度 K、SOC 为 1。
- 实测符号：`p_grid>0` 购电、`p_grid<0` 售电；`p_thermal<0` 发电；储能功率正充负放。

### 已运行验证
- 同一 `FmuSession` 连续 reset 的 19 个输出最大绝对差为 0。
- 规则策略 `[1, 0, 0]` 完成真实 FMU 168 步，无 FMU 失败和动作越界；由于所有经济费率 TODO，累计 reward=0。
- 随机合法动作在 16 步、TD3 32-step smoke 模型的确定性评估在 3 步分别因电池 SOC 越过 1 被截断；未裁剪动作。
- TD3 smoke 只运行 32 (< learning_starts=5000) 步，因此不是有效训练。正式训练未运行。
