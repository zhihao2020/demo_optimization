# Hybrid Action + Feasibility Oracle + Hybrid-GiveSafe-TD3 实现报告

日期：2026-07-14 / 更新：2026-07-15

## 1. 结论

已将训练闭环改造为 **Hybrid-GiveSafe-TD3**（仅 GiveSafe，**无** SafeFallback / 规则替换）：

- 不安全候选不进主 FMU、不推进时间、不改变状态；
- 形成 GiveSafe 自环约束样本并同状态重采样；
- 仅安全动作执行主 FMU 并写入 PhysicalReplay；
- Phase E 训练入口已启用；训练结果仍需通过全部硬门控。

## 2. GiveSafe 要点

详见 `docs/givesafe_design.md`、`docs/givesafe_replay_semantics.md`、`docs/shadow_fmu_validation.md`、`docs/givesafe_phase_e_gate.md`。

| 项 | 实现 |
|----|------|
| 控制器 | `src/safety/givesafe_controller.py` |
| 一级检查 | `GiveSafeConstraintChecker` |
| Shadow | `ShadowFmuValidator`（`canGetAndSetFMUstate=False` → reset+replay） |
| Replay | `HybridGiveSafeReplayBuffer`（70% physical / 30% givesafe） |
| 约束奖励 | 成形二次项，禁止 -1e9 |
| Fallback | **禁用**；`use_fallback: false` |

## 3. 与 190/1053 后验失败的关系

旧 smoke/short 中后验失败曾被完全丢弃，Actor 无法学习边界。GiveSafe 将执行前拒绝与 false-safe 记为**自环训练样本**（约束 reward），物理无效转移仍不进经济 PhysicalReplay。

## 4. Phase E

`formal_default_blocked=false`，正式入口可启动。最关键门控：`post_step_hard_constraint_violation_count == 0`；候选拒绝率**不要求**为 0。

## 5. 测试

`pytest tests/` 含 `tests/test_givesafe.py`（拒绝不调 FMU、自环、重采样、无 fallback、最大尝试、Shadow、混合采样、终端 SOC 时间语义）。

## 6. 运行

```bash
python scripts/train_hybrid_td3.py --mode smoke --steps 5000
python scripts/train_hybrid_td3.py --mode short --steps 20000
python scripts/train_hybrid_td3.py --mode formal   # 门控未过则 blocked
```

规则控制器仅独立评估 / C_ref 标定，**不**参与 GiveSafe 训练闭环。
