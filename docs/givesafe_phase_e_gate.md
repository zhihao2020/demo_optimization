# GiveSafe 下的 Phase E 门控

采用 GiveSafe 后，**不要求**训练期 `givesafe_rejection_count == 0`（早期拒绝正常）。

## 必须为零 / 达标

1. 非法动作进入主 FMU 次数 = 0  
2. `main_fmu_unsafe_execution_count = 0`  
3. PhysicalReplay 无无效物理转移  
4. `post_step_hard_constraint_violation_count = 0`  
5. Shadow 拒绝不得进入主 FMU  
6. `NoSafeActionFound` 率 < 配置阈值  
7. 确定性策略拒绝率接近 0  
8. 确定性评估完整 168 物理步  
9. CAES 不永久 idle  
10. Actor/Critic 无 NaN/Inf  
11. 采样比例符合配置  
12. 终端 SOC 仅在 168 物理步后触发  

## 最关键

`post_step_hard_constraint_violation_count == 0`

`proposal_rejection_rate` 应下降，但不要求为 0。

## 当前入口状态

`src/config/givesafe_config.yaml` 中 `formal_default_blocked: false`，正式训练入口可以启动。训练完成后仍必须满足以上门控，才可声称正式经济训练通过。
