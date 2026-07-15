# Hybrid-GiveSafe-TD3 设计

## 为什么不用 Fallback

Fallback（规则替换、强制 idle、最近投影、clip）会：

1. 破坏策略梯度/价值学习对真实候选动作的信用分配；
2. 掩盖 Oracle / Shadow 误判；
3. 让 Actor 依赖环境“收拾残局”，无法学会安全边界。

GiveSafe 只做：**拒绝不安全候选 → 同状态自环样本 → 重新采样 → 仅安全动作进入主 FMU**。

## 候选循环

```text
s_t → policy.sample → 一级Oracle → (可选)Shadow FMU
    → 拒绝: (s,a,r_c,s) 写入 GiveSafeReplay；时间/状态不变
    → 通过: 主FMU → (s,a,r_e,s') 写入 PhysicalReplay
    → 达 max_attempts: NoSafeActionFound，截断，无伪造物理步
```

## 自环样本

\[
(s_t, a_t^{\mathrm{unsafe}}, r_t^{\mathrm{constraint}}, s_t),\quad
\mathrm{done}=\mathrm{False},\quad
\mathrm{transition\_type}=\texttt{givesafe\_rejection}
\]

## 约束奖励 vs 经济奖励

- 物理：$r_e = -C_{\mathrm{sys}}/C_{\mathrm{ref}} + \mathbf{1}_{T}\cdot b_{\mathrm{SOC}}$
- 拒绝：$r_c = -(\mathrm{base} + \sum_j w_j v_j^2)$；**禁止** $-10^9$；**不计**货币成本与终端 SOC

## Replay

| 分区 | 内容 | 采样默认 |
|------|------|----------|
| PhysicalReplayPartition | 真实有效物理转移 | 70% |
| GiveSafeReplayPartition | 执行前拒绝自环 | 30% |

## Shadow FMU

当前 FMI `canGetAndSetFMUstate=False`。策略：独立实例 **reset + 重放本 episode 已执行物理动作 + 试探候选**，然后丢弃实例。主 FMU 永不被试探步推进。

## 配置

见 `src/config/givesafe_config.yaml`（`use_fallback: false` 强制）。
