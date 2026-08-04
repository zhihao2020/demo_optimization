# Safe Market-GHTD3：论文对齐 + 本仓库创新

对应主锚论文：*Collaborative scheduling optimization of hydrogen-enhanced integrated energy system via goal-conditioned hierarchical reinforcement learning*（Energy, **GHTD3 / Cui et al.**）。

并吸收：

| 文献思想 | 本仓库落地 |
|----------|------------|
| Cui GHTD3：高层 SoC 增量 goal + 底层设备动作 + goal 转移 / 内在奖励 | `src/training/ghtd3/` |
| Ochoa 等：多时间尺度 DA 计划 + RT 执行 | 高层每 \(c\) 步能量目标，底层 GiveSafe 执行 |
| Pei 等：price-taker 电价驱动储能 | 外生分时电价 + 市场条件 goal 先验 |
| Sun / 约束 RL + 本仓库 GiveSafe | 底层禁止 unsafe fallback |

市场深度：**price-taker 分时电价**（山东 2026 代理购电代理表），**不做 ISO 出清**。

---

## 1. 与论文的一一映射（对齐）

| 论文 | 本仓库实现 |
|------|------------|
| 高层 SMDP 动作 = 储能 SoC 增量 goal | `goal = [Δbattery_soc, Δcaes_gas_soc]` |
| 底层 GAMDP 动作 = 设备功率 | Hybrid：`u_tp, u_battery, caes_mode, caes_magnitude` |
| 子目标间隔 \(c=8\) | `ghtd3_config.yaml` → `subgoal_interval: 8` |
| goal 转移 \(g'=s+g-s'\) | `goals.goal_transition` |
| \(r^{int}=-\|soc+g-soc'\|+\alpha r^{ext}\) | `intrinsic_reward`，`intrinsic_alpha≈0.35` |
| 内外层 TD3 | `GHTD3Agent` 双 Actor-Critic |
| 高层 SMDP 折扣 \(\gamma^c\) | `agent.gamma_high = gamma ** c`（已严格使用） |
| Historical goal relabel | 实际 \(\Delta SoC\) 候选近似（`_relabel_goals`） |
| 安全执行 | 底层 **GiveSafe**（Oracle，默认无 Shadow） |

---

## 2. 本文创新点（相对原文 / 单层 Hybrid）

### I1. 市场条件高层目标先验（Market-conditioned goal prior）

- 低买价 → 正 \(\Delta SoC\)（充）；高买价 → 负 \(\Delta SoC\)（放）。
- 与高层 actor 输出 **凸组合**（`market_prior_weight`），训练与评估一致使用。
- 实现：`goals.market_conditioned_goal_prior` + `blend_goal_with_prior`。

### I2. 回收段目标（Terminal SOC recovery goals）

- 与 env 侧 `market.soc_recovery_horizon` 对齐。
- 剩余步数 ≤ `recovery_goal_horizon_steps` 时，上层强制 prior 指向 **初始 SoC**，权重抬升到 ≥0.75。
- 环境层仍可对动作做 `_apply_terminal_soc_recovery`，形成 **上层计划 + 底层扭矩** 双保险。

### I3. 严格 SMDP 折扣 \(\gamma^c\) + 高层 reward 均值归一

- 原文高层跨 \(c\) 步，目标应使用 \(\gamma^c\)；本实现 `update_high` 已用 `gamma_high`。
- 高层 reward 默认存 **周期内外在奖励均值**（`high_reward_normalize`），避免 sum 尺度炸 critic；配合 `q_clip_high`、grad clip。

### I4. 峰谷规则冷启动（Price-aware bootstrap）

- 早期探索不用纯 idle / 通用 rule，而用 `PriceAwareRuleController`，加速套利语义进入 buffer。

### I4b. 分层 BC 预热（Hierarchical BC）

- 采集峰谷规则轨迹；**BC goal = 0.5·实际ΔSoC + 0.5·市场 prior**。
- 底层 `LowLevelActor` 监督拟合规则动作；高层 `HighLevelActor` 拟合演示 goal。
- 对齐 Hybrid「BC→RL」成功路径，并适配分层结构（见 `bc_pretrain.py`）。

### I5. GiveSafe + 非凸 CAES 混合动作空间

- 底层在 Modelica/FMU 可行域 + Oracle 安全门上执行；相对氢能论文场景，本仓库对象是 **光热-电池-CAES 多能** 与 **分时购电**。

### I6. 预测价观测 / 实现价结算（可选）

- `obs_price_path`（如 BiLSTM `price_predicted.csv`）与 `price_path`（实现价）分离，对齐“预报—结算”实验范式。

---

## 3. 代码入口

```bash
# smoke ~3k
python scripts/train_ghtd3.py --mode smoke --seed 0

# short 15k（与 Hybrid market_soc_ok_15k 对齐步数）
python scripts/train_ghtd3.py --mode short --steps 15000 --run-dir runs/ghtd3_market_15k_20260803

# 全年评估
python scripts/train_ghtd3.py --mode short --steps 15000 --run-dir runs/ghtd3_market_15k_ann --annual-eval
```

配置：`src/config/ghtd3_config.yaml`  
核心包：`src/training/ghtd3/`  
环境市场与 SOC 回收：`src/config/env_config.yaml` → `market.*`

---

## 4. 与单层 Hybrid-TD3 的差异

1. **高层每 8 步**提出储能目标，减轻底层短视 SOC 调度。  
2. **底层观测条件于 goal**，内在奖励强迫跟踪。  
3. **市场先验 + 回收 goal** 把电价与期末约束显式注入 SMDP。  
4. 外在奖励仍用 `RewardCalculator`（现金流 + SOC shaping/bonus）。  
5. 评估同时对比：通用规则 / 峰谷规则 / GHTD3。

---

## 5. 论文可写的方法叙事（建议）

1. **系统**：Modelica 多能 + FMU 黑盒约束。  
2. **市场 MDP**：price-taker TOU；观测可含预测价；reward 用实现价结算；终端 SOC 回收。  
3. **学习**：Safe Market-GHTD3 = GHTD3 + \(\gamma^c\) + market/recovery goal prior + GiveSafe。  
4. **基线**：Rule / Price-aware rule / Hybrid GiveSafe-TD3（BC+RL）/ 本文 GHTD3。  
5. **指标**：周 reward、经济现金流、terminal SOC 通过率、全年窗口通过率、硬约束违规率。

---

## 6. 当前结果（2026-08-03）

| 方法 | 周 reward | 火电 MWh | 周 SOC | 全年 SOC |
|------|-----------|----------|--------|----------|
| 规则 | 67.6 | ~25200 | 是 | — |
| **Safe Market-GHTD3 50k** | **128.4** | **8958** | **是** | **16/53** |
| Hybrid BC+RL+回收 | 130.4 | 8575 | 是 | 19/53 |

最优 run：`runs/ghtd3_market_50k_annual_20260803/`  
消融表：`docs/GHTD3消融实验结果.md`

回收改进：只修 battery+gas → 到位 IDLE；禁止热/冷罐振荡式开关机。

## 7. 已知限制与后续

- Goal relabel 为 \(\Delta SoC\) 高斯近似，非原文完整 MLE 优化。  
- CAES 热/冷罐未进 goal 维（仅 battery + gas）。  
- 高层 online TD3 仍易 Q 发散；有效策略主要靠 **BC + 市场 prior + 底层 + 环境硬回收**。  
- 全年 SOC 16/53，可加长训 / 更强跨周衔接。  
- 消融中 `c=1` 与「仅改 γ」不完全等价（同时改变 goal 频率）。