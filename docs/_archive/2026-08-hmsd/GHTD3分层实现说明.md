<!-- ARCHIVED: superseded by FS-HSAC; see docs/paper_outline_and_figures.md -->

# Safe Market-GHTD3 / HMSD：论文对齐 + 本仓库创新

文档更新：2026-08-17 12:30 (+08:00)

对应主锚论文：*Collaborative scheduling optimization of hydrogen-enhanced integrated energy system via goal-conditioned hierarchical reinforcement learning*（Energy, **GHTD3 / Cui et al.**）。  
本仓库方法名：**HMSD**（Hierarchical Market-aware Safe Dispatch）= Safe Market-GHTD3 实现栈。

并吸收：

| 文献思想 | 本仓库落地 |
|----------|------------|
| Cui GHTD3：高层 SoC 增量 goal + 底层设备动作 + goal 转移 / 内在奖励 | `src/training/ghtd3/` |
| Ochoa 等：多时间尺度 DA 计划 + RT 执行 | 高层每 \(c\) 步能量目标，底层 GiveSafe 执行 |
| Pei 等：price-taker 电价驱动储能 | 外生分时电价 +（可选）市场条件 goal 先验 |
| Sun / 约束 RL + 本仓库 GiveSafe | 底层禁止 unsafe fallback |

市场深度：**price-taker 分时电价**（山东 2026 代理购电代理表），**不做 ISO 出清**。

**当前主线（代码真源）**：`execution_mode: goal_conditioned`，`goal_dim: 2`，连续 `u_caes`，`low_reward: ext`（公平对比与 flat TD3 同目标），`goal_relabel_mode: her_mix`；**无** Hybrid residual teacher / `residual_mle` / Hybrid-PPO。配置：`src/config/ghtd3_config.yaml`；季节公平：`docs/cui_seasonal_min_protocol.md`。

**Story A（2026-08-17）**：主线仍是二维 HMSD。\(J\) 增加联络线合同 \(C^{\mathrm{grid}}\)（±200 MW / 600 元/MWh）。`ghtd3_aligned.yaml` 夏天已证伪，不当主方法。禁止把高层写成 RR/LEB/SA。

```bash
python scripts/train_seasonal.py --method hmsd --season winter --episodes 200 --seed 0 --single-week
python scripts/train_seasonal.py --method hmsd --season winter --episodes 200 --seed 0 --single-week --lock-caes
```

---

## 1. 与论文的一一映射（对齐）

| 论文 | 本仓库实现 |
|------|------------|
| 高层 SMDP 动作 = 储能 SoC 增量 goal | `goal = [Δbattery_soc, Δcaes_gas_soc]`（`goal_dim: 2`） |
| 底层 GAMDP 动作 = 设备功率 | 物理三元组：`u_tp, u_battery, u_caes`（`caes_u`；mode 仅派生） |
| 子目标间隔 \(c=8\) | `ghtd3_config.yaml` → `subgoal_interval: 8` |
| goal 转移 \(g'=s+g-s'\) | `goals.goal_transition` |
| \(r^{int}=-\|soc+g-soc'\|+\alpha r^{ext}\) | `intrinsic_reward`（**消融**可用 `low_reward: intrinsic`；主线 fair 用 `ext`） |
| 内外层 TD3 | `GHTD3Agent` 双 Actor-Critic |
| 高层 SMDP 折扣 \(\gamma^c\) | `agent.gamma_high = gamma ** c`（已严格使用） |
| Historical goal relabel | HER-mix（`goal_relabel_mode: her_mix`） |
| 安全执行 | 底层 **GiveSafe**（Oracle，默认无 Shadow） |

---

## 2. 本文创新点（相对原文 / 单层 Hybrid）

### I1. 市场条件高层目标先验（Market-conditioned goal prior）

- 低买价 → 正 \(\Delta SoC\)（充）；高买价 → 负 \(\Delta SoC\)（放）。
- 与高层 actor 输出 **凸组合**（`market_prior_weight`），训练与评估一致使用。
- 实现：`goals.market_conditioned_goal_prior` + `blend_goal_with_prior`。
- **主线默认关闭**（`market_goal_prior: false`），作为可选模块；公平季节协议不依赖 prior 才能对齐 flat TD3。

### I2. 回收段（对齐崔文：电池软门）

- 电池运行盒子仍是 \([0.1, 0.9]\)（崔文式 (20)）。
- 周末回到初值：**只给固定加分，不过门为 0**（崔文式 (29)），`fail_penalty_l1=0`，电池与气库等权。
- **不劫持电池充放**。`soc_recovery_battery_horizon: 0`；气库仍可用 `soc_recovery_horizon`。
- 市场先验主线关闭时，上层也不会在末段把目标扭回初值。

### I3. 严格 SMDP 折扣 \(\gamma^c\) + 高层 reward 均值归一

- 原文高层跨 \(c\) 步，目标应使用 \(\gamma^c\)；本实现 `update_high` 已用 `gamma_high`。
- 高层 reward 默认存 **周期内外在奖励均值**（`high_reward_normalize`），避免 sum 尺度炸 critic；配合 `q_clip_high`、grad clip。

### I4. 峰谷规则冷启动（Price-aware bootstrap）

- 早期探索不用纯 idle / 通用 rule，而用 `PriceAwareRuleController`，加速套利语义进入 buffer。

### I4b. 分层 BC 预热（Hierarchical BC）

- 采集峰谷规则轨迹；**BC goal = 0.5·实际ΔSoC + 0.5·市场 prior**。
- 底层 `LowLevelActor` 监督拟合规则动作；高层 `HighLevelActor` 拟合演示 goal。
- 对齐 Hybrid「BC→RL」成功路径，并适配分层结构（见 `bc_pretrain.py`）。

### I5. GiveSafe + 非凸 CAES 连续动作

- 底层在 Modelica/FMU 可行域 + Oracle 安全门上执行；CAES 用连续 `u_caes` + 合法三段投影（`caes_u`），问题仍非凸。
- 相对氢能论文场景，本仓库对象是 **风光-火电-电池-CAES 多能** 与 **分时购电**。

### I6. 预测价观测 / 实现价结算（可选）

- `obs_price_path`（如 BiLSTM `price_predicted.csv`）与 `price_path`（实现价）分离，对齐“预报—结算”实验范式。

---

## 3. 代码入口

```bash
# smoke
python scripts/train_ghtd3.py --mode smoke --seed 0

# 公平季节对比（推荐论文表）
python scripts/train_seasonal.py --method hmsd --season winter --episodes 5000 --seed 0
python scripts/train_seasonal.py --method td3  --season winter --episodes 5000 --seed 0

# short / 全年评估（单配置实验）
python scripts/train_ghtd3.py --mode short --steps 15000 --run-dir runs/ghtd3_short
python scripts/train_ghtd3.py --mode short --steps 15000 --run-dir runs/ghtd3_short_ann --annual-eval
```

配置：`src/config/ghtd3_config.yaml`、`ghtd3_config_seasonal_min.yaml`  
历史变体：`src/config/legacy/`  
核心包：`src/training/ghtd3/`（**无** hybrid residual teacher）  
环境市场与 SOC 回收：`src/config/env_config.yaml` → `market.*`  
协议：`docs/cui_seasonal_min_protocol.md`

---

## 4. 与单层 Hybrid-TD3 的差异

1. **高层每 8 步**提出储能目标，减轻底层短视 SOC 调度。  
2. **底层观测条件于 goal**（`goal_conditioned`）。  
3. 公平对比时底层更新用 **`r_ext`**（与 flat TD3 同目标）；`intrinsic` 仅作消融。  
4. **HER-mix** 重放历史 goal；市场 prior / 回收 goal 为可选模块（主线 prior 默认关）。  
5. 外在奖励仍用 `RewardCalculator`（现金流 + SOC shaping/bonus）。  
6. 评估对比：规则 / 峰谷规则 / Hybrid-TD3 / HMSD（季节 held-out 周）。

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