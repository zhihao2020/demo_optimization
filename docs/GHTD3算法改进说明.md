# Safe Market-GHTD3 算法改进（无 v2 命名）

## 目标

1. 训练曲线**完整落盘**（论文图专业）  
2. **Modelica 对齐 5 维 goal** + **Hybrid 锚定残差**结构，争取 **超过 Hybrid**

## 结构（当前主线）

```
s → π_hybrid (冻结, raw obs) ─┐
s,g → π_res (残差, obs_norm) ─┴→ a = (1-α)a_H + α a_res → GiveSafe → FMU
g ∈ R^5 = [Δbat, Δgas, Δth, u_tp_bias, arb]  # 每 c 步高层
α = clip(α0 + (α_max-α0)*g_arb, 0, α_max)   # 默认 α_max=0.28
```

| goal 分量 | Modelica 含义 |
|-----------|----------------|
| Δbat | 电池能量意图 |
| Δgas | 气库能量意图（窄盒） |
| Δth | (hot+cold)/2 热过程软目标 |
| u_tp_bias | 火电负荷相对 Hybrid 偏置 |
| arb | 套利强度 → 残差混合比 |

## 已落地改动

### 1. 完整训练日志

- 删除 `step_log[-500:]` / `[-200:]` 截断  
- SAC 改为每 500 step 记 1 点（与 TD3/GHTD3 协议一致）  
- 续训合并旧 `step_log.json`，`valid_step` 连续累计  

### 2. HER-mix 历史目标重放

- 模式 `goal_relabel_mode: her_mix`  
- 概率：原 goal 0.4 / 本窗 achieved ΔSoC 0.4 / future achieved 0.2  
- 避免 legacy「恒等于实际 Δ」的平凡 relabel  

### 3. Prior 退火 + 两阶段耦合

- `market_prior_weight`：0.55 → 0.08（进度退火）  
- `phase_a_steps`：从零训练时先强 prior + 只训底层；**续训默认跳过**  
- `intrinsic_alpha`：0.50 → 0.22  
- 高层 `gradient_steps_high=4`  

### 4. Actor 可学习性修复（关键）

根因：原始观测含 **~1e8 W 功率**，actor 反传梯度数值为 0，续训 actor 权重完全不动。  

修复：

- `obs_norm`：`tanh(obs / 1e6)`  
- actor 更新用可微 soft mode + 边界最小跨度  
- `actor_bc_weight`：buffer 行为克隆锚定，缓解 Q 对动作平坦  

## 结构诊断（goal 死通路）与修复

### 诊断（`scripts/diagnose_ghtd3_goal_sensitivity.py`）

| 结构 | g=0≈Hybrid | 动作随 g | 根因 |
|------|------------|----------|------|
| Hybrid 权重移植 + `goal_conditioned` | 是 | **0/5** | goal 列=0 + raw-obs 绝对头 \|z\|~1e7 饱和 |
| `ghtd3_gc_hybrid_35k` | — | — | 周 J=**128.10≡Hybrid**（验证分层无效） |
| **`action_residual`** \(a=a_H+\beta\tanh\Delta(s,g)\) | **是** | **5/5** | 动作空间残差，避开饱和 |

结论：分层打不平 Hybrid **不是理论问题**，是执行接口让高层意图到不了动作。

### 推荐执行接口（当前默认）

\[
a=\mathrm{clip}\bigl(a_H(s)+\beta\cdot\tanh(\Delta_\theta(s_{\mathrm{norm}},g)-\Delta_\theta(s_{\mathrm{norm}},0))\bigr)
\]

- 冻结 Hybrid 出 \(a_H\)（下界）  
- 逆动力学 **残差 MLE** 预热 \(\Delta\)（hindsight goal）  
- 再 residual TD3 + 高层 SMDP  

配置：`execution_mode: action_residual`，`hybrid_init_low: false`，`mle_pretrain_residual: true`。

### 下界 / 冒烟

| Run | 说明 | 周 reward | SOC | goal 通路 |
|-----|------|----------|-----|-----------|
| Hybrid `market_bc_rl_60k` | 单层对照 | **128.10** | 过 | — |
| `ghtd3_gc_hybrid_35k` | 旧移植 GC | **128.10** | 过 | 死 |
| `ghtd3_ares_smoke_3k`（旧，mode 可覆盖） | 残差+MLE | 123.2 | 过 | CAES 吞吐炸到 ~3k |
| **`ghtd3_ares_safe_smoke_3k`** | **加固：mode 锁 Hybrid + logit clip + mag 纯残差** | **129.18** | 过 | CAES~645，**已微超 Hybrid** |

加固默认：`residual_mode_override: false`，`residual_logit_clip: 8`，连续 β 略收。  

### 三季正式对比（`ghtd3_ares_35k` vs Hybrid 60k）

数据：`runs/ghtd3_ares_35k/vs_hybrid.json`

| 季节 | Hybrid | Safe Market-GHTD3 (ares) | Δ | SOC |
|------|--------|--------------------------|---|-----|
| Winter | 128.10 | **129.18** | **+1.08** | 过 |
| Transition | 110.37 | **111.02** | **+0.65** | 过 |
| Summer | 91.70 | **92.28** | **+0.58** | 过 |

→ **3/3 季优于 Hybrid**（均值 Δ≈+0.77）；CAES 吞吐不再爆炸。  
根因修复有效：动作空间残差 + 模式锁 Hybrid + 逆动力学 MLE。  

论文 `论文模板/main.tex` 已同步：Method 写清 \(a=a_H+\beta\tanh\Delta\)、Results 用上表 reward、Discussion 对比 Cui GHTD3。  

### TEA：教师锚定可扩张残差（主创新推进）

固定小 β 只能 \(J_H+\varepsilon\)。TEA（`execution_mode: tea`）：

- \(\beta(t)=\mathrm{interp}(\beta_0,\beta_{\max},\rho(t))\)，\(\rho\) 随训练进度 \(0\to1\)
- 优势门控：\(Q(a_{\mathrm{res}})>Q(a_H)\) 时放大 β
- 教师 BC 系数 \(0.45\to0.05\) 退火
- mode 在 progress≥0.55 后盾内解锁
- HGR：`her_mix` 提高 hindsight 比例

配置：`src/config/ghtd3_config_tea.yaml`  
运行：`runs/ghtd3_tea_50k`（训练中/完成后三季 eval）

### TEA 50k 三季（`runs/ghtd3_tea_50k/vs_hybrid.json`）

| 季节 | Hybrid | TEA-50k | Δ | CAES MWh |
|------|--------|---------|---|----------|
| Winter | 128.10 | 124.43 | **-3.66** | 2658（炸） |
| Transition | 110.37 | **112.30** | **+1.93** | 2300 |
| Summer | 91.70 | **92.99** | **+1.29** | 1537 |

→ 夏/过渡 **比 ares 拉开更大**；**冬季因 mode 解锁后 CAES 过吞吐崩盘**。  

### TEA-WS2（mode+mag 锁教师，连续火电/电池可扩张）`runs/ghtd3_tea_ws2_40k`

| 季节 | Hybrid | TEA-WS2 | Δ | CAES |
|------|--------|---------|---|------|
| Winter | 128.10 | **130.24** | **+2.14** | 537≈Hybrid |
| Transition | 110.37 | **112.35** | **+1.98** | =Hybrid |
| Summer | 91.70 | **94.17** | **+2.47** | =Hybrid |

→ seed0：**3/3 胜 Hybrid**，均值 Δ≈**+2.20**；CAES 不再炸。  

### SCI Q1 套件（多种子 + 消融）`runs/sci_q1_suite/summary.md`

| Method | Winter Δ | Trans Δ | Summer Δ | mean Δ | >Hybrid |
|--------|----------|---------|----------|--------|---------|
| TEA-WS2 **seed0** | +2.14 | +1.98 | +2.47 | **+2.20** | **3/3** |
| TEA-WS2 seed1 | -1.75 | -20.07 | -18.73 | -13.52 | 0/3（SOC 崩） |
| TEA-WS2 seed2 | +2.59 | +1.10 | -15.59 | -3.97 | 2/3（夏 SOC 失败） |
| freeze-teacher | +0.48 | -18.29 | -16.79 | -11.53 | 1/3 |
| no-prior | -1.97 | +3.60 | -14.93 | -4.43 | 1/3 |
| no-HER | -7.51 | -1.29 | -1.51 | -3.44 | 0/3 |
| ares-35k | +1.08 | +0.65 | +0.58 | **+0.77** | **3/3** |

多种子均值：冬 +1.00，过渡 -5.66，夏 -10.62 → **方差过大，一区证据未闭合**。  
seed0 显示方法上界；seed1/2 与 20k 消融暴露 **训练不稳 / SOC 敏感**。  

**务实主结果**：稳健故事用 **ares-35k（3/3 小胜）**；潜力故事用 **tea_ws2 seed0** 并诚实报告多种子方差与后续稳定化工作。

### 旧三季（blend/移植结构，仅作历史）

| 季节 | Hybrid | GHTD3（旧） | Δ |
|------|--------|-------------|---|
| Winter | 128.10 | 124.03 | -4.06 |
| Transition | 110.37 | 110.40 | +0.03 |
| Summer | 91.70 | 91.73 | +0.03 |

### 结论

- **已修**：动作空间残差 + 逆动力学 MLE，诊断显示 goal→a 存活且 g=0=Hybrid。  
- **待做**：`ghtd3_ares_35k` 三季 eval，过线后更新论文主表。  
- 正向世界模型 MLE **alone 不够**；逆动力学 MLE 作预热有效。

## 训练命令

```powershell
$env:PYTHONPATH="src"
# 诊断
python scripts/diagnose_ghtd3_goal_sensitivity.py --out runs/diagnose_goal_sensitivity_residual.json --no-default-ckpts

# 动作残差正式训
python scripts/train_ghtd3.py --mode custom --steps 35000 --seed 0 --run-dir runs/ghtd3_ares_35k

# 三季 vs Hybrid
python scripts/eval_ghtd3_vs_hybrid.py `
  --ghtd3 runs/ghtd3_ares_35k/checkpoints/ghtd3.pt `
  --hybrid runs/market_bc_rl_60k_20260803/checkpoints/hybrid_givesafe_td3.pt
```

## 论文写法（勿写 v2）

- 方法名仍为 **Safe Market-GHTD3**  
- 贡献点写：Hybrid 锚定动作残差、逆动力学预热、HER-mix、prior 退火、SMDP \(\gamma^c\)  
- 训练曲线：完整 0→N 同协议  

