# 群智能 / 强化学习调用 FMU 与结果对比协议

---

## 1. 统一仿真接口（所有算法共用）

```
env.reset(start_time) → s_0
for t in 0 .. T-1:
    a_t ← Method.propose(s_t, context)     # 规则 / PSO / LP / RL
    a_t ← GiveSafe(Oracle, a_t)            # 可选关闭 = 消融
    if 无安全动作: 记录失败; break 或 idle 兜底
    s_{t+1}, r_t, info ← env.step(a_t)
        内部: hybrid → decode → FMU.adapter.step → 读输出
              → 分时电价结算电网现金流 → reward_terms / metrics
汇总 KPI（§3），计时 wall_s、fmu_steps
```

| 组件 | 模块 |
|------|------|
| 环境 | `PowerSystemEnv` |
| FMU | `src/fmu/` adapter |
| 安全 | `GiveSafeController` + `FeasibilityOracle` |
| 评估 | `evaluate_policy` / `evaluate_annual_policy` |

---

## 2. 强化学习如何调用 FMU

### 2.1 训练时

1. 按 `annual_episode_start_seconds` 采样周起点 `start_time`  
2. Actor 输出混合动作 → GiveSafe 过滤  
3. **仅安全动作** 调用 FMU 一步，写入 PhysicalReplay  
4. 拒绝动作写入 GiveSafeReplay（不推进物理）  
5. TD3 / GHTD3 从 buffer 更新网络  

### 2.2 评估时

- `deterministic=True`  
- 固定 seed + `start_time`  
- 指标用 **净现金流等物理 KPI**，不只看 train reward  

### 2.3 方法

| 方法 | 说明 |
|------|------|
| Hybrid-GiveSafe-TD3 | 单层混合动作 TD3 |
| Safe Market-GHTD3 | 高层 SoC goal + 底层执行 + 市场 prior |
| SAC-Hybrid（可选） | 最大熵对照 |

---

## 3. 群智能（PSO）如何调用 FMU

### 3.1 推荐编码：参数化策略（P-B）

粒子 \(\theta\in\mathbb{R}^{d}\)（\(d\sim 10{-}30\)）编码：

- 谷充/峰放电价阈值  
- 火电偏置  
- 电池充放幅值  
- CAES 参与强度  
- 末段 SOC 回收强度  

**一次适应度评价** = 用 \(\theta\) 生成 168 步动作 + **完整 FMU 滚一周**。

### 3.2 适应度（与论文 KPI 对齐）

\[
F(\theta)
=
\sum_t \Delta\mathrm{CF}_t
-
\rho_{\mathrm{uns}} E^{\mathrm{uns}}
-
\rho_{\mathrm{fail}}\mathbb{I}_{\mathrm{fail}}
-
\rho_{\mathrm{soc}}\max(0,\,e_{L1}^{\mathrm{energy}}-\tau)
\]

**不用** train 的 shaping 作主适应度（避免与报告 KPI 不一致）。

### 3.3 计算量

\[
N_{\mathrm{FMU\,steps}}
\approx
N_{\mathrm{iter}}\times N_{\mathrm{pop}}\times T
\]

例：50 代 × 20 粒子 × 168 ≈ **1.68×10⁵** 步/周场景 → 必须写入论文计算表。

### 3.4 可选：开环动作序列（P-A）

直接优化 \(168\times 4\) 维动作 → 维数灾难，仅作附录小规模试验。

---

## 4. 运筹滚动优化如何调用 FMU

1. 读当前 \(x_t\) 与未来 \(H\) 小时预报（风/光/荷/价）  
2. 解 **松弛模型**（线性平衡 + 电池 SOC + gas 等效；CAES 模式连续松弛或规则固定）  
3. 仅执行 \(z_0^*\) 对应混合动作  
4. **真实 `env.step`（FMU）**  
5. \(t\leftarrow t+1\) 滚动  

声明：求解器最优是 **代理模型** 上的最优，不是 FMU 全局最优。

---

## 5. 对比协议：相对“模型原始结果”

### 5.1 基线定义

| ID | 名称 | 定义 |
|----|------|------|
| **B0** | 原始运行 | 保守规则：高火电 + 储能 idle |
| **B1** | 峰谷规则 | 价格阈值 |
| M1 | 滚动 LP | 松弛运筹 |
| M2 | PSO | 参数化群智能 |
| M3 | Hybrid-TD3 | 单层 RL |
| M4 | GHTD3 | 本文主方法 |
| A* | 消融 | 无 GiveSafe / 无 prior 等 |

### 5.2 场景

- 冬 / 夏 / 过渡 各 1 周（`start_time = 0, 180d, 90d`）  
- 全年 53 窗  
- Perfect vs Predicted 电价观测（可选）  

### 5.3 统一输出表（每场景×方法）

| 列 | 说明 |
|----|------|
| \(J\) | 净现金流 |
| \(\Delta J\) vs B0 | 相对原始运行 |
| \(E^{\mathrm{curt}}\), \(\Delta E^{\mathrm{curt}}\) | 弃风弃光及变化 |
| \(E^{\mathrm{uns}}\) | 缺供 |
| \(E^{\mathrm{th}}\) | 火电 |
| Thr_bat, Thr_caes | 储能吞吐 |
| Buy/Sell 费用与电量 | 市场 |
| SOC_ok, L1_energy | 期末能量 SOC |
| hard_viol, gs_reject | 安全 |
| wall_s, fmu_steps | 计算成本 |

### 5.4 储能调度对比内容

对 B0 vs 最优方法画同轴图：

1. \(\lambda_t^{\mathrm{buy}}\) 与 \(p_t^{\mathrm{grid}}\)  
2. \(p_t^{\mathrm{battery}},\; \mathrm{SOC}_t^{\mathrm{bat}}\)  
3. \(p_t^{\mathrm{caes}},\; \mathrm{SOC}_t^{\mathrm{gas}}\)  
4. \(p_t^{\mathrm{thermal}}\)  

文字描述：谷充峰放是否出现、CAES 是否参与跨日转移、相对 B0 火电是否下降。

---

## 6. 复现命令（实现后）

```powershell
# 全基准（骨架）
python scripts/run_full_benchmark.py --seasons winter,summer,transition --methods b0,b1,lp,pso,hybrid,ghtd3

# 单方法
python scripts/run_rolling_lp_week.py --season winter
python scripts/run_pso_week.py --season winter --iters 30 --pop 16
```

---

## 7. 写作注意

1. **原始结果 = B0 滚 FMU**，不是“未优化的数学最优”。  
2. PSO/RL 算力差数量级 → 主表旁加 **compute table**。  
3. 弃电经济系数可为 0，但 **物理弃电量必须报**。  
4. 能量 SOC 与全状态 L1 分开写，避免被质疑“改指标刷分”。
