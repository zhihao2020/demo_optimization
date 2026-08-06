# STFR / TRAP：Safe Two-timescale Feasible Residual 形式化

> 论文方法节骨架。实现入口：`src/config/ghtd3_config_stfr.yaml`，训练开关 `high_level_mode: prior_only`。  
> 对齐运营问题：`docs/优化问题形式化说明.md`。

---

## 1. 混合动作与 FMU 动力学（回顾）

每小时动作

\[
a_t=\big(u_t^{\mathrm{tp}},\,u_t^{\mathrm{bat}},\,m_t^{\mathrm{caes}},\,\mu_t^{\mathrm{caes}}\big),
\]

经解码与 **时变合法集** \(\mathcal{F}(s_t)\)（CAES 非凸区间、mode mask、爬坡等）约束后进入 FMU：

\[
s_{t+1}=f_{\mathrm{FMU}}(s_t,\tilde a_t),\qquad
\tilde a_t=\Pi_{\mathcal{F}(s_t)}(a_t^{\mathrm{raw}}).
\]

实现中 \(\Pi_{\mathcal{F}}\) = **GiveSafe** 选择/修复算子。

周时域 \(T=168\)，主 KPI 为净现金流增量 \(J=\sum_t\Delta\mathrm{CF}_t\)，终端能量库存

\[
z_T\in\mathcal{X}_T
=\big\{z=\big(\mathrm{SoC}^{\mathrm{bat}},\mathrm{SoC}^{\mathrm{gas}}\big):
\|z-z_0\|_W\le\xi\big\}.
\]

---

## 2. 原理 I：双时间尺度（慢库存 / 快调度）

状态分解

\[
s_t=(z_t,y_t),\quad
z_t=\big(\mathrm{SoC}^{\mathrm{bat}}_t,\mathrm{SoC}^{\mathrm{gas}}_t\big)
\;\text{慢},\quad
y_t=\text{功率、电价、预报等快变量}.
\]

慢周期长度 \(H\)（配置 `subgoal_interval`，默认 24）：

\[
g_k\in\mathcal{G}\subset\mathbb{R}^{d_g}
\quad\text{在 }t\in[kH,(k+1)H)
\text{ 内保持不变}.
\]

**STFR 约束：** 慢意图只表达 **库存/市场结构 prior**，**不**自由输出 CAES 离散 mode。  
Stage A（主实现）：\(g_k\) 由 **解析 prior** 生成（电价峰谷 + 回收），**不训练**高层 actor。

---

## 3. 原理 II：教师信任域残差（TRAP）

冻结教师 \(\pi_H\)（Hybrid GiveSafe-TD3）。快策略属于锚定类

\[
\Pi(\pi_H,\bar\beta)
=\Big\{
\pi:\ 
a^{\mathrm{cont}}=\mathrm{clip}\big(a_H^{\mathrm{cont}}+\beta\odot\tanh\Delta\big),\ 
0\le\beta_i\le\bar\beta_i,\ 
m^{\mathrm{caes}}\equiv m_H
\Big\}.
\]

执行：

\[
a^{\mathrm{raw}}_t
=
\big(
a_H^{\mathrm{tp}}+\beta_{\mathrm{tp}}\tanh\Delta_{\mathrm{tp}},\;
a_H^{\mathrm{bat}}+\beta_{\mathrm{bat}}\tanh\Delta_{\mathrm{bat}},\;
m_H,\;
a_H^{\mathrm{mag}}+\beta_{\mathrm{mag}}\tanh\Delta_{\mathrm{mag}}
\big),
\quad
\tilde a_t=\Pi_{\mathcal{F}(s_t)}(a^{\mathrm{raw}}_t).
\]

**性质（设计保证 / 实现强制）：**

1. **教师恢复**：\(\Delta=0\Rightarrow a^{\mathrm{raw}}=a_H\)（再经 \(\Pi_{\mathcal{F}}\)）。  
2. **有界偏离**：连续维 \(\|a-a_H\|_\infty\le\bar\beta\)（投影前）。  
3. **模式因子化**：\(m=m_H\)，切断 CAES mode 上的自由 RL 探索（对应 TEA 吞吐爆炸的结构修复）。

代码：`execution_mode: action_residual`，`tea_expandable: false`，`residual_mode_override: false`，固定小 `residual_beta_*`。

---

## 4. 原理 III：可行投影 + 终端库存集

- \(\Pi_{\mathcal{F}(s)}\)：GiveSafe / 可行域修复（方法节对象，消融可关）。  
- \(\mathcal{X}_T\)：能量 SOC 门控（`terminal_soc.primary_keys`）。  
- 回收窗 \(t>T-H_{\mathrm{rec}}\)：抬高教师 BC、收缩 \(\beta\)（配置 `recovery_goal_horizon_steps`、`tea_recovery_*` 或固定 β + BC）。

---

## 5. 学习问题（Stage A）

\[
\max_{\theta}
\;
\mathbb{E}\Big[\sum_{t=0}^{T-1}\gamma^t r_t^{\mathrm{eco}}\Big]
\quad\text{s.t.}\quad
\tilde a_t=\Pi_{\mathcal{F}}\big(a_H+\beta\odot\tanh\Delta_\theta(s_t,g_k)\big),\;
m_t=m_H,\;
g_k=g^{\mathrm{prior}}_k.
\]

- **只更新** residual / 低层 critic。  
- **不更新**高层 actor（`high_level_mode: prior_only`）。  
- \(g^{\mathrm{prior}}\)：`market_conditioned_goal_prior` + recovery。

Stage B（可选）：仅在库存相关维上轻量学高层，仍保持 \(m=m_H\) 与 \(\bar\beta\) 上界。

---

## 6. 与 GHTD3 / TEA 对照

| | Cui GHTD3 | TEA 扩张 | **STFR Stage A** |
|--|-----------|----------|------------------|
| 慢层 | 可学 goal | 可学 + 扩张课程 | **prior 库存/市场意图** |
| 快层 | 绝对/设备动作 | 大 β 残差 | **小 β 信任域残差** |
| Mode | 连续氢流 | 可解锁 | **锁教师** |
| 安全 | 软约束为主 | 软+屏障 | **\(\Pi_{\mathcal{F}}+\mathcal{X}_T\)** |

TEA 多种子失败作为 **「无信任域扩张不必要」** 的对照证据，而非主方法。

---

## 7. 配置与命令

```powershell
$env:PYTHONPATH="src"
python scripts/train_ghtd3.py --mode custom --steps 35000 --seed 0 `
  --run-dir runs/stfr_s0_35k --config src/config/ghtd3_config_stfr.yaml
```

关键键字段：

- `high_level_mode: prior_only`
- `execution_mode: action_residual`
- `tea_expandable: false`
- `residual_mode_override: false`
- `stfr_enabled: true`（文档/创新标记；训练逻辑读 `high_level_mode`）
