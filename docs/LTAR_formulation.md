# LTAR-TD3：Lagrangian Trust-region Anchored Residual

> 文献合成：Constrained RL（库存约束）+ Safe residual exploration（教师邻域）+ TD3 + 慢库存 prior。  
> 配置：`src/config/ghtd3_config_ltar.yaml`。实现：`ltar_enabled` / `high_level_mode: prior_only`。

---

## 1. 目标

在冻结 Hybrid 教师 \(\pi_H\) 与 GiveSafe 投影 \(\Pi_{\mathcal{F}}\) 下，学习连续残差，使：

1. **多种子**周能量 SOC 门控通过；  
2. **经济** mean Δ vs Hybrid \(\ge\) ares（约 +0.5）；  
3. **CAES mode 不进入自由 RL**（\(m\equiv m_H\)）。

---

## 2. 锚定执行

\[
a_t=\Pi_{\mathcal{F}(s_t)}\!\big(a_H(s_t)+\beta_t\odot\tanh\Delta_\theta(s_t,g_k)\big),\quad m_t=m_H.
\]

慢意图 \(g_k\)：Stage A 为市场/回收 **prior**（不训练高层）。

---

## 3. 自适应信任域

\[
\beta_t=\bar\beta\cdot\sigma_{\mathrm{soc}}(z_t)\cdot\sigma_{\mathrm{adv}}\cdot\sigma_\lambda(\lambda).
\]

| 因子 | 含义 | 实现 |
|------|------|------|
| \(\bar\beta\) | 信任域上界 | `residual_beta_*` |
| \(\sigma_{\mathrm{soc}}\) | 库存安全带 | `_safe_expansion_scales` |
| \(\sigma_{\mathrm{adv}}\) | 残差 Q 优于教师才扩张 | `ltar_adv_gate` |
| \(\sigma_\lambda=1/(1+\lambda)\) | 违约收紧 | `lambda_soc` |

**无**训练进度 \(\rho\) 无界放大（区别于 TEA）。

---

## 4. Lagrangian 终端库存约束

\[
c=\big[\|z_T-z_0\|_W-\xi\big]_+,\quad
\lambda\leftarrow\mathrm{clip}\big(\lambda+\eta(c-\delta),0,\lambda_{\max}\big).
\]

- episode 结束用 `terminal_soc_l1` / 是否过门更新 \(\lambda\)；  
- 步进奖励可加 \(-\kappa\lambda\cdot e_t\)（回收窗内库存误差）。

---

## 5. 训练命令

```powershell
$env:PYTHONPATH="src"
python scripts/train_ghtd3.py --mode custom --steps 35000 --seed 0 `
  --run-dir runs/ltar_s0_35k --config src/config/ghtd3_config_ltar.yaml
```
