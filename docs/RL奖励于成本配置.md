# RL 奖励与成本配置

文档更新：2026-08-14 12:30 (+08:00)

> 硬约束（电网容量、火电爬坡、SOC/禁区等）不进入经济 reward；
> GiveSafe 拒绝使用独立约束奖励；FMU 硬失败不进经济 replay。

# 〇、分时电价（price-taker）

当 `env_config.yaml` 中 `market.available: true` 时：

- 观测追加 24h×(购电价, 售电价) 前瞻特征（`data/price_tou.csv`）。
- 经济 reward 中 **电网项** 用 Python 分时结算替换 FMU 常数购售电价增量：  
  \(\Delta C^{\mathrm{tot}\prime}=\Delta C^{\mathrm{tot}}_{\mathrm{FMU}}-\Delta C^{\mathrm{grid}}_{\mathrm{FMU}}+C^{\mathrm{grid}}_{\mathrm{market}}\)。
- **不做电网出清**；电价外生。

结算实现：`src/market/settlement.py`。项目层 NPV/IRR/LCOE：`src/economics/project_kpi.py`。

# 一、Reward 结构

物理步、GiveSafe 拒绝、硬失败是**三条路径**，不是把三项加进同一步的标量公式。训练时 Replay 按约 70% / 30% 混合采样 Physical 与 GiveSafe 转移。

| 路径 | 何时 | 奖励 | Replay |
|------|------|------|--------|
| 经济物理步 | 主 FMU 合法推进 | $r_e=-\widetilde{C}_t^{\mathrm{sys}}+b_{\mathrm{SOC}}$ | PhysicalReplay（默认 70%） |
| GiveSafe 拒绝 | Oracle/Shadow 拒不安全候选（或 false-safe 自环） | $r_c=-C^{\mathrm{c}}$ | GiveSafeReplay（默认 30%） |
| 环境硬失败 | 预检/后验硬约束/FMU 失败且无法继续 | $r_f=0$，`truncated=True` | 不进经济 PhysicalReplay |

$$
\widetilde{C}_t^{\mathrm{sys}}=\frac{C_t^{\mathrm{sys}}}{C_{\mathrm{ref}}}
$$

---

## （1）综合成本（经济项）

七项先在统一货币单位下求和，再除以**固定**基准；不按分项各自归一化后相加。

$$
C_{\mathrm{ref}} \approx 156539.84\ \text{元/步}
$$

- 功率来自 FMU，单位为 **W**。代码先乘 $10^{-6}$ 转为 **MW**，电价/吞吐系数为 **元/MWh**。
- 决策步长 $\Delta t = \texttt{decision\_interval\_seconds}/3600$（默认 $1\,\mathrm{h}$）。

$$
C_t^{\mathrm{sys}}=
C_t^{\mathrm{grid}}
+C_t^{\mathrm{thermal}}
+C_t^{\mathrm{battery}}
+C_t^{\mathrm{caes}}
+C_t^{\mathrm{curt}}
+C_t^{\mathrm{uns}}
+C_t^{\mathrm{ramp}}
$$

对应 `raw_total_cost = sum(raw_*_cost)`。

### 1）电网购售电成本

> 购电时 $p_t^{\mathrm{grid}}>0$，售电时 $p_t^{\mathrm{grid}}<0$。

$$
C_t^{\mathrm{grid}}
=
\Bigl(
\lambda^{\mathrm{buy}}\max(p_t^{\mathrm{grid}},0)
-
\lambda^{\mathrm{sell}}\max(-p_t^{\mathrm{grid}},0)
\Bigr)
\cdot 10^{-6}
\cdot \Delta t
$$

| 参数 | 值 | 说明 |
|------|-----|------|
| `buy_price_yuan_per_mwh` | $600$ | Modelica `c_buy=0.6` 元/kWh |
| `sell_price_yuan_per_mwh` | $100$ | Modelica `c_sale=0.1` 元/kWh |

### 2）火电运行成本

$$
P = |p_t^{\mathrm{thermal}}| \cdot 10^{-6}
\quad(\mathrm{MW})
$$

$$
C_t^{\mathrm{thermal}}
=
(a P^2 + b P + c)\,\Delta t
$$

当前配置：$a=0$，$b=400$，$c=0$（线性边际燃料，元/MWh 量级）。

### 3）电池吞吐成本

$$
C_t^{\mathrm{battery}}
=
c_{\mathrm{bat}}\,|p_t^{\mathrm{battery}}|\cdot 10^{-6}\cdot\Delta t
$$

当前设置 `battery_throughput_yuan_per_mwh = 0`，**当前贡献为 0**。

### 4）CAES 吞吐成本

$$
C_t^{\mathrm{caes}}
=
c_{\mathrm{caes}}\,|p_t^{\mathrm{caes}}|\cdot 10^{-6}\cdot\Delta t
$$

当前设置 `caes_throughput_yuan_per_mwh = 0`，**当前贡献为 0**。

### 5）弃电成本

$$
C_t^{\mathrm{curt}}
=
c_{\mathrm{curt}}\,p_t^{\mathrm{curtailment}}\cdot 10^{-6}\cdot\Delta t
$$

当前设置 `curtailment_yuan_per_mwh = 0`，**当前贡献为 0**。

### 6）缺供成本

$$
C_t^{\mathrm{uns}}
=
c_{\mathrm{uns}}\,p_t^{\mathrm{unserved}}\cdot 10^{-6}\cdot\Delta t
$$

当前设置 `unserved_yuan_per_mwh = 0`，**当前贡献为 0**。



---

## （2）终端 SOC 奖励 $b_{\mathrm{SOC}}$

这是经济奖励 $r_e$ 的加项，**不是** GiveSafe 的 $r_c$。

能量主状态（电池 + 气库）等权：

$$
e_{L1}=|\mathrm{SOC}_{\mathrm{bat},T}-\mathrm{SOC}_{\mathrm{bat},0}|
+|\mathrm{SOC}_{\mathrm{gas},T}-\mathrm{SOC}_{\mathrm{gas},0}|
$$

对齐崔文式 (29)：满 168 物理步、无失败、且 $e_{L1}\le\tau$ 时给固定加分；**不过门为 0，不再按 L1 乘 30 重罚**。电池充放不被环境改写。

$$
b_{\mathrm{SOC}}=b
\quad\text{（否则 }0\text{）}
$$

当前：`mode=binary_bonus`，$b=15$，$\tau=0.06$；`fail_penalty_l1=0`；`battery_soc` 权重 \(1.0\)。GiveSafe 拒绝次数**不**计入 168。

##（3）GiveSafe 约束成本 $C^{\mathrm{c}}$
> 标准 GiveSafe 要求在动作真正执行前就识别它不安全。论文也明确指出，其安全性依赖真实约束函数能够被准确表达；约束函数不准确会影响安全性和性能

```mermaid
graph TD
状态 s_t
  ↓
策略提出候选动作 a_1
  ↓
安全检查不通过
  ↓
不执行FMU
状态仍为 s_t
记录GiveSafe负向信号
  ↓
策略提出候选动作 a_2
  ↓
安全检查不通过
  ↓
不执行FMU
状态仍为 s_t
  ↓
策略提出候选动作 a_3
  ↓
安全检查通过
  ↓
执行FMU
获得 s_{t+1} 和经济reward
```

$$
C^{\mathrm{c}}=\mathrm{base}+\sum_j w_j v_j^2
,\qquad
r_c=-C^{\mathrm{c}}
$$

- `base_rejection_cost = 1.0`
- $v_j$：归一化违反量（离散拒斥时常为 $1$）
- 自环：`economic_reward = terminal_soc_bonus = 0`

- 写入 GiveSafeReplay；时间/状态不因拒绝而推进

| 权重键 | $w$ |
|--------|-----|
| `battery_soc_high` / `low` | $2.0$ |
| `caes_gas_soc_*` / `caes_pressure_*` | $2.0$ |
| `caes_hot/cold_soc_*` | $1.5$ |
| `caes_temperature_*` | $1.0$ |
| `thermal_ramp` | $1.5$ |
| `grid_capacity` | $2.0$ |
| `forbidden_mode` | $3.0$ |
| `shadow_fmu_rejection` | $2.5$ |
| `unknown` | $1.0$ |


###（3）硬边界（由 Oracle/GiveSafe 强制，不进经济成本）

**(1) 电网容量**

$$
P_{\min}^{\mathrm{grid}} \le p_t^{\mathrm{grid}} \le P_{\max}^{\mathrm{grid}}
$$

约 $\pm 500\,\mathrm{MW}$（$P_1=5\times10^8\,\mathrm{W}$，$P_2=-5\times10^8\,\mathrm{W}$）。

**(2) 火电出力与爬坡**



**(3) SOC / CAES 禁区**

电池与 CAES 气/热/冷罐 SOC、温度及禁区模式

压空连续启停约束：
