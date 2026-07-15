设计方向**基本合理**，而且与论文的结构一致：把“经济成本”和“约束/目标项”分开。但有一个关键修正：

> **电网越限、火电调控和 SOC 偏差不能取代 FMU 求解失败处理。**

它们属于三种不同层级：

1. **综合成本**：正常运行状态下评价经济性；
2. **约束惩罚和 SOC 目标**：评价物理运行质量；
3. **FMU 求解失败**：异常兜底，直接截断 episode。

论文也是“归一化综合成本 + 电网越限惩罚 + 终端 SOC 奖励”，并未把所有内容塞进同一个经济成本中。

---

# 一、推荐的最终 reward 结构

定义：

[
r_t=
-\widetilde C_t^{sys}
-\lambda_g V_t^{grid}
-\lambda_{th}V_t^{thermal}
-\lambda_{soc}V_t^{soc}
-\delta_{t,T}\lambda_TV_T^{terminal}
]

其中：

[
\widetilde C_t^{sys}
====================

\frac{C_t^{sys}}{C_{\mathrm{ref}}}
]

系统综合成本为：

[
\begin{aligned}
C_t^{sys}={}&
C_t^{grid}
+C_t^{thermal}
+C_t^{battery}
+C_t^{caes}\
&+
C_t^{curtailment}
+C_t^{unserved}
+C_t^{ramp}
\end{aligned}
]

这与你提出的七项一致。

但当 FMU 求解失败时，不再套用上述公式，而是：

[
r_t=-K_{\mathrm{fail}},
\qquad truncated=True
]

所以推荐架构是：

```text
正常物理状态：
    综合经济成本
    + 约束惩罚
    + SOC目标

FMU数值失败：
    独立失败惩罚
    + 立即截断
```

当前实现只有 `solver_failure_penalty=1e9` 生效，而且正常经济参数为空，因此暂时只能学习避免 FMU 失败。

---

# 二、系统综合成本的七项设计

## 1. 电网购售电成本

当前符号已经核验为：

* (p_{grid}>0)：购电；
* (p_{grid}<0)：售电。

因此：

[
C_t^{grid}
==========

\lambda_t^{buy}
\max(p_t^{grid},0)
\Delta t
--------

\lambda_t^{sell}
\max(-p_t^{grid},0)
\Delta t
]

若功率单位为 W、电价单位为元/kWh：

[
C_t^{grid}
==========

\left[
\lambda_t^{buy}\max(p_t^{grid},0)
---------------------------------

\lambda_t^{sell}\max(-p_t^{grid},0)
\right]
\frac{\Delta t_{\mathrm{h}}}{1000}
]

也可以先将 W 转换为 MW，再使用元/MWh。

你当前 Modelica 实例中的固定参数为：

```text
c_buy  = 0.6 元/kWh
c_sale = 0.1 元/kWh
```

这可以作为第一版固定市场价格，但如果论文目标是“参与电力市场交易”，后续必须改成随时间变化的日前/实时价格序列。固定价格很难训练出真正的峰谷套利策略。

---

## 2. 火电运行成本

当前火电实际功率为负值表示发电，因此应使用：

[
P_t^{thermal,gen}
=================

\left|p_t^{thermal}\right|
]

第一版可采用 Modelica 中已有的线性边际成本：

[
C_t^{thermal}
=============

c_{thermal}
P_t^{thermal,gen}
\Delta t
]

当前实例中火电成本系数约为：

[
c_{thermal}=0.5\ \text{元/kWh}
]

后续如果能够获得煤耗曲线，建议改成：

[
C_t^{thermal}
=============

\left(
aP_t^2+bP_t+c
\right)\Delta t
]

需要注意：固定项 (c) 只有在机组已启动时才应计入，否则会产生“停机仍付固定运行成本”的问题。

---

## 3. 电池充放电成本

这里必须谨慎。

当前 Modelica 中：

```text
c_buy  = 0.2
c_sale = 0.4
```

并通过充电和放电功率构造设备现金流。但这些参数更像是电池内部的购售电现金流，而不是电池退化成本。

如果系统综合成本已经包含：

* 电网购电成本；
* 火电发电成本；

那么再将“电池充电购电成本”加入系统成本，可能造成**重复计费**：

```text
电网向电池充电：
电网购电成本计算一次
电池 c_buy 又计算一次
```

对于系统边界优化，电池更合理的成本应是：

[
C_t^{battery}
=============

c_{bat}^{deg}
|p_t^{battery}|
\Delta t
]

也就是：

* 循环退化；
* 变流器损耗；
* 可变运维成本。

所以不能直接因为 `.mo` 中有 `c_buy/c_sale`，就把它们全部加入综合成本。必须先明确这些参数是：

* 外部市场现金流；
* 还是设备退化/运维成本；
* 还是内部结算价格。

内部结算价格不应该参与系统级总成本求和。

---

## 4. 压缩空气储能成本

CAES 同理。

当前 Modelica 中的充放电 `c_buy/c_sale` 更接近设备现金流。系统级成本建议采用：

[
C_t^{caes}
==========

c_{caes}^{var}
|p_t^{caes}|
\Delta t
]

如果有更详细的设备模型，可以拆成：

[
C_t^{caes}
==========

C_t^{compressor}
+
C_t^{expander}
+
C_t^{thermal\ storage}
+
C_t^{maintenance}
]

但不要同时计入“充电购电价格”和电网购电成本，除非明确是在计算不同主体之间的博弈收益，而不是整个系统的成本。

---

## 5. 弃电成本

[
C_t^{curtailment}
=================

c_{curt}
p_t^{curtailment}
\Delta t
]

弃电成本可以表达：

* 可再生能源补贴损失；
* 上网收益损失；
* 环境价值；
* 约束考核费用。

如果只是为了优先消纳风光，第一版可以采用固定惩罚系数。

但是当前电网边界为约 (\pm500) MW，电网会自动吸收系统残差，因此现有工况中 `p_curtailment` 和 `p_unserved` 基本始终为零。

因此，在正式训练前需要：

* 限制最大购电和售电功率；
* 或增加市场申报限额；
* 或设置禁止无限上网的场景。

否则弃电成本虽然写进 reward，却不会产生训练信号。

---

## 6. 缺供成本

[
C_t^{unserved}
==============

VOLL,
p_t^{unserved}
\Delta t
]

其中 VOLL 是失负荷价值。

应满足：

[
VOLL
\gg
c_{buy},c_{thermal},c_{curt}
]

否则智能体可能发现：

> 少供负荷比购电或发电更便宜。

通常缺供惩罚应是正常电价的数倍到数十倍，但最终参数需要有文献或市场规则依据。

---

## 7. 火电爬坡成本

[
C_t^{ramp}
==========

c_{ramp}
\left|
P_t^{thermal}
-------------

P_{t-1}^{thermal}
\right|
]

或者用二次形式：

[
C_t^{ramp}
==========

c_{ramp}
\left(
P_t^{thermal}
-------------

P_{t-1}^{thermal}
\right)^2
]

线性形式更容易解释为调节磨损成本，二次形式会更强烈抑制大幅波动。

如果机组还存在最大爬坡约束：

[
|\Delta P_t|
\leq R_{\max}
]

则应区分：

* 范围内爬坡：进入经济成本；
* 超过最大爬坡率：进入约束惩罚。

不要对同一个爬坡量重复使用两个完全相同的惩罚。

---

# 三、电网功率越限惩罚

定义电网合同或物理交换范围：

[
P_{\min}^{grid}
\leq p_t^{grid}
\leq P_{\max}^{grid}
]

使用无量纲二次惩罚：

[
V_t^{grid}
==========

\left(
\frac{
[p_t^{grid}-P_{\max}^{grid}]*+
}{
P*{\mathrm{ref}}^{grid}
}
\right)^2
+
\left(
\frac{
[P_{\min}^{grid}-p_t^{grid}]*+
}{
P*{\mathrm{ref}}^{grid}
}
\right)^2
]

这样比直接对 W 级功率惩罚更稳定。

当前 FMU 的电网范围过大，极端测试仍能通过电网平衡系统。因此应先定义更有意义的市场或联络线限制，否则：

[
V_t^{grid}=0
]

几乎始终成立。

---

# 四、“火力发电调控惩罚”需要明确含义

这个名称现在过于宽泛。至少可能代表三类东西：

1. 火电出力上下限违反；
2. 火电指令与实际出力不一致；
3. 火电爬坡率违反。

你的动作 `u_tp` 已经被 Python 限制在：

[
u_{tp}\in[1/3,1]
]

对应火电：

[
P_{\min}=50\ \mathrm{MW},
\qquad
P_{\max}=150\ \mathrm{MW}
]

因此，**动作层面的出力上下限一般不会被违反**。

更有价值的是实际调控偏差：

[
V_t^{thermal}
=============

\left(
\frac{
P_t^{thermal,actual}
--------------------

P_t^{thermal,command}
}{
P_{\mathrm{cap}}^{thermal}
}
\right)^2
]

或者实际出力边界违反：

[
V_t^{thermal}
=============

\left(
\frac{
[P_{\min}-P_t^{thermal,gen}]*+
}{
P*{\mathrm{cap}}
}
\right)^2
+
\left(
\frac{
[P_t^{thermal,gen}-P_{\max}]*+
}{
P*{\mathrm{cap}}
}
\right)^2
]

若火电模型当前满足：

[
P_{actual}=P_{plan}
]

且没有真实惯性、延迟和爬坡限制，那么这项也会长期为零。此时最应该先完善火电动态，而不是添加一个永远不会触发的 penalty。

---

# 五、SOC奖励的推荐设计

“奖励”可以做，但建议分成两部分。

## 1. 每步 SOC 越界惩罚

对于电池和 CAES 各储能状态：

[
V_t^{soc}
=========

\sum_i
\left[
\left(
\frac{[SOC_{i,t}-SOC_{i,\max}]*+}
{SOC*{i,\max}-SOC_{i,\min}}
\right)^2
+
\left(
\frac{[SOC_{i,\min}-SOC_{i,t}]*+}
{SOC*{i,\max}-SOC_{i,\min}}
\right)^2
\right]
]

这用于防止超出物理范围。

你的随机和 TD3 smoke 已经出现 SOC 超过1后截断，因此这个连续惩罚是有必要的。

但要认识到：如果 SOC 越界后 FMU 已直接 assert 或求解失败，智能体可能来不及收到平滑惩罚。因此还需要动作设计或动态可行域解决根本问题。

## 2. 终端 SOC 回归

建议使用连续终端成本：

[
V_T^{terminal}
==============

\sum_i
\left(
\frac{
SOC_{i,T}-SOC_{i,0}
}{
SOC_{i,\max}-SOC_{i,\min}
}
\right)^2
]

终端 reward 为：

[
r_T^{soc}
=========

-\lambda_TV_T^{terminal}
]

这种形式比论文中的二元奖励更平滑。论文使用的是终端 SOC 偏差落在容差内时获得固定正奖励，但它对“轻微超出”和“严重超出”无法区分。

不建议每一步都奖励 SOC 高，因为智能体可能为了获得 SOC 奖励而一直充电。SOC 本身不是越高越好，关键是：

* 不越界；
* 保持合理储备；
* 终端恢复到指定水平。

---

# 六、FMU 求解失败仍然必须保留

不能把：

```text
solver_failure_penalty
```

直接改名成：

```text
grid_violation + thermal_violation + SOC_reward
```

因为 FMU 还可能因以下原因失败：

* 非线性方程不收敛；
* 物性计算超出定义域；
* 温度、压力异常；
* 事件迭代失败；
* 数值溢出；
* 模型内部 assert；
* 模型代码缺陷。

这些未必能够通过三个正常惩罚项提前解释。

建议：

[
r_{\mathrm{failure}}=-K_{\mathrm{fail}}
]

并：

```python
truncated = True
```

但不应继续使用毫无尺度关系的：

[
K_{\mathrm{fail}}=10^9
]

假设归一化后正常单步成本约为 (0\sim3)，168步 episode 的回报量级约为数百，则可先测试：

[
K_{\mathrm{fail}}\in[200,500]
]

使失败显著劣于正常完整运行，但不把 Critic 的 Q 值推到 (10^9) 量级。

---

# 七、综合成本怎么归一化

建议先在统一货币单位下求和：

[
C_t^{sys}
=========

\sum_j C_{j,t}
]

再统一除以固定基准：

[
\boxed{
\widetilde C_t^{sys}
====================

\frac{C_t^{sys}}{C_{\mathrm{ref}}}
}
]

不要把七项分别除以各自最大值后直接相加，否则会改变原始经济关系。

## 初始参考值估计

按你当前模型的一小时步长和现有参数粗略估计：

* 最大购电：(500\text{ MW}\times0.6\text{ 元/kWh}\approx300000) 元/h；
* 最大火电：(150\text{ MW}\times0.5\text{ 元/kWh}\approx75000) 元/h；
* 电池额定功率：约100 MW；
* CAES额定功率：约150 MW。

只考虑电网和火电，正常单步经济量级就可能达到：

[
3.75\times10^5\ \text{元/h}
]

因此可以临时取：

[
C_{\mathrm{ref}}=4\times10^5\ \text{元/步}
]

作为 smoke test 初值。

但正式值更应该由规则控制器和代表性场景得到：

[
C_{\mathrm{ref}}
================

P_{95}
\left(
|C_t^{sys,baseline}|
\right)
]

固定后整个训练期间不再变化。

---

# 八、对当前 `.mo` 成本参数的关键判断

不能直接把 `.mo` 中所有 `C` 相加后当作综合成本，因为当前 `C` 的本质是**部件现金流**：

```modelica
der(Income) =
    Power_PV.C
  + Power_WT.C
  + Power_TP.C
  + Power_BT.C
  + Power_CAES.C
  + Power_Eload.C
  + Power_Grid.C
  - C_penality;
```

这里混合了：

* 发电收益；
* 用电收入或支出；
* 电网交易；
* 储能内部购售电；
* penalty。

这更接近多主体现金流汇总，不是严格的系统边界经济成本。

所以 Python reward 应重新构建，不要直接使用：

```text
bus.Income
OPT_goal
组件 C 的简单求和
```

尤其要检查储能 `c_buy/c_sale` 是否与电网和火电成本重复计费。

---

# 结论

你的框架应调整为：

[
\boxed{
r_t
===

-\frac{C_t^{sys}}{C_{\mathrm{ref}}}
-\lambda_gV_t^{grid}
-\lambda_{th}V_t^{thermal}
-\lambda_{soc}V_t^{soc}
-\delta_{t,T}\lambda_TV_T^{terminal}
}
]

其中：

[
C_t^{sys}
=========

C_t^{grid}
+C_t^{thermal}
+C_t^{battery,degradation}
+C_t^{caes,operation}
+C_t^{curtailment}
+C_t^{unserved}
+C_t^{ramp}
]

另外保留独立的：

[
\boxed{
\text{FMU失败}
\Rightarrow
r_t=-K_{\mathrm{fail}},
\quad truncated=True
}
]

因此，**经济成本归一化 + 电网越限惩罚 + 火电调控惩罚 + SOC约束/终端目标**是合理设计；不合理的是完全删除 FMU 失败兜底，或者直接把 Modelica 中全部组件现金流当作互不重复的系统成本。
