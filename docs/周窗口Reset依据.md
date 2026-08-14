# 为什么用「周窗口 reset」而不是连续 8760h 一条轨迹？

文档更新：2026-08-14 12:30 (+08:00)

## 1. 本仓库在做什么

- **训练 / 评估 episode**：`episode_steps = 168`（7×24 h）  
- **全年评价**：`evaluate_annual_policy` 按  
  `start_hour = 0, 168, 336, …` 依次 **reset** 53 个周窗，覆盖 8760 h  
- **每个窗起点**：储能 SOC 回到环境初始（约 bat=0.5, gas=0.85），**不是**上一窗末状态连续传递  

这是 **周时域闭环调度 + 全年拼贴评价**，不是物理上的「一年不中断运行」。

---

## 2. 论文与领域依据（可写 Related Work / Experimental Setup）

### 2.1 与对标文一致的「典型周 / 季节周」设定

| 依据 | 做法 |
|------|------|
| **Cui et al., Energy 2025 (GHTD3)** | Case study 用 **夏 / 冬 / 过渡 典型周** 展示调度与 SoC；分层 horizon 与周尺度运营一致，而非单条年轨迹 |
| **多数 IES + DRL 调度文**（Energy / Applied Energy / CSEE） | 常用 **24h 或 168h episode** 训练 MDP；全年用 **多起点滚动周** 或季节代表周统计 |
| **储能套利 / price-taker 文** | 常按 **日/周结算周期** 定义运营问题；周重置对应「运营期末库存回收」商业假设 |

### 2.2 运营与建模理由（可写 Method / Assumption）

1. **运营周期**  
   分时电价、周结算与「期末库存回到初值附近」是常见运营假设。本仓库对齐崔文：电池运行盒子 \([0.1,0.9]\) 仍硬；周末门是软加分（过则 +15，不过为 0），**不劫持电池指令**。气库仍可用末段回收窗。  
   周 reset = 每个运营周从同一初值出发，而不是假设全年连续持仓。

2. **MDP 可学习性**  
   168 步回报方差仍可控；若单 episode=8760，信用分配极难，样本效率差（深度 RL 能源调度实践中普遍避免超长 episode）。

3. **公平对比与可复现**  
   固定 53 个起点，所有方法（规则 / LP / PSO / RL）在 **同一周边界与同一初值** 下比净现金流与 SOC，避免「初始 SOC 偶然占优」。

4. **与 FMU 数字孪生的匹配**  
   高保真 FMU 步进贵；周 episode 便于短训探路与消融，全年用 **拼贴周** 控制总仿真预算（亦便于 E: 缓存）。

5. **不是缺陷，是评价协议**  
   正文应明确：

   > *We evaluate closed-loop weekly operation (\(T=168\) h) and report annual metrics by tiling 53 weekly windows over the 8760 h horizon with SOC reset at each window start (weekly-horizon evaluation). This matches a finite operating horizon with terminal storage recovery, and is standard in hierarchical RL and multi-energy scheduling case studies.*

---

## 3. 与「连续年 SOC」的区别（Discussion 可写）

| 协议 | 含义 | 适用 |
|------|------|------|
| **周 reset（本仓库）** | 每窗独立运营 + 期末回收 | 调度策略、分时套利、算法对比 |
| **连续年（可选扩展）** | 窗与窗 SOC 传递 | 研究季节性库存转移、跨周价值 |

若审稿人要求连续年：**附录协议已实现**——见 **`docs/连续年SOC附录协议.md`**  
与 `evaluate_annual_policy(..., continuous_soc=True)` / `evaluate_continuous_annual_policy`  
（单次 FMU 实例化连续 8760 h，SOC 物理传递；**勿与主表 weekly_reset 混表**）。

CLI：`scripts/eval_continuous_annual.py`

---

## 4. 本仓库代码锚点

| 配置/代码 | 内容 |
|-----------|------|
| `env_config.yaml` → `fmu.episode_steps: 168` | 周长 |
| `fmu.annual_horizon_hours: 8760` | 年长 |
| `annual_episode_start_seconds` | 训练周起点循环 |
| `evaluate_annual_policy` | 53 窗拼贴评估（`protocol=weekly_reset`） |
| `evaluate_continuous_annual_policy` | 附录连续年（`protocol=continuous_soc`） |
| `terminal_soc` / `soc_recovery_*` | 运营期末回收 |

---

## 5. 推荐引用写法（示例句）

> Following common practice in multi-energy DRL scheduling and hierarchical control studies (e.g., weekly/seasonal case studies in GHTD3-type works), each training and evaluation episode spans one week. Annual performance is obtained by evaluating 53 successive weekly windows covering 8760 h, with storage states re-initialized at each window to enforce a terminal SoC recovery constraint consistent with weekly market operation.
