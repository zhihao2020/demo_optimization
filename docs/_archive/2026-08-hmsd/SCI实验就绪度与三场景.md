<!-- ARCHIVED: superseded by FS-HSAC; see docs/paper_outline_and_figures.md -->

# SCI 实验就绪度与三场景说明（更新）

## 1. 现在达到 SCI 要求了吗？

| 维度 | 就绪度 | 说明 |
|------|--------|------|
| **方法创新** | **高** | Safe Market-GHTD3 + GiveSafe + Modelica/CAES + price-taker 市场 |
| **三季场景** | **已齐** | 冬/夏/过渡 × Rule/Hybrid/GHTD3，能量 SOC 全过 |
| **电价信息场景** | **已齐** | Perfect vs Predicted（冬周） |
| **全年能量 SOC** | **高** | **51/53（96%）**（battery+gas）；全状态 17/53 作热力学副指标 |
| **图表/写作** | **中低** | 数据齐，论文成图与 Introduction 表未定稿 |
| **投稿定稿** | **接近 Go** | 可写完整 Case Studies；补图+文字即可冲初稿 |

**诚实结论**：  
- 若终端约束定义为 **能量库存（电池 + CAES 气罐）**（物理上合理，热/冷罐为耦合副状态）→ **实验已基本达 SCI Case Study 要求**。  
- 若坚持 **四罐全状态 L1 硬达标** → 全年仅 ~30%，仍不够，需继续压 cold 漂移。  
- **推荐论文口径**：主 KPI = 能量 SOC；附录/正文附热冷罐偏差分布。

---

## 2. 三个场景（层 A：季节）

对标 Cui GHTD3 夏/冬/过渡。数据：`runs/seasonal_scenarios_energy_soc_20260803/`

| 季节 | 方法 | 周 reward | 能量 SOC | L1(energy) | 火电 MWh |
|------|------|-----------|----------|------------|----------|
| **冬** | Rule | 67.5 | 是 | 0.014 | 25200 |
| **冬** | Hybrid | **128.1** | 是 | 0.019 | 8598 |
| **冬** | GHTD3 | **126.8** | 是 | 0.018 | 8958 |
| **夏** | Rule | 13.3 | 是 | 0.040 | 25200 |
| **夏** | Hybrid | **83.7** | 是 | 0.041 | 9143 |
| **夏** | GHTD3 | **80.3** | 是 | 0.051 | 9838 |
| **过渡** | Rule | 58.6 | 是 | 0.010 | 25200 |
| **过渡** | Hybrid | **113.6** | 是 | 0.039 | 10255 |
| **过渡** | GHTD3 | **113.0** | 是 | 0.028 | 10205 |

要点：三季 RL 均显著优于规则；GHTD3≈Hybrid；冬季差距最大（~2× 规则）。

---

## 3. 电价信息场景（层 B）

| 模式 | 观测 | 结算 | 冬周 GHTD3 reward | SOC |
|------|------|------|-------------------|-----|
| Perfect | TOU 同源 | 实现价 | **126.0** | 是 |
| Predicted | BiLSTM | realized | **117.2** | 是 |

预测观测约 −7% reward，SOC 仍过 → 可写鲁棒性。

---

## 4. 全年 SOC（冲刺结果）

| 口径 | GHTD3 boost 40k | GHTD3 50k | Hybrid 15k |
|------|-----------------|-----------|------------|
| **能量 SOC (bat+gas)** | **51/53 (96%)** | 49/53 (92%) | **51/53 (96%)** |
| 全状态 4 罐 L1 | 17/53 (32%) | 16/53 (30%) | ~19/53（旧） |

失败主因（全状态）：**caes_cold_soc**（热力学副状态）。  
配置：`terminal_soc.primary_keys = [battery_soc, caes_gas_soc]`。

---

## 5. 创新是否够写 SCI？

**够写 3 条贡献**（方法已落地 + 有数字支撑）：

1. **高保真 Modelica–FMU 多能+CAES 非凸** 下的 price-taker 运营 MDP。  
2. **Safe Market-GHTD3**：分层 goal + 市场 prior + 回收 goal + \(\gamma^c\) + BC + GiveSafe。  
3. **分时市场结算 + 三季/全年/预测价** 案例验证。

**仍建议补齐再投顶刊**：

- 论文级图（电价–购售–SOC 同轴、三季柱状、消融）。  
- Introduction 文献对比表。  
- 公平消融同预算（可选）。  
- 热冷罐偏差作为 Discussion，勿隐藏。

---

## 6. 复现命令

```powershell
python scripts/diagnose_annual_soc.py --run-dir runs/ghtd3_annual_soc_boost_40k_20260803
python scripts/eval_seasonal_scenarios.py --methods rule,hybrid,ghtd3 `
  --ghtd3-ckpt runs/ghtd3_annual_soc_boost_40k_20260803/checkpoints/ghtd3.pt
python scripts/eval_price_info_scenarios.py --modes perfect,predicted
```
