<!-- ARCHIVED: superseded by FS-HSAC; see docs/paper_outline_and_figures.md -->

# Applied Energy 加深：五轮自主迭代报告

文档更新：2026-08-12

## 总目标

针对“算法与问题深度不够”，围绕 **非凸压空 + 硬安全 + 同目标层次调度** 做五轮可落地加深，服务 *Applied Energy* 投稿。

---

## Round 1 — 问题深度与贡献钉死

**产出：** `docs/ae_problem_depth.md`

- 科学问题一句话钉死：硬安全与非凸动作下，如何在与扁平 TD3 **相同外在经济目标** 上稳定学到库存协调。  
- 难点 D1–D4、机制 M1–M4、可证伪命题 P1–P4。  
- 投稿贡献 C1–C3（问题 / 方法 / 实证），避免“未开启的 market prior”冒充主贡献。

**深度提升：** 从“训了一个层次模型”变为“检验层次+安全+非凸表示是否解决有效样本稀疏与长时域协调”。

---

## Round 2 — 算法加固（稳定性 + 安全学习可测）

### 2.1 SAC 温度爆炸修复

**文件：** `src/training/hybrid_sac/algorithm.py`

| 措施 | 作用 |
|------|------|
| `alpha ∈ [1e-4, 10]`（log 空间 clamp） | 阻止 α→1e17 |
| target Q / actor Q clip | 抑制 Q 发散 |
| 梯度 clip | 数值稳定 |
| non-finite 时 **跳过更新** 而非杀进程 | 过渡/夏季 SAC 可继续跑完 |

对应此前 `transition_sac` / `summer_sac` 失败根因。

### 2.2 HMSD 安全—学习耦合指标

**文件：** `src/training/ghtd3/train.py`

`train_result` 新增 `safety_learning`：

- `reject_rate`、`givesafe_reject`、`physical_ok`、`learn_from_reject` 等  

**深度提升：** 结果不仅比经济 KPI，还能报告“安全拒绝是否随学习下降”（P2 证据通道）。

---

## Round 3 — 消融实验入口（P1–P3）

**配置：**

| 文件 | 命题 |
|------|------|
| `src/config/ghtd3_config.yaml` | 完整 HMSD |
| `src/config/ablation/ghtd3_no_her.yaml` | P3：无历史目标重标 |
| `src/config/ablation/ghtd3_no_reject_learn.yaml` | P2：安全只过滤不入库 |
| `train_seasonal --method td3` | P1：无层次 |

**脚本：** `scripts/run_paper_ablations.py`（支持 smoke 200 ep 与 dry-run）  
**说明：** `src/config/ablation/README.md`

正式建议：冬季 5000 ep × seed0（或 0–2），再汇总。

---

## Round 4 — 结果汇聚工具

**脚本：** `scripts/aggregate_fair_results.py`

- 扫描 `runs/seasonal_v1/**/train_result.json`  
- 输出 per-seed 表 + season×method 均值±标准差  
- HMSD vs TD3 分季节均值比较  
- 兼容 HMSD / TD3 / SAC / PSO / linprog 字段差异  

用法：

```bash
# 将远程 runs/seasonal_v1 拉回后
python scripts/aggregate_fair_results.py --root runs/seasonal_v1 --out docs/ae_results_table.md
```

---

## Round 5 — 投稿判断与后续清单

### 5.1 服务器 fair 队列现状（2026-08-12 探活）

- `QUEUE_DONE`：调度结束  
- 完成约 28/33；SAC 4 失败 + 1 异常（数值发散，**应用 Round2 后需重跑**）  
- **HMSD 三季×3 seed 全完成**；**TD3 三季×3 seed 全完成**

### 5.2 深度是否够投 Applied Energy？

| 项 | 迭代前 | 迭代后 |
|----|--------|--------|
| 问题锋利度 | 偏“能跑的调度” | 有可证伪命题与 C1–C3 |
| 算法独特性 | 组合件为主 | 安全—学习可测 + SAC 可稳定基线 |
| 消融 | 缺入口 | 配置+脚本就绪（**实验待跑**） |
| 结果叙事 | 零散探活 | 汇聚脚本就绪；**正式总表待拉远程结果生成** |

**结论：** 深度**明显补了一层“可投稿骨架”**，但 **消融实验与远程结果正式总表仍未跑完**，不能声称“已达投稿终态”。  
合理表述：**具备 Applied Energy 应用方法文的问题—方法—实验框架；再完成消融 + 结果表 + 英文撰写即可进入投稿冲刺。**

### 5.3 建议你方立刻执行的 3 步

1. 同步 Round2 SAC 修复到服务器，重跑失败的 `transition/summer sac`（可选但推荐）。  
2. 拉回 `runs/seasonal_v1`，运行 `aggregate_fair_results.py` 生成主表。  
3. 开 `run_paper_ablations.py --episodes 5000`（至少冬季 seed0 全套消融）。

### 5.4 英文 Highlights 草案（可改）

1. Hierarchical goal-conditioned RL for multi-energy dispatch under hard safety and nonconvex CAES sets.  
2. Same external economic objective as flat TD3 for fair comparison; held-out seasonal weeks.  
3. Reject-aware learning couples GiveSafe filtering with low-level value updates.  
4. Multi-season multi-seed evidence of improved stability and economics vs flat TD3.  
5. Ablations isolate hierarchy, hindsight relabeling, and reject learning.

---

## 文件清单（本五轮）

| 路径 | 轮次 |
|------|------|
| `docs/ae_problem_depth.md` | R1 |
| `src/training/hybrid_sac/algorithm.py` | R2 |
| `src/training/ghtd3/train.py` | R2 |
| `src/config/ablation/*` | R3 |
| `scripts/run_paper_ablations.py` | R3 |
| `scripts/aggregate_fair_results.py` | R4 |
| `docs/ae_five_round_iteration.md` | R5（本文） |
