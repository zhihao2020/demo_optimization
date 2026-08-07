# 历史说明：现金目标（非综合成本）下的 SAC / HMSD

> **失效声明**：下列数字与 checkpoint 对应的是 **TOU 现金 / 旧 reward** 优化问题，  
> **不是** 当前默认的综合货币目标  
> \(J^{\mathrm{gen}} =\) cash − ETS 碳 − \(C^{CUT}\) − \(C^{\mathrm{deg}}\)。  
> **不可与主表、新 5000×168 全量训练结果混用或直接对比。**

当前主协议（与 Cui 对齐）：

- \(E_{\max} = 5000\) 周 episode，\(T = 168\) h  
- 等价 valid steps \(= 5000 \times 168 = 840{,}000\)  
- run 命名 **无 `cc` 后缀**（综合成本为默认目标）

## 保留的历史说明文档

| 文件 | 内容（旧问题） |
|------|----------------|
| `runs/paper_hmsd80k_scratch_summary.md` | HMSD from-scratch 80k vs SAC-80k |
| `runs/paper_hsac_status.md` | Hierarchical-SAC 35k/80k |
| `docs/论文对照_弱单层SAC.md` | 弱单层 SAC 角色与旧主表数字 |
| `docs/扩展基准_linprog_SAC.md` | 旧 linprog/SAC 扩展表 |
| `docs/论文表格草稿_三季PSO对比.md` | 旧三季 PSO 草稿 |

## 旧预算习惯（已废弃）

| 预算 | 旧用途 |
|------|--------|
| 15k | 短训消融/敏感性 |
| 35k | 旧主表 |
| 80k | 加长对照 SAC |

Cui 原文使用 **5000 episodes × 168 h**，并非 35k/80k。

## 清理

旧目标下的 `runs/*/checkpoints`、轨迹与训练 log 已按协议删除；仅保留上表文字说明。
