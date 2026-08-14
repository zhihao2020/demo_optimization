# 文档索引与代码对齐

文档更新：2026-08-10 21:15 (+08:00)

本目录是 **活文档 + 实验快照**。代码真源在 `src/`；训练/评测入口在 `scripts/`。  
用 **qmd** 管理检索与对齐，不要凭记忆改 API/入口叙述。

## 主线事实（与代码一致）

| 项 | 真源 |
| --- | --- |
| 方法名 | **HMSD / GHTD3**（`src/training/ghtd3/`） |
| 基线 | Hybrid-TD3、Hybrid-SAC（**无 Hybrid-PPO**） |
| 物理动作 | `u_tp`, `u_battery`, `u_caes`（连续 CAES；mode 仅派生用于锁/日志） |
| CAES 表示 | `src/actions/caes_u.py`（合法三段投影） |
| 主配置 | `src/config/ghtd3_config.yaml`、`ghtd3_config_seasonal_min.yaml` |
| 旧配置 | `src/config/legacy/` |
| 季节公平协议 | [cui_seasonal_min_protocol.md](cui_seasonal_min_protocol.md) + `scripts/train_seasonal.py` |
| 安全 | GiveSafe（Oracle；默认无 Shadow） |

## 文档与代码对齐流程

```text
改代码（语义 / 契约 / 行为）
  → 仓库根 qmd search/query 找齐 living docs / README
  → 同工作单元改文档（按代码真源）
  → 实质修改页戳：文档更新：YYYY-MM-DD HH:MM (+08:00)
  → powershell -File scripts\qmd\reindex.ps1
  → git add 代码 + 文档同一 commit
```

| 变更类型 | 优先文档 |
| --- | --- |
| 动作空间 / CAES / Validator / GiveSafe | 根 `README.md`、`FMU输入上下限.md`、算法 living |
| 训练算法 / goal / buffer / 网络 | `GHTD3分层实现说明.md`、`principle_innovation_MIF_HRL.md` |
| 训练入口 / seasonal / fair eval | `cui_seasonal_min_protocol.md`、根 `README.md` |
| 配置路径 / legacy | 根 `README.md`、本页 |
| 仅内部重构、无外部行为 | **不**改 living doc、不戳时间戳 |
| 新实验结果 | results 快照页 + `paper_results_manifest.md` |

**Living**（须随行为变更更新）：算法/协议/接口/数据字典。  
**Snapshot**（可滞后，但禁止与当前 API/入口矛盾）：消融表、三季结果、论文表格草稿。

## qmd 用法

前置：`npm install -g @tobilu/qmd`（本机已装可跳过）。

```powershell
# 首次 / 改 collection 模板后
powershell -File scripts\qmd\setup_collections.ps1

# 文档改完后
powershell -File scripts\qmd\reindex.ps1

# 验收
powershell -File scripts\qmd\smoke_search.ps1

# 检索（务必在仓库根）
qmd search "HMSD GHTD3 seasonal" -c docs-algo --format files -n 8
qmd search "FMU u_caes" -c docs-env --format files -n 8
qmd search "GiveSafe" -c docs-all --format files -n 10
```

| Collection | 范围 |
| --- | --- |
| `readme` | 根 `README.md` |
| `docs-algo` | 算法 / 原理 / seasonal protocol（英前缀为主） |
| `docs-env` | FMU / 奖励 / 数据 / Modelica（英前缀为主） |
| `docs-protocol` | 协议与就绪度 |
| `docs-results` | 结果快照与论文草稿 |
| `docs-all` | `docs/**/*.md` 全量（中文文件名兜底） |

索引目录 `.qmd/` 已 gitignore；模板在 `scripts/qmd/qmd.index.yml`。

Agent skill：`.agents/skills/optimal-docs-search/SKILL.md`。

## 推荐阅读顺序

1. 根 [README.md](../README.md) — 安装与 CLI  
2. [cui_seasonal_min_protocol.md](cui_seasonal_min_protocol.md) — 公平季节对比  
3. [GHTD3分层实现说明.md](GHTD3分层实现说明.md) — 与论文映射 + 创新点（**代码真源对齐**）  
4. [优化问题形式化说明.md](优化问题形式化说明.md) — 决策变量 / 目标  
5. [FMU输入上下限.md](FMU输入上下限.md) / [RL奖励于成本配置.md](RL奖励于成本配置.md)  
6. [paper_results_manifest.md](paper_results_manifest.md) — 结果文件索引  

## 已删除（与代码不一致的过时方法稿）

下列文档描述已移除或非主线实现（LTAR / STFR / TEA residual / 5D-MSGP-F-MLE 主叙事 / ares 对照表），**已删除**，勿再引用：

- `LTAR_formulation.md`、`STFR_TRAP_formulation.md`
- `GHTD3算法改进说明.md`（历史 residual/TEA 日志）
- `Safe_Market_GHTD3_principles.md`（旧 5D + F-MLE 主栈叙述）
- `论文对照_典型TD3_absGHTD3.md`、`论文对照_弱单层SAC.md`

归档配置仅在 `src/config/legacy/`，**无**对应可运行训练路径承诺。

## Living vs snapshot 清单（简表）

**Living**：`GHTD3分层实现说明.md`、`principle*`、`cui_*`、`FMU*`、`RL*`、`data_dictionary*`、`economic*`、`modelica*`、形式化/协议类、根 README。  
**Snapshot**：`*结果*.md`、`paper_*`、`P0_*`、`*草稿*`、消融/三季数字表（可滞后，但禁止写已删 API）。
