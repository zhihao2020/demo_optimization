# 风光火储综合能源系统 · FMU + RL 优化 Demo

文档更新：2026-08-30 22:40 (+08:00)

基于 **Modelica/FMU 物理仿真** 的厂级风光–火电–BESS–绝热 CAES 周调度仓库。电是唯一出售载体。论文 live 方法是 **PC-HybridTD3**：单层参数化混合 TD3，Actor 把 `(mode, magnitude)` 解码到状态相关支撑 \(\mathcal A_f(s)\)，Critic 看 hybrid packing，经济 Bellman 只用 FMU 真转移。**不是** FS-HSAC/SAC，**不是** HMSD/GHTD3/库存 HRL。对照：投影连续 TD3、滚动 MILP、price-aware rule。GiveSafe 采用；`storage_use` 关闭。结果表在 Stage D 之前保持空。

> 物理动作三元组：`u_tp`, `u_battery`, `u_caes`。CAES 合法集仍是非凸三段（放电 / 待机 / 充电）。论文策略用混合 `(mode, magnitude)` 直接落在 \(\mathcal A_f(s)\) 上，而不是盒动作再投影。  
> 论文入口：`scripts/train_seasonal.py --method td3 --season all`。投影消融 `--ablation projection`；静态支撑 `--ablation static-support`。FS-HSAC / HMSD 仍在代码树中，但不是论文身份。

文档索引：[docs/README.md](docs/README.md)。口径：[docs/paper_outline_and_figures.md](docs/paper_outline_and_figures.md)、[docs/ae_contributions_zh.md](docs/ae_contributions_zh.md)。

---

## 功能概览

- **物理层**：FMI 3.0 Co-Simulation（`fmpy`），固定步长默认 3600 s
- **环境**：Gymnasium，`PowerSystemEnv`，Dict 物理动作 + 可选 24 h 日前 forecast 观测
- **安全**：GiveSafe（一级 FeasibilityOracle + 可选 Shadow FMU），禁止规则 fallback
- **论文主线**：PC-HybridTD3（`--method td3`）；**对照**：投影 TD3、滚动 MILP、price-aware rule；FS-HSAC / HMSD 仍可跑，但不是 live claim
- **报告**：训练后自动生成 `report/report.md`（收益元、动作摘要、对比图）
- **规则基线**：高火电 + 储能 IDLE 与随机可行采样

---

## 目录结构

```text
demo_optimization/
├── data/                      # FMU、风光负荷/环境 CSV
├── docs/                      # 活文档 + 实验快照（见 docs/README.md）
├── scripts/                   # CLI：训练、评估、标定、探查；scripts/qmd 文档索引
├── src/
│   ├── actions/               # 物理动作、caes_u、可行域 Oracle、validator
│   ├── config/                # YAML 主配置；legacy/ 为归档变体
│   ├── controllers/           # 规则基线
│   ├── envs/                  # PowerSystemEnv、reward、forecast
│   ├── fmu/                   # FMI 会话、校验、变量注册
│   ├── replay/                # Physical + GiveSafe 分区 buffer
│   ├── safety/                # GiveSafe 控制器与 Shadow FMU
│   └── training/              # hybrid_td3（论文 PC-HybridTD3）、fs_hsac / hybrid_sac / ghtd3 归档可跑
├── tests/                     # pytest
├── requirements.txt
└── README.md
```

本地运行产物默认落在 `runs/`（部分实验产物已入库；拉数目录见 `.gitignore`）。

---

## 环境与依赖

- **Python**：建议 3.10+
- **平台**：仓库自带的 `.fmu` 需包含当前 OS 原生二进制（Windows 为 `.dll`）。在无对应二进制的机器上会直接报错。
- **安装**（仓库根目录）：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

多数脚本会把 `src/` 加入 `sys.path`；交互调试时可设置：

```bash
# Windows PowerShell
$env:PYTHONPATH = "src"
# Linux/macOS
export PYTHONPATH=src
```

---

## 快速开始

以下命令均在**仓库根目录**执行。

### 1. 检查 FMU 接口

```bash
python scripts/inspect_fmu.py
```

### 2. 环境 smoke（reset + 规则一步）

```bash
# 需能 import envs（脚本未插 path 时请先设 PYTHONPATH=src）
$env:PYTHONPATH = "src"   # PowerShell
python scripts/smoke_test_env.py
```

### 3. 规则基线滚一周

```bash
python scripts/rollout_rule_controller.py
```

轨迹与摘要写入 `runs/rule_controller/`。

### 4. 基线对比（规则 vs 随机可行）

```bash
python scripts/evaluate_baselines.py
```

### 5. compare 开环方案 → FMU 回放

将 `compare/output1.py` / `output2.py` 生成的全年调度序列映射为 FMU 输入并开环回放。  
电池/CAES 会按符号约定取反后写入；若落在 CAES 禁区等非法区间，**接入校验阶段直接报错**（不投影）。

```bash
python scripts/rollout_compare.py --scheme both --hours 168
# 全年：
python scripts/rollout_compare.py --scheme both --hours 8760
```

轨迹与摘要写入 `runs/compare/`。

### 6. GiveSafe 训练（论文矩阵）

论文主线与对照均走物理动作三元组 + GiveSafe，soft_shell OFF。

```bash
# 论文 live：PC-HybridTD3
python scripts/train_seasonal.py --method td3 --season all --episodes 200 --seed 0

# 消融：连续投影 / 静态支撑
python scripts/train_seasonal.py --method td3 --ablation projection --season winter --seed 0
python scripts/train_seasonal.py --method td3 --ablation static-support --season winter --seed 0

# 滚动 MILP（代理优化，FMU 评估）
python scripts/train_seasonal.py --method milp --season winter --seed 0

# 仍可用、但不是论文身份
python scripts/train_seasonal.py --method fs_hsac --support --season winter --seed 0
python scripts/train_ghtd3.py --mode smoke
```

训练结束后默认阅读入口：

```text
runs/<run_name>/report/report.md
```

内含策略 vs 规则的 **累计现金流（元）**、动作行为摘要，以及 `actions.png` / `cashflow.png` / `soc.png`。  
对已有 run 可补生成：

```bash
python scripts/generate_policy_report.py --run-dir runs/givesafe_td3_smoke
```

### 7. 测试

```bash
$env:PYTHONPATH = "src"
pytest tests/ -q
```

---

## 配置指针

| 文件 | 作用 |
|------|------|
| [`src/config/ghtd3_config.yaml`](src/config/ghtd3_config.yaml) | HMSD 栈（非论文身份）：`goal_conditioned`、2D goal、`low_reward: ext` |
| [`src/config/ghtd3_config_seasonal_min.yaml`](src/config/ghtd3_config_seasonal_min.yaml) | 季节公平训练用（与主线同栈） |
| [`src/config/legacy/`](src/config/legacy/) | 历史 TEA / residual / 消融配置归档（非主线） |
| [`src/config/env_config.yaml`](src/config/env_config.yaml) | FMU 路径、步长、episode 长度、动作/观测清单、forecast CSV |
| [`src/config/reward_config.yaml`](src/config/reward_config.yaml) | 电价、火电成本、`C_ref`、终端 SOC bonus |
| [`src/config/givesafe_config.yaml`](src/config/givesafe_config.yaml) | GiveSafe 重采样次数；经济 replay `physical_fraction: 1.0` |
| [`src/config/device_params.yaml`](src/config/device_params.yaml) | 设备物理参数（Oracle / 预测） |
| [`src/config/feasibility_margins.yaml`](src/config/feasibility_margins.yaml) | 可行域裕度（可标定） |

更细的文档：

- [docs/README.md](docs/README.md) — 文档索引 + qmd 对齐协议
- [docs/cui_seasonal_min_protocol.md](docs/cui_seasonal_min_protocol.md) — 公平季节对比
- [docs/FMU输入上下限.md](docs/FMU输入上下限.md) — `u_tp` / `u_battery` / `u_caes` 边界
- [docs/RL奖励于成本配置.md](docs/RL奖励于成本配置.md) — 经济奖励与 GiveSafe 约束奖励路径
- [docs/风光负荷数据说明.md](docs/风光负荷数据说明.md) — `data/*.csv` 字段与单位
- [docs/修改modelica模型.md](docs/修改modelica模型.md) — 改 Modelica / 重导 FMU 时注意点

---

## 架构与数据流

```text
策略输出 PhysicalAction {u_tp, u_battery, u_caes}
        │  caes_u.project / mode_from_u（mode 仅锁/日志）
        ▼
 validator / physical_from_dict ──► PhysicalFmuAction
        │
        ▼
 GiveSafeController ──► Oracle 预检 (+ 可选 Shadow FMU)
        │ 安全
        ▼
   FmuAdapter / FmuSession ── doStep(3600s) ──► 物理输出
        │
        ├── 合法物理步 → RewardCalculator → PhysicalReplay（经济）
        └── 拒绝 / 硬失败 → 约束奖励或 truncated，不进经济 buffer
```

要点：

1. **FMU 是真值**：风光荷边界由模型内嵌时序驱动；`data/*.csv` 只扩展策略观测（完美日前），不是替代物理源。
2. **动作合法化**：`caes_u.project_u_caes` 把连续标量落到合法三段；越界/Oracle 拒绝由 GiveSafe 重采样或失败，**无规则 fallback**。
3. **奖励三条路径**：物理经济步 / GiveSafe 拒绝自环 / 环境硬失败（详见奖励文档）。
4. **论文策略**：PC-HybridTD3 把 `(m,z)` 解码到 \(\mathcal A_f(s)\)；GiveSafe 是采用的执行筛，不是贡献。FS-HSAC / HMSD 仍在代码里，但不是论文身份。

```mermaid
flowchart LR
  Actor[PC-HybridTD3 on A_f(s)] --> GS[GiveSafe / Oracle]
  GS -->|safe| FMU[FMU Session]
  GS -->|reject| Reject[safety audit; no Bellman self-loop]
  FMU --> Env[PowerSystemEnv]
  Env -->|r_ext| PhysBuf[Bellman Replay]
  PhysBuf --> Upd[Hybrid TD3 update]
```

---

## 注意事项

- **FMU 路径**：默认 `data/TypicalScensrio_Example_TypicalScene_PowerSystem_8760h.fmu`（文件名拼写沿用现有产物）。更换模型需同步 `env_config` 变量名与边界文档。
- **Windows**：需 FMU 内含 Windows 二进制；解压目录由 `fmpy.extract` 管理，`close()` 时清理。
- **时长**：`episode_steps: 168`（一周）；`--annual-eval` 覆盖 8760 h，耗时显著增加。
- **标定脚本**：`scripts/calibrate_reward*.py`、`calibrate_feasibility_margins.py` 会改写配置或生成报告，正式训练前请确认 `reward_config` 中 `cost_reference` / `terminal_soc` 已就绪。

---

## 许可与归属

本仓库为综合能源调度 RL 演示与研究代码。Modelica 源与 FMU 导出流程以 `docs/` 为准。
