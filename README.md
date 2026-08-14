# 风光火储综合能源系统 · FMU + RL 优化 Demo

文档更新：2026-08-10 20:45 (+08:00)

基于 **Modelica/FMU 物理仿真** 与 **HMSD/GHTD3 + Hybrid 基线** 的电力系统调度强化学习示例仓库。  
FMU 负责火电、电池、CAES（压缩空气储能）与风光荷的一小时步物理演化；Python 侧负责动作空间、动态可行域、GiveSafe 安全过滤与经济奖励。

> 物理动作三元组：`u_tp`, `u_battery`, `u_caes`。CAES 合法集仍是非凸三段（放电 / 待机 / 充电）；策略直接输出连续 `u_caes`，由 `src/actions/caes_u.py` 做合法投影，**mode 仅派生**（锁/最短运行/日志），**不再有独立 mode/magnitude 动作维**。  
> 主线：HMSD/GHTD3（`execution_mode: goal_conditioned`，2D SoC goal，`low_reward: ext`）。基线：Hybrid-TD3 / Hybrid-SAC。**无 Hybrid-PPO、无 hybrid residual teacher**。入口见 `src/training/ghtd3/`、`hybrid_td3/`、`hybrid_sac/`。

文档索引与 qmd 对齐流程：[docs/README.md](docs/README.md)。

---

## 功能概览

- **物理层**：FMI 3.0 Co-Simulation（`fmpy`），固定步长默认 3600 s
- **环境**：Gymnasium，`PowerSystemEnv`，Dict 物理动作 + 可选 24 h 日前 forecast 观测
- **安全**：GiveSafe（一级 FeasibilityOracle + 可选 Shadow FMU），禁止规则 fallback
- **训练主线**：HMSD/GHTD3（连续 `u_caes` + 2D 目标条件层次 TD3）；**基线**：Hybrid-TD3 / Hybrid-SAC
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
│   └── training/              # ghtd3 (HMSD)、hybrid_td3、hybrid_sac、评估报告
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

### 6. GiveSafe 训练（HMSD / Hybrid-TD3 / Hybrid-SAC）

主线与基线均走物理动作三元组 + GiveSafe。CLI 常用参数：`--mode smoke|short|formal`、`--steps`、`--seed`、`--run-dir`、`--no-shadow`、`--annual-eval`。

```bash
# 主线 HMSD / GHTD3（连续 u_caes；公平季节协议见 docs/cui_seasonal_min_protocol.md）
python scripts/train_ghtd3.py --mode smoke
python scripts/train_seasonal.py --method hmsd --season winter --episodes 200 --seed 0
python scripts/train_seasonal.py --method td3  --season winter --episodes 200 --seed 0

# 基线 Hybrid-TD3 / Hybrid-SAC（无 PPO）
python scripts/train_hybrid_td3.py --mode smoke
python scripts/train_hybrid_sac.py --mode smoke
python scripts/train_hybrid_td3.py --mode short --seed 0
python scripts/train_hybrid_sac.py --mode formal --annual-eval
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
| [`src/config/ghtd3_config.yaml`](src/config/ghtd3_config.yaml) | HMSD 主线：`goal_conditioned`、2D goal、`low_reward: ext`、HER-mix |
| [`src/config/ghtd3_config_seasonal_min.yaml`](src/config/ghtd3_config_seasonal_min.yaml) | 季节公平训练用（与主线同栈） |
| [`src/config/legacy/`](src/config/legacy/) | 历史 TEA / residual / 消融配置归档（非主线） |
| [`src/config/env_config.yaml`](src/config/env_config.yaml) | FMU 路径、步长、episode 长度、动作/观测清单、forecast CSV |
| [`src/config/reward_config.yaml`](src/config/reward_config.yaml) | 电价、火电成本、`C_ref`、终端 SOC bonus |
| [`src/config/givesafe_config.yaml`](src/config/givesafe_config.yaml) | GiveSafe 重采样次数、约束奖励、replay 混合比例（约 70/30） |
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
4. **HMSD 分层**：高层每 `subgoal_interval`（默认 8）发 2D SoC goal；底层 goal-conditioned 出物理三元组；公平对比时底层用 `r_ext`（与 flat TD3 同目标）。

```mermaid
flowchart LR
  High[High-level Actor goal] --> Low[Low-level Actor]
  Low --> GS[GiveSafe / Oracle]
  GS -->|safe| FMU[FMU Session]
  GS -->|reject| GSBuf[GiveSafe Replay]
  FMU --> Env[PowerSystemEnv]
  Env -->|economic r_ext| PhysBuf[Physical Replay]
  PhysBuf --> Upd[HMSD / Hybrid TD3 Update]
  GSBuf --> Upd
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
