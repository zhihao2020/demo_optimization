# 论文结果清单与跨机使用说明

**目的**：另一台电脑 `git pull` / `clone` 后，无需访问 `172.16.1.80`、无需重训，即可用仓库内训练结果复现论文表图、加载 checkpoint 做评测。

**主文**：`Paper/main.tex`（Safe Market-GHTD3 vs 典型单层 TD3-scratch）。

---

## 1. 保留的 `runs/` 目录地图

| 路径 | 用途 |
|------|------|
| `ghtd3_abs_s{0,1,2}_35k/` | 主 multi-seed GHTD3：`checkpoints/ghtd3.pt`、`vs_td3.json`、`train/step_log.json` |
| `td3_scratch_s{0,1,2}_35k/` | 主 multi-seed TD3-scratch：`checkpoints/hybrid_givesafe_td3.pt`、`step_log` |
| `ghtd3_abs/multi_seed_summary_paired.json` | 配对 multi-seed 汇总 |
| `paper_baselines_or_pso.json` | B0 / linprog / PSO 主表数字 |
| `paper_dispatch_traj/` | 三季功率/SoC/经济轨迹 CSV（画平衡图） |
| `paper_annual_weekly_reset.json` | 协议 A：52 周 + 尾窗滑动年评 |
| `ghtd3_sens_summary.json` | \(c,\alpha\) 敏感性汇总 |
| `ghtd3_sens_*_s0_15k/` | 敏感性各点 ckpt + `vs_td3.json` |
| `ghtd3_abs_abl_*_s0_15k/` | 消融 MSGP / MS-HER / F-MLE |
| `ghtd3_abs_s2_lambda_v3_35k/` | \(\lambda\)-SoC 敏感性（seed 2） |

每个 run 内建议关注：`checkpoints/`、`summary.json`、`vs_td3*.json`、`config/`、`train/step_log.json`（主 run）。

---

## 2. 冻结数字（与正文主表一致）

### 2.1 主表（episode reward，3 seed mean；RL 为 \(3.5\times10^4\) steps）

| Season | B0 | linprog | PSO | TD3-scratch | Safe Market-GHTD3 | \(\Delta\) vs TD3 |
|--------|-----|---------|-----|-------------|-------------------|-------------------|
| Winter | 67.5 | 54.7 | 1.8 | \(99.6\pm8.2\) | \(113.7\pm8.5\) | +14.0 |
| Transition | 52.7 | 51.4 | 69.6 | \(42.2\pm33.0\) | \(98.4\pm5.7\) | +56.1 |
| Summer | 20.7 | 5.9 | 24.0 | \(37.1\pm20.6\) | \(73.6\pm12.4\) | +36.5 |

来源：`paper_baselines_or_pso.json` + `ghtd3_abs/multi_seed_summary_paired.json`。

### 2.2 GHTD3 分 seed（`vs_td3`）

| seed | Winter | Transition | Summer |
|------|--------|------------|--------|
| 0 | 106.8 | 94.7 | 71.0 |
| 1 | 123.1 | 95.5 | 87.1 |
| 2 | 111.1 | 104.9 | 62.6（SoC fail） |

### 2.3 年评协议 A（8760 h 滑动周 reset，seed 1）

| Method | Windows | Reward mean±std | SoC pass |
|--------|---------|-----------------|----------|
| B0 | 53 | 50.6±18.2 | 51/53 (96.2%) |
| TD3-scratch | 53 | 56.7±31.9 | 27/53 (50.9%) |
| GHTD3 | 53 | 99.1±17.6 | 48/53 (90.6%) |

来源：`paper_annual_weekly_reset.json`。

### 2.4 敏感性 \(c\) / \(\alpha_{\mathrm{end}}\)（seed 0，15k，三季 mean）

| \(c\) | mean | \(\alpha_{\mathrm{end}}\) | mean |
|-------|------|---------------------------|------|
| 4 | 75.5 | 0.10 | 92.6 |
| **8** | **106.1** | **0.22** | **106.1** |
| 12 | 105.0 | 0.35 | 100.2 |
| 24 | 102.7 | 0.50 | 90.3 |
| | | 0.70 | 88.1 |

来源：`ghtd3_sens_summary.json`。

---

## 3. 另一台电脑：最短用法

### 3.1 只重画论文图（通常不需要 FMU）

```bash
# 依赖：Python + matplotlib + numpy + pandas
python scripts/plot_paper_figures_cui_style.py
python scripts/plot_paper_figures_v2.py   # 主柱图/训练曲线等
```

数据来自 `runs/paper_dispatch_traj/`、`runs/ghtd3_sens_summary.json`、各 run 的 `step_log`。

### 3.2 核对表数字

直接读 JSON：

- `runs/paper_baselines_or_pso.json`
- `runs/ghtd3_abs/multi_seed_summary_paired.json`
- `runs/*/vs_td3.json`
- `runs/paper_annual_weekly_reset.json`

### 3.3 加载 checkpoint 再 eval（需要本机 FMU / 环境配置）

```bash
# 示例：三季 GHTD3 vs TD3
python scripts/eval_ghtd3_vs_td3.py \
  --ghtd3 runs/ghtd3_abs_s1_35k/checkpoints/ghtd3.pt \
  --td3 runs/td3_scratch_s1_35k/checkpoints/hybrid_givesafe_td3.pt \
  --config src/config/ghtd3_config_abs.yaml \
  --out runs/recheck_s1_vs_td3.json

# 年评协议 A
python scripts/eval_annual_weekly_reset.py \
  --ghtd3-ckpt runs/ghtd3_abs_s1_35k/checkpoints/ghtd3.pt \
  --td3-ckpt runs/td3_scratch_s1_35k/checkpoints/hybrid_givesafe_td3.pt \
  --out runs/paper_annual_weekly_reset_recheck.json
```

FMU 路径与缓存见 `src/config/paths.py` / `env_config.yaml`；仅画图时可跳过。

### 3.4 配置入口

- 主算法：`src/config/ghtd3_config_abs.yaml`
- 消融：`ghtd3_config_abs_no{prior,her,fmle}.yaml`
- 敏感性：`ghtd3_config_abs_sens_*.yaml`
- \(\lambda\)-SoC：`ghtd3_config_abs_lambda.yaml`

---

## 4. 刻意未纳入仓库的内容

- 旧 Hybrid / TEA / LTAR / STFR / 调试 / smoke / probe 全量 run  
- `runs/remote_pull*` 重复镜像  
- 会话脚本 `_remote_*.py`、`_check_*.py`（运维本机 172.16.1.80）  
- 真连续年 carry 全轨迹（主文以周 reset 为主；数字见年评 JSON）

远程训练机 `172.16.1.80` 可另作备份，**论文复现以本 git 树为准**。

---

## 5. 相关脚本

| 脚本 | 作用 |
|------|------|
| `scripts/plot_paper_figures_v2.py` | 主经济柱图、训练曲线、SoC 等 |
| `scripts/plot_paper_figures_cui_style.py` | 平衡/SoC 矩阵/成本/敏感性/消融条 |
| `scripts/export_paper_dispatch_traj.py` | 导出调度 CSV（需 FMU） |
| `scripts/eval_ghtd3_vs_td3.py` | 三季对照 |
| `scripts/eval_paired_seeds.py` | multi-seed 配对 |
| `scripts/eval_annual_weekly_reset.py` | 年评协议 A |
| `scripts/run_ghtd3_sensitivity_grid.py` | \(c/\alpha\) 网格训练编排 |

---

*生成目的：跨机论文复现；与 `Paper/main.tex` 当前主线一致。*
