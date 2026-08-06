# 论文对照：典型单层 TD3 vs 绝对 Safe Market-GHTD3

## 拍板

| 角色 | 方法 | 配置 / 入口 |
|------|------|-------------|
| **典型单层 baseline** | TD3 from-scratch（GiveSafe，无规则示范主导） | `scripts/train_hybrid_td3.py --mode scratch` |
| **主方法** | Safe Market-GHTD3 = 绝对 GC + **MSGP + MS-HER + F-MLE + GiveSafe** | `src/config/ghtd3_config_abs.yaml` |
| **去掉** | Hybrid 强教师、`a=a_H+βΔ` 残差主线 | 代码保留，非主叙事 |

原理全文：`docs/Safe_Market_GHTD3_principles.md`。

## 训练命令

### 本地 / 远程单 seed

```bash
# 1) 典型单层 TD3
python scripts/train_hybrid_td3.py --mode scratch --steps 35000 --seed 0 \
  --run-dir runs/td3_scratch_s0_35k --no-shadow

# 2) 绝对 GHTD3
python scripts/train_ghtd3.py --mode custom --steps 35000 --seed 0 \
  --run-dir runs/ghtd3_abs_s0_35k --config src/config/ghtd3_config_abs.yaml

# 3) 三季对比
python scripts/eval_ghtd3_vs_td3.py \
  --ghtd3 runs/ghtd3_abs_s0_35k/checkpoints/ghtd3.pt \
  --td3 runs/td3_scratch_s0_35k/checkpoints/hybrid_givesafe_td3.pt \
  --out runs/ghtd3_abs_s0_35k/vs_td3.json
```

### 远程打包（`_remote_bootstrap_tea.py`）

```bash
# TD3-scratch multi-seed
python _remote_bootstrap_tea.py --kind td3_scratch --steps 35000 --seeds 0,1,2 \
  --run-prefix runs/td3_scratch --log-prefix td3 --config src/config/ghtd3_config_abs.yaml

# abs GHTD3（训完后用 s0 TD3 ckpt 做 eval）
python _remote_bootstrap_tea.py --kind ghtd3 --steps 35000 --seeds 0,1,2 \
  --run-prefix runs/ghtd3_abs --log-prefix abs \
  --config src/config/ghtd3_config_abs.yaml \
  --td3-ckpt runs/td3_scratch_s0_35k/checkpoints/hybrid_givesafe_td3.pt
```

> 建议：先训完 `td3_scratch_s0`，再开 GHTD3 多 seed，以便 bat 内 eval 指向固定 TD3 ckpt。

## 训前门禁

```bash
python scripts/diagnose_ghtd3_goal_sensitivity.py \
  --config src/config/ghtd3_config_abs.yaml \
  --fresh-only \
  --out runs/diagnose_abs_goal.json
```

期望：fresh `goal_conditioned` 下扫 5 维 goal 时动作有响应（非死通路）。

## 成功标准

- 三季 GHTD3 reward **显著高于** TD3-scratch
- 主文 **不依赖** Hybrid 数字
- 多 seed SOC / reward 可报告

## 与旧 ares 关系

`ghtd3_ares_35k` + Hybrid 残差仅作工程附录；**不**作主 claim。
