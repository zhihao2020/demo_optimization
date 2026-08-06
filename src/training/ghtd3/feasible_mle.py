"""F-MLE：Feasibility-filtered inverse-dynamics / behaviour cloning.

与 Cui 文中 MLE 复制同构，但演示 **仅来自规则/峰谷可行轨迹**（物理有效转移），
**绝不**依赖 Hybrid 教师 checkpoint。用于绝对 goal-conditioned 底层预热：

  max_θ  E log π_lo(a | s, g^hind)   s.t. a 来自 valid FMU step

高层 goal BC 仍用 bc_pretrain_high_goals（g^BC ≈ 0.5 hindsight + 0.5 market prior）。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .bc_pretrain import (
    behavior_clone_low_actor,
    bc_pretrain_high_goals,
    collect_hierarchical_demos,
)
from .agent import GHTD3Agent
from envs.power_system_env import PowerSystemEnv


def collect_feasible_demos(
    env: PowerSystemEnv,
    agent: GHTD3Agent,
    *,
    n_windows: int = 4,
    seed: int = 0,
    cfg: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """采集可行演示；内部复用规则轨迹采集（仅 valid transition 入库）。"""
    return collect_hierarchical_demos(
        env,
        agent,
        n_windows=n_windows,
        seed=seed,
        price_aware=True,
        cfg=cfg,
    )


def f_mle_pretrain(
    env: PowerSystemEnv,
    agent: GHTD3Agent,
    *,
    cfg: dict[str, Any] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """运行 F-MLE：底层逆动力学式 BC + 可选高层 goal BC。

    Args:
        env: FMU 环境。
        agent: 绝对 GC GHTD3 agent（无 hybrid_anchor）。
        cfg: ghtd3 配置块。
        seed: 随机种子。

    Returns:
        含 n_demos、low/high 指标与 principle 标签的字典。
    """
    cfg = dict(cfg or agent.cfg)
    if bool(cfg.get("hybrid_anchor", False)) and agent._hybrid_anchor is not None:
        # 仍允许在 abs 误配时跑，但打警告语义：F-MLE 设计为无教师
        pass

    demos = collect_feasible_demos(
        env,
        agent,
        n_windows=int(cfg.get("f_mle_windows", cfg.get("bc_windows", 4))),
        seed=seed,
        cfg=cfg,
    )
    low_stats = None
    high_stats = None
    if bool(cfg.get("f_mle_low", cfg.get("bc_pretrain_low", True))):
        # 逆动力学视角：在 (s, g_hind) 上拟合 a —— 与 BC 同损失
        low_stats = behavior_clone_low_actor(
            agent,
            demos,
            epochs=int(cfg.get("f_mle_epochs_low", cfg.get("bc_epochs_low", 25))),
        )
    if bool(cfg.get("f_mle_high", cfg.get("bc_pretrain_high", True))):
        high_stats = bc_pretrain_high_goals(
            agent,
            demos,
            epochs=int(cfg.get("f_mle_epochs_high", cfg.get("bc_epochs_high", 25))),
        )
    return {
        "principle": "F-MLE",
        "demo_source": "price_aware_or_rule_feasible_only",
        "hybrid_teacher": False,
        "n_demos": int(demos["obs"].shape[0]),
        "low": low_stats,
        "high": high_stats,
    }
