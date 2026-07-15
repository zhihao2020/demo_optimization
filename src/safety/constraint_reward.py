"""GiveSafe 约束奖励：与经济 reward 严格分离，禁止 -1e9。

拒绝样本写入 GiveSafeReplay，供 Critic 学习「不安全动作更差」，不污染经济成本尺度。
"""

from __future__ import annotations

from typing import Any, Mapping

from .safety_result import SafetyCheckResult


class ConstraintRewardCalculator:
    """按 violation 类型加权：``r_c = -base * weight``（配置见 givesafe_config.yaml）。"""
    def __init__(self, config: Mapping[str, Any] | None = None):
        cfg = dict(config or {})
        self.base = float(cfg.get("base_rejection_cost", 1.0))
        self.weights = dict(cfg.get("weights") or {})

    def calculate(self, safety_result: SafetyCheckResult) -> dict[str, float]:
        cost = self.base
        terms: dict[str, float] = {"base_rejection_cost": self.base}
        viols = safety_result.normalized_violations or {}
        if not viols and safety_result.violation_type:
            key = safety_result.violation_type
            w = float(self.weights.get(key, self.weights.get("unknown", 1.0)))
            v = max(float(safety_result.violation_severity), 1.0)
            part = w * (v ** 2)
            cost += part
            terms[f"w_{key}"] = part
        else:
            for key, v in viols.items():
                w = float(self.weights.get(key, self.weights.get("unknown", 1.0)))
                part = w * (float(v) ** 2)
                cost += part
                terms[f"w_{key}"] = part
        reward = -float(cost)
        return {
            "constraint_reward": reward,
            "constraint_cost": float(cost),
            "economic_reward": 0.0,
            "terminal_soc_bonus": 0.0,
            "total_training_reward": reward,
            "reward": reward,
            **terms,
        }
