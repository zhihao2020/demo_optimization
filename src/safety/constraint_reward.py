"""安全给予(GiveSafe) 约束奖励：与经济 reward 严格分离，禁止 -1e9 硬截断。"""

from __future__ import annotations

from typing import Any, Mapping

from .safety_result import SafetyCheckResult


class ConstraintRewardCalculator:
    """根据安全检查结果的归一化违规量计算约束惩罚奖励。"""

    def __init__(self, config: Mapping[str, Any] | None = None):
        """初始化约束奖励计算器。

        Args:
            config: 可选配置字典，含 base_rejection_cost 与 weights 键。

        Returns:
            无。

        Raises:
            无。
        """
        cfg = dict(config or {})
        self.base = float(cfg.get("base_rejection_cost", 1.0))
        self.weights = dict(cfg.get("weights") or {})

    def calculate(self, safety_result: SafetyCheckResult) -> dict[str, float]:
        """根据单次安全检查失败结果计算约束奖励与分项成本。

        Args:
            safety_result: 被拒绝候选对应的安全检查结果。

        Returns:
            含 constraint_reward、constraint_cost、economic_reward 及各分项权重的字典。

        Raises:
            无。
        """
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
