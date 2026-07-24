"""经济 reward：只使用 Modelica 导出的累计现金流，Python 不重复结算。"""

from __future__ import annotations

from typing import Any


SOC_KEYS = ("battery_soc", "caes_gas_soc", "caes_hot_soc", "caes_cold_soc")
ECONOMIC_TOTAL = "economic_cashflow_total"
ECONOMIC_COMPONENTS = (
    "economic_cashflow_wind",
    "economic_cashflow_pv",
    "economic_cashflow_thermal",
    "economic_cashflow_battery",
    "economic_cashflow_caes",
    "economic_cashflow_load",
    "economic_cashflow_grid",
)


class IncompleteRewardConfigError(RuntimeError):
    """正式经济训练所需参数缺失(IncompleteRewardConfigError)。"""


class RewardCalculator:
    """经济奖励计算器(RewardCalculator)。

    以 FMU 累计现金流差分为步奖励，可选终端 SOC 奖励/惩罚；不做 Python 侧重算。
    """

    def __init__(self, config: dict[str, Any], *, require_complete: bool = False) -> None:
        """加载 reward 配置并校验。

        Args:
            config: ``reward_config.yaml`` 解析后的字典。
            require_complete: 为 ``True`` 时缺少 ``cost_reference`` 或终端 SOC 参数则抛错。

        Raises:
            IncompleteRewardConfigError: ``require_complete`` 且参数不完整。
            ValueError: 未知 ``terminal_soc.mode``。
        """
        self.config = config
        self.initial_soc: dict[str, float] | None = None
        self.previous_cashflow: dict[str, float] | None = None
        self.step_in_episode = 0
        self.episode_steps = int(config.get("episode_steps", 168))
        self.require_complete = require_complete
        self._validate_config()

    def _validate_config(self) -> None:
        """校验 cost_reference 与 terminal_soc 配置完整性。

        Raises:
            IncompleteRewardConfigError: 正式训练模式缺少必需键。
            ValueError: 未知终端 SOC 模式。
        """
        cref = self._nested(self.config, "cost_reference", "value")
        term = self.config.get("terminal_soc") or {}
        if self.require_complete:
            missing = []
            if cref is None or float(cref) <= 0:
                missing.append("cost_reference.value")
            if term.get("enabled", True):
                if term.get("bonus") is None:
                    missing.append("terminal_soc.bonus")
                if term.get("tolerance") is None:
                    missing.append("terminal_soc.tolerance")
            if missing:
                raise IncompleteRewardConfigError(
                    f"正式经济训练参数不完整: {missing}；仅允许接口 smoke"
                )
        mode = term.get("mode", "binary_bonus")
        if mode not in ("binary_bonus", "quadratic_penalty", None):
            raise ValueError(f"未知 terminal_soc.mode={mode}")

    @staticmethod
    def _nested(cfg: dict, *keys: str) -> Any:
        """按键路径嵌套取值。

        Args:
            cfg: 根配置字典。
            *keys:  successive 键名。

        Returns:
            末级值；路径中断则 ``None``。
        """
        cur: Any = cfg
        for key in keys:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
        return cur

    @staticmethod
    def _cashflows(outputs: dict[str, float]) -> dict[str, float]:
        """提取 FMU 累计经济输出。

        Args:
            outputs: FMU 输出字典。

        Returns:
            总量与各分量现金流字典。

        Raises:
            KeyError: 缺少必需经济输出键。
        """
        required = (ECONOMIC_TOTAL, *ECONOMIC_COMPONENTS)
        missing = [name for name in required if name not in outputs]
        if missing:
            raise KeyError(f"FMU 缺少累计经济输出: {missing}")
        return {name: float(outputs[name]) for name in required}

    def reset(self, outputs: dict[str, float]) -> None:
        """在 episode 起点记录初始 SOC 与累计现金流基线。

        Args:
            outputs: reset 后 FMU 输出快照。
        """
        self.initial_soc = {key: float(outputs[key]) for key in SOC_KEYS}
        self.previous_cashflow = self._cashflows(outputs)
        self.step_in_episode = 0

    def terminal_soc_diagnostics(self, outputs: dict[str, float]) -> dict[str, float]:
        """计算终端 SOC 相对初始值的 L1/L2 误差与是否满足容差。

        Args:
            outputs: 当前步 FMU 输出。

        Returns:
            含 ``terminal_soc_l1_error``、``terminal_soc_l2_error`` 等诊断键的字典。

        Raises:
            AssertionError: 未先 ``reset``（``initial_soc`` 为 ``None``）。
        """
        term = self.config.get("terminal_soc") or {}
        weights = term.get("weights") or {key: 1.0 for key in SOC_KEYS}
        assert self.initial_soc is not None
        l1 = l2 = 0.0
        for key in SOC_KEYS:
            delta = float(outputs[key]) - self.initial_soc[key]
            weight = float(weights.get(key, 1.0))
            l1 += weight * abs(delta)
            l2 += weight * delta * delta
        tol = term.get("tolerance")
        return {
            "terminal_soc_l1_error": l1,
            "terminal_soc_l2_error": l2,
            "terminal_soc_tolerance": float(tol) if tol is not None else float("nan"),
            "terminal_soc_satisfied": float(tol is not None and l1 <= float(tol)),
        }

    def calculate(
        self,
        outputs: dict[str, float],
        previous_thermal: float | None = None,
        *,
        is_final_step: bool,
        episode_completed: bool,
        no_failure: bool,
        valid_episode_steps: int | None = None,
    ) -> tuple[float, dict[str, float]]:
        """计算步奖励与分项审计字典。

        奖励为累计现金流增量除以 ``cost_reference``，满足门控时叠加终端 SOC 项。
        ``previous_thermal`` 保留仅为调用兼容。

        Args:
            outputs: 当前步 FMU 输出。
            previous_thermal: 上一步火电功率（未使用，兼容旧接口）。
            is_final_step: 是否为 episode 最后一步。
            episode_completed: episode 是否正常跑满步数且无失败。
            no_failure: 本 episode 是否未发生硬失败。
            valid_episode_steps: 有效步数；默认 ``step_in_episode + 1``。

        Returns:
            ``(reward, terms)``；``terms`` 含经济增量、归一化成本与终端 SOC 诊断。

        Raises:
            RuntimeError: 未先调用 ``reset``。
            KeyError: FMU 缺少经济输出。
        """
        if self.previous_cashflow is None:
            raise RuntimeError("RewardCalculator 必须先 reset")
        current = self._cashflows(outputs)
        delta = {name: current[name] - self.previous_cashflow[name] for name in current}
        self.previous_cashflow = current
        cref = self._nested(self.config, "cost_reference", "value")
        if cref is None or float(cref) <= 0:
            reference, reference_missing = 1.0, True
        else:
            reference, reference_missing = float(cref), False
        economic_reward = delta[ECONOMIC_TOTAL] / reference
        # 离线报告以“成本”为正，严格等于现金流增量的相反数。
        raw_total_cost = -delta[ECONOMIC_TOTAL]
        terminal_bonus = 0.0
        diag = self.terminal_soc_diagnostics(outputs) if self.initial_soc else {
            "terminal_soc_l1_error": 0.0, "terminal_soc_l2_error": 0.0,
            "terminal_soc_tolerance": float("nan"), "terminal_soc_satisfied": 0.0,
        }
        term = self.config.get("terminal_soc") or {}
        steps_ok = valid_episode_steps if valid_episode_steps is not None else self.step_in_episode + 1
        gates = bool(term.get("enabled", True)) and is_final_step and episode_completed and no_failure and steps_ok >= self.episode_steps
        if gates and term.get("mode", "binary_bonus") == "binary_bonus":
            if diag["terminal_soc_l1_error"] <= float(term.get("tolerance", float("-inf"))):
                terminal_bonus = float(term.get("bonus", 0.0))
        elif gates and term.get("mode") == "quadratic_penalty":
            terminal_bonus = -float(term.get("quadratic_weight", 1.0)) * diag["terminal_soc_l2_error"]
        reward = economic_reward + terminal_bonus
        terms: dict[str, float] = {
            "economic_cashflow_total": current[ECONOMIC_TOTAL],
            "economic_cashflow_delta": delta[ECONOMIC_TOTAL],
            "economic_reward": economic_reward,
            "raw_total_cost": raw_total_cost,
            "normalized_cost": raw_total_cost / reference,
            "cost_reference": reference,
            "cost_reference_missing": float(reference_missing),
            "terminal_soc_bonus": terminal_bonus,
            "reward": reward,
            **diag,
        }
        for name in ECONOMIC_COMPONENTS:
            suffix = name.removeprefix("economic_cashflow_")
            terms[f"economic_cashflow_{suffix}"] = current[name]
            terms[f"economic_cashflow_delta_{suffix}"] = delta[name]
            terms[f"raw_{suffix}_cost"] = -delta[name]
        return reward, terms
