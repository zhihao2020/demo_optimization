"""经济 reward：仅评价合法物理轨迹。硬约束与 FMU 失败不进入 reward。

正式公式：``r = -C_sys / C_ref + b_SOC``（仅完整合法 episode 末步可发终端 SOC 奖励）。
成本分项与标定见 ``docs/RL奖励于成本配置.md``、``src/config/reward_config.yaml``。
"""

from __future__ import annotations

from typing import Any


SOC_KEYS = ("battery_soc", "caes_gas_soc", "caes_hot_soc", "caes_cold_soc")


class IncompleteRewardConfigError(RuntimeError):
    """正式经济训练所需参数缺失。"""


class RewardCalculator:
    """把一步 FMU 输出折算为系统成本与归一化经济奖励。

    ``require_complete=True`` 时强制 C_ref / 电价 / 终端 SOC 参数齐全，防止冒充正式训练。
    """

    def __init__(self, config: dict[str, Any], *, require_complete: bool = False):
        self.config = config
        self.dt_hours = float(config["decision_interval_seconds"]) / 3600.0
        self.initial_soc: dict[str, float] | None = None
        self.step_in_episode = 0
        self.episode_steps = int(config.get("episode_steps", 168))
        self.require_complete = require_complete
        self._validate_config()

    def _validate_config(self) -> None:
        cref = self._nested(self.config, "cost_reference", "value")
        term = self.config.get("terminal_soc") or {}
        if self.require_complete:
            required = [
                "buy_price_yuan_per_mwh",
                "sell_price_yuan_per_mwh",
                "thermal_a",
                "thermal_b",
                "thermal_c",
            ]
            missing = [k for k in required if self.config.get(k) is None]
            if cref is None:
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
        cur: Any = cfg
        for key in keys:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
        return cur

    def reset(self, outputs: dict[str, float]) -> None:
        self.initial_soc = {key: float(outputs[key]) for key in SOC_KEYS}
        self.step_in_episode = 0

    def _rate(self, key: str, default: float = 0.0) -> float:
        value = self.config.get(key)
        return default if value is None else float(value)

    def raw_costs(self, outputs: dict[str, float], previous_thermal: float) -> dict[str, float]:
        """七项原始成本（元/步）：电网、火电、储能吞吐、弃电、缺供、爬坡。"""
        dt, power_to_mw = self.dt_hours, 1e-6
        p_grid = float(outputs.get("p_grid", 0.0))
        p_thermal = abs(float(outputs.get("p_thermal", 0.0)))
        grid = (
            self._rate("buy_price_yuan_per_mwh") * max(p_grid, 0.0)
            - self._rate("sell_price_yuan_per_mwh") * max(-p_grid, 0.0)
        ) * power_to_mw * dt
        a, b, c = (self._rate(k) for k in ("thermal_a", "thermal_b", "thermal_c"))
        thermal_mw = p_thermal * power_to_mw
        thermal = (a * thermal_mw ** 2 + b * thermal_mw + c) * dt
        battery = self._rate("battery_throughput_yuan_per_mwh") * abs(float(outputs.get("p_battery", 0.0))) * power_to_mw * dt
        caes = self._rate("caes_throughput_yuan_per_mwh") * abs(float(outputs.get("p_caes", 0.0))) * power_to_mw * dt
        curtailment = self._rate("curtailment_yuan_per_mwh") * float(outputs.get("p_curtailment", 0.0)) * power_to_mw * dt
        unserved = self._rate("unserved_yuan_per_mwh") * float(outputs.get("p_unserved", 0.0)) * power_to_mw * dt
        ramp = self._rate("ramp_yuan_per_mw") * abs(float(outputs.get("p_thermal", 0.0)) - previous_thermal) * power_to_mw
        terms = {
            "raw_grid_cost": grid,
            "raw_thermal_cost": thermal,
            "raw_battery_cost": battery,
            "raw_caes_cost": caes,
            "raw_curtailment_cost": curtailment,
            "raw_unserved_cost": unserved,
            "raw_ramp_cost": ramp,
        }
        terms["raw_total_cost"] = sum(terms.values())
        return terms

    def terminal_soc_diagnostics(self, outputs: dict[str, float]) -> dict[str, float]:
        term = self.config.get("terminal_soc") or {}
        weights = term.get("weights") or {
            "battery_soc": 1.0,
            "caes_gas_soc": 1.0,
            "caes_hot_soc": 1.0,
            "caes_cold_soc": 1.0,
        }
        assert self.initial_soc is not None
        l1 = 0.0
        l2 = 0.0
        for key in SOC_KEYS:
            w = float(weights.get(key, 1.0))
            delta = float(outputs[key]) - self.initial_soc[key]
            l1 += w * abs(delta)
            l2 += w * delta * delta
        tol = term.get("tolerance")
        satisfied = float(tol is not None and l1 <= float(tol))
        return {
            "terminal_soc_l1_error": l1,
            "terminal_soc_l2_error": l2,
            "terminal_soc_tolerance": float(tol) if tol is not None else float("nan"),
            "terminal_soc_satisfied": satisfied,
        }

    def calculate(
        self,
        outputs: dict[str, float],
        previous_thermal: float,
        *,
        is_final_step: bool,
        episode_completed: bool,
        no_failure: bool,
        valid_episode_steps: int | None = None,
    ) -> tuple[float, dict[str, float]]:
        """返回 (reward, 分项字典)。终端 bonus 受完整步数/无失败门控。"""
        costs = self.raw_costs(outputs, previous_thermal)
        cref = self._nested(self.config, "cost_reference", "value")
        if cref is None or float(cref) <= 0:
            # smoke 允许：尚未标定则 normalized_cost 用 raw（不伪装 C_ref）
            cost_reference = 1.0
            cref_missing = True
        else:
            cost_reference = float(cref)
            cref_missing = False
        normalized = costs["raw_total_cost"] / cost_reference
        terminal_bonus = 0.0
        diag = self.terminal_soc_diagnostics(outputs) if self.initial_soc else {
            "terminal_soc_l1_error": 0.0,
            "terminal_soc_l2_error": 0.0,
            "terminal_soc_tolerance": float("nan"),
            "terminal_soc_satisfied": 0.0,
        }
        term = self.config.get("terminal_soc") or {}
        mode = term.get("mode", "binary_bonus")
        steps_ok = valid_episode_steps if valid_episode_steps is not None else self.step_in_episode + 1
        gates = (
            bool(term.get("enabled", True))
            and is_final_step
            and episode_completed
            and no_failure
            and steps_ok >= self.episode_steps
            and (not term.get("require_complete_episode", True) or episode_completed)
            and (not term.get("require_no_failure", True) or no_failure)
        )
        if gates and mode == "binary_bonus":
            bonus = term.get("bonus")
            tol = term.get("tolerance")
            if bonus is not None and tol is not None and diag["terminal_soc_l1_error"] <= float(tol):
                terminal_bonus = float(bonus)
        elif gates and mode == "quadratic_penalty":
            weight = float(term.get("quadratic_weight", 1.0))
            terminal_bonus = -weight * diag["terminal_soc_l2_error"]

        reward = -normalized + terminal_bonus
        terms = {
            **costs,
            "cost_reference": cost_reference,
            "cost_reference_missing": float(cref_missing),
            "normalized_cost": normalized,
            "terminal_soc_bonus": terminal_bonus,
            "reward": reward,
            **diag,
        }
        return reward, terms
