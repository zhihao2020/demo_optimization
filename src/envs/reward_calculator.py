"""经济 reward：默认用 Modelica 累计现金流；可选 price-taker 分时电价替换电网项。"""

from __future__ import annotations

from typing import Any, Mapping

from market.settlement import settle_grid_step


SOC_KEYS = ("battery_soc", "caes_gas_soc", "caes_hot_soc", "caes_cold_soc")
ECONOMIC_TOTAL = "economic_cashflow_total"
ECONOMIC_GRID = "economic_cashflow_grid"
ECONOMIC_COMPONENTS = (
    "economic_cashflow_wind",
    "economic_cashflow_pv",
    "economic_cashflow_thermal",
    "economic_cashflow_battery",
    "economic_cashflow_caes",
    "economic_cashflow_load",
    ECONOMIC_GRID,
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
        self.previous_soc_l1: float = 0.0
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
        shaping = term.get("shaping") or {}
        if shaping.get("enabled") and shaping.get("mode", "potential") not in (
            "potential",
            "absolute",
            "hybrid",
        ):
            raise ValueError(f"未知 terminal_soc.shaping.mode={shaping.get('mode')}")

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
        self.previous_soc_l1 = 0.0  # 起点相对自身 L1=0
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
        # 全状态 L1（含热/冷罐，用于 shaping 与诊断）
        l1 = l2 = 0.0
        for key in SOC_KEYS:
            delta = float(outputs[key]) - self.initial_soc[key]
            weight = float(weights.get(key, 1.0))
            l1 += weight * abs(delta)
            l2 += weight * delta * delta
        # 能量主状态 L1：battery + CAES gas（运营期末回收硬指标）
        # 热/冷罐为 CAES 热力学耦合副状态，单独报告，不否决 energy pass
        primary = term.get("primary_keys") or ["battery_soc", "caes_gas_soc"]
        l1_energy = l2_energy = 0.0
        for key in primary:
            if key not in self.initial_soc:
                continue
            delta = float(outputs[key]) - self.initial_soc[key]
            weight = float(weights.get(key, 1.0))
            l1_energy += weight * abs(delta)
            l2_energy += weight * delta * delta
        tol = term.get("tolerance")
        # 达标判定默认用能量主状态；full_state_pass 要求可用 use_full_l1_for_pass=true
        use_full = bool(term.get("use_full_l1_for_pass", False))
        score = l1 if use_full else l1_energy
        return {
            "terminal_soc_l1_error": float(score),
            "terminal_soc_l1_full": float(l1),
            "terminal_soc_l1_energy": float(l1_energy),
            "terminal_soc_l2_error": float(l2 if use_full else l2_energy),
            "terminal_soc_tolerance": float(tol) if tol is not None else float("nan"),
            "terminal_soc_satisfied": float(tol is not None and score <= float(tol)),
        }

    def _soc_shaping(
        self,
        l1: float,
        *,
        steps_done: int | None = None,
    ) -> tuple[float, dict[str, float]]:
        """SOC 过程奖励。

        - potential: r = coef * (L1_prev - L1)，整周望远镜和 ≈ -coef * L1_final
        - absolute: r = -absolute_coef * L1（每步）
        - recovery_horizon: 末段放大 coef/absolute，推动回到初始 SOC
        """
        term = self.config.get("terminal_soc") or {}
        shaping = term.get("shaping") or {}
        if not shaping.get("enabled", False):
            return 0.0, {
                "soc_shaping_reward": 0.0,
                "soc_shaping_coef": 0.0,
                "soc_shaping_absolute": 0.0,
                "soc_recovery_scale": 1.0,
                "soc_l1_prev": float(self.previous_soc_l1),
            }
        coef = float(shaping.get("coef", 1.0))
        abs_coef = float(shaping.get("absolute_coef", 0.0))
        mode = str(shaping.get("mode", "potential"))
        # 末段回收：剩余步数 ≤ recovery_horizon 时放大 shaping
        episode_steps = int(self.config.get("episode_steps", self.episode_steps))
        done = int(steps_done if steps_done is not None else self.step_in_episode + 1)
        remaining = max(episode_steps - done, 0)
        recovery_h = int(shaping.get("recovery_horizon_steps", 0) or 0)
        recovery_scale = 1.0
        if recovery_h > 0 and remaining <= recovery_h:
            # 线性升温：刚进回收窗 scale→1，最后一步接近 recovery_coef_scale
            base = float(shaping.get("recovery_coef_scale", 3.0))
            frac = 1.0 - (remaining / max(recovery_h, 1))
            recovery_scale = 1.0 + (base - 1.0) * float(frac)
            coef *= recovery_scale
            abs_coef *= float(shaping.get("recovery_absolute_scale", recovery_scale))
        prev = float(self.previous_soc_l1)
        l1f = float(l1)
        pot = 0.0
        abs_pen = 0.0
        if mode in ("potential", "hybrid"):
            pot = coef * (prev - l1f)
        if mode in ("absolute", "hybrid") or (mode == "potential" and abs_coef > 0.0):
            abs_pen = -abs_coef * l1f
        if mode == "absolute" and abs_coef <= 0.0:
            abs_pen = -coef * l1f
        value = pot + abs_pen
        self.previous_soc_l1 = l1f
        return float(value), {
            "soc_shaping_reward": float(value),
            "soc_shaping_potential": float(pot),
            "soc_shaping_absolute": float(abs_pen),
            "soc_shaping_coef": coef,
            "soc_recovery_scale": float(recovery_scale),
            "soc_l1_prev": prev,
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
        market_prices: Mapping[str, float] | None = None,
        decision_interval_hours: float | None = None,
    ) -> tuple[float, dict[str, float]]:
        """奖励 = 经济现金流增量/C_ref + SOC shaping + 终端 bonus。

        market_prices 若提供 ``buy_yuan_per_kwh`` / ``sell_yuan_per_kwh``，则用分时电价
        重算电网现金流，并替换 FMU 电网增量：  
        ``delta_total' = delta_total - delta_grid_fmu + market_grid_cashflow``。
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

        market_terms: dict[str, float] = {"market_settlement_enabled": 0.0}
        economic_delta = float(delta[ECONOMIC_TOTAL])
        if market_prices is not None:
            buy = market_prices.get("buy_yuan_per_kwh")
            sell = market_prices.get("sell_yuan_per_kwh")
            if buy is None or sell is None:
                raise ValueError("market_prices 需含 buy_yuan_per_kwh 与 sell_yuan_per_kwh")
            dt_h = decision_interval_hours
            if dt_h is None:
                dt_h = float(self.config.get("decision_interval_seconds", 3600)) / 3600.0
            p_grid = float(outputs.get("p_grid", 0.0))
            settled = settle_grid_step(p_grid, float(buy), float(sell), float(dt_h))
            fmu_grid_delta = float(delta[ECONOMIC_GRID])
            economic_delta = economic_delta - fmu_grid_delta + float(settled["market_grid_cashflow"])
            delta = dict(delta)
            delta[ECONOMIC_TOTAL] = economic_delta
            delta[ECONOMIC_GRID] = float(settled["market_grid_cashflow"])
            market_terms = {
                "market_settlement_enabled": 1.0,
                **settled,
                "fmu_grid_cashflow_delta": fmu_grid_delta,
            }

        economic_reward = economic_delta / reference
        # 离线报告以“成本”为正，严格等于有效现金流增量的相反数。
        raw_total_cost = -economic_delta
        terminal_bonus = 0.0
        diag = self.terminal_soc_diagnostics(outputs) if self.initial_soc else {
            "terminal_soc_l1_error": 0.0, "terminal_soc_l2_error": 0.0,
            "terminal_soc_tolerance": float("nan"), "terminal_soc_satisfied": 0.0,
        }
        steps_ok = valid_episode_steps if valid_episode_steps is not None else self.step_in_episode + 1
        shaping_reward, shaping_terms = self._soc_shaping(
            float(diag["terminal_soc_l1_error"]), steps_done=int(steps_ok)
        )
        term = self.config.get("terminal_soc") or {}
        gates = bool(term.get("enabled", True)) and is_final_step and episode_completed and no_failure and steps_ok >= self.episode_steps
        tol = float(term.get("tolerance", float("-inf")))
        l1_err = float(diag["terminal_soc_l1_error"])
        if gates and term.get("mode", "binary_bonus") == "binary_bonus":
            if l1_err <= tol:
                terminal_bonus = float(term.get("bonus", 0.0))
            else:
                # 未达标：按 L1 罚，避免“永远不回收仍拿高经济 reward”
                fail_pen = float(term.get("fail_penalty_l1", 0.0) or 0.0)
                terminal_bonus = -fail_pen * l1_err
        elif gates and term.get("mode") == "quadratic_penalty":
            terminal_bonus = -float(term.get("quadratic_weight", 1.0)) * diag["terminal_soc_l2_error"]
        reward = economic_reward + shaping_reward + terminal_bonus
        terms: dict[str, float] = {
            "economic_cashflow_total": current[ECONOMIC_TOTAL],
            "economic_cashflow_delta": economic_delta,
            "economic_reward": economic_reward,
            "raw_total_cost": raw_total_cost,
            "normalized_cost": raw_total_cost / reference,
            "cost_reference": reference,
            "cost_reference_missing": float(reference_missing),
            "terminal_soc_bonus": terminal_bonus,
            "reward": reward,
            **diag,
            **shaping_terms,
            **market_terms,
        }
        for name in ECONOMIC_COMPONENTS:
            suffix = name.removeprefix("economic_cashflow_")
            terms[f"economic_cashflow_{suffix}"] = current[name]
            terms[f"economic_cashflow_delta_{suffix}"] = delta[name]
            terms[f"raw_{suffix}_cost"] = -delta[name]
        return reward, terms
