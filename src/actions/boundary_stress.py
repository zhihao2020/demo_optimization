"""边界应力测试：近界动作必须预检拒绝或后验成功。"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np
from actions import CaesMode, FeasibilityOracle, HybridAction
from actions.validator import hybrid_from_dict
from envs.failures import ConstraintFailure


@dataclass
class BoundaryStressResult:
    n_attempted: int = 0
    n_precheck_rejected: int = 0
    n_oracle_legal: int = 0
    n_post_step_success: int = 0
    n_post_step_fail: int = 0
    n_fmu_fail: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)
    scenarios: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        # Oracle 判合法的动作不得后验硬失败
        return self.n_post_step_fail == 0 and self.n_attempted > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_attempted": self.n_attempted,
            "n_precheck_rejected": self.n_precheck_rejected,
            "n_oracle_legal": self.n_oracle_legal,
            "n_post_step_success": self.n_post_step_success,
            "n_post_step_fail": self.n_post_step_fail,
            "n_fmu_fail": self.n_fmu_fail,
            "passed": self.passed,
            "scenarios": self.scenarios,
            "failure_count": len(self.failures),
        }


class BoundaryStressTester:
    """生成近边界/多约束应力动作；目标规模 ≥20000（Phase E 门控）。"""

    SCENARIOS = (
        "battery_soc_near_min",
        "battery_soc_near_max",
        "caes_gas_near_min",
        "caes_gas_near_max",
        "caes_hot_near_bound",
        "caes_cold_near_bound",
        "caes_pressure_near_bound",
        "caes_temp_near_bound",
        "caes_mode_switch",
        "thermal_ramp_limit",
        "grid_near_buy_cap",
        "grid_near_sell_cap",
        "multi_constraint",
    )

    def __init__(self, oracle: FeasibilityOracle | None = None, seed: int = 0):
        self.oracle = oracle or FeasibilityOracle.from_root()
        self.rng = np.random.default_rng(seed)

    def sample_boundary_action(
        self,
        outputs: dict[str, float],
        previous_thermal_w: float,
        scenario: str | None = None,
    ) -> tuple[dict, str]:
        scenario = scenario or str(self.rng.choice(self.SCENARIOS))
        feasible = self.oracle.compute(outputs, previous_thermal_w)
        # 故意偏置到边界：50% 贴齐动态界，50% 可能越界以验证预检
        bias_illegal = self.rng.random() < 0.35
        u_tp = float(self.rng.uniform(feasible.u_tp_low, feasible.u_tp_high))
        u_bat = float(self.rng.uniform(feasible.u_battery_low, feasible.u_battery_high))
        modes = [
            m
            for m, ok in zip(
                (CaesMode.DISCHARGE, CaesMode.IDLE, CaesMode.CHARGE),
                (
                    feasible.mode_mask.discharge,
                    feasible.mode_mask.idle,
                    feasible.mode_mask.charge,
                ),
            )
            if ok
        ] or [CaesMode.IDLE]
        mode = modes[int(self.rng.integers(0, len(modes)))]
        mag = 0.0 if mode == CaesMode.IDLE else float(self.rng.uniform(0.0, 1.0))
        if scenario.startswith("battery_soc_near"):
            if "min" in scenario:
                u_bat = float(feasible.u_battery_low)
            else:
                u_bat = float(feasible.u_battery_high)
            if bias_illegal:
                u_bat = float(
                    np.clip(u_bat + (0.2 if "max" in scenario else -0.2), -1.5, 1.5)
                )
        elif scenario.startswith("caes_gas") or "caes_" in scenario:
            if feasible.mode_mask.charge and "max" in scenario:
                mode, mag = CaesMode.CHARGE, 1.0
            elif feasible.mode_mask.discharge and "min" in scenario:
                mode, mag = CaesMode.DISCHARGE, 1.0
            elif scenario == "caes_mode_switch":
                mode = (
                    CaesMode.CHARGE
                    if mode == CaesMode.DISCHARGE
                    else CaesMode.DISCHARGE
                )
                if mode == CaesMode.DISCHARGE and not feasible.mode_mask.discharge:
                    mode = CaesMode.IDLE
                if mode == CaesMode.CHARGE and not feasible.mode_mask.charge:
                    mode = CaesMode.IDLE
                mag = 1.0 if mode != CaesMode.IDLE else 0.0
        elif scenario == "thermal_ramp_limit":
            u_tp = float(
                feasible.u_tp_high if self.rng.random() < 0.5 else feasible.u_tp_low
            )
            if bias_illegal:
                u_tp = float(np.clip(u_tp + (0.1 if u_tp > 0.5 else -0.1), 0.0, 1.2))
        elif scenario.startswith("grid_near"):
            # 推高储能同向以逼近联络线
            if "buy" in scenario:
                mode, mag = (
                    (CaesMode.CHARGE, 1.0)
                    if feasible.mode_mask.charge
                    else (CaesMode.IDLE, 0.0)
                )
                u_bat = float(feasible.u_battery_high)
            else:
                mode, mag = (
                    (CaesMode.DISCHARGE, 1.0)
                    if feasible.mode_mask.discharge
                    else (CaesMode.IDLE, 0.0)
                )
                u_bat = float(feasible.u_battery_low)
        elif scenario == "multi_constraint":
            u_tp = float(feasible.u_tp_high)
            u_bat = float(
                feasible.u_battery_high
                if self.rng.random() < 0.5
                else feasible.u_battery_low
            )
            if feasible.mode_mask.charge:
                mode, mag = CaesMode.CHARGE, 1.0
            elif feasible.mode_mask.discharge:
                mode, mag = CaesMode.DISCHARGE, 1.0
        action = {
            "u_tp": np.asarray([u_tp], dtype=np.float32),
            "u_battery": np.asarray([u_bat], dtype=np.float32),
            "caes_mode": int(mode),
            "caes_magnitude": np.asarray([mag], dtype=np.float32),
        }
        return action, scenario

    def run(
        self,
        env,
        *,
        n_actions: int = 20000,
        reset_every: int = 50,
        step_fn: Callable | None = None,
    ) -> BoundaryStressResult:
        """对真实 env 做边界偏置步进。非法须预检拒绝；Oracle 合法须后验成功。"""
        result = BoundaryStressResult()
        obs, info = env.reset(seed=0)
        episode_steps = 0
        for i in range(n_actions):
            if env.last_outputs is None:
                obs, info = env.reset(seed=i)
            outputs = dict(env.last_outputs)
            prev_th = float(env.previous_thermal)
            action, scenario = self.sample_boundary_action(outputs, prev_th)
            sc = result.scenarios.setdefault(
                scenario,
                {"attempted": 0, "precheck_rejected": 0, "post_ok": 0, "post_fail": 0},
            )
            sc["attempted"] += 1
            result.n_attempted += 1
            # 预检（与 env 一致：validator + oracle）
            feasible = env.get_feasible_action_spec()
            try:
                hybrid = hybrid_from_dict(action)
                env.hybrid_validator.validate(hybrid, feasible)
                ok, reason = self.oracle.check_action_executable(
                    hybrid, outputs, feasible, prev_th
                )
                if not ok:
                    raise ConstraintFailure(reason or "precheck")
            except Exception:
                result.n_precheck_rejected += 1
                sc["precheck_rejected"] += 1
                episode_steps += 1
                if episode_steps >= reset_every:
                    obs, info = env.reset(seed=i + 1)
                    episode_steps = 0
                continue
            result.n_oracle_legal += 1
            if step_fn is not None:
                obs, reward, term, trunc, info = step_fn(env, action)
            else:
                obs, reward, term, trunc, info = env.step(action)
            episode_steps += 1
            if info.get("physically_valid") and info.get("transition_valid"):
                result.n_post_step_success += 1
                sc["post_ok"] += 1
            else:
                ft = info.get("failure_type") or ""
                if "Fmu" in ft or "Fmi" in ft:
                    result.n_fmu_fail += 1
                else:
                    result.n_post_step_fail += 1
                    sc["post_fail"] += 1
                    result.failures.append(
                        {
                            "scenario": scenario,
                            "failure_type": ft,
                            "fine_failure_type": info.get("fine_failure_type"),
                            "reason": info.get("failure_reason"),
                            "action": {
                                k: (float(v[0]) if hasattr(v, "__len__") else v)
                                for k, v in action.items()
                            },
                        }
                    )
                obs, info = env.reset(seed=i + 17)
                episode_steps = 0
                continue
            if term or trunc or episode_steps >= reset_every:
                obs, info = env.reset(seed=i + 3)
                episode_steps = 0
        return result
