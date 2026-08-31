"""一级快速解析安全检查：可行性神谕(FeasibilityOracle) 与归一化越界量。"""

from __future__ import annotations

from typing import Any, Mapping

from actions import FeasibilityOracle, PhysicalFmuAction
from actions.caes_u import mode_from_u
from actions.validator import physical_from_dict

from .safety_result import SafetyCheckResult


class GiveSafeConstraintChecker:
    """一级约束检查器：静态范围、模式掩码、动态 SOC、压力温度、爬坡与电网。"""

    def __init__(self, oracle: FeasibilityOracle):
        self.oracle = oracle

    def check(
        self,
        observation_outputs: Mapping[str, float],
        action: dict | PhysicalFmuAction,
        previous_thermal_w: float,
    ) -> SafetyCheckResult:
        physical = (
            action if isinstance(action, PhysicalFmuAction) else physical_from_dict(action)
        )
        feasible = self.oracle.compute(observation_outputs, previous_thermal_w)
        physical = self.oracle.canonicalize_physical(physical, feasible)
        predicted = self.oracle.predict_next_state(
            observation_outputs, physical, previous_thermal_w
        )
        physical_dist, safe_dist = self.oracle.distances_to_bounds(observation_outputs)
        bounds = {
            "u_tp_low": feasible.u_tp_low,
            "u_tp_high": feasible.u_tp_high,
            "u_battery_low": feasible.u_battery_low,
            "u_battery_high": feasible.u_battery_high,
        }
        ok, reason = self.oracle.check_action_executable(
            physical, observation_outputs, feasible, previous_thermal_w
        )
        normalized = self._normalized_violations(
            observation_outputs, physical, predicted, feasible, ok, reason
        )
        if not ok:
            vtype = self._map_reason(reason or "unknown")
            severity = max(normalized.values()) if normalized else 1.0
            return SafetyCheckResult(
                safe=False,
                rejection_stage="oracle",
                violation_type=vtype,
                violation_severity=float(severity),
                normalized_violations=normalized,
                predicted_next_state=dict(predicted),
                dynamic_bounds=bounds,
                boundary_margins={
                    **physical_dist,
                    **{f"safe_{k}": v for k, v in safe_dist.items()},
                },
                mode_mask=feasible.mode_mask.as_dict(),
                oracle_safe=False,
                oracle_rejection_reason=reason,
                metadata={"oracle_version": self.oracle.oracle_version},
            )
        post_pred_ok, post_reason = self._predicted_hard_ok(predicted)
        if not post_pred_ok:
            normalized = self._normalized_from_predicted(predicted)
            vtype = self._map_reason(post_reason or "unknown")
            return SafetyCheckResult(
                safe=False,
                rejection_stage="oracle",
                violation_type=vtype,
                violation_severity=max(normalized.values()) if normalized else 1.0,
                normalized_violations=normalized,
                predicted_next_state=dict(predicted),
                dynamic_bounds=bounds,
                boundary_margins={**physical_dist},
                mode_mask=feasible.mode_mask.as_dict(),
                oracle_safe=False,
                oracle_rejection_reason=post_reason,
                metadata={"oracle_version": self.oracle.oracle_version},
            )
        return SafetyCheckResult(
            safe=True,
            rejection_stage=None,
            violation_type=None,
            violation_severity=0.0,
            normalized_violations={},
            predicted_next_state=dict(predicted),
            dynamic_bounds=bounds,
            boundary_margins={
                **physical_dist,
                **{f"safe_{k}": v for k, v in safe_dist.items()},
            },
            mode_mask=feasible.mode_mask.as_dict(),
            oracle_safe=True,
            oracle_rejection_reason=None,
            metadata={"oracle_version": self.oracle.oracle_version},
        )

    def _predicted_hard_ok(self, predicted: Mapping[str, float]) -> tuple[bool, str | None]:
        fake = {
            k: float(predicted.get(k, 0.0))
            for k in (
                "battery_soc",
                "caes_gas_soc",
                "caes_hot_soc",
                "caes_cold_soc",
                "caes_gas_pressure",
                "caes_gas_temperature",
                "caes_hot_temperature",
                "caes_cold_temperature",
            )
        }
        for k in ("p_thermal", "p_battery", "p_caes", "p_grid"):
            fake[k] = float(predicted.get(k, 0.0))
        return self.oracle.post_step_hard_ok(fake)

    def _normalized_violations(
        self,
        outputs: Mapping[str, float],
        physical: PhysicalFmuAction,
        predicted: Mapping[str, float],
        feasible,
        ok: bool,
        reason: str | None,
    ) -> dict[str, float]:
        _ = outputs
        if ok:
            return {}
        out: dict[str, float] = {}
        reason_l = (reason or "").lower()
        if "u_tp" in reason_l or "thermal" in reason_l or "爬坡" in (reason or ""):
            span = max(feasible.u_tp_high - feasible.u_tp_low, 1e-6)
            if physical.u_tp > feasible.u_tp_high:
                out["u_tp"] = (physical.u_tp - feasible.u_tp_high) / span
            elif physical.u_tp < feasible.u_tp_low:
                out["u_tp"] = (feasible.u_tp_low - physical.u_tp) / span
        if "battery" in reason_l or "u_battery" in reason_l:
            span = max(feasible.u_battery_high - feasible.u_battery_low, 1e-6)
            if physical.u_battery > feasible.u_battery_high:
                out["u_battery"] = (physical.u_battery - feasible.u_battery_high) / span
            elif physical.u_battery < feasible.u_battery_low:
                out["u_battery"] = (feasible.u_battery_low - physical.u_battery) / span
        mode = mode_from_u(physical.u_caes)
        if "charge" in reason_l or "discharge" in reason_l or "mask" in reason_l:
            out["caes_mode"] = 1.0
            _ = mode
        if "p_grid" in reason_l or "联络线" in (reason or ""):
            out["p_grid"] = 1.0
        if "硬约束" in (reason or "") or "soc" in reason_l:
            out.update(self._normalized_from_predicted(predicted))
        if not out:
            out["unknown"] = 1.0
        return {k: float(min(max(v, 0.0), 10.0)) for k, v in out.items()}

    def _normalized_from_predicted(self, predicted: Mapping[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, lo, hi in (
            ("battery_soc", 0.1, 0.9),
            ("caes_gas_soc", 0.05, 0.95),
            ("caes_hot_soc", 0.05, 0.95),
            ("caes_cold_soc", 0.05, 0.95),
        ):
            v = float(predicted.get(key, 0.5))
            if v < lo:
                out[key] = (lo - v) / max(lo, 1e-6)
            elif v > hi:
                out[key] = (v - hi) / max(1.0 - hi, 1e-6)
        return out

    @staticmethod
    def _map_reason(reason: str) -> str:
        r = reason.lower()
        if "p_grid" in r or "联络线" in reason:
            return "grid"
        if "charge" in r or "discharge" in r or "mask" in r or "mode" in r:
            return "caes_mode"
        if "battery" in r or "u_battery" in r:
            return "battery_soc"
        if "thermal" in r or "u_tp" in r or "爬坡" in reason:
            return "thermal_ramp"
        if "gas" in r or "caes" in r or "pressure" in r or "temp" in r:
            return "caes_state"
        return "unknown"
