"""一级快速解析安全检查（Oracle + 规范化越界量）。"""

from __future__ import annotations

from typing import Any, Mapping

from actions import FeasibilityOracle, HybridAction
from actions.validator import hybrid_from_dict

from .safety_result import SafetyCheckResult


class GiveSafeConstraintChecker:
    """一级：静态范围 / mode mask / 动态 SOC / 压力温度 / 爬坡 / 电网。"""

    def __init__(self, oracle: FeasibilityOracle):
        self.oracle = oracle

    def check(
        self,
        observation_outputs: Mapping[str, float],
        action: dict | HybridAction,
        previous_thermal_w: float,
    ) -> SafetyCheckResult:
        hybrid = action if isinstance(action, HybridAction) else hybrid_from_dict(action)
        feasible = self.oracle.compute(observation_outputs, previous_thermal_w)
        predicted = self.oracle.predict_next_state(observation_outputs, hybrid, previous_thermal_w)
        physical_dist, safe_dist = self.oracle.distances_to_bounds(observation_outputs)
        bounds = {
            "u_tp_low": feasible.u_tp_low,
            "u_tp_high": feasible.u_tp_high,
            "u_battery_low": feasible.u_battery_low,
            "u_battery_high": feasible.u_battery_high,
        }
        ok, reason = self.oracle.check_action_executable(
            hybrid, observation_outputs, feasible, previous_thermal_w
        )
        normalized = self._normalized_violations(observation_outputs, hybrid, predicted, feasible, ok, reason)
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
                boundary_margins={**physical_dist, **{f"safe_{k}": v for k, v in safe_dist.items()}},
                mode_mask=feasible.mode_mask.as_dict(),
                oracle_safe=False,
                oracle_rejection_reason=reason,
                metadata={"oracle_version": self.oracle.oracle_version},
            )
        # 额外：根据预测下一状态相对物理界的越界程度
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
            boundary_margins={**physical_dist, **{f"safe_{k}": v for k, v in safe_dist.items()}},
            mode_mask=feasible.mode_mask.as_dict(),
            oracle_safe=True,
            oracle_rejection_reason=None,
            metadata={"oracle_version": self.oracle.oracle_version},
        )

    def _predicted_hard_ok(self, predicted: Mapping[str, float]) -> tuple[bool, str | None]:
        # 复用 oracle 物理界检查字段
        fake = {k: float(predicted.get(k, 0.0)) for k in (
            "battery_soc", "caes_gas_soc", "caes_hot_soc", "caes_cold_soc",
            "caes_gas_pressure", "caes_gas_temperature", "caes_hot_temperature", "caes_cold_temperature",
        )}
        # pad required keys for post_step_hard_ok
        for k in ("p_thermal", "p_battery", "p_caes", "p_grid"):
            fake[k] = float(predicted.get(k, 0.0))
        return self.oracle.post_step_hard_ok(fake)

    def _normalized_violations(
        self,
        outputs: Mapping[str, float],
        hybrid: HybridAction,
        predicted: Mapping[str, float],
        feasible,
        ok: bool,
        reason: str | None,
    ) -> dict[str, float]:
        if ok:
            return {}
        if reason and "mode" in reason.lower():
            return {"forbidden_mode": 1.0}
        if reason and "grid" in reason.lower() or (reason and "p_grid" in (reason or "")):
            return {"grid_capacity": 1.0}
        if reason and "u_tp" in (reason or ""):
            return {"thermal_ramp": 1.0}
        if reason and "u_battery" in (reason or ""):
            # 方向从动作与边界推断
            if hybrid.u_battery > feasible.u_battery_high:
                return {"battery_soc_high": 1.0}
            if hybrid.u_battery < feasible.u_battery_low:
                return {"battery_soc_low": 1.0}
            return {"battery_soc_high": 0.5}
        return self._normalized_from_predicted(predicted) or {"unknown": 1.0}

    def _normalized_from_predicted(self, predicted: Mapping[str, float]) -> dict[str, float]:
        p = self.oracle.params
        b, c = p["battery"], p["caes"]
        out: dict[str, float] = {}
        def soc_high(name, val, lo, hi, key):
            span = max(float(hi) - float(lo), 1e-9)
            v = max(float(val) - float(hi), 0.0) / span
            if v > 0:
                out[key] = v
        def soc_low(name, val, lo, hi, key):
            span = max(float(hi) - float(lo), 1e-9)
            v = max(float(lo) - float(val), 0.0) / span
            if v > 0:
                out[key] = v
        soc_high("b", predicted.get("battery_soc", 0.5), b["SOC_min"], b["SOC_max"], "battery_soc_high")
        soc_low("b", predicted.get("battery_soc", 0.5), b["SOC_min"], b["SOC_max"], "battery_soc_low")
        soc_high("g", predicted.get("caes_gas_soc", 0.8), c["gas_SOC_min"], c["gas_SOC_max"], "caes_gas_soc_high")
        soc_low("g", predicted.get("caes_gas_soc", 0.8), c["gas_SOC_min"], c["gas_SOC_max"], "caes_gas_soc_low")
        soc_high("h", predicted.get("caes_hot_soc", 0.5), c["hot_SOC_min"], c["hot_SOC_max"], "caes_hot_soc_high")
        soc_low("h", predicted.get("caes_hot_soc", 0.5), c["hot_SOC_min"], c["hot_SOC_max"], "caes_hot_soc_low")
        soc_high("c", predicted.get("caes_cold_soc", 0.5), c["cold_SOC_min"], c["cold_SOC_max"], "caes_cold_soc_high")
        soc_low("c", predicted.get("caes_cold_soc", 0.5), c["cold_SOC_min"], c["cold_SOC_max"], "caes_cold_soc_low")
        pmin, pmax = float(c["gas_pressure_min_Pa"]), float(c["gas_pressure_max_Pa"])
        span_p = max(pmax - pmin, 1.0)
        pr = float(predicted.get("caes_gas_pressure", 8e6))
        if pr > pmax:
            out["caes_pressure_high"] = (pr - pmax) / span_p
        if pr < pmin:
            out["caes_pressure_low"] = (pmin - pr) / span_p
        return out

    @staticmethod
    def _map_reason(reason: str) -> str:
        r = (reason or "").lower()
        if "forbidden" in r or "mode" in r or "charge" in r or "discharge" in r:
            return "forbidden_mode"
        if "grid" in r or "联络" in r:
            return "grid_capacity"
        if "u_tp" in r or "ramp" in r or "thermal" in r:
            return "thermal_ramp"
        if "battery" in r and ("high" in r or "max" in r or "上限" in r):
            return "battery_soc_high"
        if "battery" in r:
            return "battery_soc_low"
        if "gas" in r and ("press" in r or "压力" in r):
            return "caes_pressure_low" if "low" in r or "下限" in r else "caes_pressure_high"
        if "gas" in r:
            return "caes_gas_soc_low" if "low" in r or "下限" in r or ">" in r else "caes_gas_soc_high"
        if "hot" in r:
            return "caes_hot_soc_high" if "high" in r or "上限" in r else "caes_hot_soc_low"
        if "cold" in r:
            return "caes_cold_soc_high" if "high" in r or "上限" in r else "caes_cold_soc_low"
        if "temp" in r or "温度" in r:
            return "caes_temperature_high"
        return "unknown"
