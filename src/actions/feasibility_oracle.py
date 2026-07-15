"""根据当前状态计算动态可行域 A(s_t)；逐设备方程 + residual 方向裕度。"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import yaml
from .decoder import HybridActionDecoder
from .feasible_set import DynamicFeasibleActionSet
from .mode_mask import ModeMask
from .types import CaesMode, HybridAction, PhysicalFmuAction
OBS_NAMES = (
    "battery_soc",
    "caes_gas_soc",
    "caes_hot_soc",
    "caes_cold_soc",
    "caes_gas_pressure",
    "caes_gas_temperature",
    "caes_hot_temperature",
    "caes_cold_temperature",
    "p_thermal",
    "p_battery",
    "p_caes",
    "p_grid",
    "p_wind_available",
    "p_wind_actual",
    "p_pv_available",
    "p_pv_actual",
    "p_load_actual",
    "p_curtailment",
    "p_unserved",
)
PREDICTED_STATE_KEYS = (
    "battery_soc",
    "caes_gas_soc",
    "caes_hot_soc",
    "caes_cold_soc",
    "caes_gas_pressure",
    "caes_gas_temperature",
    "caes_hot_temperature",
    "caes_cold_temperature",
    "p_thermal",
    "p_battery",
    "p_caes",
    "p_grid",
)
def load_margins(path: Path | None = None) -> dict[str, Any]:
    p = path or Path(__file__).resolve().parents[1] / "config" / "feasibility_margins.yaml"
    with Path(p).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}
class FeasibilityOracle:
    def __init__(
        self,
        params: Mapping[str, Any] | None = None,
        params_path: str | Path | None = None,
        margins: Mapping[str, Any] | None = None,
        margins_path: str | Path | None = None,
    ):
        if params is None:
            path = Path(params_path) if params_path else Path(__file__).resolve().parents[1] / "config" / "device_params.yaml"
            with Path(path).open(encoding="utf-8") as stream:
                params = yaml.safe_load(stream)
        self.params = dict(params)
        self.dt = float(self.params.get("decision_interval_seconds", 3600))
        self.decoder = HybridActionDecoder()
        if margins is None:
            mpath = Path(margins_path) if margins_path else Path(__file__).resolve().parents[1] / "config" / "feasibility_margins.yaml"
            self.margins = load_margins(mpath)
        else:
            self.margins = dict(margins)
        self.oracle_version = str(self.margins.get("oracle_version", "unknown"))
    @classmethod
    def from_root(cls, root: Path | None = None) -> "FeasibilityOracle":
        root = root or Path(__file__).resolve().parents[2]
        return cls(
            params_path=root / "src" / "config" / "device_params.yaml",
            margins_path=root / "src" / "config" / "feasibility_margins.yaml",
        )
    def compute(
        self,
        outputs: Mapping[str, float],
        previous_thermal_w: float | None = None,
    ) -> DynamicFeasibleActionSet:
        # Thermal: ALWAYS from actual previous p_thermal (caller must pass last FMU output)
        p_prev = float(previous_thermal_w if previous_thermal_w is not None else outputs.get("p_thermal", 0.0))
        bat = self._battery_bounds(float(outputs["battery_soc"]))
        mode = self._caes_mode_mask(outputs)
        tp = self._thermal_bounds_from_actual(p_prev)
        # CAES 幅值联合可行性记入 metadata（mask 已编码模式可否）
        caes_mag = self._caes_magnitude_caps(outputs)
        empty = (
            bat[0] > bat[1] + 1e-12
            or tp[0] > tp[1] + 1e-12
            or not (mode.discharge or mode.idle or mode.charge)
        )
        meta = {
            "battery_bounds_source": "exact_1step_modelica_SOC_with_direction_margins",
            "thermal_ramp_enforced_in_python": True,
            "thermal_ramp_active_in_fmu": False,
            "thermal_uses_actual_previous_p_thermal": True,
            "caes_feasibility": "mode_specific_joint_soc_pressure_temp",
            "oracle_version": self.oracle_version,
            "feasible_set_empty": empty,
            "caes_magnitude_caps": caes_mag,
        }
        return DynamicFeasibleActionSet(
            u_tp_low=tp[0],
            u_tp_high=tp[1],
            u_battery_low=bat[0],
            u_battery_high=bat[1],
            mode_mask=mode,
            grid_violation_predicted=False,
            metadata=meta,
        )
    def is_feasible_set_empty(self, feasible: DynamicFeasibleActionSet) -> bool:
        if feasible.metadata and feasible.metadata.get("feasible_set_empty"):
            return True
        if feasible.u_tp_low > feasible.u_tp_high + 1e-12:
            return True
        if feasible.u_battery_low > feasible.u_battery_high + 1e-12:
            return True
        m = feasible.mode_mask
        return not (m.discharge or m.idle or m.charge)
    def check_action_executable(
        self,
        action: HybridAction,
        outputs: Mapping[str, float],
        feasible: DynamicFeasibleActionSet | None = None,
        previous_thermal_w: float | None = None,
    ) -> tuple[bool, str | None]:
        """预执行电网容量等状态相关检查。返回 (ok, reason)。"""
        feasible = feasible or self.compute(outputs, previous_thermal_w)
        if self.is_feasible_set_empty(feasible):
            return False, "可行集为空"
        physical = self.decoder.decode(action)
        pred = self.predict_p_grid(outputs, physical)
        g = self.params["grid"]
        gm = float(self.margins.get("grid", {}).get("margin_W", 0.0))
        if pred > float(g["P_max_buy_W"]) - gm + 1.0 or pred < float(g["P_max_sell_W"]) + gm - 1.0:
            return False, f"预测 p_grid={pred} W 超出联络线安全界"
        if action.caes_mode == CaesMode.CHARGE and not feasible.mode_mask.charge:
            return False, "CHARGE 被 mask 禁止"
        if action.caes_mode == CaesMode.DISCHARGE and not feasible.mode_mask.discharge:
            return False, "DISCHARGE 被 mask 禁止"
        # 联合预测：若预测下一状态越物理界则拒绝
        predicted = self.predict_next_state(outputs, action, previous_thermal_w)
        ok, reason = self.post_step_hard_ok(predicted, use_safe=False)
        if not ok:
            return False, f"Oracle 预测下一状态违反硬约束: {reason}"
        return True, None
    def predict_p_grid(self, outputs: Mapping[str, float], physical: PhysicalFmuAction) -> float:
        """功率符号约定：负荷为正，火电/风光发电为负，储能正充负放，电网正购负售。"""
        p_tp = -physical.u_tp * float(self.params["thermal"]["P_cap_W"])
        p_bat = physical.u_battery * float(self.params["battery"]["P_cap_W"])
        p_caes = physical.u_caes * float(self.params["caes"]["P_cap_W"])
        p_wind = float(outputs.get("p_wind_actual", outputs.get("p_wind_available", 0.0)))
        p_pv = float(outputs.get("p_pv_actual", outputs.get("p_pv_available", 0.0)))
        p_load = float(outputs.get("p_load_actual", 0.0))
        return -(p_tp + p_bat + p_caes + p_wind + p_pv + p_load)
    def predict_next_state(
        self,
        outputs: Mapping[str, float],
        action: HybridAction | PhysicalFmuAction | Mapping[str, Any],
        previous_thermal_w: float | None = None,
    ) -> dict[str, float]:
        """一阶预测下一决策步状态（residual = actual - predicted）。"""
        if isinstance(action, HybridAction):
            physical = self.decoder.decode(action)
            mode = action.caes_mode
            mag = 0.0 if mode == CaesMode.IDLE else float(action.caes_magnitude)
        elif isinstance(action, PhysicalFmuAction):
            physical = action
            mode = self._mode_from_u(physical.u_caes)
            mag = self._mag_from_u(physical.u_caes, mode)
        else:
            # dict hybrid or physical
            if "caes_mode" in action:
                ha = HybridAction(
                    float(action["u_tp"][0] if hasattr(action["u_tp"], "__len__") else action["u_tp"]),
                    float(action["u_battery"][0] if hasattr(action["u_battery"], "__len__") else action["u_battery"]),
                    CaesMode(int(action["caes_mode"])),
                    float(action["caes_magnitude"][0] if hasattr(action["caes_magnitude"], "__len__") else action["caes_magnitude"]),
                )
                return self.predict_next_state(outputs, ha, previous_thermal_w)
            physical = PhysicalFmuAction(float(action["u_tp"]), float(action["u_battery"]), float(action["u_caes"]))
            mode = self._mode_from_u(physical.u_caes)
            mag = self._mag_from_u(physical.u_caes, mode)
        bat = self.params["battery"]
        p_cap_b = float(bat["P_cap_W"])
        e_cap = float(bat["E_cap_J"])
        eta = float(bat["eta"])
        soc = float(outputs["battery_soc"])
        u_b = physical.u_battery
        if u_b >= 0:
            soc_next = soc + u_b * p_cap_b * self.dt / e_cap
        else:
            soc_next = soc + u_b * p_cap_b * self.dt / (e_cap * eta)
        p_cap_c = float(self.params["caes"]["P_cap_W"])
        u_c = physical.u_caes
        p_caes = u_c * p_cap_c
        em = self.margins.get("caes", {}).get("energy_model", {})
        e_ref = float(em.get("E_ref_J", 5.4e12))
        energy = p_caes * self.dt / max(e_ref, 1.0)
        gas = float(outputs["caes_gas_soc"]) + float(em.get("alpha_gas", 1.0)) * energy
        hot = float(outputs["caes_hot_soc"]) + float(em.get("alpha_hot", 0.35)) * energy
        cold = float(outputs["caes_cold_soc"]) + float(em.get("alpha_cold", 0.35)) * energy
        # 压力与 SOC 近似耦合
        d_soc_gas = gas - float(outputs["caes_gas_soc"])
        pressure = float(outputs["caes_gas_pressure"]) + float(em.get("alpha_pressure_Pa_per_soc", 3e6)) * d_soc_gas
        dT = float(em.get("alpha_temp_K_per_u", 2.0)) * u_c
        tg = float(outputs["caes_gas_temperature"]) + dT
        th = float(outputs["caes_hot_temperature"]) + 0.5 * dT
        tc = float(outputs["caes_cold_temperature"]) - 0.3 * dT
        p_tp = -physical.u_tp * float(self.params["thermal"]["P_cap_W"])
        p_bat = u_b * p_cap_b
        pred_grid = self.predict_p_grid(outputs, physical)
        return {
            "battery_soc": float(soc_next),
            "caes_gas_soc": float(gas),
            "caes_hot_soc": float(hot),
            "caes_cold_soc": float(cold),
            "caes_gas_pressure": float(pressure),
            "caes_gas_temperature": float(tg),
            "caes_hot_temperature": float(th),
            "caes_cold_temperature": float(tc),
            "p_thermal": float(p_tp),
            "p_battery": float(p_bat),
            "p_caes": float(p_caes),
            "p_grid": float(pred_grid),
            "caes_mode": int(mode),
            "caes_magnitude": float(mag),
        }
    def residual(
        self,
        predicted: Mapping[str, float],
        actual: Mapping[str, float],
    ) -> dict[str, float]:
        out = {}
        for k in PREDICTED_STATE_KEYS:
            if k in predicted and k in actual:
                out[k] = float(actual[k]) - float(predicted[k])
        return out
    def dangerous_residual(
        self,
        residual: Mapping[str, float],
        *,
        mode: CaesMode | int | None = None,
        u_battery: float | None = None,
    ) -> dict[str, float]:
        """方向感知：仅保留推向物理界的危险残差分量。"""
        dang: dict[str, float] = {}
        rb = residual.get("battery_soc")
        if rb is not None and u_battery is not None:
            if u_battery > 0 and rb > 0:
                dang["battery_soc_high"] = float(rb)
            if u_battery < 0 and rb < 0:
                dang["battery_soc_low"] = float(rb)
        mode_i = int(mode) if mode is not None else None
        if mode_i == int(CaesMode.CHARGE):
            for k, key in (
                ("caes_gas_soc", "caes_gas_soc_high"),
                ("caes_hot_soc", "caes_hot_soc_high"),
                ("caes_cold_soc", "caes_cold_soc_high"),
                ("caes_gas_pressure", "caes_pressure_high"),
            ):
                v = residual.get(k)
                if v is not None and v > 0:
                    dang[key] = float(v)
        elif mode_i == int(CaesMode.DISCHARGE):
            for k, key in (
                ("caes_gas_soc", "caes_gas_soc_low"),
                ("caes_hot_soc", "caes_hot_soc_low"),
                ("caes_cold_soc", "caes_cold_soc_low"),
                ("caes_gas_pressure", "caes_pressure_low"),
            ):
                v = residual.get(k)
                if v is not None and v < 0:
                    dang[key] = float(v)
        return dang
    def distances_to_bounds(self, outputs: Mapping[str, float]) -> tuple[dict[str, float], dict[str, float]]:
        b = self.params["battery"]
        c = self.params["caes"]
        physical: dict[str, float] = {}
        safe: dict[str, float] = {}
        soc = float(outputs["battery_soc"])
        physical["battery_soc_to_min"] = soc - float(b["SOC_min"])
        physical["battery_soc_to_max"] = float(b["SOC_max"]) - soc
        safe["battery_soc_to_safe_min"] = soc - float(b["SOC_safe_min"])
        safe["battery_soc_to_safe_max"] = float(b["SOC_safe_max"]) - soc
        for key, lo, hi, slo, shi in (
            ("caes_gas_soc", c["gas_SOC_min"], c["gas_SOC_max"], c["gas_SOC_safe_min"], c["gas_SOC_safe_max"]),
            ("caes_hot_soc", c["hot_SOC_min"], c["hot_SOC_max"], c["hot_SOC_safe_min"], c["hot_SOC_safe_max"]),
            ("caes_cold_soc", c["cold_SOC_min"], c["cold_SOC_max"], c["cold_SOC_safe_min"], c["cold_SOC_safe_max"]),
        ):
            val = float(outputs[key])
            physical[f"{key}_to_min"] = val - float(lo)
            physical[f"{key}_to_max"] = float(hi) - val
            safe[f"{key}_to_safe_min"] = val - float(slo)
            safe[f"{key}_to_safe_max"] = float(shi) - val
        p = float(outputs.get("caes_gas_pressure", 8e6))
        physical["caes_pressure_to_min"] = p - float(c["gas_pressure_min_Pa"])
        physical["caes_pressure_to_max"] = float(c["gas_pressure_max_Pa"]) - p
        return physical, safe
    def post_step_hard_ok(
        self,
        outputs: Mapping[str, float],
        *,
        use_safe: bool = False,
    ) -> tuple[bool, str | None]:
        """FMU 步进后硬状态检查。use_safe=True 时用 safe 界（Oracle 预检更严）。"""
        b = self.params["battery"]
        c = self.params["caes"]
        soc_b = float(outputs["battery_soc"])
        lo_b = float(b["SOC_safe_min"] if use_safe else b["SOC_min"])
        hi_b = float(b["SOC_safe_max"] if use_safe else b["SOC_max"])
        if not (lo_b - 1e-6 <= soc_b <= hi_b + 1e-6):
            return False, f"battery_soc={soc_b} 越物理界 [{lo_b}, {hi_b}]" if not use_safe else f"battery_soc={soc_b} 越安全界 [{lo_b}, {hi_b}]"
        gas = float(outputs["caes_gas_soc"])
        glo = float(c["gas_SOC_safe_min"] if use_safe else c["gas_SOC_min"])
        ghi = float(c["gas_SOC_safe_max"] if use_safe else c["gas_SOC_max"])
        if not (glo - 1e-6 <= gas <= ghi + 1e-6):
            return False, f"caes_gas_soc={gas} 越 assert 界 [{glo}, {ghi}]"
        for key, lo_k, hi_k, slo_k, shi_k in (
            ("caes_hot_soc", "hot_SOC_min", "hot_SOC_max", "hot_SOC_safe_min", "hot_SOC_safe_max"),
            ("caes_cold_soc", "cold_SOC_min", "cold_SOC_max", "cold_SOC_safe_min", "cold_SOC_safe_max"),
        ):
            val = float(outputs[key])
            lo = float(c[slo_k if use_safe else lo_k])
            hi = float(c[shi_k if use_safe else hi_k])
            if not (lo - 1e-6 <= val <= hi + 1e-6):
                return False, f"{key}={val} 越界 [{lo}, {hi}]"
        p = float(outputs.get("caes_gas_pressure", 8e6))
        plo, phi = float(c["gas_pressure_min_Pa"]), float(c["gas_pressure_max_Pa"])
        if not (plo - 1.0 <= p <= phi + 1.0):
            return False, f"caes_gas_pressure={p} 越界 [{plo}, {phi}]"
        for tname, lo_k, hi_k in (
            ("caes_gas_temperature", "gas_temp_min_K", "gas_temp_max_K"),
            ("caes_hot_temperature", "hot_temp_min_K", "hot_temp_max_K"),
            ("caes_cold_temperature", "cold_temp_min_K", "cold_temp_max_K"),
        ):
            if tname not in outputs:
                continue
            val = float(outputs[tname])
            lo, hi = float(c[lo_k]), float(c[hi_k])
            if not (lo - 1e-6 <= val <= hi + 1e-6):
                return False, f"{tname}={val} 越界 [{lo}, {hi}]"
        g = self.params["grid"]
        if "p_grid" in outputs:
            pg = float(outputs["p_grid"])
            if pg > float(g["P_max_buy_W"]) + 1.0 or pg < float(g["P_max_sell_W"]) - 1.0:
                return False, f"p_grid={pg} 越联络线 [{g['P_max_sell_W']}, {g['P_max_buy_W']}]"
        for name, val in outputs.items():
            if name in ("caes_mode", "caes_magnitude"):
                continue
            if not np.isfinite(float(val)):
                return False, f"{name} 非有限"
        return True, None
    def _battery_bounds(self, soc: float) -> tuple[float, float]:
        p = self.params["battery"]
        m = self.margins.get("battery", {})
        p_cap = float(p["P_cap_W"])
        e_cap = float(p["E_cap_J"])
        eta = float(p["eta"])
        # 方向裕度：把危险 residual P99 叠到 safe 界上（非统一 0.02）
        margin_hi = float(m.get("margin_charge_high", 0.0)) + float(m.get("residual_p99_charge_high", 0.0))
        margin_lo = float(m.get("margin_discharge_low", 0.0)) + float(m.get("residual_p99_discharge_low", 0.0))
        safe_min = float(p["SOC_min"]) + margin_lo
        safe_max = float(p["SOC_max"]) - margin_hi
        # 不得宽于配置的 SOC_safe_*（device_params 中的运行界）
        safe_min = max(safe_min, float(p["SOC_safe_min"]))
        safe_max = min(safe_max, float(p["SOC_safe_max"]))
        dt = self.dt
        u_charge_max = (safe_max - soc) * e_cap / (p_cap * dt) if p_cap * dt > 0 else 0.0
        u_discharge_min = (safe_min - soc) * e_cap * eta / (p_cap * dt) if p_cap * dt > 0 else 0.0
        u_low = max(-1.0, float(u_discharge_min))
        u_high = min(1.0, float(u_charge_max))
        if u_low > u_high:
            return 0.0, 0.0
        return u_low, u_high
    def _thermal_bounds_from_actual(self, previous_thermal_w: float) -> tuple[float, float]:
        t = self.params["thermal"]
        u_min = float(t["u_min"])
        u_max = float(t["u_max"])
        p_cap = float(t["P_cap_W"])
        rate = float(t["rate_max_per_s"])
        # CRITICAL: previous_thermal_w 必须是上一时刻 FMU 实际 p_thermal
        p_prev = float(previous_thermal_w)
        u_prev = float(np.clip(-p_prev / p_cap, u_min, u_max))
        du = rate * self.dt
        margin = float(self.margins.get("thermal", {}).get("margin_u", 0.0))
        low = max(u_min, u_prev - du + margin)
        high = min(u_max, u_prev + du - margin)
        if low > high:
            low = high = float(np.clip(u_prev, u_min, u_max))
        return low, high
    def _caes_mode_mask(self, outputs: Mapping[str, float]) -> ModeMask:
        """模式特定联合可行性：gas/hot/cold SOC + pressure + temps。"""
        c = self.params["caes"]
        cm = self.margins.get("caes", {})
        gas = float(outputs["caes_gas_soc"])
        hot = float(outputs["caes_hot_soc"])
        cold = float(outputs["caes_cold_soc"])
        p = float(outputs.get("caes_gas_pressure", 8.5e6))
        tg = float(outputs.get("caes_gas_temperature", 300.0))
        th = float(outputs.get("caes_hot_temperature", 400.0))
        tc = float(outputs.get("caes_cold_temperature", 290.0))
        # 用最大可行动作预测下一状态能否留在物理界内
        charge_ok = self._caes_mode_feasible(
            outputs,
            CaesMode.CHARGE,
            mag=1.0,
            margins=cm.get("charge", {}),
            direction="high",
        )
        discharge_ok = self._caes_mode_feasible(
            outputs,
            CaesMode.DISCHARGE,
            mag=1.0,
            margins=cm.get("discharge", {}),
            direction="low",
        )
        # IDLE：当前态在物理界内则始终允许（残差裕度不得禁止待机）
        idle_ok, _ = self.post_step_hard_ok(outputs, use_safe=False)
        # 即使预测失败，仍保留温度硬门（禁充放，不禁 idle）
        temp_ok = (
            float(c["gas_temp_min_K"]) <= tg <= float(c["gas_temp_max_K"])
            and float(c["hot_temp_min_K"]) <= th <= float(c["hot_temp_max_K"])
            and float(c["cold_temp_min_K"]) <= tc <= float(c["cold_temp_max_K"])
        )
        if not temp_ok:
            charge_ok = discharge_ok = False
        # 额外：接近物理界时用联合阈值（仅充/放）
        chg = cm.get("charge", {})
        dis = cm.get("discharge", {})
        charge_ok = charge_ok and (
            gas < float(c["gas_SOC_max"]) - float(chg.get("margin_gas", 0.0)) - float(chg.get("residual_p99_gas_high", 0.0))
            and hot < float(c["hot_SOC_max"]) - float(chg.get("margin_hot", 0.0)) - float(chg.get("residual_p99_hot_high", 0.0))
            and cold < float(c["cold_SOC_max"]) - float(chg.get("margin_cold", 0.0)) - float(chg.get("residual_p99_cold_high", 0.0))
            and p < float(c["gas_pressure_max_Pa"]) - float(chg.get("margin_pressure_Pa", 0.0)) - float(chg.get("residual_p99_pressure_high", 0.0))
        )
        discharge_ok = discharge_ok and (
            gas > float(c["gas_SOC_min"]) + float(dis.get("margin_gas", 0.0)) + float(dis.get("residual_p99_gas_low", 0.0))
            and hot > float(c["hot_SOC_min"]) + float(dis.get("margin_hot", 0.0)) + float(dis.get("residual_p99_hot_low", 0.0))
            and cold > float(c["cold_SOC_min"]) + float(dis.get("margin_cold", 0.0)) + float(dis.get("residual_p99_cold_low", 0.0))
            and p > float(c["gas_pressure_min_Pa"]) + float(dis.get("margin_pressure_Pa", 0.0)) + float(dis.get("residual_p99_pressure_low", 0.0))
        )
        return ModeMask(discharge=bool(discharge_ok), idle=bool(idle_ok), charge=bool(charge_ok))
    def _caes_mode_feasible(
        self,
        outputs: Mapping[str, float],
        mode: CaesMode,
        mag: float,
        margins: Mapping[str, Any],
        direction: str,
    ) -> bool:
        action = HybridAction(u_tp=1.0, u_battery=0.0, caes_mode=mode, caes_magnitude=mag)
        pred = self.predict_next_state(outputs, action)
        ok, _ = self.post_step_hard_ok(pred, use_safe=False)
        if not ok:
            return False
        # 对危险方向叠加 residual 裕度再验
        c = self.params["caes"]
        if direction in ("high", "both"):
            if float(pred["caes_gas_soc"]) > float(c["gas_SOC_max"]) - float(margins.get("residual_p99_gas_high", margins.get("residual_p99_gas", 0.0))):
                return False
        if direction in ("low", "both"):
            if float(pred["caes_gas_soc"]) < float(c["gas_SOC_min"]) + float(margins.get("residual_p99_gas_low", margins.get("residual_p99_gas", 0.0))):
                return False
        return True
    def _caes_magnitude_caps(self, outputs: Mapping[str, float]) -> dict[str, float]:
        """模式内最大安全幅值（0–1）；用于 action pipeline 收紧。"""
        caps = {"discharge": 0.0, "charge": 0.0}
        for mode, key in ((CaesMode.DISCHARGE, "discharge"), (CaesMode.CHARGE, "charge")):
            lo, hi = 0.0, 1.0
            best = 0.0
            for _ in range(8):
                mid = 0.5 * (lo + hi)
                action = HybridAction(1.0, 0.0, mode, mid)
                pred = self.predict_next_state(outputs, action)
                ok, _ = self.post_step_hard_ok(pred, use_safe=False)
                if ok:
                    best = mid
                    lo = mid
                else:
                    hi = mid
            caps[key] = float(best)
        return caps
    @staticmethod
    def _mode_from_u(u: float) -> CaesMode:
        if abs(u) <= 1e-9:
            return CaesMode.IDLE
        return CaesMode.DISCHARGE if u < 0 else CaesMode.CHARGE
    @staticmethod
    def _mag_from_u(u: float, mode: CaesMode) -> float:
        if mode == CaesMode.IDLE:
            return 0.0
        if mode == CaesMode.DISCHARGE:
            return float(np.clip((u - (-1.0)) / (-0.33 - (-1.0)), 0.0, 1.0))
        return float(np.clip((u - 0.86) / (1.0 - 0.86), 0.0, 1.0))
