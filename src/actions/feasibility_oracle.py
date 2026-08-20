"""根据当前状态计算动态可行域 A(s_t)；逐设备方程 + residual 方向裕度。"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import yaml
from .caes_u import (
    CHARGE_HI,
    CHARGE_LO,
    DISCHARGE_HI,
    DISCHARGE_LO,
    mode_from_u,
    u_from_mode_mag,
)
from .feasible_set import DynamicFeasibleActionSet
from .mode_mask import ModeMask
from .types import CaesMode, PhysicalFmuAction
from .validator import physical_from_dict

# FMU 输出变量
OBS_NAMES = (
    "battery_soc",  # 电池剩余电量
    "caes_gas_soc",  # 压空气体剩余容量
    "caes_hot_soc",  # 压空热能剩余容量
    "caes_cold_soc",  # 压空冷能剩余容量
    "caes_gas_pressure",  # 压空气体压力
    "caes_gas_temperature",  # 压空气体温度
    "caes_hot_temperature",  # 压空热能温度
    "caes_cold_temperature",  # 压空冷能温度
    "p_thermal",  # 火电功率
    "p_battery",  # 电池功率
    "p_caes",  # 压空功率
    "p_grid",  # 电网功率
    "p_wind_available",  # 风电可用功率
    "p_wind_actual",  # 风电实际功率
    "p_pv_available",  # 光伏可用功率
    "p_pv_actual",  # 光伏实际功率
    "p_load_actual",  # 负荷实际功率
    "p_curtailment",  # 弃风功率
    "p_unserved",  # 未供电功率
)

# 预测
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
    """从 YAML 加载可行性裕度配置(feasibility_margins)。

    Args:
        path: 可选，裕度配置文件路径；None 则使用默认 src/config/feasibility_margins.yaml。

    Returns:
        解析后的裕度字典；文件为空则返回 {}。

    Raises:
        无。
    """
    p = (
        path
        or Path(__file__).resolve().parents[1] / "config" / "feasibility_margins.yaml"
    )
    with Path(p).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


class FeasibilityOracle:
    """可行性神谕(FeasibilityOracle)：计算动态可行域、预检动作与一步状态预测。"""

    def __init__(
        self,
        params: Mapping[str, Any] | None = None,
        params_path: str | Path | None = None,
        margins: Mapping[str, Any] | None = None,
        margins_path: str | Path | None = None,
    ):
        """初始化 Oracle：加载设备参数与 residual 方向裕度。

        Args:
            params: 可选，设备参数字典；None 则从 params_path 或默认 YAML 加载。
            params_path: 设备参数 YAML 路径。
            margins: 可选，裕度字典；None 则从 margins_path 或默认 YAML 加载。
            margins_path: 裕度配置 YAML 路径。

        Returns:
            无（构造器）。

        Raises:
            无。
        """
        if params is None:
            path = (
                Path(params_path)
                if params_path
                else Path(__file__).resolve().parents[1]
                / "config"
                / "device_params.yaml"
            )
            with Path(path).open(encoding="utf-8") as stream:
                params = yaml.safe_load(stream)
        self.params = dict(params)
        self.dt = float(self.params.get("decision_interval_seconds", 3600))
        if margins is None:
            mpath = (
                Path(margins_path)
                if margins_path
                else Path(__file__).resolve().parents[1]
                / "config"
                / "feasibility_margins.yaml"
            )
            self.margins = load_margins(mpath)
        else:
            self.margins = dict(margins)
        self.oracle_version = str(self.margins.get("oracle_version", "unknown"))

    @classmethod
    def from_root(cls, root: Path | None = None) -> "FeasibilityOracle":
        """从项目根目录加载默认 config 路径构造 Oracle。

        Args:
            root: 可选，项目根 Path；None 则取本文件上两级目录。

        Returns:
            配置好默认 YAML 路径的 FeasibilityOracle 实例。

        Raises:
            无。
        """
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
        """根据当前观测与实际上一火电功率计算动态可行动作集 A(s_t)。

        Args:
            outputs: 当前 FMU 输出字典。
            previous_thermal_w: 上一决策步 FMU 实际 p_thermal（W）；None 则回退 outputs["p_thermal"]。

        Returns:
            动态可行动作集(DynamicFeasibleActionSet)，含 u_tp/u_battery 界与 CAES 模式掩码。

        Raises:
            无。
        """
        # 火电：始终基于实际上一时刻 p_thermal（调用方须传入 last FMU output）
        p_prev = float(
            previous_thermal_w
            if previous_thermal_w is not None
            else outputs.get("p_thermal", 0.0)
        )
        bat = self._battery_bounds(float(outputs["battery_soc"]))
        mode, intervals = self._caes_mask_and_intervals(outputs)
        tp = self._thermal_bounds_from_actual(p_prev)
        caes_mag = self._caes_magnitude_caps(intervals)
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
            "caes_feasibility": "mode_specific_joint_soc_pressure_temp_with_magnitude_interval",
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
            u_caes_discharge=intervals["discharge"],
            u_caes_charge=intervals["charge"],
            grid_violation_predicted=False,
            metadata=meta,
        )

    def is_feasible_set_empty(self, feasible: DynamicFeasibleActionSet) -> bool:
        """判断动态可行域是否为空（无合法动作）。

        Args:
            feasible: 待检查的动态可行动作集。

        Returns:
            若 metadata 标记 empty、界 inverted 或无任何 CAES 模式可用则为 True。

        Raises:
            无。
        """
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
        action: PhysicalFmuAction,
        outputs: Mapping[str, float],
        feasible: DynamicFeasibleActionSet | None = None,
        previous_thermal_w: float | None = None,
    ) -> tuple[bool, str | None]:
        """预执行电网容量与联合下一状态硬约束检查。"""
        feasible = feasible or self.compute(outputs, previous_thermal_w)
        if self.is_feasible_set_empty(feasible):
            return False, "可行集为空"
        physical = action
        pred = self.predict_p_grid(outputs, physical)
        g = self.params["grid"]
        gm = float(self.margins.get("grid", {}).get("margin_W", 0.0))
        if (
            pred > float(g["P_max_buy_W"]) - gm + 1.0
            or pred < float(g["P_max_sell_W"]) + gm - 1.0
        ):
            return False, f"预测 p_grid={pred} W 超出联络线安全界"
        mode = mode_from_u(physical.u_caes)
        if mode == CaesMode.CHARGE and not feasible.mode_mask.charge:
            return False, "CHARGE 被 mask 禁止"
        if mode == CaesMode.DISCHARGE and not feasible.mode_mask.discharge:
            return False, "DISCHARGE 被 mask 禁止"
        span = (
            feasible.u_caes_charge
            if mode == CaesMode.CHARGE
            else feasible.u_caes_discharge
            if mode == CaesMode.DISCHARGE
            else None
        )
        if span is not None:
            lo, hi = min(span), max(span)
            if not (lo - 1e-9 <= physical.u_caes <= hi + 1e-9):
                return False, (
                    f"u_caes={physical.u_caes} 超出该方向安全幅值区间 [{lo}, {hi}]"
                )
        predicted = self.predict_next_state(outputs, action, previous_thermal_w)
        ok, reason = self.post_step_hard_ok(predicted, use_safe=False)
        if not ok:
            return False, f"Oracle 预测下一状态违反硬约束: {reason}"
        return True, None

    def predict_p_grid(
        self, outputs: Mapping[str, float], physical: PhysicalFmuAction
    ) -> float:
        """由物理动作与当前可再生/负荷出力预测联络线功率 p_grid。

        功率符号约定：负荷为正，火电/风光发电为负，储能正充负放，电网正购负售。

        Args:
            outputs: 当前 FMU 输出（含风光、负荷）。
            physical: 解码后的物理动作(PhysicalFmuAction)。

        Returns:
            预测 p_grid（W）。

        Raises:
            无。
        """
        p_tp = -physical.u_tp * float(self.params["thermal"]["P_cap_W"])
        p_bat = physical.u_battery * float(self.params["battery"]["P_cap_W"])
        p_caes = physical.u_caes * float(self.params["caes"]["P_cap_W"])
        p_wind = float(
            outputs.get("p_wind_actual", outputs.get("p_wind_available", 0.0))
        )
        p_pv = float(outputs.get("p_pv_actual", outputs.get("p_pv_available", 0.0)))
        p_load = float(outputs.get("p_load_actual", 0.0))
        return -(p_tp + p_bat + p_caes + p_wind + p_pv + p_load)

    def predict_next_state(
        self,
        outputs: Mapping[str, float],
        action: PhysicalFmuAction | Mapping[str, Any],
        previous_thermal_w: float | None = None,
    ) -> dict[str, float]:
        """一阶预测下一决策步状态（residual = actual - predicted）。"""
        _ = previous_thermal_w
        if isinstance(action, PhysicalFmuAction):
            physical = action
        else:
            physical = physical_from_dict(dict(action))
        mode = mode_from_u(physical.u_caes)
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
        alpha = self._caes_alphas(em, energy)
        gas = float(outputs["caes_gas_soc"]) + alpha["gas"] * energy
        hot = float(outputs["caes_hot_soc"]) + alpha["hot"] * energy
        cold = float(outputs["caes_cold_soc"]) + alpha["cold"] * energy
        # 压力与 SOC 近似耦合
        d_soc_gas = gas - float(outputs["caes_gas_soc"])
        pressure = (
            float(outputs["caes_gas_pressure"])
            + float(em.get("alpha_pressure_Pa_per_soc", 3e6)) * d_soc_gas
        )
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
            "u_caes": float(physical.u_caes),
            "caes_mode": int(mode),  # 派生诊断字段
        }

    @staticmethod
    def _caes_alphas(
        energy_model: Mapping[str, Any], energy: float
    ) -> dict[str, float]:
        """按 energy 符号取充/放电各自的库存斜率。

        充放电斜率并不对称（实测放电幅度比充电大 ~40%），因此按符号分支。
        配置缺少 charge/discharge 子段时回落到合并标量值。

        注意 alpha_cold 必须为负：充电抽冷罐、放电回灌冷罐，与热罐反号。
        默认值取自 scripts/fit_caes_transition.py 的实测拟合，即使配置文件过期
        也不会退回到符号错误的旧值。

        Args:
            energy_model: caes.energy_model 配置子字典。
            energy: 带符号的无量纲能量增量，>0 表示充电。

        Returns:
            含 "gas" / "hot" / "cold" 三个斜率的字典。

        Raises:
            无。
        """
        fallback = {"gas": 0.213, "hot": 0.106, "cold": -0.103}
        section = "charge" if energy > 0 else "discharge"
        sub = energy_model.get(section)
        sub = sub if isinstance(sub, Mapping) else {}
        out: dict[str, float] = {}
        for name, default in fallback.items():
            legacy = energy_model.get(f"alpha_{name}", default)
            out[name] = float(sub.get(f"alpha_{name}", legacy))
        return out

    def residual(
        self,
        predicted: Mapping[str, float],
        actual: Mapping[str, float],
    ) -> dict[str, float]:
        """计算 actual - predicted 逐步 residual。

        Args:
            predicted: Oracle 或模型预测状态。
            actual: FMU 实际输出。

        Returns:
            键为 PREDICTED_STATE_KEYS 子集的 residual 字典。

        Raises:
            无。
        """
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
        """方向感知：仅保留推向物理界的危险 residual 分量。

        Args:
            residual: 完整 residual 字典。
            mode: 可选，CAES 模式，用于 SOC/压力方向筛选。
            u_battery: 可选，电池功率指令，用于 SOC 方向筛选。

        Returns:
            细粒度键（如 battery_soc_high）到危险 residual 分量的字典。

        Raises:
            无。
        """
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

    def distances_to_bounds(
        self, outputs: Mapping[str, float]
    ) -> tuple[dict[str, float], dict[str, float]]:
        """计算当前状态到物理界与安全界的 signed 距离。

        Args:
            outputs: 当前 FMU 输出。

        Returns:
            (physical_distances, safe_distances) 两个字典。

        Raises:
            无。
        """
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
            (
                "caes_gas_soc",
                c["gas_SOC_min"],
                c["gas_SOC_max"],
                c["gas_SOC_safe_min"],
                c["gas_SOC_safe_max"],
            ),
            (
                "caes_hot_soc",
                c["hot_SOC_min"],
                c["hot_SOC_max"],
                c["hot_SOC_safe_min"],
                c["hot_SOC_safe_max"],
            ),
            (
                "caes_cold_soc",
                c["cold_SOC_min"],
                c["cold_SOC_max"],
                c["cold_SOC_safe_min"],
                c["cold_SOC_safe_max"],
            ),
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
        """FMU 步进后硬状态检查；use_safe=True 时使用更严的安全界。

        Args:
            outputs: 待检状态（实际或预测）。
            use_safe: True 时用 SOC_safe_* 等安全界；False 用物理 assert 界。

        Returns:
            (ok, reason)：满足界则 (True, None)，否则 (False, 原因)。

        Raises:
            无。
        """
        b = self.params["battery"]
        c = self.params["caes"]
        soc_b = float(outputs["battery_soc"])
        lo_b = float(b["SOC_safe_min"] if use_safe else b["SOC_min"])
        hi_b = float(b["SOC_safe_max"] if use_safe else b["SOC_max"])
        if not (lo_b - 1e-6 <= soc_b <= hi_b + 1e-6):
            return False, (
                f"battery_soc={soc_b} 越物理界 [{lo_b}, {hi_b}]"
                if not use_safe
                else f"battery_soc={soc_b} 越安全界 [{lo_b}, {hi_b}]"
            )
        gas = float(outputs["caes_gas_soc"])
        glo = float(c["gas_SOC_safe_min"] if use_safe else c["gas_SOC_min"])
        ghi = float(c["gas_SOC_safe_max"] if use_safe else c["gas_SOC_max"])
        if not (glo - 1e-6 <= gas <= ghi + 1e-6):
            return False, f"caes_gas_soc={gas} 越 assert 界 [{glo}, {ghi}]"
        for key, lo_k, hi_k, slo_k, shi_k in (
            (
                "caes_hot_soc",
                "hot_SOC_min",
                "hot_SOC_max",
                "hot_SOC_safe_min",
                "hot_SOC_safe_max",
            ),
            (
                "caes_cold_soc",
                "cold_SOC_min",
                "cold_SOC_max",
                "cold_SOC_safe_min",
                "cold_SOC_safe_max",
            ),
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
        for tname, lo, hi in self._temp_bounds():
            if tname not in outputs:
                continue
            val = float(outputs[tname])
            if not (lo - 1e-6 <= val <= hi + 1e-6):
                return False, f"{tname}={val} 越界 [{lo}, {hi}]"
        g = self.params["grid"]
        if "p_grid" in outputs:
            pg = float(outputs["p_grid"])
            if (
                pg > float(g["P_max_buy_W"]) + 1.0
                or pg < float(g["P_max_sell_W"]) - 1.0
            ):
                return (
                    False,
                    f"p_grid={pg} 越联络线 [{g['P_max_sell_W']}, {g['P_max_buy_W']}]",
                )
        for name, val in outputs.items():
            if name in ("caes_mode", "u_caes"):
                continue
            if not np.isfinite(float(val)):
                return False, f"{name} 非有限"
        return True, None

    def _temp_bounds(self) -> tuple[tuple[str, float, float], ...]:
        """三个罐体温度的可行性判定区间。

        气罐温度下界在 Modelica 里是 ``T = max(Medium.T_ph(p, h), 253)`` 这样的
        数值钳位而非物理约束，FMU 不可能输出低于它的值、也不会因此报错。把它当硬
        约束会让环境温度（全年最低 248.36 K）在寒冷时段永久锁死 CAES 放电，
        故当 ``gas_temp_min_is_numeric_clamp`` 为真时下界取 -inf。

        Returns:
            (输出名, 下界, 上界) 三元组序列。

        Raises:
            无。
        """
        c = self.params["caes"]
        gas_lo = (
            float("-inf")
            if bool(c.get("gas_temp_min_is_numeric_clamp", False))
            else float(c["gas_temp_min_K"])
        )
        return (
            ("caes_gas_temperature", gas_lo, float(c["gas_temp_max_K"])),
            ("caes_hot_temperature", float(c["hot_temp_min_K"]), float(c["hot_temp_max_K"])),
            ("caes_cold_temperature", float(c["cold_temp_min_K"]), float(c["cold_temp_max_K"])),
        )

    def _battery_bounds(self, soc: float) -> tuple[float, float]:
        """由当前 SOC 与方向裕度计算 u_battery 动态上下界。

        Args:
            soc: 当前电池 SOC。

        Returns:
            (u_low, u_high) 归一化电池功率指令界；不可行时可能 inverted。

        Raises:
            无。
        """
        p = self.params["battery"]
        m = self.margins.get("battery", {})
        p_cap = float(p["P_cap_W"])
        e_cap = float(p["E_cap_J"])
        eta = float(p["eta"])
        # 方向裕度：把危险 residual P99 叠到 safe 界上（非统一 0.02）
        margin_hi = float(m.get("margin_charge_high", 0.0)) + float(
            m.get("residual_p99_charge_high", 0.0)
        )
        margin_lo = float(m.get("margin_discharge_low", 0.0)) + float(
            m.get("residual_p99_discharge_low", 0.0)
        )
        safe_min = float(p["SOC_min"]) + margin_lo
        safe_max = float(p["SOC_max"]) - margin_hi
        # 不得宽于配置的 SOC_safe_*（device_params 中的运行界）
        safe_min = max(safe_min, float(p["SOC_safe_min"]))
        safe_max = min(safe_max, float(p["SOC_safe_max"]))
        dt = self.dt
        u_charge_max = (
            (safe_max - soc) * e_cap / (p_cap * dt) if p_cap * dt > 0 else 0.0
        )
        u_discharge_min = (
            (safe_min - soc) * e_cap * eta / (p_cap * dt) if p_cap * dt > 0 else 0.0
        )
        u_low = max(-1.0, float(u_discharge_min))
        u_high = min(1.0, float(u_charge_max))
        if u_low > u_high:
            return 0.0, 0.0
        return u_low, u_high

    def _thermal_bounds_from_actual(
        self, previous_thermal_w: float
    ) -> tuple[float, float]:
        """由实际上一火电功率与爬坡率计算 u_tp 动态界。

        Args:
            previous_thermal_w: 上一决策步 FMU 实际 p_thermal（W）。

        Returns:
            (u_low, u_high) 归一化火电指令界。

        Raises:
            无。
        """
        t = self.params["thermal"]
        u_min = float(t["u_min"])
        u_max = float(t["u_max"])
        p_cap = float(t["P_cap_W"])
        rate = float(t["rate_max_per_s"])
        # 关键：previous_thermal_w 必须是上一时刻 FMU 实际 p_thermal
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
        """模式特定联合可行性，并附加一步生存性过滤。

        单步硬界检查在周重置下够用，但连续年运行会暴露递归不可行：某个动作本身合法，
        却把系统推进「非 idle 模式全被封死」的死角（典型是冷罐见底同时气罐见底，
        既不能充也不能放，只能永久 idle）。这里在基础掩码之上多看一步：
        若后继状态连一个非 idle 模式都不剩，则当前模式判为不可行。

        Args:
            outputs: 当前 FMU 输出。

        Returns:
            允许的 CAES 模式掩码(ModeMask)。

        Raises:
            无。
        """
        return self._caes_mask_and_intervals(outputs)[0]

    def _caes_mask_and_intervals(
        self, outputs: Mapping[str, float]
    ) -> tuple[ModeMask, dict[str, tuple[float, float] | None]]:
        """同时给出模式掩码与各方向的安全幅值子区间。

        掩码与区间必须由同一次扫描导出，否则会出现「方向被判合法、但带内多数
        幅值其实会越界」的不一致——这正是历史上智能体动作被大量拒绝、并在最短
        运行锁下走进死角的原因。

        Args:
            outputs: 当前 FMU 输出。

        Returns:
            (mask, intervals)，intervals 的键为 "charge" / "discharge"，
            值为 (u_low, u_high) 或 None（该方向不可行）。

        Raises:
            无。
        """
        base = self._caes_mode_mask_base(outputs)
        intervals: dict[str, tuple[float, float] | None] = {
            "charge": self._caes_feasible_u_interval(outputs, CaesMode.CHARGE)
            if base.charge
            else None,
            "discharge": self._caes_feasible_u_interval(outputs, CaesMode.DISCHARGE)
            if base.discharge
            else None,
        }
        charge_ok = intervals["charge"] is not None
        discharge_ok = intervals["discharge"] is not None
        # 一步生存性：某方向即使本步合法，若会把系统推进「非 idle 全被封死」的
        # 死角也判为不可行。这是连续年运行相对周重置的额外要求，周重置会掩盖
        # 递归不可行。用该方向区间内最激进的一端做最坏情况前推。
        if charge_ok and not self._successor_has_reversible_mode(
            outputs, intervals["charge"][1]
        ):
            charge_ok = False
            intervals["charge"] = None
        if discharge_ok and not self._successor_has_reversible_mode(
            outputs, intervals["discharge"][0]
        ):
            discharge_ok = False
            intervals["discharge"] = None
        mask = ModeMask(
            discharge=bool(discharge_ok), idle=bool(base.idle), charge=bool(charge_ok)
        )
        return mask, intervals

    def _successor_has_reversible_mode(
        self, outputs: Mapping[str, float], u_caes: float
    ) -> bool:
        """以给定 u_caes 走一步后，后继状态是否还剩至少一个非 idle 模式。

        只向前看一步，且后继状态用不含生存性过滤的基础掩码判定，避免无限递归。

        Args:
            outputs: 当前 FMU 输出。
            u_caes: 待检的 CAES 指令（取该方向区间内最激进的一端）。

        Returns:
            后继状态仍允许充或放则为 True。

        Raises:
            无。
        """
        action = PhysicalFmuAction(u_tp=1.0, u_battery=0.0, u_caes=float(u_caes))
        pred = self.predict_next_state(outputs, action)
        nxt = self._caes_mode_mask_base(pred)
        return bool(nxt.charge or nxt.discharge)

    def _caes_feasible_u_interval(
        self, outputs: Mapping[str, float], mode: CaesMode
    ) -> tuple[float, float] | None:
        """扫描该方向的合法带，返回一步预测仍在硬界内的 u_caes 子区间。

        用等距网格扫描而非二分：一步预测对 u 是线性的，可行集通常是从带内最弱端
        出发的一段区间，但当前状态已越界时（需要反向动作把库存拉回）单调性会反转，
        二分会给出错误答案。网格扫描对此免疫，且预测只是几次算术运算，开销可忽略。

        Args:
            outputs: 当前 FMU 输出。
            mode: CHARGE 或 DISCHARGE。

        Returns:
            (u_low, u_high) 安全子区间；该方向无任何合法幅值时为 None。

        Raises:
            无。
        """
        cm = self.margins.get("caes", {})
        if mode == CaesMode.CHARGE:
            lo_band, hi_band = CHARGE_LO, CHARGE_HI
            margins, direction = cm.get("charge", {}), "high"
        else:
            lo_band, hi_band = DISCHARGE_LO, DISCHARGE_HI
            margins, direction = cm.get("discharge", {}), "low"
        grid = np.linspace(lo_band, hi_band, self._CAES_SCAN_POINTS)
        feasible = [
            float(u)
            for u in grid
            if self._u_caes_step_ok(outputs, float(u), margins, direction)
        ]
        if not feasible:
            return None
        return min(feasible), max(feasible)

    _CAES_SCAN_POINTS = 33

    def _caes_mode_mask_base(self, outputs: Mapping[str, float]) -> ModeMask:
        """基础模式掩码：gas/hot/cold SOC + pressure + temps，不含生存性过滤。

        Args:
            outputs: 当前 FMU 输出。

        Returns:
            允许的 CAES 模式掩码(ModeMask)。

        Raises:
            无。
        """
        c = self.params["caes"]
        cm = self.margins.get("caes", {})
        gas = float(outputs["caes_gas_soc"])
        hot = float(outputs["caes_hot_soc"])
        cold = float(outputs["caes_cold_soc"])
        p = float(outputs.get("caes_gas_pressure", 8.5e6))
        tg = float(outputs.get("caes_gas_temperature", 300.0))
        th = float(outputs.get("caes_hot_temperature", 400.0))
        tc = float(outputs.get("caes_cold_temperature", 290.0))
        # 方向是否可行 = 带内「最温和」的动作能否留在物理界内。
        # 历史实现用 u_from_mode_mag(mode, 1.0) 当最坏情况探针，但该函数是在合法带内
        # 线性插值：充电 mag=1.0 -> u=1.0（最强），放电 mag=1.0 -> u=-0.33（最弱）。
        # 于是放电方向拿最温和动作判「合法」、充电方向拿最强动作判「合法」，两侧语义
        # 相反且都不符合注释。现在统一为最弱端探针，具体幅值由安全子区间约束。
        charge_ok = self._u_caes_step_ok(
            outputs, CHARGE_LO, cm.get("charge", {}), "high"
        )
        discharge_ok = self._u_caes_step_ok(
            outputs, DISCHARGE_HI, cm.get("discharge", {}), "low"
        )
        # IDLE：当前态在物理界内则始终允许（残差裕度不得禁止待机）
        idle_ok, _ = self.post_step_hard_ok(outputs, use_safe=False)
        # 即使预测失败，仍保留温度硬门（禁充放，不禁 idle）
        current_temps = {
            "caes_gas_temperature": tg,
            "caes_hot_temperature": th,
            "caes_cold_temperature": tc,
        }
        temp_ok = all(
            lo <= current_temps[name] <= hi for name, lo, hi in self._temp_bounds()
        )
        if not temp_ok:
            charge_ok = discharge_ok = False
        # 额外：接近物理界时用联合阈值（仅充/放）
        # 方向约定：充电抬升气罐与热罐、抽低冷罐；放电反向。
        # 冷罐守卫方向必须与气/热罐相反，否则充电时冷罐被抽干却无人拦截。
        chg = cm.get("charge", {})
        dis = cm.get("discharge", {})
        charge_ok = charge_ok and (
            gas
            < float(c["gas_SOC_max"])
            - float(chg.get("margin_gas", 0.0))
            - float(chg.get("residual_p99_gas_high", 0.0))
            and hot
            < float(c["hot_SOC_max"])
            - float(chg.get("margin_hot", 0.0))
            - float(chg.get("residual_p99_hot_high", 0.0))
            and cold
            > float(c["cold_SOC_min"])
            + float(chg.get("margin_cold", 0.0))
            + float(chg.get("residual_p99_cold_low", 0.0))
            and p
            < float(c["gas_pressure_max_Pa"])
            - float(chg.get("margin_pressure_Pa", 0.0))
            - float(chg.get("residual_p99_pressure_high", 0.0))
        )
        discharge_ok = discharge_ok and (
            gas
            > float(c["gas_SOC_min"])
            + float(dis.get("margin_gas", 0.0))
            + float(dis.get("residual_p99_gas_low", 0.0))
            and hot
            > float(c["hot_SOC_min"])
            + float(dis.get("margin_hot", 0.0))
            + float(dis.get("residual_p99_hot_low", 0.0))
            and cold
            < float(c["cold_SOC_max"])
            - float(dis.get("margin_cold", 0.0))
            - float(dis.get("residual_p99_cold_high", 0.0))
            and p
            > float(c["gas_pressure_min_Pa"])
            + float(dis.get("margin_pressure_Pa", 0.0))
            + float(dis.get("residual_p99_pressure_low", 0.0))
        )
        return ModeMask(
            discharge=bool(discharge_ok), idle=bool(idle_ok), charge=bool(charge_ok)
        )

    def _caes_mode_feasible(
        self,
        outputs: Mapping[str, float],
        mode: CaesMode,
        mag: float,
        margins: Mapping[str, Any],
        direction: str,
    ) -> bool:
        """预测给定模式与 ``mag`` 带内位置下一步是否仍在物理界内。

        注意 ``mag`` 是合法带内的插值位置而非物理幅值：放电 mag=1.0 对应 u=-0.33
        （最弱），充电 mag=1.0 对应 u=1.0（最强）。需要按物理幅值判定时请直接用
        ``_u_caes_step_ok``。

        Args:
            outputs: 当前 FMU 输出。
            mode: 待检 CAES 模式。
            mag: 合法带内插值位置 [0, 1]。
            margins: 该模式方向的裕度子字典。
            direction: "high" / "low" / "both"，危险 SOC 方向。

        Returns:
            若预测状态满足硬界与方向裕度则为 True。

        Raises:
            无。
        """
        return self._u_caes_step_ok(
            outputs, u_from_mode_mag(mode, mag), margins, direction
        )

    def _u_caes_step_ok(
        self,
        outputs: Mapping[str, float],
        u_caes: float,
        margins: Mapping[str, Any],
        direction: str,
    ) -> bool:
        """给定具体 u_caes，一步预测是否仍在物理界内（含 residual 方向裕度）。

        Args:
            outputs: 当前 FMU 输出。
            u_caes: 具体 CAES 指令。
            margins: 该方向的裕度子字典。
            direction: "high" / "low" / "both"，危险 SOC 方向。

        Returns:
            若预测状态满足硬界与方向裕度则为 True。

        Raises:
            无。
        """
        action = PhysicalFmuAction(u_tp=1.0, u_battery=0.0, u_caes=float(u_caes))
        pred = self.predict_next_state(outputs, action)
        ok, _ = self.post_step_hard_ok(pred, use_safe=False)
        if not ok:
            return False
        c = self.params["caes"]
        if direction in ("high", "both"):
            if float(pred["caes_gas_soc"]) > float(c["gas_SOC_max"]) - float(
                margins.get(
                    "residual_p99_gas_high", margins.get("residual_p99_gas", 0.0)
                )
            ):
                return False
        if direction in ("low", "both"):
            if float(pred["caes_gas_soc"]) < float(c["gas_SOC_min"]) + float(
                margins.get(
                    "residual_p99_gas_low", margins.get("residual_p99_gas", 0.0)
                )
            ):
                return False
        return True

    @staticmethod
    def _caes_magnitude_caps(
        intervals: Mapping[str, tuple[float, float] | None],
    ) -> dict[str, float]:
        """各方向最大安全物理幅值 |u|（metadata/诊断用）。

        原实现对 ``u_from_mode_mag(mode, mid)`` 做二分，隐含假设「mid 越大越危险」。
        该假设对充电成立、对放电反号（mid=1.0 是最弱放电），因此放电方向的
        cap 恒为 0，诊断值一直是错的。现在直接由安全子区间导出。

        Args:
            intervals: ``_caes_mask_and_intervals`` 给出的方向区间。

        Returns:
            键为 "discharge" / "charge" 的最大安全 |u|；该方向不可行时为 0。

        Raises:
            无。
        """
        caps = {"discharge": 0.0, "charge": 0.0}
        for key in caps:
            span = intervals.get(key)
            if span is not None:
                caps[key] = float(max(abs(span[0]), abs(span[1])))
        return caps
