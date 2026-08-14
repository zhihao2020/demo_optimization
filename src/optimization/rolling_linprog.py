"""滚动线性规划：短视域代理 + 只执行第一步 + 真仿真闭环。

相对旧实现的修复：
  - 风光发电按 FMU 惯例（发电为负瓦）折成正的注入兆瓦，不再把残差加反
  - 用日前表做 24 小时负荷 / 风 / 光形状，不再把当前小时重复 8 次
  - 目标对齐综合收益：购售电 + 燃料 + 碳 + 弃电 / 缺供 + 电池放电磨损
  - 周末库存用线性绝对值软罚（真正进目标）
  - 去掉最后 40 小时强行抬火电
  - 压空小功率走待机，不再把幅值抬到 0.5
这不是一周全局最优，也不是综合收益的严格上界。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import linprog

from actions import CaesMode
from actions.caes_u import u_from_mode_mag


@dataclass
class LinprogMPCConfig:
    horizon: int = 24
    terminal_soc_yuan: float = 2.0e5
    fuel_yuan_per_mwh: float = 400.0
    carbon_yuan_per_t: float = 80.0
    eta_thermal_t_per_mwh: float = 0.85
    eta_grid_t_per_mwh: float = 0.5703
    nu_curt_yuan_per_mwh: float = 300.0
    nu_uns_yuan_per_mwh: float = 1000.0
    deg_yuan_per_mwh: float = 250.0
    caes_hours_at_rated: float = 4.0
    time_limit_s: float = 8.0


def gen_mw(power_w: float) -> float:
    """发电通道：FMU 多为负瓦，转成正的注入兆瓦。"""
    return float(max(0.0, -float(power_w))) * 1e-6


def demand_mw(power_w: float) -> float:
    """负荷：FMU 为正瓦需求。"""
    return float(max(0.0, float(power_w))) * 1e-6


def wind_power_proxy(speed_ms: float, *, vin: float = 3.0, vr: float = 12.0, vout: float = 25.0) -> float:
    """标准三段风速–功率标幺，只用于把预报风速换成相对形状。"""
    v = float(speed_ms)
    if v < vin or v >= vout:
        return 0.0
    if v >= vr:
        return 1.0
    x = (v - vin) / max(vr - vin, 1e-6)
    return float(x * x * x)


def _load_yaml(rel: str) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    with (root / rel).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


class RollingLinprogController:
    """滚动线性规划控制器。"""

    def __init__(self, env: Any, cfg: LinprogMPCConfig | None = None):
        self.env = env
        self.cfg = cfg or LinprogMPCConfig()
        self.params = _load_yaml("src/config/device_params.yaml")
        self._apply_reward_defaults()
        self._name = "rolling_linprog"

    def _apply_reward_defaults(self) -> None:
        try:
            rew = _load_yaml("src/config/reward_config.yaml")
        except OSError:
            return
        carbon = rew.get("carbon") or {}
        cut = rew.get("curtailment") or {}
        if carbon.get("enabled", True):
            self.cfg.carbon_yuan_per_t = float(carbon.get("price_cny_per_t", self.cfg.carbon_yuan_per_t))
            self.cfg.eta_thermal_t_per_mwh = float(
                carbon.get("eta_thermal_t_per_mwh", self.cfg.eta_thermal_t_per_mwh)
            )
            self.cfg.eta_grid_t_per_mwh = float(
                carbon.get("eta_grid_t_per_mwh", self.cfg.eta_grid_t_per_mwh)
            )
        if cut.get("enabled", True):
            self.cfg.nu_curt_yuan_per_mwh = float(cut.get("nu_curt_cny_per_mwh", self.cfg.nu_curt_yuan_per_mwh))
            self.cfg.nu_uns_yuan_per_mwh = float(cut.get("nu_uns_cny_per_mwh", self.cfg.nu_uns_yuan_per_mwh))

    def on_episode_reset(self, info: dict) -> None:
        return None

    def _prices_horizon(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        env = self.env
        buy = np.zeros(horizon, dtype=np.float64)
        sell = np.zeros(horizon, dtype=np.float64)
        t0 = float(env.adapter.time)
        dt = float(env.config["fmu"]["decision_interval_seconds"])
        for k in range(horizon):
            t = t0 + k * dt
            if getattr(env, "price_profile", None) is not None:
                try:
                    b, s = env.price_profile.prices_at(t)
                    buy[k], sell[k] = float(b), float(s)
                except Exception:
                    buy[k], sell[k] = 0.6, 0.2
            else:
                buy[k], sell[k] = 0.6, 0.2
        return buy, sell

    def _boundary_horizon(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回负荷、风电可用、光伏可用（正兆瓦），长度 = 视界。"""
        env = self.env
        h = int(self.cfg.horizon)
        outs = env.last_outputs or {}
        load0 = demand_mw(outs.get("p_load_actual", 2e8))
        wind0 = gen_mw(outs.get("p_wind_available", 0.0))
        pv0 = gen_mw(outs.get("p_pv_available", 0.0))
        load = np.full(h, load0)
        wind = np.full(h, wind0)
        pv = np.full(h, pv0)

        fp = getattr(env, "forecast_provider", None)
        if fp is None:
            return load, wind, pv
        try:
            t0 = float(env.adapter.time)
            feat = np.asarray(fp.at_time(t0), dtype=np.float64)
            nch = len(getattr(fp, "sources", ()) or (0, 1, 2, 3))
            fh = int(getattr(fp, "horizon_hours", h))
            if feat.size < nch:
                return load, wind, pv
            mat = feat.reshape(feat.size // nch, nch)
            take = min(h, len(mat))
            src = list(fp.sources)
            wind_ms = mat[:take, 0] * float(src[0].scale) + float(src[0].offset)
            irr = mat[:take, 1] * float(src[1].scale) + float(src[1].offset)
            load_w = mat[:take, 3] * float(src[3].scale) + float(src[3].offset)
            load_mw = np.maximum(load_w, 0.0) * 1e-6
            if float(load_mw[0]) > 1e-6:
                load[:take] = load0 * (load_mw / float(load_mw[0]))
            else:
                load[:take] = load_mw
            p0 = wind_power_proxy(float(wind_ms[0]))
            if p0 > 1e-8:
                wind[:take] = wind0 * np.array([wind_power_proxy(v) for v in wind_ms]) / p0
            irr0 = max(float(irr[0]), 1.0)
            pv[:take] = pv0 * np.maximum(irr, 0.0) / irr0
            wind = np.clip(wind, 0.0, 350.0)
            pv = np.clip(pv, 0.0, 350.0)
            load = np.clip(load, 0.0, 800.0)
        except Exception:
            return np.full(h, load0), np.full(h, wind0), np.full(h, pv0)
        return load, wind, pv

    def _solve(self) -> dict[str, float]:
        env = self.env
        cfg = self.cfg
        h = int(cfg.horizon)
        bat = self.params["battery"]
        th = self.params["thermal"]
        caes = self.params["caes"]
        grid = self.params["grid"]

        e_bat = float(bat["E_cap_J"]) / 3.6e9
        eta = float(bat["eta"])
        p_bat_max = float(bat["P_cap_W"]) * 1e-6
        p_tp_min = float(th["P_min_W"]) * 1e-6
        p_tp_max = float(th["P_max_W"]) * 1e-6
        p_caes_max = float(caes["P_cap_W"]) * 1e-6
        soc_g_lo = float(caes["gas_SOC_min"])
        soc_g_hi = float(caes["gas_SOC_max"])
        e_gas = p_caes_max * float(cfg.caes_hours_at_rated) * max(soc_g_hi - soc_g_lo, 0.05)
        p_buy_max = float(grid["P_max_buy_W"]) * 1e-6
        p_sell_max = abs(float(grid["P_max_sell_W"])) * 1e-6

        outs = env.last_outputs or {}
        soc_b0 = float(outs.get("battery_soc", 0.5))
        soc_g0 = float(outs.get("caes_gas_soc", 0.85))
        p_tp_prev = gen_mw(outs.get("p_thermal", -p_tp_min * 1e6))
        if p_tp_prev < p_tp_min * 0.5:
            p_tp_prev = demand_mw(abs(float(outs.get("p_thermal", p_tp_min * 1e6))))
        p_tp_prev = float(np.clip(p_tp_prev, p_tp_min, p_tp_max))

        load, wind, pv = self._boundary_horizon()
        buy_p, sell_p = self._prices_horizon(h)
        buy_mwh = buy_p * 1000.0
        sell_mwh = sell_p * 1000.0
        c_th = (
            cfg.fuel_yuan_per_mwh
            + cfg.carbon_yuan_per_t * cfg.eta_thermal_t_per_mwh
        )
        c_buy_extra = cfg.carbon_yuan_per_t * cfg.eta_grid_t_per_mwh

        # 每步 10 个控制量：火电、电池充/放、气充/放、购、售、弃风、弃光、缺供
        n_u = 10
        n_soc = (h + 1) * 2
        n_term = 2
        n = h * n_u + n_soc + n_term
        c = np.zeros(n)
        for k in range(h):
            base = k * n_u
            c[base + 0] = c_th
            c[base + 2] = cfg.deg_yuan_per_mwh
            c[base + 5] = buy_mwh[k] + c_buy_extra
            c[base + 6] = -sell_mwh[k]
            c[base + 7] = cfg.nu_curt_yuan_per_mwh
            c[base + 8] = cfg.nu_curt_yuan_per_mwh
            c[base + 9] = cfg.nu_uns_yuan_per_mwh
        c[-2] = cfg.terminal_soc_yuan
        c[-1] = cfg.terminal_soc_yuan

        bounds: list[tuple[float, float]] = []
        for k in range(h):
            bounds += [
                (p_tp_min, p_tp_max),
                (0.0, p_bat_max),
                (0.0, p_bat_max),
                (0.0, p_caes_max),
                (0.0, p_caes_max),
                (0.0, p_buy_max),
                (0.0, p_sell_max),
                (0.0, float(wind[k])),
                (0.0, float(pv[k])),
                (0.0, float(load[k])),
            ]
        sb_lo, sb_hi = float(bat["SOC_min"]), float(bat["SOC_max"])
        for _ in range(h + 1):
            bounds.append((sb_lo, sb_hi))
        for _ in range(h + 1):
            bounds.append((soc_g_lo, soc_g_hi))
        bounds.append((0.0, None))
        bounds.append((0.0, None))

        i_sb0 = h * n_u
        i_sg0 = h * n_u + (h + 1)
        i_eb = n - 2
        i_eg = n - 1
        a_eq: list[np.ndarray] = []
        b_eq: list[float] = []
        row = np.zeros(n)
        row[i_sb0] = 1.0
        a_eq.append(row)
        b_eq.append(soc_b0)
        row = np.zeros(n)
        row[i_sg0] = 1.0
        a_eq.append(row)
        b_eq.append(soc_g0)

        dt = 1.0
        for k in range(h):
            base = k * n_u
            # 火电 + 风实用 + 光实用 + 放电 - 充电 + 购 - 售 + 缺供 = 负荷
            # 风实用 = 可用 - 弃风
            row = np.zeros(n)
            row[base + 0] = 1.0
            row[base + 1] = -1.0
            row[base + 2] = 1.0
            row[base + 3] = -1.0
            row[base + 4] = 1.0
            row[base + 5] = 1.0
            row[base + 6] = -1.0
            row[base + 7] = -1.0
            row[base + 8] = -1.0
            row[base + 9] = 1.0
            a_eq.append(row)
            b_eq.append(float(load[k] - wind[k] - pv[k]))

            row = np.zeros(n)
            row[i_sb0 + k + 1] = 1.0
            row[i_sb0 + k] = -1.0
            row[base + 1] = -eta * dt / max(e_bat, 1e-6)
            row[base + 2] = dt / (max(eta, 1e-6) * max(e_bat, 1e-6))
            a_eq.append(row)
            b_eq.append(0.0)

            row = np.zeros(n)
            row[i_sg0 + k + 1] = 1.0
            row[i_sg0 + k] = -1.0
            row[base + 3] = -dt / max(e_gas, 1e-6)
            row[base + 4] = dt / max(e_gas, 1e-6)
            a_eq.append(row)
            b_eq.append(0.0)

        a_ub: list[np.ndarray] = []
        b_ub: list[float] = []
        du_max = float(th["rate_max_per_s"]) * 3600.0
        dp_max = du_max * p_tp_max
        row = np.zeros(n)
        row[0] = 1.0
        a_ub.append(row)
        b_ub.append(p_tp_prev + dp_max)
        row = np.zeros(n)
        row[0] = -1.0
        a_ub.append(row)
        b_ub.append(-p_tp_prev + dp_max)
        for k in range(1, h):
            b0 = (k - 1) * n_u
            b1 = k * n_u
            row = np.zeros(n)
            row[b1] = 1.0
            row[b0] = -1.0
            a_ub.append(row)
            b_ub.append(dp_max)
            row = np.zeros(n)
            row[b1] = -1.0
            row[b0] = 1.0
            a_ub.append(row)
            b_ub.append(dp_max)

        s_ref_b = float((env.initial_soc or {}).get("battery_soc", soc_b0))
        s_ref_g = float((env.initial_soc or {}).get("caes_gas_soc", soc_g0))
        # |s_H - s_ref| <= e
        row = np.zeros(n)
        row[i_sb0 + h] = 1.0
        row[i_eb] = -1.0
        a_ub.append(row)
        b_ub.append(s_ref_b)
        row = np.zeros(n)
        row[i_sb0 + h] = -1.0
        row[i_eb] = -1.0
        a_ub.append(row)
        b_ub.append(-s_ref_b)
        row = np.zeros(n)
        row[i_sg0 + h] = 1.0
        row[i_eg] = -1.0
        a_ub.append(row)
        b_ub.append(s_ref_g)
        row = np.zeros(n)
        row[i_sg0 + h] = -1.0
        row[i_eg] = -1.0
        a_ub.append(row)
        b_ub.append(-s_ref_g)

        res = linprog(
            c,
            A_ub=np.asarray(a_ub),
            b_ub=np.asarray(b_ub),
            A_eq=np.asarray(a_eq),
            b_eq=np.asarray(b_eq),
            bounds=bounds,
            method="highs",
            options={"presolve": True, "time_limit": float(cfg.time_limit_s)},
        )
        if not res.success or res.x is None:
            return {"ok": 0.0, "p_tp": p_tp_prev, "p_bat": 0.0, "p_gas": 0.0}
        x = res.x
        return {
            "ok": 1.0,
            "p_tp": float(x[0]),
            "p_bat": float(x[1] - x[2]),
            "p_gas": float(x[3] - x[4]),
            "obj": float(res.fun) if res.fun is not None else 0.0,
        }

    def predict(self, obs, deterministic: bool = True) -> dict:
        env = self.env
        try:
            feas = env.get_feasible_action_spec()
        except Exception:
            return {
                "u_tp": np.asarray([1.0], np.float32),
                "u_battery": np.asarray([0.0], np.float32),
                "u_caes": np.asarray([0.0], np.float32),
            }

        sol = self._solve()
        p_tp_max = float(self.params["thermal"]["P_max_W"]) * 1e-6
        p_bat_max = float(self.params["battery"]["P_cap_W"]) * 1e-6
        p_caes_max = float(self.params["caes"]["P_cap_W"]) * 1e-6

        u_tp = float(np.clip(sol["p_tp"] / max(p_tp_max, 1e-6), feas.u_tp_low, feas.u_tp_high))
        u_bat = float(np.clip(sol["p_bat"] / max(p_bat_max, 1e-6), feas.u_battery_low, feas.u_battery_high))

        p_gas = float(sol["p_gas"])
        ratio = abs(p_gas) / max(p_caes_max, 1e-6)
        mode = CaesMode.IDLE
        mag = 0.0
        if p_gas > 8.0 and feas.mode_mask.charge and ratio >= 0.50:
            mode = CaesMode.CHARGE
            u_ratio = float(np.clip(ratio, 0.86, 1.0))
            mag = (u_ratio - 0.86) / 0.14
        elif p_gas < -8.0 and feas.mode_mask.discharge and ratio >= 0.25:
            mode = CaesMode.DISCHARGE
            u_ratio = float(np.clip(ratio, 0.33, 1.0))
            mag = (1.0 - u_ratio) / 0.67
        elif not feas.mode_mask.idle:
            if feas.mode_mask.discharge:
                mode, mag = CaesMode.DISCHARGE, 0.15
            elif feas.mode_mask.charge:
                mode, mag = CaesMode.CHARGE, 0.0

        return {
            "u_tp": np.asarray([u_tp], dtype=np.float32),
            "u_battery": np.asarray([u_bat], dtype=np.float32),
            "u_caes": np.asarray([float(u_from_mode_mag(mode, mag))], dtype=np.float32),
        }
