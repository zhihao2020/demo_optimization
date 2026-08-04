"""参数化策略 PSO：每个粒子评价 = 完整一周 FMU 滚仿真。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from actions import CaesMode
from envs.power_system_env import PowerSystemEnv
from training.evaluate_td3 import evaluate_policy

from .metrics import extract_kpi_from_eval


@dataclass
class PSOConfig:
    n_particles: int = 12
    n_iters: int = 25
    w: float = 0.6
    c1: float = 1.4
    c2: float = 1.4
    seed: int = 0
    # θ = [charge_th, discharge_th, bat_mag, caes_mag, tp_bias, recovery_strength]
    dim: int = 6
    rho_uns: float = 1e3
    rho_fail: float = 1e5
    rho_soc: float = 5e4
    soc_tol: float = 0.06


class ParametricPricePolicy:
    """低维 θ 的峰谷+回收策略，供 PSO 优化。"""

    def __init__(self, env: PowerSystemEnv, theta: np.ndarray):
        self.env = env
        th = np.asarray(theta, dtype=np.float64).ravel()
        self.charge_th = float(np.clip(th[0], 0.15, 0.7))
        self.discharge_th = float(np.clip(th[1], 0.5, 1.5))
        if self.discharge_th < self.charge_th + 0.05:
            self.discharge_th = self.charge_th + 0.05
        self.bat_mag = float(np.clip(th[2], 0.2, 1.0))
        self.caes_mag = float(np.clip(th[3], 0.0, 1.0))
        self.tp_bias = float(np.clip(th[4], 0.0, 1.0))  # 0=偏下限 1=偏上限
        self.recovery = float(np.clip(th[5], 0.0, 1.0))

    def predict(self, obs, deterministic: bool = True) -> dict:
        env = self.env
        feas = env.get_feasible_action_spec()
        outs = env.last_outputs or {}
        buy = None
        if getattr(env, "price_profile", None) is not None:
            try:
                buy, _ = env.price_profile.prices_at(float(env.adapter.time))
            except Exception:
                buy = None
        bat = float(outs.get("battery_soc", 0.5))
        gas = float(outs.get("caes_gas_soc", 0.5))
        lo, hi = float(feas.u_tp_low), float(feas.u_tp_high)
        u_tp = lo + self.tp_bias * (hi - lo)
        u_bat = 0.0
        mode = CaesMode.IDLE
        mag = 0.0

        rem = int(env.episode_steps - env.step_index)
        horizon = 40
        recover = rem <= horizon and self.recovery > 0.3

        if recover and env.initial_soc is not None:
            b0 = float(env.initial_soc.get("battery_soc", 0.5))
            g0 = float(env.initial_soc.get("caes_gas_soc", 0.85))
            strength = self.recovery
            if bat < b0 - 0.03:
                u_bat = float(np.clip(self.bat_mag * strength, feas.u_battery_low, feas.u_battery_high))
            elif bat > b0 + 0.03:
                u_bat = float(np.clip(-self.bat_mag * strength, feas.u_battery_low, feas.u_battery_high))
            if gas < g0 - 0.03 and feas.mode_mask.charge:
                mode, mag = CaesMode.CHARGE, self.caes_mag * strength
            elif gas > g0 + 0.03 and feas.mode_mask.discharge:
                mode, mag = CaesMode.DISCHARGE, self.caes_mag * strength
            u_tp = hi
        elif buy is not None:
            if buy <= self.charge_th and bat < 0.9:
                u_bat = float(np.clip(self.bat_mag, feas.u_battery_low, feas.u_battery_high))
                u_tp = lo + 0.4 * (hi - lo)
                if self.caes_mag > 0.05 and gas < 0.92 and feas.mode_mask.charge:
                    mode, mag = CaesMode.CHARGE, self.caes_mag
            elif buy >= self.discharge_th and bat > 0.2:
                u_bat = float(np.clip(-self.bat_mag, feas.u_battery_low, feas.u_battery_high))
                u_tp = lo + 0.5 * (hi - lo)
                if self.caes_mag > 0.05 and gas > 0.3 and feas.mode_mask.discharge:
                    mode, mag = CaesMode.DISCHARGE, self.caes_mag

        return {
            "u_tp": np.asarray([float(np.clip(u_tp, lo, hi))], dtype=np.float32),
            "u_battery": np.asarray([u_bat], dtype=np.float32),
            "caes_mode": int(mode),
            "caes_magnitude": np.asarray([float(mag)], dtype=np.float32),
        }


def _fitness_from_eval(kpi: dict[str, Any], cfg: PSOConfig) -> float:
    j = float(kpi.get("net_cashflow_j") or 0.0)
    uns = float(kpi.get("unserved_mwh") or 0.0)
    fail = float(kpi.get("fmu_failure_count") or 0.0) + float(kpi.get("invalid_transition_count") or 0.0)
    l1 = float(kpi.get("terminal_soc_l1") or 0.0)
    soc_pen = max(0.0, l1 - cfg.soc_tol)
    return j - cfg.rho_uns * uns - cfg.rho_fail * fail - cfg.rho_soc * soc_pen


def evaluate_theta(
    theta: np.ndarray,
    *,
    start_time: float,
    seed: int,
    cfg: PSOConfig,
) -> tuple[float, dict[str, Any]]:
    env = PowerSystemEnv(run_id="pso_eval", forecast_enabled=True)
    try:
        pol = ParametricPricePolicy(env, theta)
        t0 = time.perf_counter()
        res = evaluate_policy(env, pol, reset_options={"start_time": float(start_time)})
        wall = time.perf_counter() - t0
        kpi = extract_kpi_from_eval(res, wall_s=wall, fmu_steps=res.get("valid_steps"))
        # net_cashflow: evaluate 的 economic_cashflow_delta 在 terms 里是累计
        # 若为末步单值，用 weekly_raw_total_cost 反号近似
        if abs(float(kpi.get("net_cashflow_j") or 0.0)) < 1.0:
            raw = res.get("weekly_raw_total_cost")
            if raw is not None:
                kpi["net_cashflow_j"] = -float(raw)
        fit = _fitness_from_eval(kpi, cfg)
        kpi["fitness"] = fit
        kpi["theta"] = np.asarray(theta, dtype=np.float64).tolist()
        return fit, kpi
    finally:
        env.close()


def run_pso(
    *,
    start_time: float = 0.0,
    cfg: PSOConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or PSOConfig()
    rng = np.random.default_rng(cfg.seed)
    # 初始边界
    low = np.array([0.2, 0.6, 0.3, 0.0, 0.3, 0.5], dtype=np.float64)
    high = np.array([0.55, 1.2, 1.0, 0.9, 1.0, 1.0], dtype=np.float64)
    pos = rng.uniform(low, high, size=(cfg.n_particles, cfg.dim))
    vel = rng.normal(0.0, 0.05, size=pos.shape)
    pbest = pos.copy()
    pbest_f = np.full(cfg.n_particles, -np.inf)
    gbest = pos[0].copy()
    gbest_f = -np.inf
    gbest_kpi: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    fmu_steps_total = 0
    t0 = time.perf_counter()

    for it in range(cfg.n_iters):
        for i in range(cfg.n_particles):
            fit, kpi = evaluate_theta(pos[i], start_time=start_time, seed=cfg.seed + i, cfg=cfg)
            fmu_steps_total += int(kpi.get("fmu_steps") or 168)
            if fit > pbest_f[i]:
                pbest_f[i] = fit
                pbest[i] = pos[i].copy()
            if fit > gbest_f:
                gbest_f = fit
                gbest = pos[i].copy()
                gbest_kpi = kpi
        # velocity update
        r1 = rng.random(pos.shape)
        r2 = rng.random(pos.shape)
        vel = cfg.w * vel + cfg.c1 * r1 * (pbest - pos) + cfg.c2 * r2 * (gbest - pos)
        pos = np.clip(pos + vel, low, high)
        history.append({"iter": it, "gbest_f": float(gbest_f)})
        print(f"[PSO] iter={it+1}/{cfg.n_iters} gbest_f={gbest_f:.4e}")

    wall = time.perf_counter() - t0
    # 最终再评估一次 gbest
    _, final_kpi = evaluate_theta(gbest, start_time=start_time, seed=cfg.seed, cfg=cfg)
    final_kpi["method"] = "pso"
    final_kpi["fitness"] = gbest_f
    final_kpi["wall_s_search"] = wall
    final_kpi["fmu_steps_search"] = fmu_steps_total
    final_kpi["history"] = history
    final_kpi["theta_best"] = gbest.tolist()
    return final_kpi
