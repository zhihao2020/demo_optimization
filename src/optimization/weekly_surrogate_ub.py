"""Weekly perfect-foresight convex upper bound on a *relaxed cash-flow surrogate*.

Aligned with FMU trajectory economics (see ``paper_dispatch_traj``):
  - Load power > 0 (demand); thermal/wind/PV actual < 0 (generation).
  - Weekly J ≈ load revenue + grid TOU cash-flow + thermal/wind/PV O&M
    (+ small storage terms). Load revenue is exogenous for fixed load.
  - Thermal O&M ≈ 400 CNY/MWh; wind ≈ 341; PV ≈ 250 (calibrated on B0 winter).

**Not** a rigorous global optimum of the Modelica twin. Gap vs closed-loop
FMU J mixes model mismatch and suboptimality — state this in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class WeeklyUBConfig:
    horizon: int = 168
    fuel_yuan_per_mwh: float = 400.0  # calibrated on FMU B0 winter thermal
    wind_om_yuan_per_mwh: float = 340.72
    pv_om_yuan_per_mwh: float = 250.18
    terminal_eps: float = 0.06
    E_gas_mwh_proxy: float = 80.0
    time_limit_s: float = 30.0
    must_take_renewables: bool = True


def _load_device_params(root: Path | None = None) -> dict[str, Any]:
    root = root or ROOT
    with (root / "src" / "config" / "device_params.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _gen_mw(series: np.ndarray) -> np.ndarray:
    """FMU generation channels are ≤0 W; convert to ≥0 MW injection."""
    s = np.asarray(series, dtype=np.float64)
    return np.maximum(0.0, -s) * 1e-6


def exogenous_from_dispatch_csv(
    csv_path: Path | str,
    *,
    horizon: int = 168,
) -> dict[str, np.ndarray | float]:
    """Exogenous week from a closed-loop traj (B0 recommended for boundary)."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    n = min(int(horizon), len(df))
    df = df.iloc[:n].copy()

    load_mw = df["obs_p_load_actual"].to_numpy(dtype=np.float64) * 1e-6
    wind_mw = _gen_mw(df["obs_p_wind_available"].to_numpy(dtype=np.float64))
    pv_mw = _gen_mw(df["obs_p_pv_available"].to_numpy(dtype=np.float64))
    buy = df["rt_market_buy_yuan_per_kwh"].to_numpy(dtype=np.float64)
    sell = df["rt_market_sell_yuan_per_kwh"].to_numpy(dtype=np.float64)

    # Fixed FMU-aligned components for this week (exogenous under fixed load / must-take RE)
    load_rev = float(df["rt_economic_cashflow_delta_load"].sum())
    # If we force must-take full available RE, O&M ≈ rate * available MWh
    wind_om_full = -float(WeeklyUBConfig.wind_om_yuan_per_mwh) * float(wind_mw.sum())
    pv_om_full = -float(WeeklyUBConfig.pv_om_yuan_per_mwh) * float(pv_mw.sum())

    soc_b0 = float(df["obs_battery_soc"].iloc[0])
    soc_g0 = float(df["obs_caes_gas_soc"].iloc[0])
    # thermal gen MW at t0
    p_tp0 = float(_gen_mw(np.array([df["obs_p_thermal"].iloc[0]]))[0])
    j_fmu = float(df["rt_economic_cashflow_delta"].sum())

    return {
        "load_mw": load_mw,
        "wind_mw": wind_mw,
        "pv_mw": pv_mw,
        "buy_yuan_per_kwh": buy,
        "sell_yuan_per_kwh": sell,
        "soc_bat0": soc_b0,
        "soc_gas0": soc_g0,
        "p_tp0_mw": p_tp0,
        "j_fmu": j_fmu,
        "load_revenue_fixed": load_rev,
        "wind_om_if_full": wind_om_full,
        "pv_om_if_full": pv_om_full,
        "n": n,
    }


def solve_weekly_lp(
    exo: dict[str, np.ndarray | float],
    *,
    cfg: WeeklyUBConfig | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Maximise surrogate weekly cash-flow (HiGHS LP).

    Decision: thermal, battery c/d, CAES-eq c/d, buy/sell.
    Renewables: must-take available MW (reduces residual demand).
    J_surr = load_rev_fixed + RE_om_full + (-linprog cost of fuel/buy/sell).
    """
    cfg = cfg or WeeklyUBConfig()
    params = params or _load_device_params()
    H = int(min(cfg.horizon, int(exo["n"])))
    bat = params["battery"]
    th = params["thermal"]
    caes = params["caes"]
    grid = params["grid"]

    E_bat_mwh = float(bat["E_cap_J"]) / 3.6e9
    eta = float(bat["eta"])
    p_bat_max = float(bat["P_cap_W"]) * 1e-6
    p_tp_min = float(th["P_min_W"]) * 1e-6
    p_tp_max = float(th["P_max_W"]) * 1e-6
    p_caes_max = float(caes["P_cap_W"]) * 1e-6
    E_gas_mwh = float(cfg.E_gas_mwh_proxy)
    p_buy_max = float(grid["P_max_buy_W"]) * 1e-6
    p_sell_max = abs(float(grid["P_max_sell_W"])) * 1e-6

    load = np.asarray(exo["load_mw"], dtype=np.float64)[:H]
    wind = np.asarray(exo["wind_mw"], dtype=np.float64)[:H]
    pv = np.asarray(exo["pv_mw"], dtype=np.float64)[:H]
    residual = load - wind - pv  # MW still to serve by thermal/storage/grid
    buy_mwh = np.asarray(exo["buy_yuan_per_kwh"], dtype=np.float64)[:H] * 1000.0
    sell_mwh = np.asarray(exo["sell_yuan_per_kwh"], dtype=np.float64)[:H] * 1000.0
    soc_b0 = float(exo["soc_bat0"])
    soc_g0 = float(exo["soc_gas0"])
    p_tp_prev = float(exo["p_tp0_mw"])
    if p_tp_prev < p_tp_min:
        p_tp_prev = p_tp_min

    n_u = 7
    n = H * n_u + (H + 1) + (H + 1)
    # minimise cost = fuel + buy - sell  ⇒  maximise cash-flow proxy = -cost
    c = np.zeros(n)
    for h in range(H):
        base = h * n_u
        c[base + 0] = cfg.fuel_yuan_per_mwh
        c[base + 5] = buy_mwh[h]
        c[base + 6] = -sell_mwh[h]

    bounds: list[tuple[float | None, float | None]] = []
    for _ in range(H):
        bounds += [
            (p_tp_min, p_tp_max),
            (0.0, p_bat_max),
            (0.0, p_bat_max),
            (0.0, p_caes_max),
            (0.0, p_caes_max),
            (0.0, p_buy_max),
            (0.0, p_sell_max),
        ]
    sb_lo, sb_hi = float(bat["SOC_min"]), float(bat["SOC_max"])
    sg_lo, sg_hi = float(caes["gas_SOC_min"]), float(caes["gas_SOC_max"])
    for _ in range(H + 1):
        bounds.append((sb_lo, sb_hi))
    for _ in range(H + 1):
        bounds.append((sg_lo, sg_hi))

    A_eq: list[np.ndarray] = []
    b_eq: list[float] = []
    i_sb0 = H * n_u
    i_sg0 = H * n_u + (H + 1)

    row = np.zeros(n)
    row[i_sb0] = 1.0
    A_eq.append(row)
    b_eq.append(soc_b0)
    row = np.zeros(n)
    row[i_sg0] = 1.0
    A_eq.append(row)
    b_eq.append(soc_g0)

    dt = 1.0
    for h in range(H):
        base = h * n_u
        # FMU/settlement: p_grid>0 buy, p_grid<0 sell.
        # Identity: tp + bdis - bch + gdis - gch - buy + sell = residual
        # with residual = load - wind - pv (must-take RE).
        row = np.zeros(n)
        row[base + 0] = 1.0
        row[base + 1] = -1.0  # charge
        row[base + 2] = 1.0   # discharge
        row[base + 3] = -1.0
        row[base + 4] = 1.0
        row[base + 5] = -1.0  # buy
        row[base + 6] = 1.0   # sell
        A_eq.append(row)
        b_eq.append(float(residual[h]))

        row = np.zeros(n)
        row[i_sb0 + h + 1] = 1.0
        row[i_sb0 + h] = -1.0
        row[base + 1] = -eta * dt / max(E_bat_mwh, 1e-6)
        row[base + 2] = dt / (max(eta, 1e-6) * max(E_bat_mwh, 1e-6))
        A_eq.append(row)
        b_eq.append(0.0)

        row = np.zeros(n)
        row[i_sg0 + h + 1] = 1.0
        row[i_sg0 + h] = -1.0
        row[base + 3] = -dt / max(E_gas_mwh, 1e-6)
        row[base + 4] = dt / max(E_gas_mwh, 1e-6)
        A_eq.append(row)
        b_eq.append(0.0)

    du_max = float(th["rate_max_per_s"]) * 3600.0
    dP_max = du_max * p_tp_max
    A_ub: list[np.ndarray] = []
    b_ub: list[float] = []
    row = np.zeros(n)
    row[0] = 1.0
    A_ub.append(row)
    b_ub.append(p_tp_prev + dP_max)
    row = np.zeros(n)
    row[0] = -1.0
    A_ub.append(row)
    b_ub.append(-p_tp_prev + dP_max)
    for h in range(1, H):
        b0 = (h - 1) * n_u
        b1 = h * n_u
        row = np.zeros(n)
        row[b1] = 1.0
        row[b0] = -1.0
        A_ub.append(row)
        b_ub.append(dP_max)
        row = np.zeros(n)
        row[b1] = -1.0
        row[b0] = 1.0
        A_ub.append(row)
        b_ub.append(dP_max)

    eps = float(cfg.terminal_eps)
    for idx0, s0, lo_box, hi_box in (
        (i_sb0 + H, soc_b0, sb_lo, sb_hi),
        (i_sg0 + H, soc_g0, sg_lo, sg_hi),
    ):
        hi = min(hi_box, s0 + eps)
        lo = max(lo_box, s0 - eps)
        row = np.zeros(n)
        row[idx0] = 1.0
        A_ub.append(row)
        b_ub.append(hi)
        row = np.zeros(n)
        row[idx0] = -1.0
        A_ub.append(row)
        b_ub.append(-lo)

    res = linprog(
        c,
        A_ub=np.asarray(A_ub),
        b_ub=np.asarray(b_ub),
        A_eq=np.asarray(A_eq),
        b_eq=np.asarray(b_eq),
        bounds=bounds,
        method="highs",
        options={"presolve": True, "time_limit": float(cfg.time_limit_s)},
    )

    out: dict[str, Any] = {
        "success": bool(res.success),
        "message": str(res.message),
        "horizon": H,
        "fuel_yuan_per_mwh": cfg.fuel_yuan_per_mwh,
        "terminal_eps": eps,
        "residual_mean_mw": float(np.mean(residual)),
    }
    if not res.success or res.x is None:
        out["j_surr_ub"] = None
        out["obj_cost"] = None
        return out

    x = res.x
    cost = float(res.fun)  # fuel + buy - sell
    j_var = -cost
    load_rev = float(exo.get("load_revenue_fixed") or 0.0)
    re_om = float(exo.get("wind_om_if_full") or 0.0) + float(exo.get("pv_om_if_full") or 0.0)
    # Recalculate RE om from series * cfg rates (consistent with this solve)
    re_om = -cfg.wind_om_yuan_per_mwh * float(wind.sum()) - cfg.pv_om_yuan_per_mwh * float(pv.sum())
    j_ub = load_rev + re_om + j_var

    p_tp = np.array([x[h * n_u + 0] for h in range(H)])
    p_bch = np.array([x[h * n_u + 1] for h in range(H)])
    p_bdis = np.array([x[h * n_u + 2] for h in range(H)])
    p_gch = np.array([x[h * n_u + 3] for h in range(H)])
    p_gdis = np.array([x[h * n_u + 4] for h in range(H)])
    p_buy = np.array([x[h * n_u + 5] for h in range(H)])
    p_sell = np.array([x[h * n_u + 6] for h in range(H)])
    out.update(
        {
            "j_surr_ub": j_ub,
            "j_var_grid_fuel": j_var,
            "load_revenue_fixed": load_rev,
            "renewable_om_fixed": re_om,
            "obj_cost": cost,
            "thermal_mwh": float(p_tp.sum()),
            "bat_charge_mwh": float(p_bch.sum()),
            "bat_discharge_mwh": float(p_bdis.sum()),
            "gas_charge_mwh": float(p_gch.sum()),
            "gas_discharge_mwh": float(p_gdis.sum()),
            "buy_mwh": float(p_buy.sum()),
            "sell_mwh": float(p_sell.sum()),
            "soc_bat_T": float(x[i_sb0 + H]),
            "soc_gas_T": float(x[i_sg0 + H]),
        }
    )
    return out


def proxy_j_from_dispatch_csv(
    csv_path: Path | str,
    exo: dict[str, np.ndarray | float],
    *,
    cfg: WeeklyUBConfig | None = None,
) -> dict[str, Any]:
    """Evaluate the *same* linear cash-flow proxy on a closed-loop trajectory.

    Uses trajectory thermal generation and signed grid exchange; renewables
    charged at full available O&M (must-take); load revenue from ``exo``.
    This makes \(J_{\mathrm{surr}}^\star - J_{\mathrm{surr}}(\pi)\) a true gap
    on one model (not a claim about FMU global optimality).
    """
    import pandas as pd

    cfg = cfg or WeeklyUBConfig()
    df = pd.read_csv(csv_path)
    H = int(min(cfg.horizon, len(df), int(exo["n"])))
    df = df.iloc[:H]
    buy_mwh = np.asarray(exo["buy_yuan_per_kwh"], dtype=np.float64)[:H] * 1000.0
    sell_mwh = np.asarray(exo["sell_yuan_per_kwh"], dtype=np.float64)[:H] * 1000.0

    th_gen = _gen_mw(df["obs_p_thermal"].to_numpy(dtype=np.float64))
    # grid: positive W in traj ≈ export in FMU algebraic convention for this twin;
    # market columns are authoritative when present.
    if "rt_market_energy_buy_mwh" in df.columns and "rt_market_energy_sell_mwh" in df.columns:
        # per-step MWh (not cumulative): p_grid>0 buy, p_grid<0 sell
        buy_e = np.clip(df["rt_market_energy_buy_mwh"].to_numpy(dtype=np.float64), 0.0, None)
        sell_e = np.clip(df["rt_market_energy_sell_mwh"].to_numpy(dtype=np.float64), 0.0, None)
    else:
        g = df["obs_p_grid"].to_numpy(dtype=np.float64) * 1e-6
        buy_e = np.clip(g, 0.0, None)
        sell_e = np.clip(-g, 0.0, None)

    fuel = float(cfg.fuel_yuan_per_mwh * th_gen.sum())
    grid_cf = float(np.sum(sell_mwh * sell_e - buy_mwh * buy_e))
    load_rev = float(exo.get("load_revenue_fixed") or 0.0)
    wind = np.asarray(exo["wind_mw"], dtype=np.float64)[:H]
    pv = np.asarray(exo["pv_mw"], dtype=np.float64)[:H]
    re_om = -cfg.wind_om_yuan_per_mwh * float(wind.sum()) - cfg.pv_om_yuan_per_mwh * float(pv.sum())
    j_surr = load_rev + re_om + grid_cf - fuel
    j_fmu = float(df["rt_economic_cashflow_delta"].sum())
    return {
        "j_surr": j_surr,
        "j_fmu": j_fmu,
        "fuel_cost": fuel,
        "grid_cashflow_proxy": grid_cf,
        "load_revenue_fixed": load_rev,
        "renewable_om_fixed": re_om,
        "thermal_mwh": float(th_gen.sum()),
        "buy_mwh": float(buy_e.sum()),
        "sell_mwh": float(sell_e.sum()),
    }


def try_weekly_milp_modes(
    exo: dict[str, np.ndarray | float],
    *,
    cfg: WeeklyUBConfig | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Optional CAES mode-exclusive MILP (PuLP/CBC)."""
    try:
        import pulp  # type: ignore
    except Exception as exc:  # pragma: no cover
        return {"success": False, "message": f"pulp unavailable: {exc}", "j_surr_milp": None}

    cfg = cfg or WeeklyUBConfig()
    params = params or _load_device_params()
    H = int(min(cfg.horizon, int(exo["n"])))
    bat = params["battery"]
    th = params["thermal"]
    caes = params["caes"]
    grid = params["grid"]

    E_bat_mwh = float(bat["E_cap_J"]) / 3.6e9
    eta = float(bat["eta"])
    p_bat_max = float(bat["P_cap_W"]) * 1e-6
    p_tp_min = float(th["P_min_W"]) * 1e-6
    p_tp_max = float(th["P_max_W"]) * 1e-6
    p_caes_max = float(caes["P_cap_W"]) * 1e-6
    E_gas_mwh = float(cfg.E_gas_mwh_proxy)
    p_buy_max = float(grid["P_max_buy_W"]) * 1e-6
    p_sell_max = abs(float(grid["P_max_sell_W"])) * 1e-6

    load = np.asarray(exo["load_mw"], dtype=np.float64)[:H]
    wind = np.asarray(exo["wind_mw"], dtype=np.float64)[:H]
    pv = np.asarray(exo["pv_mw"], dtype=np.float64)[:H]
    residual = load - wind - pv
    buy_mwh = np.asarray(exo["buy_yuan_per_kwh"], dtype=np.float64)[:H] * 1000.0
    sell_mwh = np.asarray(exo["sell_yuan_per_kwh"], dtype=np.float64)[:H] * 1000.0
    soc_b0 = float(exo["soc_bat0"])
    soc_g0 = float(exo["soc_gas0"])
    eps = float(cfg.terminal_eps)
    load_rev = float(exo.get("load_revenue_fixed") or 0.0)
    re_om = -cfg.wind_om_yuan_per_mwh * float(wind.sum()) - cfg.pv_om_yuan_per_mwh * float(pv.sum())

    m = pulp.LpProblem("weekly_caes_mode_milp", pulp.LpMaximize)
    p_tp = [pulp.LpVariable(f"tp_{h}", p_tp_min, p_tp_max) for h in range(H)]
    p_bch = [pulp.LpVariable(f"bch_{h}", 0, p_bat_max) for h in range(H)]
    p_bdis = [pulp.LpVariable(f"bdis_{h}", 0, p_bat_max) for h in range(H)]
    p_gch = [pulp.LpVariable(f"gch_{h}", 0, p_caes_max) for h in range(H)]
    p_gdis = [pulp.LpVariable(f"gdis_{h}", 0, p_caes_max) for h in range(H)]
    p_buy = [pulp.LpVariable(f"buy_{h}", 0, p_buy_max) for h in range(H)]
    p_sell = [pulp.LpVariable(f"sell_{h}", 0, p_sell_max) for h in range(H)]
    z_c = [pulp.LpVariable(f"zc_{h}", cat="Binary") for h in range(H)]
    z_i = [pulp.LpVariable(f"zi_{h}", cat="Binary") for h in range(H)]
    z_d = [pulp.LpVariable(f"zd_{h}", cat="Binary") for h in range(H)]
    s_b = [pulp.LpVariable(f"sb_{h}", float(bat["SOC_min"]), float(bat["SOC_max"])) for h in range(H + 1)]
    s_g = [
        pulp.LpVariable(f"sg_{h}", float(caes["gas_SOC_min"]), float(caes["gas_SOC_max"]))
        for h in range(H + 1)
    ]

    m += pulp.lpSum(
        sell_mwh[h] * p_sell[h] - buy_mwh[h] * p_buy[h] - cfg.fuel_yuan_per_mwh * p_tp[h]
        for h in range(H)
    )
    m += s_b[0] == soc_b0
    m += s_g[0] == soc_g0
    for h in range(H):
        m += z_c[h] + z_i[h] + z_d[h] == 1
        m += p_gch[h] <= p_caes_max * z_c[h]
        m += p_gdis[h] <= p_caes_max * z_d[h]
        m += p_tp[h] + p_bdis[h] - p_bch[h] + p_gdis[h] - p_gch[h] - p_buy[h] + p_sell[h] == residual[h]
        m += s_b[h + 1] == s_b[h] + eta * p_bch[h] / E_bat_mwh - p_bdis[h] / (eta * E_bat_mwh)
        m += s_g[h + 1] == s_g[h] + p_gch[h] / E_gas_mwh - p_gdis[h] / E_gas_mwh
    m += s_b[H] <= soc_b0 + eps
    m += s_b[H] >= soc_b0 - eps
    m += s_g[H] <= soc_g0 + eps
    m += s_g[H] >= soc_g0 - eps

    status = m.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=int(cfg.time_limit_s)))
    ok = pulp.LpStatus[status] == "Optimal"
    j_var = float(pulp.value(m.objective)) if ok else None
    j = (load_rev + re_om + j_var) if j_var is not None else None
    return {
        "success": ok,
        "message": pulp.LpStatus.get(status, str(status)),
        "j_surr_milp": j,
        "j_var": j_var,
        "horizon": H,
    }
