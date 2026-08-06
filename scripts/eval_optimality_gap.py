#!/usr/bin/env python
"""Surrogate weekly LP optimum vs same-proxy trajectory gaps + FMU J.

Primary diagnostic (honest):
  gap_surr% = (J_surr* - J_surr(method)) / |J_surr*|
on one linear cash-flow model (perfect foresight, continuous relaxation).

Secondary: report FMU closed-loop J (different model — not a true gap to J_surr*).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optimization.weekly_surrogate_ub import (  # noqa: E402
    WeeklyUBConfig,
    exogenous_from_dispatch_csv,
    proxy_j_from_dispatch_csv,
    solve_weekly_lp,
    try_weekly_milp_modes,
)

SEASONS = ("winter", "transition", "summer")
METHODS = ("b0", "linprog", "pso", "td3", "ghtd3")
METHOD_LABEL = {
    "b0": "B0 (rule)",
    "linprog": "linprog MPC (H=8)",
    "pso": "PSO",
    "td3": "TD3 (seed1)",
    "ghtd3": "HMSD (seed1)",
}


def gap_pct(j_star: float | None, j: float | None) -> float | None:
    if j_star is None or j is None or abs(j_star) < 1e-9:
        return None
    return float((j_star - j) / abs(j_star) * 100.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", type=str, default="runs/paper_dispatch_traj")
    ap.add_argument("--horizon", type=int, default=168)
    ap.add_argument("--milp", action="store_true")
    ap.add_argument("--out", type=str, default="runs/paper_optimality_gap.json")
    ap.add_argument("--sac-j-fmu", type=str, default="", help="optional season->FMU J JSON")
    args = ap.parse_args()

    traj_dir = ROOT / args.traj_dir
    cfg = WeeklyUBConfig(horizon=int(args.horizon))
    sac_fmu: dict[str, float] = {}
    if args.sac_j_fmu:
        sac_fmu = {k: float(v) for k, v in json.loads(args.sac_j_fmu).items()}

    payload: dict = {
        "note": (
            "J_surr_star: optimal linear cash-flow proxy (weekly LP, perfect foresight, "
            "continuous thermal/storage/grid, must-take RE, terminal SoC band). "
            "gap_surr compares methods on the SAME proxy (trajectory replay). "
            "J_fmu is closed-loop twin cash flow and is NOT bounded by J_surr_star."
        ),
        "horizon": cfg.horizon,
        "fuel_yuan_per_mwh": cfg.fuel_yuan_per_mwh,
        "terminal_eps": cfg.terminal_eps,
        "seasons": {},
    }

    for season in SEASONS:
        exo_path = traj_dir / f"{season}_b0.csv"
        if not exo_path.is_file():
            print(f"[skip] {exo_path}")
            continue
        exo = exogenous_from_dispatch_csv(exo_path, horizon=cfg.horizon)
        lp = solve_weekly_lp(exo, cfg=cfg)
        milp = (
            try_weekly_milp_modes(exo, cfg=cfg)
            if args.milp
            else {"success": False, "j_surr_milp": None}
        )
        j_star = lp.get("j_surr_ub")

        methods = {}
        for m in METHODS:
            path = traj_dir / f"{season}_{m}.csv"
            if not path.is_file():
                continue
            proxy = proxy_j_from_dispatch_csv(path, exo, cfg=cfg)
            methods[m] = {
                "label": METHOD_LABEL[m],
                "j_surr": proxy["j_surr"],
                "j_surr_1e6": proxy["j_surr"] / 1e6,
                "j_fmu": proxy["j_fmu"],
                "j_fmu_1e6": proxy["j_fmu"] / 1e6,
                "gap_surr_pct": gap_pct(j_star, proxy["j_surr"]),
                "thermal_mwh": proxy["thermal_mwh"],
                "buy_mwh": proxy["buy_mwh"],
                "sell_mwh": proxy["sell_mwh"],
            }
        if sac_fmu and season in sac_fmu:
            methods["sac"] = {
                "label": "SAC-Hybrid (80k, hist.)",
                "j_surr": None,
                "j_fmu": sac_fmu[season],
                "j_fmu_1e6": sac_fmu[season] / 1e6,
                "gap_surr_pct": None,
            }

        payload["seasons"][season] = {
            "j_surr_star": j_star,
            "j_surr_star_1e6": None if j_star is None else j_star / 1e6,
            "lp": {
                k: lp.get(k)
                for k in (
                    "success",
                    "message",
                    "thermal_mwh",
                    "bat_charge_mwh",
                    "bat_discharge_mwh",
                    "gas_charge_mwh",
                    "gas_discharge_mwh",
                    "buy_mwh",
                    "sell_mwh",
                    "j_var_grid_fuel",
                    "load_revenue_fixed",
                    "renewable_om_fixed",
                )
            },
            "milp": milp,
            "methods": methods,
        }
        g_h = methods.get("ghtd3", {}).get("gap_surr_pct")
        print(f"{season}: J*={j_star}  HMSD gap_surr%={g_h}  HMSD J_fmu={methods.get('ghtd3',{}).get('j_fmu')}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Surrogate weekly optimality gap (same-proxy)",
        "",
        payload["note"],
        "",
        f"- Horizon: {cfg.horizon} h | fuel: {cfg.fuel_yuan_per_mwh} CNY/MWh | terminal ε={cfg.terminal_eps}",
        "",
        "| Season | \(J^*_{surr}\) (10⁶) | B0 gap% | linprog gap% | PSO gap% | TD3 gap% | HMSD gap% | HMSD \(J_{surr}/J^*\) | HMSD \(J_{FMU}\) (10⁶) |",
        "|--------|---------------------:|--------:|-------------:|---------:|---------:|----------:|----------------------:|------------------------:|",
    ]
    for season in SEASONS:
        s = payload["seasons"].get(season)
        if not s:
            continue
        m = s["methods"]

        def g(key: str) -> str:
            v = m.get(key, {}).get("gap_surr_pct")
            return "—" if v is None else f"{v:.1f}"

        ju = s["j_surr_star_1e6"]
        ratio = None
        if m.get("ghtd3") and ju not in (None, 0) and m["ghtd3"].get("j_surr") is not None:
            ratio = m["ghtd3"]["j_surr"] / (ju * 1e6)
        jf = m.get("ghtd3", {}).get("j_fmu_1e6")
        lines.append(
            f"| {season} | {ju:.2f} | {g('b0')} | {g('linprog')} | {g('pso')} | {g('td3')} | {g('ghtd3')} | "
            f"{ratio if ratio is not None else float('nan'):.2f} | {jf if jf is not None else float('nan'):.2f} |"
        )
    md = out.with_suffix(".md")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", out)
    print(md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
