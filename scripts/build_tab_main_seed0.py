#!/usr/bin/env python
"""Build docs/tab_main_seed0.md/.json from seasonal_v1 (GHTD3-style cost breakdown).

Economics (tab:main): valid_steps == 168 only; cost components aligned with
GHTD3 Table 4 (CC / C_ET / C_ops / C_CUT / C_CO2 / C_DEG).
Executability (tab:run): all methods (hours / eval_failed) — auxiliary, not main KPI.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "runs" / "seasonal_v1"
SEASONS = ("winter", "transition", "summer")
# Main matrix methods. sac_param / td3_param preferred when present.
METHODS = (
    "sac_param",
    "td3_param",
    "sac",
    "td3",
    "pso",
    "linprog",
    "milp",
)
FULL_WEEK = 168
METHOD_LABEL = {
    "sac_param": "hybrid SAC",
    "td3_param": "hybrid TD3",
    "sac": "proj. SAC",
    "td3": "proj. TD3",
    "pso": "pso",
    "linprog": "linprog",
    "milp": "milp",
}


def _run_dir(season: str, method: str) -> Path:
    return V1 / season / f"{method}_s0"


def _load_row(season: str, method: str) -> dict:
    p = _run_dir(season, method) / "train_result.json"
    out = {
        "season": season,
        "method": method,
        "label": METHOD_LABEL.get(method, method),
        "status": "missing",
        "R": None,
        "Jgen": None,
        "CC": None,
        "C_ET": None,
        "C_ops": None,
        "C_CUT": None,
        "C_CO2": None,
        "C_DEG": None,
        "eth_mwh": None,
        "bat_mwh": None,
        "caes_mwh": None,
        "steps": None,
        "obs": None,
        "full_week": False,
        "note": "",
    }
    if not p.is_file():
        return out
    j = json.loads(p.read_text(encoding="utf-8"))
    out["status"] = str(
        j.get("status") or ("completed" if j.get("episode_reward") is not None else "unknown")
    )
    out["obs"] = j.get("observation_dim")
    if out["status"] == "eval_failed":
        out["note"] = j.get("failure_type") or "eval_failed"
        return out
    ev = j.get("eval") or {}
    kpi = j.get("kpi") or {}
    terms = ev.get("cost_terms") or {}
    metrics = ev.get("metrics") or {}
    r = ev.get("episode_reward")
    if r is None:
        r = j.get("episode_reward")
    if r is None:
        r = kpi.get("episode_reward")
    jgen = terms.get("generalized_cashflow_delta")
    if jgen is None:
        jgen = j.get("sum_delta_j_gen")
    out["R"] = float(r) if r is not None else None
    out["Jgen"] = float(jgen) if jgen is not None else None
    # Cost framing (lower better): CC = -Jgen when Jgen present
    if out["Jgen"] is not None:
        out["CC"] = -float(out["Jgen"])
    elif terms.get("raw_generalized_cost") is not None:
        out["CC"] = float(terms["raw_generalized_cost"])
    # Electricity trading cost (buy − sell revenue)
    if terms.get("market_grid_cost") is not None:
        out["C_ET"] = float(terms["market_grid_cost"])
    # Operating / external (fuel-dominated proxy used by reward)
    if terms.get("external_cost_cny") is not None:
        out["C_ops"] = float(terms["external_cost_cny"])
    cut = terms.get("cut_total_cost_cny")
    if cut is None:
        cut = (terms.get("curtailment_cost_cny") or 0.0) + (terms.get("unserved_cost_cny") or 0.0)
    out["C_CUT"] = float(cut) if cut is not None else None
    if terms.get("carbon_cost_cny") is not None:
        out["C_CO2"] = float(terms["carbon_cost_cny"])
    if terms.get("battery_deg_cost_cny") is not None:
        out["C_DEG"] = float(terms["battery_deg_cost_cny"])
    eth = metrics.get("thermal_generation_mwh", j.get("thermal_mwh"))
    bat = metrics.get("battery_throughput_mwh", j.get("battery_throughput_mwh"))
    caes = metrics.get("caes_throughput_mwh", j.get("caes_throughput_mwh"))
    out["eth_mwh"] = float(eth) if eth is not None else None
    out["bat_mwh"] = float(bat) if bat is not None else None
    out["caes_mwh"] = float(caes) if caes is not None else None
    out["steps"] = ev.get("valid_steps") or j.get("valid_steps") or j.get("fmu_steps")
    try:
        out["full_week"] = out["steps"] is not None and int(out["steps"]) >= FULL_WEEK
    except (TypeError, ValueError):
        out["full_week"] = False
    if out["steps"] is not None and not out["full_week"]:
        out["note"] = f"incomplete {out['steps']}/{FULL_WEEK} h"
    return out


def _fmt(x, spec=".1f"):
    if x is None:
        return "—"
    return format(x, spec)


def main() -> None:
    rows = [_load_row(s, m) for s in SEASONS for m in METHODS]
    (ROOT / "docs" / "tab_main_seed0.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    econ = [r for r in rows if r["full_week"] and r["CC"] is not None]
    lines = [
        "# seasonal_v1 seed 0 — GHTD3-style cost tables (hybrid SAC identity)",
        "",
        "Source: `runs/seasonal_v1/**/train_result.json` and `docs/matrix_status_hybrid_sac.md`.",
        f"Only `valid_steps={FULL_WEEK}` rows enter `tab:main`. Truncated weeks are not ranked on cash.",
        "**Main method: hybrid SAC (`sac_param`).** No HMSD. `milp` = binary CAES + min-load on energy surrogate.",
        "",
        "Cost signs: **lower is better** for CC / C_ET / C_ops / C_CUT / C_CO2 / C_DEG "
        "(CC = −J^gen). Jgen kept for cross-check (higher better).",
        "",
        "## tab:main (full 168 h; cost breakdown)",
        "",
        "| season | method | CC (CNY) | C_ET | C_ops | C_CUT | C_CO2 | C_DEG | Jgen |",
        "|--------|--------|----------|------|-------|-------|-------|-------|------|",
    ]
    for r in econ:
        lines.append(
            f"| {r['season']} | {r['label']} | {_fmt(r['CC'], '.3e')} | "
            f"{_fmt(r['C_ET'], '.3e')} | {_fmt(r['C_ops'], '.3e')} | "
            f"{_fmt(r['C_CUT'], '.1f')} | {_fmt(r['C_CO2'], '.1f')} | "
            f"{_fmt(r['C_DEG'], '.1f')} | {_fmt(r['Jgen'], '.3e')} |"
        )
    lines += [
        "",
        "## tab:run (executability; auxiliary)",
        "",
        "| season | method | status | hours | full_week | note |",
        "|--------|--------|--------|-------|-----------|------|",
    ]
    for r in rows:
        h = r["steps"] if r["steps"] is not None else "—"
        lines.append(
            f"| {r['season']} | {r['label']} | {r['status']} | {h} | "
            f"{'Y' if r['full_week'] else 'N'} | {r['note']} |"
        )
    lines += [
        "",
        "## Forbidden claims",
        "",
        "- Do not rank truncated weeks against full weeks on cash.",
        "- Do not claim 8760 h RL safe + best economics.",
        "- Do not mix obs=163 archives with this matrix.",
        "- Do not report HMSD in the paper (body or appendix).",
        "- Do not claim unconditional RL > MILP; if MILP CC is lower, explain "
        "energy linearization / missing thermal coupling / twin non-executability.",
        "",
    ]
    out = ROOT / "docs" / "tab_main_seed0.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out, "econ_rows", len(econ), "all_rows", len(rows))


if __name__ == "__main__":
    main()
