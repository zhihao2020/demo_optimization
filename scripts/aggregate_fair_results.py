#!/usr/bin/env python
"""Aggregate seasonal_v1 train_result.json into paper tables (local or copied runs).

Usage:
  python scripts/aggregate_fair_results.py --root runs/seasonal_v1
  python scripts/aggregate_fair_results.py --root D:/path/to/remote_copy/seasonal_v1 --out docs/ae_results_table.md
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def extract_row(path: Path) -> dict | None:
    try:
        j = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    # path patterns:
    #   .../seasonal_v1/{season}/{method}_s{seed}/train_result.json
    #   .../ablation/{tag}_winter_s0/train_result.json
    parts = path.resolve().parts
    season, tag = "?", path.parent.name
    try:
        if "seasonal_v1" in parts:
            i = parts.index("seasonal_v1")
            season = parts[i + 1]
            tag = parts[i + 2]  # hmsd_s0
        elif "ablation" in parts:
            i = parts.index("ablation")
            tag = parts[i + 1]  # ablation_no_her_winter_s0
            # parse ..._winter_s0 / ..._transition_s1
            for sname in ("winter", "transition", "summer"):
                if f"_{sname}_s" in tag or tag.endswith(f"_{sname}_s0"):
                    season = sname
                    break
            else:
                season = "ablation"
    except (ValueError, IndexError):
        season, tag = "?", path.parent.name
    method, _, seed = tag.partition("_s")
    if not seed:
        # tag like ablation_no_her_winter_s0
        if "_s" in tag:
            method, seed = tag.rsplit("_s", 1)
        else:
            method, seed = tag, "?"

    eval_ = j.get("eval") or {}
    kpi = j.get("kpi") or {}
    terms = eval_.get("cost_terms") or {}
    metrics = eval_.get("metrics") or {}

    r = (
        eval_.get("episode_reward")
        if eval_.get("episode_reward") is not None
        else j.get("episode_reward")
        if j.get("episode_reward") is not None
        else kpi.get("episode_reward")
    )
    jgen = (
        terms.get("generalized_cashflow_delta")
        if terms.get("generalized_cashflow_delta") is not None
        else kpi.get("sum_delta_j_gen")
        if kpi.get("sum_delta_j_gen") is not None
        else j.get("sum_delta_j_gen")
    )
    cf = (
        terms.get("economic_cashflow_delta")
        or terms.get("cashflow_delta")
        or kpi.get("sum_delta_cf")
        or j.get("net_cashflow_j")
    )
    soc = eval_.get("terminal_soc_satisfied")
    if soc is None:
        soc = kpi.get("terminal_soc_satisfied")
    if soc is None:
        soc = j.get("terminal_soc_satisfied")
    uns = metrics.get("unserved_energy_mwh")
    if uns is None:
        uns = kpi.get("unserved_energy_mwh")
    if uns is None:
        uns = j.get("unserved_mwh")

    safety = j.get("safety_learning") or {}
    return {
        "season": season,
        "method": method,
        "seed": seed,
        "status": j.get("status", "unknown"),
        "R": float(r) if r is not None else None,
        "Jgen": float(jgen) if jgen is not None else None,
        "CF": float(cf) if cf is not None else None,
        "SOC_ok": bool(soc) if soc is not None else None,
        "unserved": float(uns) if uns is not None else None,
        "reject_rate": safety.get("reject_rate"),
        "path": str(path),
    }


def mean_std(vals: list[float]) -> tuple[float | None, float | None]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="runs/seasonal_v1")
    ap.add_argument("--out", type=str, default="docs/ae_results_table.md")
    args = ap.parse_args()
    root = Path(args.root)
    rows = []
    for p in sorted(root.rglob("train_result.json")):
        r = extract_row(p)
        if r:
            rows.append(r)

    # group method x season
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["season"], r["method"])].append(r)

    lines = [
        "# Fair seasonal results aggregate",
        "",
        f"Source root: `{root}`",
        f"n_results: {len(rows)}",
        "",
        "## Per seed",
        "",
        "| season | method | seed | R | Jgen | CF | SOC_ok | unserved | reject_rate |",
        "|--------|--------|------|---|------|----|--------|----------|-------------|",
    ]
    for r in sorted(rows, key=lambda x: (x["season"], x["method"], x["seed"])):
        lines.append(
            "| {season} | {method} | {seed} | {R} | {Jgen} | {CF} | {SOC_ok} | {unserved} | {reject_rate} |".format(
                season=r["season"],
                method=r["method"],
                seed=r["seed"],
                R=f"{r['R']:.3f}" if r["R"] is not None else "—",
                Jgen=f"{r['Jgen']:.3e}" if r["Jgen"] is not None else "—",
                CF=f"{r['CF']:.3e}" if r["CF"] is not None else "—",
                SOC_ok=r["SOC_ok"] if r["SOC_ok"] is not None else "—",
                unserved=f"{r['unserved']:.3f}" if r["unserved"] is not None else "—",
                reject_rate=f"{r['reject_rate']:.3f}" if r["reject_rate"] is not None else "—",
            )
        )

    lines += ["", "## Mean ± std (by season × method)", ""]
    lines += [
        "| season | method | n | R mean±std | Jgen mean±std | SOC_ok rate |",
        "|--------|--------|---|------------|---------------|-------------|",
    ]
    for (season, method), rs in sorted(groups.items()):
        rm, rs_ = mean_std([x["R"] for x in rs if x["R"] is not None])
        jm, js = mean_std([x["Jgen"] for x in rs if x["Jgen"] is not None])
        socs = [x["SOC_ok"] for x in rs if x["SOC_ok"] is not None]
        soc_rate = (sum(1 for s in socs if s) / len(socs)) if socs else None
        r_s = f"{rm:.2f}±{rs_:.2f}" if rm is not None else "—"
        j_s = f"{jm:.3e}±{js:.3e}" if jm is not None else "—"
        s_s = f"{soc_rate:.0%}" if soc_rate is not None else "—"
        lines.append(f"| {season} | {method} | {len(rs)} | {r_s} | {j_s} | {s_s} |")

    # simple win rate HMSD vs TD3 per season on R
    lines += ["", "## HMSD vs TD3 (mean R)", ""]
    for season in ("winter", "transition", "summer"):
        h = groups.get((season, "hmsd"), [])
        t = groups.get((season, "td3"), [])
        hm, _ = mean_std([x["R"] for x in h if x["R"] is not None])
        tm, _ = mean_std([x["R"] for x in t if x["R"] is not None])
        if hm is None or tm is None:
            lines.append(f"- {season}: insufficient data")
        else:
            flag = "HMSD higher" if hm > tm else "TD3 higher or equal"
            lines.append(f"- {season}: HMSD={hm:.2f}, TD3={tm:.2f} → **{flag}**")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
