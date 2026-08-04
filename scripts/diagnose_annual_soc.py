#!/usr/bin/env python
"""诊断 annual_eval 各周窗 terminal SOC 失败原因。"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


def _season(start_hour: int) -> str:
    day = start_hour / 24.0
    if day < 60 or day >= 335:
        return "winter"
    if 60 <= day < 152:
        return "spring"
    if 152 <= day < 244:
        return "summer"
    return "autumn"


def diagnose_run(run_dir: Path) -> dict:
    annual_dir = run_dir / "trajectories" / "annual_eval"
    if not annual_dir.is_dir():
        raise FileNotFoundError(f"无 annual_eval: {annual_dir}")

    rows = []
    for path in sorted(annual_dir.glob("window_*.csv")):
        m = re.search(r"window_(\d+)h", path.name)
        start_h = int(m.group(1)) if m else -1
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            last = None
            first = None
            for r in reader:
                if first is None:
                    first = r
                last = r
        if last is None:
            continue
        init = {
            "battery_soc": float(first.get("obs_battery_soc", 0.5)),
            "caes_gas_soc": float(first.get("obs_caes_gas_soc", 0.85)),
            "caes_hot_soc": float(first.get("obs_caes_hot_soc", 0.5)),
            "caes_cold_soc": float(first.get("obs_caes_cold_soc", 0.5)),
        }
        # first row is after first step; use config defaults for init when unavailable
        term = {
            "battery_soc": float(last.get("obs_battery_soc", 0.5)),
            "caes_gas_soc": float(last.get("obs_caes_gas_soc", 0.5)),
            "caes_hot_soc": float(last.get("obs_caes_hot_soc", 0.5)),
            "caes_cold_soc": float(last.get("obs_caes_cold_soc", 0.5)),
        }
        # Prefer reward-terms L1 if present
        # 轨迹中的 rt_* 可能是旧 full-L1 口径；诊断同时重算 energy / full
        nominal = {"battery_soc": 0.5, "caes_gas_soc": 0.85, "caes_hot_soc": 0.5, "caes_cold_soc": 0.5}
        weights = {"battery_soc": 1.5, "caes_gas_soc": 1.0, "caes_hot_soc": 0.35, "caes_cold_soc": 0.35}
        comps = {k: weights[k] * abs(term[k] - nominal[k]) for k in nominal}
        l1_full = float(sum(comps.values()))
        l1_energy = float(comps["battery_soc"] + comps["caes_gas_soc"])
        tol = 0.06
        ok_energy = l1_energy <= tol
        ok_full = l1_full <= tol
        main = max(comps, key=comps.get)
        rows.append(
            {
                "window": path.name,
                "start_hour": start_h,
                "season": _season(start_h),
                "soc_ok": ok_energy,
                "soc_ok_full": ok_full,
                "l1": l1_energy,
                "l1_full": l1_full,
                "l1_energy": l1_energy,
                "terminal": term,
                "component_weighted": comps,
                "main_component": main,
                "week_reward_sum": None,
            }
        )

    n = len(rows)
    n_ok = sum(1 for r in rows if r["soc_ok"])
    n_ok_full = sum(1 for r in rows if r["soc_ok_full"])
    fail = [r for r in rows if not r["soc_ok"]]
    by_season: dict[str, dict] = {}
    for r in rows:
        s = r["season"]
        by_season.setdefault(s, {"n": 0, "ok_energy": 0, "ok_full": 0, "l1_energy": [], "l1_full": []})
        by_season[s]["n"] += 1
        by_season[s]["ok_energy"] += int(r["soc_ok"])
        by_season[s]["ok_full"] += int(r["soc_ok_full"])
        by_season[s]["l1_energy"].append(r["l1_energy"])
        by_season[s]["l1_full"].append(r["l1_full"])
    for s, v in by_season.items():
        v["pass_rate_energy"] = v["ok_energy"] / max(v["n"], 1)
        v["pass_rate_full"] = v["ok_full"] / max(v["n"], 1)
        v["mean_l1_energy"] = float(np.mean(v["l1_energy"])) if v["l1_energy"] else None
        v["mean_l1_full"] = float(np.mean(v["l1_full"])) if v["l1_full"] else None
        del v["l1_energy"]
        del v["l1_full"]

    main_fail_counts: dict[str, int] = {}
    for r in fail:
        main_fail_counts[r["main_component"]] = main_fail_counts.get(r["main_component"], 0) + 1

    l1s = [r["l1_energy"] for r in rows]
    l1f = [r["l1_full"] for r in rows]
    report = {
        "run_dir": str(run_dir),
        "criterion": "energy_primary = battery + caes_gas (thermal tanks diagnostic only)",
        "n_windows": n,
        "n_pass_energy": n_ok,
        "n_pass_full": n_ok_full,
        "pass_rate_energy": n_ok / max(n, 1),
        "pass_rate_full": n_ok_full / max(n, 1),
        "n_pass": n_ok,
        "pass_rate": n_ok / max(n, 1),
        "l1_mean": float(np.mean(l1s)) if l1s else None,
        "l1_p50": float(np.median(l1s)) if l1s else None,
        "l1_p90": float(np.percentile(l1s, 90)) if l1s else None,
        "l1_full_mean": float(np.mean(l1f)) if l1f else None,
        "by_season": by_season,
        "fail_main_component_counts": main_fail_counts,
        "worst_10": sorted(rows, key=lambda x: -x["l1_energy"])[:10],
        "fail_windows": [
            {
                "start_hour": r["start_hour"],
                "season": r["season"],
                "l1_energy": r["l1_energy"],
                "l1_full": r["l1_full"],
                "main": r["main_component"],
                "terminal": r["terminal"],
            }
            for r in sorted(fail, key=lambda x: -x["l1_energy"])
        ],
    }
    out = run_dir / "annual_soc_diagnosis.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, required=True)
    args = p.parse_args()
    rep = diagnose_run(Path(args.run_dir))
    print(
        json.dumps(
            {
                "n_pass": f"{rep['n_pass']}/{rep['n_windows']}",
                "pass_rate": rep["pass_rate"],
                "l1_mean": rep["l1_mean"],
                "l1_p90": rep["l1_p90"],
                "by_season": rep["by_season"],
                "fail_main_component_counts": rep["fail_main_component_counts"],
                "worst_3": rep["worst_10"][:3],
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
