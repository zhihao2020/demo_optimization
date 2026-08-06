#!/usr/bin/env python
"""Offline GiveSafe / feasibility statistics from closed-loop trajectory CSVs.

Uses requested_* vs decoded_* columns logged by evaluate_policy as a proxy for
pre/post shield actions, plus validity / CAES lock fields.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEASONS = ("winter", "transition", "summer")
METHODS = ("ghtd3", "td3", "b0", "pso", "linprog")


def _num(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").fillna(0.0).to_numpy(dtype=float)


def stats_one(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    n = len(df)
    if n == 0:
        return {"steps": 0}

    req_tp = _num(df["requested_u_tp"]) if "requested_u_tp" in df else np.zeros(n)
    dec_tp = _num(df["decoded_u_tp"]) if "decoded_u_tp" in df else np.zeros(n)
    req_bat = _num(df["requested_u_battery"]) if "requested_u_battery" in df else np.zeros(n)
    dec_bat = _num(df["decoded_u_battery"]) if "decoded_u_battery" in df else np.zeros(n)
    req_mode = _num(df["requested_caes_mode"]) if "requested_caes_mode" in df else np.zeros(n)
    # decoded mode not always separate; use requested if missing
    req_mag = _num(df["requested_caes_magnitude"]) if "requested_caes_magnitude" in df else np.zeros(n)
    dec_mag = _num(df["decoded_u_caes"]) if "decoded_u_caes" in df else req_mag

    cont_diff = np.sqrt((req_tp - dec_tp) ** 2 + (req_bat - dec_bat) ** 2 + (req_mag - dec_mag) ** 2)
    cont_changed = cont_diff > 1e-4
    mode_changed = np.abs(req_mode - req_mode)  # placeholder
    # mode: compare requested mode to mask-implied execution via lock / allow flags if present
    if "caes_locked_mode" in df.columns:
        locked = df["caes_locked_mode"].notna() & (df["caes_locked_mode"].astype(str) != "nan")
    else:
        locked = pd.Series(False, index=df.index)
    if "caes_locked_steps_remaining" in df.columns:
        lock_rem = _num(df["caes_locked_steps_remaining"]) > 0
    else:
        lock_rem = np.zeros(n, dtype=bool)

    valid = df["transition_valid"].astype(bool) if "transition_valid" in df else pd.Series(True, index=df.index)
    fail = (df["fmu_status"].astype(str) == "failure") if "fmu_status" in df else pd.Series(False, index=df.index)
    has_fail_type = df["failure_type"].notna() if "failure_type" in df else pd.Series(False, index=df.index)

    # projection proxy: continuous command changed by shield/decoder
    proj_rate = float(cont_changed.mean())
    return {
        "steps": int(n),
        "projection_rate_cont": proj_rate,
        "action_l2_mean": float(cont_diff.mean()),
        "action_l2_p50": float(np.median(cont_diff)),
        "action_l2_p95": float(np.percentile(cont_diff, 95)),
        "action_l2_max": float(cont_diff.max()),
        "frac_cont_changed": proj_rate,
        "invalid_transition_count": int((~valid).sum()),
        "invalid_transition_rate": float((~valid).mean()),
        "fmu_failure_count": int(fail.sum()),
        "failure_type_count": int(has_fail_type.sum()),
        "caes_lock_active_frac": float(np.mean(lock_rem.astype(float))),
        "caes_lock_active_steps": int(np.sum(lock_rem)),
        "caes_locked_mode_nonnull_frac": float(locked.mean()) if hasattr(locked, "mean") else 0.0,
        "episode_reward": float(pd.to_numeric(df["reward"], errors="coerce").fillna(0).sum())
        if "reward" in df
        else None,
        "terminal_soc_satisfied": bool(
            float(df["rt_terminal_soc_satisfied"].iloc[-1]) > 0.5
            if "rt_terminal_soc_satisfied" in df
            else False
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", type=str, default="runs/paper_dispatch_traj")
    ap.add_argument("--out", type=str, default="runs/paper_givesafe_stats.json")
    args = ap.parse_args()
    traj = ROOT / args.traj_dir
    out: dict = {"protocol": "offline_from_eval_csv", "traj_dir": str(traj), "by_method": {}}

    for method in METHODS:
        seasons = {}
        for season in SEASONS:
            p = traj / f"{season}_{method}.csv"
            if not p.is_file():
                continue
            seasons[season] = stats_one(p)
        if not seasons:
            continue
        # aggregate over seasons (equal weight)
        keys = [
            "projection_rate_cont",
            "action_l2_mean",
            "invalid_transition_rate",
            "fmu_failure_count",
            "caes_lock_active_frac",
        ]
        agg = {}
        for k in keys:
            vals = [seasons[s][k] for s in seasons if k in seasons[s]]
            agg[k + "_mean"] = float(np.mean(vals)) if vals else None
        agg["seasons"] = seasons
        out["by_method"][method] = agg
        print(
            f"{method:8s} proj={agg['projection_rate_cont_mean']:.3f} "
            f"l2={agg['action_l2_mean_mean']:.4f} "
            f"inv={agg['invalid_transition_rate_mean']:.4f} "
            f"fail={agg['fmu_failure_count_mean']:.2f} "
            f"lock={agg['caes_lock_active_frac_mean']:.3f}",
            flush=True,
        )

    out_path = ROOT / args.out
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    # markdown table
    lines = [
        "# GiveSafe / feasibility offline stats (from dispatch CSVs)",
        "",
        "Projection proxy: continuous command change between `requested_*` and `decoded_*` (shield/decoder).",
        "",
        "| Method | Proj. rate | Action L2 mean | Invalid trans. rate | FMU fail / week (mean) | CAES lock frac |",
        "|--------|------------|----------------|---------------------|------------------------|----------------|",
    ]
    for method, agg in out["by_method"].items():
        lines.append(
            f"| {method} | {agg['projection_rate_cont_mean']:.3f} | {agg['action_l2_mean_mean']:.4f} | "
            f"{agg['invalid_transition_rate_mean']:.4f} | {agg['fmu_failure_count_mean']:.2f} | "
            f"{agg['caes_lock_active_frac_mean']:.3f} |"
        )
    md = out_path.with_suffix(".md")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", out_path)
    print(md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
