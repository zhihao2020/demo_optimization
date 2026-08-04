#!/usr/bin/env python
"""评估 SAC + linprog 三季，并与既有 merged 表合并成加固对照表。"""
from __future__ import annotations

import json  # noqa: I001
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config.paths import apply_process_cache_env, resolve_run_dir  # noqa: E402

apply_process_cache_env()

from actions import CaesMode  # noqa: E402
from envs.power_system_env import PowerSystemEnv  # noqa: E402
from optimization.metrics import extract_kpi_from_eval, relative_to_baseline  # noqa: E402
from optimization.rolling_linprog import RollingLinprogController  # noqa: E402
from safety import GiveSafeController, NoSafeActionFoundError, load_givesafe_config  # noqa: E402
from training.evaluate_td3 import evaluate_policy  # noqa: E402
from training.hybrid_sac.algorithm import HybridSAC  # noqa: E402

SEASONS = {
    "winter": 0.0,
    "summer": 180 * 24 * 3600.0,
    "transition": 90 * 24 * 3600.0,
}


class SacPolicy:
    def __init__(self, agent, env, ctrl):
        self.agent, self.env, self.ctrl = agent, env, ctrl

    def predict(self, obs, deterministic=True):
        try:
            feas = self.env.get_feasible_action_spec()
        except Exception:
            return {
                "u_tp": np.asarray([1.0], np.float32),
                "u_battery": np.asarray([0.0], np.float32),
                "caes_mode": 1,
                "caes_magnitude": np.asarray([0.0], np.float32),
            }

        def prop():
            return self.agent.select_action(obs, feas, deterministic=deterministic)

        try:
            return self.ctrl.select_safe_action(
                self.env.last_outputs,
                self.env.previous_thermal,
                prop,
                deterministic=deterministic,
                feasible_override=feas,
            ).safe_action
        except NoSafeActionFoundError:
            return {
                "u_tp": np.asarray([float(feas.u_tp_high)], np.float32),
                "u_battery": np.asarray([0.0], np.float32),
                "caes_mode": int(CaesMode.IDLE),
                "caes_magnitude": np.asarray([0.0], np.float32),
            }


def fix_j(kpi, res):
    raw = res.get("weekly_raw_total_cost")
    if raw is not None:
        kpi["net_cashflow_j"] = -float(raw)
    return kpi


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Eval SAC + linprog three seasons into extended table")
    ap.add_argument(
        "--sac-ckpt",
        type=str,
        default="runs/givesafe_sac_15k_20260804/checkpoints/hybrid_givesafe_sac.pt",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="runs/benchmark_extended_linprog_sac_20260804",
    )
    ap.add_argument("--skip-linprog", action="store_true", help="仅重评 SAC（沿用既有 linprog 行时用）")
    args = ap.parse_args()

    out = resolve_run_dir(args.out_dir)
    gs_cfg = load_givesafe_config(ROOT / "src/config/givesafe_config.yaml")
    probe = PowerSystemEnv(run_id="probe", forecast_enabled=True)
    dim = int(np.prod(probe.observation_space.shape))
    probe.close()
    sac = HybridSAC(obs_dim=dim)
    ckpt = Path(args.sac_ckpt)
    if not ckpt.is_file():
        raise FileNotFoundError(f"SAC ckpt not found: {ckpt}")
    sac.load(ckpt)

    methods = ("sac",) if args.skip_linprog else ("linprog", "sac")
    rows = []
    for season, start in SEASONS.items():
        for method in methods:
            print(f"=== {season} {method} ===", flush=True)
            env = PowerSystemEnv(run_id=f"ext_{season}_{method}", forecast_enabled=True)
            try:
                if method == "linprog":
                    pol = RollingLinprogController(env)
                else:
                    ctrl = GiveSafeController(oracle=env.oracle, shadow=None, config=gs_cfg)
                    pol = SacPolicy(sac, env, ctrl)
                t0 = time.perf_counter()
                res = evaluate_policy(env, pol, reset_options={"start_time": start})
                wall = time.perf_counter() - t0
                kpi = extract_kpi_from_eval(res, wall_s=wall, fmu_steps=res.get("valid_steps"))
                kpi = fix_j(kpi, res)
                kpi["season"] = season
                kpi["method"] = method
                rows.append(kpi)
                print(
                    f"  J={kpi.get('net_cashflow_j')} rew={kpi.get('episode_reward')} "
                    f"soc={kpi.get('terminal_soc_satisfied')} th={kpi.get('thermal_mwh')}",
                    flush=True,
                )
            finally:
                env.close()

    # merge with previous main table if present
    merged_path = Path("runs/benchmark_merged_3season_pso_20260804/benchmark_merged.json")
    base_rows = []
    if merged_path.exists():
        base_rows = json.loads(merged_path.read_text(encoding="utf-8")).get("rows") or []

    # 若 skip-linprog：保留旧 extended 表中的 linprog 行
    if args.skip_linprog:
        prev_ext = out / "extended_table.json"
        if prev_ext.is_file():
            prev_rows = json.loads(prev_ext.read_text(encoding="utf-8")).get("rows") or []
            linprog_rows = [r for r in prev_rows if r.get("method") == "linprog"]
            rows = linprog_rows + rows

    # 新 SAC 行替换 base 中旧 sac（若有）
    all_rows = [r for r in base_rows if r.get("method") != "sac"] + rows
    # attach vs_b0
    b0 = {r["season"]: r for r in all_rows if r.get("method") == "b0"}
    for r in all_rows:
        if r.get("method") == "b0":
            continue
        if r.get("season") in b0 and r.get("net_cashflow_j") is not None:
            r["vs_b0"] = relative_to_baseline(r, b0[r["season"]])

    order = ["b0", "b1", "lp", "linprog", "pso", "sac", "hybrid", "ghtd3"]
    seasons = ["winter", "summer", "transition"]
    sorted_rows = []
    for s in seasons:
        for m in order:
            for r in all_rows:
                if r.get("season") == s and r.get("method") == m:
                    sorted_rows.append(r)

    sac_label = "M5 Hybrid-SAC"
    ckpt_s = str(ckpt).replace("\\", "/")
    if "80k" in ckpt_s:
        sac_label = "M5 Hybrid-SAC (80k)"
    elif "15k" in ckpt_s:
        sac_label = "M5 Hybrid-SAC (15k)"

    payload = {
        "rows": sorted_rows,
        "note": f"Extended with true linprog MPC and SAC-Hybrid ({ckpt_s}).",
        "sac_ckpt": str(ckpt),
    }
    (out / "extended_table.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    labels = {
        "b0": "B0 Rule (original)",
        "b1": "B1 Price rule",
        "lp": "M1 Heuristic rolling",
        "linprog": "M1b True linprog MPC",
        "pso": "M2 PSO parametric",
        "sac": sac_label,
        "hybrid": "M3 Hybrid-TD3",
        "ghtd3": "M4 Safe Market-GHTD3",
    }
    slab = {"winter": "Winter", "summer": "Summer", "transition": "Transition"}

    def sci(x):
        try:
            return f"{float(x):.3e}"
        except Exception:
            return "—"

    def f1(x):
        try:
            return f"{float(x):.1f}"
        except Exception:
            return "—"

    lines = [
        "# Extended baseline table: linprog + SAC-Hybrid",
        "",
        "| Season | Method | Net cash flow J | ΔJ vs B0 | Reward | SOC | Thermal MWh | Bat thr. | CAES thr. |",
        "|--------|--------|----------------:|---------:|-------:|:---:|------------:|---------:|----------:|",
    ]
    for r in sorted_rows:
        vs = r.get("vs_b0") or {}
        lines.append(
            "| {s} | {m} | {j} | {dj} | {rew} | {soc} | {th} | {bt} | {ct} |".format(
                s=slab.get(r.get("season"), r.get("season")),
                m=labels.get(r.get("method"), r.get("method")),
                j=sci(r.get("net_cashflow_j")),
                dj=sci(vs.get("delta_j_vs_b0")) if vs else "—",
                rew=f1(r.get("episode_reward")),
                soc="Y" if r.get("terminal_soc_satisfied") else "N",
                th=f1(r.get("thermal_mwh")).replace(".0", "") if r.get("thermal_mwh") is not None else "—",
                bt=f1(r.get("battery_throughput_mwh")).replace(".0", "") if r.get("battery_throughput_mwh") is not None else "—",
                ct=f1(r.get("caes_throughput_mwh")).replace(".0", "") if r.get("caes_throughput_mwh") is not None else "—",
            )
        )
    lines.append("")
    md = "\n".join(lines) + "\n"
    (out / "extended_table.md").write_text(md, encoding="utf-8")
    (ROOT / "docs" / "扩展基准_linprog_SAC.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
