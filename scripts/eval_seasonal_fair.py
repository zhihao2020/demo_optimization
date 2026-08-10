#!/usr/bin/env python
"""Evaluate a seasonal checkpoint on explicit week starts.

Primary KPI: sum generalized_cashflow_delta (ΔJ_gen). Also unserved / terminal SOC.

Examples:
  python scripts/eval_seasonal_fair.py --method hmsd --ckpt path/to/ghtd3.pt --weeks 5,6
  python scripts/eval_seasonal_fair.py --method td3 --ckpt path/to/hybrid_givesafe_td3.pt --weeks 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config.paths import apply_process_cache_env  # noqa: E402

apply_process_cache_env()

from envs.power_system_env import PowerSystemEnv  # noqa: E402
from safety import GiveSafeController, load_givesafe_config  # noqa: E402
from training.evaluate_td3 import evaluate_policy  # noqa: E402
from training.ghtd3.agent import GHTD3Agent  # noqa: E402
from training.ghtd3.train import GHTD3PolicyWrapper, load_ghtd3_config  # noqa: E402
from training.hybrid_td3.algorithm import HybridTD3  # noqa: E402
from training.hybrid_td3.train import HybridPolicyWrapper, annual_episode_start_seconds  # noqa: E402


def week_start(week: int) -> float:
    env = PowerSystemEnv(run_id="eval_meta", forecast_enabled=True)
    try:
        return float(annual_episode_start_seconds(env.config["fmu"], env.episode_steps, week))
    finally:
        env.close()


def kpi(ev: dict) -> dict:
    terms = ev.get("cost_terms") or {}
    metrics = ev.get("metrics") or {}
    return {
        "steps": ev.get("steps"),
        "sum_delta_j_gen": terms.get("generalized_cashflow_delta"),
        "sum_delta_cf": terms.get("economic_cashflow_delta") or terms.get("cashflow_delta"),
        "episode_reward": ev.get("episode_reward"),
        "unserved_energy_mwh": metrics.get("unserved_energy_mwh"),
        "curtailment_energy_mwh": metrics.get("curtailment_energy_mwh"),
        "battery_throughput_mwh": metrics.get("battery_throughput_mwh"),
        "caes_throughput_mwh": metrics.get("caes_throughput_mwh"),
        "thermal_generation_mwh": metrics.get("thermal_generation_mwh"),
        "terminal_soc_satisfied": ev.get("terminal_soc_satisfied"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["hmsd", "td3"], required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--weeks", type=str, required=True)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    weeks = [int(x.strip()) for x in args.weeks.split(",") if x.strip()]
    ckpt = Path(args.ckpt)
    gs = load_givesafe_config(ROOT / "src/config/givesafe_config.yaml")
    rows = []

    for w in weeks:
        start = week_start(w)
        env = PowerSystemEnv(run_id=f"fair_eval_w{w}", forecast_enabled=True)
        try:
            obs_dim = int(env.observation_space.shape[0])
            ctrl = GiveSafeController(oracle=env.oracle, shadow=None, config=gs)
            if args.method == "hmsd":
                full = load_ghtd3_config(args.config)
                cfg = dict(full.get("ghtd3") or full)
                agent = GHTD3Agent(obs_dim, cfg)
                agent.load(ckpt, strict=False)
                policy = GHTD3PolicyWrapper(agent, env, ctrl, cfg)
            else:
                agent = HybridTD3(obs_dim)
                agent.load(ckpt)
                policy = HybridPolicyWrapper(agent, env, ctrl, deterministic=True)
            ev = evaluate_policy(env, policy, reset_options={"start_time": start})
            row = {"week": w, "start_time_seconds": start, **kpi(ev)}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        finally:
            env.close()

    jvals = [float(r["sum_delta_j_gen"]) for r in rows if r.get("sum_delta_j_gen") is not None]
    out = {
        "method": args.method,
        "ckpt": str(ckpt),
        "weeks": rows,
        "mean_sum_delta_j_gen": (sum(jvals) / len(jvals)) if jvals else None,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
