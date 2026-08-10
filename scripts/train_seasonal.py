#!/usr/bin/env python
"""Unified seasonal suite: HMSD / TD3 / SAC (RL) + PSO / linprog (open-loop baselines).

Same env reward and explicit week protocol for all methods.

RL:
  multi-week train + held-out eval week (default)

PSO / linprog:
  optimize or control on eval week (and PSO may search on that week); same KPI dump

Examples:
  python scripts/train_seasonal.py --method hmsd --season winter --episodes 5000 --seed 0
  python scripts/train_seasonal.py --method sac --season winter --episodes 5000 --seed 0
  python scripts/train_seasonal.py --method pso --season winter --seed 0
  python scripts/train_seasonal.py --method linprog --season winter --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config.paths import apply_process_cache_env, resolve_run_dir  # noqa: E402

apply_process_cache_env()

from envs.power_system_env import PowerSystemEnv  # noqa: E402
from training.evaluate_td3 import evaluate_policy  # noqa: E402
from training.ghtd3.train import run_ghtd3_training  # noqa: E402
from training.hybrid_sac.train import run_hybrid_sac_training  # noqa: E402
from training.hybrid_td3.train import annual_episode_start_seconds, run_td3_scratch  # noqa: E402

SEASON_WEEKS = {
    "winter": {"train": [0, 1, 2, 3, 4], "eval": 5},
    "transition": {"train": [13, 14, 15, 16, 17], "eval": 18},
    "summer": {"train": [26, 27, 28, 29, 30], "eval": 31},
}
EPISODE_HOURS = 168
RL_METHODS = ("hmsd", "td3", "sac")
ALL_METHODS = ("hmsd", "td3", "sac", "pso", "linprog")


def week_start_seconds(week_index: int) -> float:
    env = PowerSystemEnv(run_id="season_meta", forecast_enabled=True)
    try:
        return float(annual_episode_start_seconds(env.config["fmu"], env.episode_steps, week_index))
    finally:
        env.close()


def parse_weeks(raw: str | None, default: list[int]) -> list[int]:
    if not raw:
        return list(default)
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def kpi_from_eval(ev: dict) -> dict:
    terms = ev.get("cost_terms") or {}
    metrics = ev.get("metrics") or {}
    return {
        "sum_delta_j_gen": terms.get("generalized_cashflow_delta"),
        "sum_delta_cf": terms.get("economic_cashflow_delta") or terms.get("cashflow_delta"),
        "episode_reward": ev.get("episode_reward"),
        "unserved_energy_mwh": metrics.get("unserved_energy_mwh"),
        "terminal_soc_satisfied": ev.get("terminal_soc_satisfied"),
        "battery_throughput_mwh": metrics.get("battery_throughput_mwh"),
        "caes_throughput_mwh": metrics.get("caes_throughput_mwh"),
        "thermal_generation_mwh": metrics.get("thermal_generation_mwh"),
    }


def run_pso_job(run_dir: Path, eval_start: float, seed: int, pso_iters: int, pso_particles: int) -> dict:
    from optimization.pso_fmu import PSOConfig, run_pso

    cfg = PSOConfig(n_particles=pso_particles, n_iters=pso_iters, seed=seed)
    t0 = time.perf_counter()
    out = run_pso(start_time=eval_start, cfg=cfg)
    out["wall_s"] = time.perf_counter() - t0
    out["status"] = "completed"
    out["method"] = "pso"
    (run_dir / "train_result.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return out


def run_linprog_job(run_dir: Path, eval_start: float) -> dict:
    from optimization.rolling_linprog import RollingLinprogController

    env = PowerSystemEnv(run_id=run_dir.name, forecast_enabled=True)
    try:
        pol = RollingLinprogController(env)
        t0 = time.perf_counter()
        ev = evaluate_policy(
            env,
            pol,
            run_dir / "trajectories" / "eval.csv",
            reset_options={"start_time": eval_start},
        )
        wall = time.perf_counter() - t0
    finally:
        env.close()
    out = {
        "status": "completed",
        "method": "linprog",
        "eval": ev,
        "kpi": kpi_from_eval(ev),
        "eval_start_time_seconds": eval_start,
        "wall_s": wall,
    }
    (run_dir / "train_result.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Fair seasonal suite (RL + baselines)")
    p.add_argument("--method", choices=list(ALL_METHODS), required=True)
    p.add_argument("--season", choices=list(SEASON_WEEKS.keys()), required=True)
    p.add_argument("--episodes", type=int, default=5000, help="RL E_max; steps=E*168 (ignored for pso/linprog)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--config", type=str, default=str(ROOT / "src/config/ghtd3_config.yaml"))
    p.add_argument("--train-weeks", type=str, default=None)
    p.add_argument("--eval-week", type=int, default=None)
    p.add_argument("--single-week", action="store_true")
    p.add_argument("--pso-iters", type=int, default=25)
    p.add_argument("--pso-particles", type=int, default=12)
    args = p.parse_args()

    meta = SEASON_WEEKS[args.season]
    if args.single_week:
        train_weeks = [meta["train"][0]]
        eval_week = train_weeks[0]
    else:
        train_weeks = parse_weeks(args.train_weeks, meta["train"])
        eval_week = int(args.eval_week) if args.eval_week is not None else int(meta["eval"])

    train_starts = [week_start_seconds(w) for w in train_weeks]
    eval_start = week_start_seconds(eval_week)

    steps = int(args.episodes) * EPISODE_HOURS if args.method in RL_METHODS else 0
    run_dir = args.run_dir or f"runs/seasonal/{args.season}/{args.method}_s{args.seed}"
    run_dir = str(resolve_run_dir(run_dir))
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "trajectories").mkdir(parents=True, exist_ok=True)

    protocol = {
        "protocol": "fair_seasonal_v1",
        "method": args.method,
        "season": args.season,
        "train_weeks": train_weeks,
        "eval_week": eval_week,
        "train_start_seconds": train_starts,
        "eval_start_seconds": eval_start,
        "single_week_debug": bool(args.single_week),
        "episodes": args.episodes if args.method in RL_METHODS else None,
        "steps": steps or None,
        "seed": args.seed,
    }
    (Path(run_dir) / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps(protocol, indent=2), flush=True)

    os.environ["OPTIMAL_DEMO_SEASON"] = args.season
    os.environ["OPTIMAL_DEMO_TRAIN_WEEK_STARTS"] = ",".join(str(s) for s in train_starts)
    os.environ["OPTIMAL_DEMO_EVAL_EPISODE_START"] = str(eval_start)
    os.environ["OPTIMAL_DEMO_JOB_ID"] = f"seasonal_{args.season}_{args.method}_s{args.seed}"
    os.environ["OPTIMAL_DEMO_FMU_ISOLATE"] = "1"
    if args.single_week:
        os.environ["OPTIMAL_DEMO_FORCE_EPISODE_START"] = str(train_starts[0])
    else:
        os.environ.pop("OPTIMAL_DEMO_FORCE_EPISODE_START", None)

    if args.method == "hmsd":
        result = run_ghtd3_training(
            total_valid_steps=steps,
            run_dir=run_dir,
            seed=args.seed,
            annual_evaluation=False,
            skip_bc=True,
            config_path=args.config,
        )
    elif args.method == "td3":
        result = run_td3_scratch(
            total_valid_steps=steps,
            run_dir=run_dir,
            seed=args.seed,
            enable_shadow=False,
        )
    elif args.method == "sac":
        result = run_hybrid_sac_training(
            total_valid_steps=steps,
            run_dir=run_dir,
            seed=args.seed,
            enable_shadow=False,
            annual_evaluation=False,
        )
    elif args.method == "pso":
        result = run_pso_job(Path(run_dir), eval_start, args.seed, args.pso_iters, args.pso_particles)
    else:
        result = run_linprog_job(Path(run_dir), eval_start)

    if args.method in RL_METHODS:
        (Path(run_dir) / "train_result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    print("status", result.get("status"), "run_dir", run_dir, flush=True)


if __name__ == "__main__":
    main()
