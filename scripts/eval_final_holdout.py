#!/usr/bin/env python
"""Evaluate Rule / rolling MILP / PSO / PC-HybridTD3 on one final-holdout week.

Final weeks: 12, 25, 38, 51 (second TEST week of each quarter).
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
sys.path.insert(0, str(ROOT / "scripts"))

from config.paths import apply_process_cache_env, resolve_run_dir  # noqa: E402

apply_process_cache_env()

from envs.power_system_env import PowerSystemEnv  # noqa: E402
from optimization.metrics import extract_kpi_from_eval  # noqa: E402
from training.evaluate_td3 import evaluate_policy  # noqa: E402
from training.hybrid_td3.train import annual_episode_start_seconds  # noqa: E402

FINAL_WEEKS = (12, 25, 38, 51)
WEEK_SEASON = {12: "winter", 25: "transition", 38: "summer", 51: "autumn"}


def week_start_seconds(week_index: int) -> float:
    env = PowerSystemEnv(run_id="holdout_meta", forecast_enabled=True)
    try:
        return float(annual_episode_start_seconds(env.config["fmu"], env.episode_steps, week_index))
    finally:
        env.close()


def write_out(run_dir: Path, payload: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "trajectories").mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    (run_dir / "train_result.json").write_text(text, encoding="utf-8")
    (run_dir / "summary.json").write_text(text, encoding="utf-8")


def eval_rule(run_dir: Path, start: float) -> dict:
    from controllers.price_aware_rule import PriceAwareRuleController

    env = PowerSystemEnv(run_id=run_dir.name, forecast_enabled=True)
    try:
        t0 = time.perf_counter()
        ev = evaluate_policy(
            env,
            PriceAwareRuleController(env),
            run_dir / "trajectories" / "eval.csv",
            reset_options={"start_time": start},
        )
        wall = time.perf_counter() - t0
    finally:
        env.close()
    return {"status": "completed", "method": "rule", "eval": ev, "kpi": extract_kpi_from_eval(ev), "wall_s": wall}


def eval_milp(run_dir: Path, start: float) -> dict:
    from optimization.rolling_milp import RollingMilpController

    env = PowerSystemEnv(run_id=run_dir.name, forecast_enabled=True)
    try:
        t0 = time.perf_counter()
        ev = evaluate_policy(
            env,
            RollingMilpController(env),
            run_dir / "trajectories" / "eval.csv",
            reset_options={"start_time": start},
        )
        wall = time.perf_counter() - t0
    finally:
        env.close()
    return {"status": "completed", "method": "milp", "eval": ev, "kpi": extract_kpi_from_eval(ev), "wall_s": wall}


def eval_pso(run_dir: Path, start: float, seed: int, n_iters: int, n_particles: int) -> dict:
    import numpy as np
    from optimization.pso_fmu import PSOConfig, ParametricPricePolicy, run_pso

    cfg = PSOConfig(n_particles=int(n_particles), n_iters=int(n_iters), seed=int(seed))
    t_search = time.perf_counter()
    search = run_pso(start_time=start, cfg=cfg)
    search_s = time.perf_counter() - t_search
    theta = np.asarray(search.get("theta_best"), dtype=np.float64)
    env = PowerSystemEnv(run_id=run_dir.name, forecast_enabled=True)
    try:
        pol = ParametricPricePolicy(env, theta)
        t0 = time.perf_counter()
        ev = evaluate_policy(
            env,
            pol,
            run_dir / "trajectories" / "eval.csv",
            reset_options={"start_time": start},
        )
        wall = time.perf_counter() - t0
    finally:
        env.close()
    return {
        "status": "completed",
        "method": "pso",
        "eval": ev,
        "kpi": extract_kpi_from_eval(ev),
        "wall_s": wall,
        "wall_s_search": search_s,
        "theta_best": theta.tolist(),
        "pso_iters": int(n_iters),
        "pso_particles": int(n_particles),
        "fmu_steps_search": search.get("fmu_steps_search"),
        "search_history": search.get("history"),
        "search_fitness": search.get("fitness"),
    }


def eval_td3(run_dir: Path, start: float, ckpt: Path) -> dict:
    import numpy as np
    from safety import GiveSafeController, load_givesafe_config
    from training.hybrid_common.policy_wrapper import HybridGiveSafePolicyWrapper
    from training.hybrid_td3.algorithm import HybridTD3
    from training.hybrid_td3.train import _paper_algo_cfg, _pin_torch_threads, _torch_device

    env = PowerSystemEnv(run_id=run_dir.name, forecast_enabled=True)
    gs_cfg = load_givesafe_config()
    ctrl = GiveSafeController(oracle=env.oracle, shadow=None, config=gs_cfg)
    paper_algo = _paper_algo_cfg(ROOT)
    _pin_torch_threads()
    agent = HybridTD3(
        obs_dim=int(np.prod(env.observation_space.shape)),
        gamma=float(paper_algo.get("gamma", 0.99)),
        actor_lr=float(paper_algo.get("actor_lr", 3e-4)),
        critic_lr=float(paper_algo.get("critic_lr", 3e-4)),
        tau=float(paper_algo.get("tau", 0.005)),
        policy_delay=int(paper_algo.get("policy_delay", 2)),
        parameterized_caes=True,
        use_dynamic_support=True,
        device=_torch_device(),
    )
    agent.load(ckpt)
    pol = HybridGiveSafePolicyWrapper(agent, env, ctrl, deterministic=True, soft_shell=False)
    try:
        t0 = time.perf_counter()
        ev = evaluate_policy(
            env,
            pol,
            run_dir / "trajectories" / "eval.csv",
            reset_options={"start_time": start},
        )
        wall = time.perf_counter() - t0
    finally:
        env.close()
    return {
        "status": "completed",
        "method": "pc_hybrid_td3",
        "checkpoint": str(ckpt),
        "eval": ev,
        "kpi": extract_kpi_from_eval(ev),
        "wall_s": wall,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=("rule", "milp", "pso", "td3"), required=True)
    p.add_argument("--week", type=int, required=True, choices=list(FINAL_WEEKS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--pso-iters", type=int, default=25)
    p.add_argument("--pso-particles", type=int, default=12)
    args = p.parse_args()
    os.environ["OPTIMAL_DEMO_FMU_ISOLATE"] = "1"
    os.environ["OPTIMAL_DEMO_JOB_ID"] = f"holdout_{args.method}_w{args.week}_s{args.seed}"
    run_dir = Path(resolve_run_dir(args.run_dir))
    start = week_start_seconds(args.week)
    if args.method == "rule":
        out = eval_rule(run_dir, start)
    elif args.method == "milp":
        out = eval_milp(run_dir, start)
    elif args.method == "pso":
        out = eval_pso(run_dir, start, args.seed, args.pso_iters, args.pso_particles)
    else:
        ckpt = Path(args.ckpt or "")
        if not ckpt.is_file():
            raise SystemExit(f"missing checkpoint: {ckpt}")
        out = eval_td3(run_dir, start, ckpt)
    out.update(
        {
            "eval_week": args.week,
            "season": WEEK_SEASON[args.week],
            "seed": args.seed,
            "eval_start_time_seconds": start,
            "protocol": "paper_min_final_holdout_w12_25_38_51",
        }
    )
    write_out(run_dir, out)
    ev = out.get("eval") or {}
    print(
        "status", out.get("status"),
        "week", args.week,
        "eval", ev.get("eval_status"),
        "steps", ev.get("valid_steps"),
        "cost", ev.get("weekly_raw_total_cost"),
        flush=True,
    )
    if str(ev.get("eval_status")) != "ok" or int(ev.get("valid_steps") or 0) < 168:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
