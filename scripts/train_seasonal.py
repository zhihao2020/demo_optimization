#!/usr/bin/env python
"""Unified seasonal suite: PC-HybridTD3 / projection TD3 + classical baselines.

Paper mainline:
  --method td3 --season all            PC-HybridTD3 on A_f(s)
  --method td3 --ablation projection   continuous box + clamp
  --method td3 --ablation static-support   hybrid heads on static CAES bands
  --method milp                        rolling MILP (optimise surrogate, eval on FMU)

Train weeks come only from the 36/8/8 split (9/2/2 per quarter). Tables use TEST.
FS-HSAC / HMSD remain in the tree as archive methods, not the paper identity.
GiveSafe on; soft_shell OFF; storage_use off.

Examples:
  python scripts/train_seasonal.py --method td3 --season winter --stage A
  python scripts/train_seasonal.py --method td3 --season all --stage B --seed 0
  python scripts/train_seasonal.py --method td3 --season all --stage D --seed 0
  python scripts/train_seasonal.py --method td3 --ablation projection --season all --stage D --seed 0
  python scripts/train_seasonal.py --method milp --season winter --seed 0
  python scripts/train_seasonal.py --method rule --season winter --seed 0
  python scripts/train_seasonal.py --method td3 --season all --forecast-mode noisy --annual-eval
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from seasonal_cli import (  # noqa: E402
    ALL_METHODS,
    EPISODE_HOURS,
    RL_METHODS,
    SEASON_WEEKS,
    STAGE_STEPS,
    parse_args,
)
from config.paths import apply_process_cache_env, resolve_run_dir  # noqa: E402

apply_process_cache_env()

from envs.power_system_env import PowerSystemEnv  # noqa: E402
from training.evaluate_td3 import evaluate_policy  # noqa: E402
from training.ghtd3.train import run_ghtd3_training  # noqa: E402
from training.hybrid_sac.train import run_hybrid_sac_training  # noqa: E402
from training.fs_hsac.train import run_fs_hsac_training  # noqa: E402
from training.hybrid_td3.train import annual_episode_start_seconds, run_td3_scratch  # noqa: E402


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
    from optimization.metrics import extract_kpi_from_eval

    return extract_kpi_from_eval(ev)



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


def run_milp_job(run_dir: Path, eval_start: float) -> dict:
    from optimization.rolling_milp import RollingMilpController

    env = PowerSystemEnv(run_id=run_dir.name, forecast_enabled=True)
    try:
        pol = RollingMilpController(env)
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
        "method": "milp",
        "eval": ev,
        "kpi": kpi_from_eval(ev),
        "eval_start_time_seconds": eval_start,
        "wall_s": wall,
        "baseline_notes": {
            "forecast": "24 h rolling horizon from forecast_provider; first hour applied to the same FMU",
            "horizon_hours": 24,
            "evaluate_on": "same Sysplorer FMU as RL (surrogate is used only to propose a_t)",
            "caes": "binary commitment + min-load bands; energy SoC only (no hot/cold/pressure DAE)",
            "min_run_steps": 1,
        },
    }
    (run_dir / "train_result.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return out


def run_rule_job(run_dir: Path, eval_start: float, forecast_mode: str | None) -> dict:
    from controllers.price_aware_rule import PriceAwareRuleController

    env = PowerSystemEnv(
        run_id=run_dir.name,
        forecast_enabled=True,
        forecast_mode=forecast_mode,
    )
    try:
        pol = PriceAwareRuleController(env)
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
        "method": "rule",
        "eval": ev,
        "kpi": kpi_from_eval(ev),
        "eval_start_time_seconds": eval_start,
        "wall_s": wall,
        "forecast_mode": forecast_mode,
    }
    (run_dir / "train_result.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return out


def run_stage_a(run_dir: Path, seed: int) -> dict:
    from training.hybrid_td3.stage_a import run_stage_a_support

    out = run_stage_a_support(n=10_000, seed=seed)
    payload = json.dumps(out, indent=2, ensure_ascii=False, default=str)
    (run_dir / "train_result.json").write_text(payload, encoding="utf-8")
    (run_dir / "summary.json").write_text(payload, encoding="utf-8")
    return out


def main() -> None:
    args = parse_args()

    if args.stage == "A":
        method_tag = "pc_hybrid_td3_stageA"
        run_dir = args.run_dir or f"runs/seasonal/{args.season}/{method_tag}_s{args.seed}"
        run_dir = str(resolve_run_dir(run_dir))
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        result = run_stage_a(Path(run_dir), args.seed)
        print("status", result.get("status"), "run_dir", run_dir, flush=True)
        if result.get("status") != "completed":
            raise SystemExit(2)
        return

    meta = SEASON_WEEKS[args.season]
    if args.single_week:
        train_weeks = [meta["train"][0]]
        val_weeks = train_weeks
        test_weeks = train_weeks
        eval_week = train_weeks[0]
    else:
        train_weeks = parse_weeks(args.train_weeks, meta["train"])
        val_weeks = parse_weeks(args.val_weeks, meta.get("val", meta["train"][-2:]))
        test_weeks = parse_weeks(args.test_weeks, meta.get("test", [meta["eval"]]))
        if not test_weeks or not val_weeks:
            raise SystemExit("formal split: val/test weeks must be configured; no train-week fallback")
        eval_week = int(args.eval_week) if args.eval_week is not None else int(test_weeks[0])
        if eval_week in train_weeks:
            raise SystemExit(f"formal split: eval week {eval_week} is a training week")

    train_starts = [week_start_seconds(w) for w in train_weeks]
    eval_start = week_start_seconds(eval_week)

    steps = int(args.episodes) * EPISODE_HOURS if args.method in RL_METHODS else 0
    if args.stage:
        steps = int(STAGE_STEPS[args.stage])
    method_name = args.method
    if args.method == "fs_hsac" and args.support_only:
        method_name = "fs_hsac_support"
    if args.method == "td3" and args.ablation == "projection":
        method_name = "td3_proj"
    elif args.method == "td3" and args.ablation == "static-support":
        method_name = "td3_static"
    elif args.method == "td3":
        method_name = "pc_hybrid_td3"
    method_tag = f"{method_name}_lockcaes" if args.lock_caes else method_name
    run_dir = args.run_dir or f"runs/seasonal/{args.season}/{method_tag}_s{args.seed}"
    run_dir = str(resolve_run_dir(run_dir))
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "trajectories").mkdir(parents=True, exist_ok=True)

    protocol = {
        "protocol": "pc_hybridtd3_36_8_8",
        "method": args.method,
        "method_tag": method_name,
        "ablation": args.ablation if args.method == "td3" else None,
        "season": args.season,
        "train_weeks": train_weeks,
        "val_weeks": val_weeks,
        "test_weeks": test_weeks,
        "eval_week": eval_week,
        "train_start_seconds": train_starts,
        "eval_start_seconds": eval_start,
        "single_week_debug": bool(args.single_week),
        "episodes": args.episodes if args.method in RL_METHODS else None,
        "steps": steps or None,
        "seed": args.seed,
        "lock_caes": bool(args.lock_caes),
        "support_only": bool(args.support_only) if args.method == "fs_hsac" else None,
        "use_feasibility_penalty": (
            (not bool(args.support_only)) if args.method == "fs_hsac" else None
        ),
        "soft_shell": False,
        "paper_mainline": args.method == "td3" and args.ablation == "none",
        "parameterized_caes": (args.ablation != "projection") if args.method == "td3" else None,
        "use_dynamic_support": (args.ablation == "none") if args.method == "td3" else None,
        "forecast_mode": args.forecast_mode,
        "stage": args.stage,
        "annual_eval": bool(args.annual_eval),
        "story": "A_grid_contract",
    }
    (Path(run_dir) / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps(protocol, indent=2), flush=True)

    os.environ["OPTIMAL_DEMO_SEASON"] = args.season
    os.environ["OPTIMAL_DEMO_TRAIN_WEEK_STARTS"] = ",".join(str(s) for s in train_starts)
    os.environ["OPTIMAL_DEMO_EVAL_EPISODE_START"] = str(eval_start)
    os.environ["OPTIMAL_DEMO_VAL_WEEK_STARTS"] = ",".join(
        str(week_start_seconds(w)) for w in val_weeks
    )
    os.environ["OPTIMAL_DEMO_TEST_WEEK_STARTS"] = ",".join(
        str(week_start_seconds(w)) for w in test_weeks
    )
    os.environ["OPTIMAL_DEMO_JOB_ID"] = (
        f"seasonal_{args.season}_{method_tag}_s{args.seed}"
    )
    os.environ["OPTIMAL_DEMO_FMU_ISOLATE"] = "1"
    if args.lock_caes:
        os.environ["OPTIMAL_DEMO_LOCK_CAES"] = "1"
    else:
        os.environ.pop("OPTIMAL_DEMO_LOCK_CAES", None)
    if args.method == "fs_hsac" and args.support_only:
        os.environ["FS_HSAC_NO_FEAS"] = "1"
    elif args.method == "fs_hsac":
        os.environ.pop("FS_HSAC_NO_FEAS", None)
    if args.single_week:
        os.environ["OPTIMAL_DEMO_FORCE_EPISODE_START"] = str(train_starts[0])
        os.environ.pop("OPTIMAL_DEMO_FORMAL_SPLIT", None)
    else:
        os.environ.pop("OPTIMAL_DEMO_FORCE_EPISODE_START", None)
        os.environ["OPTIMAL_DEMO_FORMAL_SPLIT"] = "1"

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
            parameterized_caes=args.ablation != "projection",
            use_dynamic_support=args.ablation == "none",
            forecast_mode=args.forecast_mode,
            require_gas_swing=0.05 if args.stage == "B" else None,
            annual_evaluation=bool(args.annual_eval),
        )
    elif args.method == "sac":
        result = run_hybrid_sac_training(
            total_valid_steps=steps,
            run_dir=run_dir,
            seed=args.seed,
            enable_shadow=False,
            annual_evaluation=False,
            soft_shell=False,
        )
    elif args.method == "fs_hsac":
        result = run_fs_hsac_training(
            total_valid_steps=steps,
            run_dir=run_dir,
            seed=args.seed,
            enable_shadow=False,
            annual_evaluation=False,
            use_feasibility_penalty=not bool(args.support_only),
            soft_shell=False,
        )
    elif args.method == "pso":
        result = run_pso_job(Path(run_dir), eval_start, args.seed, args.pso_iters, args.pso_particles)
    elif args.method == "milp":
        result = run_milp_job(Path(run_dir), eval_start)
    elif args.method == "rule":
        result = run_rule_job(Path(run_dir), eval_start, args.forecast_mode)
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
