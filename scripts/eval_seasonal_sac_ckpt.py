#!/usr/bin/env python
"""Re-run held-out eval for a trained seasonal SAC checkpoint — NO GiveSafe fallback.

If GiveSafe cannot find a safe action, records status=eval_failed (does not invent
a rule/thermal fallback). This preserves algorithm-quality evidence for the paper.

Usage:
  python scripts/eval_seasonal_sac_ckpt.py \\
    --run-dir runs/seasonal_v1/winter/sac_s0 \\
    --ckpt runs/seasonal_v1/winter/sac_s0/checkpoints/hybrid_givesafe_sac.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config.paths import apply_process_cache_env  # noqa: E402

apply_process_cache_env()

from envs.power_system_env import PowerSystemEnv  # noqa: E402
from safety import GiveSafeController, NoSafeActionFoundError, load_givesafe_config  # noqa: E402
from training.evaluate_td3 import evaluate_policy  # noqa: E402
from training.hybrid_common.policy_wrapper import HybridGiveSafePolicyWrapper  # noqa: E402
from training.hybrid_sac.algorithm import HybridSAC  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument(
        "--eval-start-seconds",
        type=float,
        default=None,
        help="Override; default from protocol.json eval_start_seconds",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    ckpt = Path(args.ckpt) if args.ckpt else run_dir / "checkpoints" / "hybrid_givesafe_sac.pt"
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint missing: {ckpt}")

    protocol_path = run_dir / "protocol.json"
    protocol = {}
    if protocol_path.is_file():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    eval_start = args.eval_start_seconds
    if eval_start is None:
        eval_start = float(protocol.get("eval_start_seconds", 3024000.0))

    gs_cfg = load_givesafe_config(ROOT / "src/config/givesafe_config.yaml")
    if bool((gs_cfg.get("givesafe") or {}).get("use_fallback", False)):
        raise SystemExit("refuse: givesafe.use_fallback must stay false for paper eval")

    probe = PowerSystemEnv(run_id=f"{run_dir.name}_sac_eval_probe", forecast_enabled=True)
    obs_dim = int(np.prod(probe.observation_space.shape))
    probe.close()

    agent = HybridSAC(obs_dim=obs_dim)
    agent.load(ckpt)
    print(
        f"loaded {ckpt} obs_dim={obs_dim} parameterized_caes={agent.parameterized_caes}",
        flush=True,
    )

    env = PowerSystemEnv(run_id=f"{run_dir.name}_sac_reeval", forecast_enabled=True)
    ctrl = GiveSafeController(oracle=env.oracle, shadow=None, config=gs_cfg)
    policy = HybridGiveSafePolicyWrapper(agent, env, ctrl, deterministic=True)

    traj = run_dir / "trajectories"
    traj.mkdir(parents=True, exist_ok=True)
    eval_csv = traj / "eval.csv"

    result: dict = {
        "status": "unknown",
        "algorithm": "hybrid_givesafe_sac",
        "parameterized_caes": bool(agent.parameterized_caes),
        "observation_dim": obs_dim,
        "checkpoint": str(ckpt.as_posix()),
        "eval_start_time_seconds": eval_start,
        "protocol": protocol,
        "givesafe_fallback": False,
        "reeval": True,
        "reeval_note": "post-hoc held-out eval; NoSafeActionFound recorded as eval_failed (no fallback)",
    }

    t0 = time.perf_counter()
    try:
        eval_res = evaluate_policy(
            env,
            policy,
            eval_csv,
            reset_options={"start_time": eval_start},
        )
        wall = time.perf_counter() - t0
        result["status"] = "completed"
        result["eval"] = eval_res
        result["eval_wall_seconds"] = wall
        result["episode_reward"] = eval_res.get("episode_reward")
        terms = eval_res.get("cost_terms") or {}
        result["sum_delta_j_gen"] = terms.get("generalized_cashflow_delta")
        result["terminal_soc_satisfied"] = eval_res.get("terminal_soc_satisfied")
        print(
            f"EVAL_OK R={result['episode_reward']} "
            f"Jgen={result['sum_delta_j_gen']} "
            f"SOC={result['terminal_soc_satisfied']}",
            flush=True,
        )
    except NoSafeActionFoundError as exc:
        wall = time.perf_counter() - t0
        result["status"] = "eval_failed"
        result["failure_type"] = "NoSafeActionFoundError"
        result["failure_reason"] = str(exc)
        result["failure_attempts"] = int(getattr(exc, "attempts", 0) or 0)
        result["eval_wall_seconds"] = wall
        result["traceback"] = traceback.format_exc()
        # partial trajectory if any rows were written before crash inside evaluate_policy
        if eval_csv.is_file() and eval_csv.stat().st_size > 0:
            result["partial_eval_csv"] = str(eval_csv.as_posix())
        print(f"EVAL_FAILED NoSafeActionFound: {exc}", flush=True)
    except Exception as exc:
        wall = time.perf_counter() - t0
        result["status"] = "eval_failed"
        result["failure_type"] = type(exc).__name__
        result["failure_reason"] = str(exc)
        result["eval_wall_seconds"] = wall
        result["traceback"] = traceback.format_exc()
        print(f"EVAL_FAILED {type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        env.close()

    out = run_dir / "train_result.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"wrote {out} status={result['status']}", flush=True)
    if result["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
