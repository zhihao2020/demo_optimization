#!/usr/bin/env python
"""Held-out soft-shell eval for seasonal_v1 checkpoints (does NOT overwrite hard-protocol results).

Replays existing weights under SoftConstraintEnv + SoftShellGiveSafePolicy.
Writes to runs/seasonal_v1_soft_shell/{season}/{method}_s{seed}/.

Usage:
  python scripts/eval_soft_shell_seasonal.py --season transition --method td3
  python scripts/eval_soft_shell_seasonal.py --all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config.paths import apply_process_cache_env  # noqa: E402

apply_process_cache_env()

from envs.power_system_env import PowerSystemEnv  # noqa: E402
from safety import GiveSafeController, load_givesafe_config  # noqa: E402
from training.evaluate_td3 import evaluate_policy  # noqa: E402
from training.ghtd3.agent import GHTD3Agent  # noqa: E402
from training.ghtd3.train import GHTD3PolicyWrapper, load_ghtd3_config  # noqa: E402
from training.hybrid_common.policy_wrapper import SoftShellGiveSafePolicy  # noqa: E402
from training.hybrid_sac.algorithm import HybridSAC  # noqa: E402
from training.hybrid_td3.algorithm import HybridTD3  # noqa: E402

SEASONS = ("winter", "transition", "summer")
METHODS = ("hmsd", "td3", "sac")
SRC_ROOT = ROOT / "runs" / "seasonal_v1"
OUT_ROOT = ROOT / "runs" / "seasonal_v1_soft_shell"

CKPT_NAME = {
    "hmsd": "ghtd3.pt",
    "td3": "hybrid_givesafe_td3.pt",
    "sac": "hybrid_givesafe_sac.pt",
}


def _load_protocol(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "protocol.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _make_policy(method: str, env: PowerSystemEnv, ckpt: Path, gs_cfg: dict) -> Any:
    obs_dim = int(np.prod(env.observation_space.shape))
    ctrl = GiveSafeController(oracle=env.oracle, shadow=None, config=gs_cfg)
    if method == "hmsd":
        cfg = load_ghtd3_config(ROOT / "src/config/ghtd3_config.yaml")
        agent = GHTD3Agent(obs_dim=obs_dim, cfg=cfg)
        agent.load(ckpt)
        # Soft shell at evaluate_policy level; wrapper still recovers NoSafeAction
        # for HMSD historical behavior — evaluate_policy soft_shell covers env retry.
        return GHTD3PolicyWrapper(agent, env, ctrl, cfg)
    if method == "td3":
        agent = HybridTD3(obs_dim=obs_dim, explore_noise=0.0)
        agent.load(ckpt)
        return SoftShellGiveSafePolicy(agent, env, ctrl, deterministic=True, soft_shell=True)
    if method == "sac":
        agent = HybridSAC(obs_dim=obs_dim)
        agent.load(ckpt)
        return SoftShellGiveSafePolicy(agent, env, ctrl, deterministic=True, soft_shell=True)
    raise ValueError(f"unknown method {method}")


def eval_one(season: str, method: str, seed: int = 0) -> dict[str, Any]:
    src = SRC_ROOT / season / f"{method}_s{seed}"
    ckpt = src / "checkpoints" / CKPT_NAME[method]
    out = OUT_ROOT / season / f"{method}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "trajectories").mkdir(exist_ok=True)

    result: dict[str, Any] = {
        "status": "running",
        "protocol": "soft_shell_eval_v1",
        "source_run": str(src.as_posix()),
        "season": season,
        "method": method,
        "seed": seed,
        "soft_shell": True,
        "use_fallback": False,
        "note": "eval-only soft shell on frozen weights; does not replace hard-protocol seasonal_v1",
    }
    if not ckpt.is_file():
        result["status"] = "skipped_missing_checkpoint"
        result["error"] = f"missing {ckpt}"
        (out / "soft_shell_eval.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        return result

    protocol = _load_protocol(src)
    eval_start = float(protocol.get("eval_start_seconds", 0.0))
    result["eval_start_seconds"] = eval_start
    result["source_protocol"] = protocol

    gs_cfg = load_givesafe_config(ROOT / "src/config/givesafe_config.yaml")
    if bool((gs_cfg.get("givesafe") or {}).get("use_fallback", False)):
        raise SystemExit("refuse: givesafe.use_fallback must stay false")

    os.environ["OPTIMAL_DEMO_SEASON"] = season
    if protocol.get("eval_start_seconds") is not None:
        os.environ["OPTIMAL_DEMO_EVAL_EPISODE_START"] = str(eval_start)

    env = PowerSystemEnv(run_id=f"soft_shell_{season}_{method}_s{seed}", forecast_enabled=True)
    t0 = time.perf_counter()
    try:
        policy = _make_policy(method, env, ckpt, gs_cfg)
        ev = evaluate_policy(
            env,
            policy,
            out / "trajectories" / "eval.csv",
            reset_options={"start_time": eval_start},
            soft_shell=True,
        )
        result["status"] = "completed"
        result["eval"] = ev
        result["kpi"] = {
            "steps": ev.get("steps"),
            "valid_steps": ev.get("valid_steps"),
            "episode_reward": ev.get("episode_reward"),
            "soft_shell_count": ev.get("soft_shell_count"),
            "soft_shell_hours": ev.get("soft_shell_hours"),
            "terminal_soc_satisfied": ev.get("terminal_soc_satisfied"),
        }
    except Exception as exc:  # noqa: BLE001
        result["status"] = "eval_failed"
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
    finally:
        result["wall_s"] = time.perf_counter() - t0
        env.close()

    (out / "soft_shell_eval.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    # Do not write train_result.json — avoid clobbering hard-protocol naming.
    print(
        f"{season}/{method}: status={result['status']} "
        f"valid={result.get('kpi', {}).get('valid_steps')} "
        f"shell={result.get('kpi', {}).get('soft_shell_count')}",
        flush=True,
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Soft-shell held-out eval (separate run dir)")
    ap.add_argument("--season", choices=list(SEASONS), default=None)
    ap.add_argument("--method", choices=list(METHODS), default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="All seasons × methods")
    args = ap.parse_args()

    jobs: list[tuple[str, str]] = []
    if args.all:
        jobs = [(s, m) for s in SEASONS for m in METHODS]
    elif args.season and args.method:
        jobs = [(args.season, args.method)]
    else:
        raise SystemExit("provide --season and --method, or --all")

    summary = []
    for season, method in jobs:
        summary.append(eval_one(season, method, args.seed))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
