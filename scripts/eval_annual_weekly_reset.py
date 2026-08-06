#!/usr/bin/env python
"""Protocol A: 52-week sliding annual evaluation with weekly SoC reset.

Compares B0 / TD3-scratch / Safe Market-GHTD3 under the same FMU weekly-reset
protocol used in the main paper tables (not continuous-year inventory carry).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config.paths import apply_process_cache_env  # noqa: E402

apply_process_cache_env()

from controllers.rule_based_controller import RuleBasedController  # noqa: E402
from envs.power_system_env import PowerSystemEnv  # noqa: E402
from safety import GiveSafeController, load_givesafe_config  # noqa: E402
from training.evaluate_td3 import evaluate_annual_policy, evaluate_policy  # noqa: E402
from training.ghtd3.agent import GHTD3Agent  # noqa: E402
from training.ghtd3.train import GHTD3PolicyWrapper, load_ghtd3_config  # noqa: E402
from training.hybrid_td3.algorithm import HybridTD3  # noqa: E402


def make_td3_policy(env: PowerSystemEnv, ckpt: Path):
    dim = int(np.prod(env.observation_space.shape))
    agent = HybridTD3(obs_dim=dim, explore_noise=0.0)
    agent.load(ckpt)
    gs = load_givesafe_config(ROOT / "src/config/givesafe_config.yaml")
    ctrl = GiveSafeController(oracle=env.oracle, shadow=None, config=gs)

    class Pol:
        def on_episode_reset(self, info=None):
            pass

        def predict(self, obs, deterministic=True):
            feas = env.get_feasible_action_spec()

            def propose():
                return agent.select_action(obs, feas, deterministic=True)

            return ctrl.select_safe_action(
                env.last_outputs,
                env.previous_thermal,
                propose,
                deterministic=True,
                feasible_override=feas,
            ).safe_action

    return Pol()


def make_ghtd3_policy(env: PowerSystemEnv, ckpt: Path, config_path: Path | None):
    dim = int(np.prod(env.observation_space.shape))
    cfg_file = config_path
    if cfg_file is None:
        cand = Path(ckpt).resolve().parents[1] / "config" / "ghtd3_config.yaml"
        if cand.is_file():
            cfg_file = cand
    cfg = dict(load_ghtd3_config(cfg_file).get("ghtd3") or {})
    agent = GHTD3Agent(dim, cfg)
    agent.load(ckpt, strict=False)
    agent.execution_mode = str(cfg.get("execution_mode", "goal_conditioned")).lower()
    agent.low_use_raw_obs = bool(cfg.get("low_use_raw_obs", False))
    gs = load_givesafe_config(ROOT / "src/config/givesafe_config.yaml")
    ctrl = GiveSafeController(oracle=env.oracle, shadow=None, config=gs)
    return GHTD3PolicyWrapper(agent, env, ctrl, cfg)


def per_window_stats(env, policy, annual_hours: int) -> list[dict[str, Any]]:
    """Re-run window loop to keep per-window reward/SoC (annual API only returns aggregates)."""
    step_hours = float(env.config["fmu"]["decision_interval_seconds"]) / 3600.0
    episode_hours = int(env.episode_steps * step_hours)
    rows = []
    for start_hour in range(0, annual_hours, episode_hours):
        hours = min(episode_hours, annual_hours - start_hour)
        res = evaluate_policy(
            env,
            policy,
            output_csv=None,
            reset_options={"start_time": float(start_hour * 3600)},
            max_steps=hours,
        )
        rew = float(res.get("episode_reward") or res.get("weekly_episode_reward") or 0.0)
        rows.append(
            {
                "start_hour": start_hour,
                "hours": hours,
                "episode_reward": rew,
                "terminal_soc_satisfied": bool(res.get("terminal_soc_satisfied")),
                "valid_steps": int(res.get("valid_steps") or 0),
                "fmu_failure_count": int(res.get("fmu_failure_count") or 0),
            }
        )
        print(
            f"  win start={start_hour:4d}h rew={rew:8.2f} soc={rows[-1]['terminal_soc_satisfied']} "
            f"steps={rows[-1]['valid_steps']}",
            flush=True,
        )
    return rows


def summarize(method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    rews = [float(r["episode_reward"]) for r in rows]
    soc = sum(1 for r in rows if r["terminal_soc_satisfied"])
    return {
        "method": method,
        "protocol": "weekly_reset_sliding",
        "n_windows": len(rows),
        "reward_mean": float(np.mean(rews)) if rews else 0.0,
        "reward_std": float(np.std(rews, ddof=1)) if len(rews) > 1 else 0.0,
        "reward_sum": float(np.sum(rews)) if rews else 0.0,
        "reward_p25": float(np.percentile(rews, 25)) if rews else 0.0,
        "reward_p50": float(np.percentile(rews, 50)) if rews else 0.0,
        "reward_p75": float(np.percentile(rews, 75)) if rews else 0.0,
        "soc_pass": soc,
        "soc_pass_rate": float(soc / max(len(rows), 1)),
        "windows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Annual weekly-reset evaluation (protocol A)")
    ap.add_argument("--hours", type=int, default=8760, help="annual horizon hours")
    ap.add_argument("--ghtd3-ckpt", type=str, default="runs/ghtd3_abs_s1_35k/checkpoints/ghtd3.pt")
    ap.add_argument("--td3-ckpt", type=str, default="runs/td3_scratch_s1_35k/checkpoints/hybrid_givesafe_td3.pt")
    ap.add_argument("--ghtd3-config", type=str, default="src/config/ghtd3_config_abs.yaml")
    ap.add_argument("--methods", type=str, default="b0,td3,ghtd3")
    ap.add_argument("--out", type=str, default="runs/paper_annual_weekly_reset.json")
    ap.add_argument("--max-windows", type=int, default=0, help="if >0, only first N windows (smoke)")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # optional smoke: truncate by rewriting hours so only N full weeks
    hours = int(args.hours)
    if args.max_windows and args.max_windows > 0:
        hours = min(hours, args.max_windows * 168)

    results: dict[str, Any] = {
        "protocol": "weekly_reset_sliding",
        "annual_horizon_hours": hours,
        "note": "Each 168h window resets FMU/SoC; same protocol as main seasonal table.",
        "methods": {},
    }

    for method in methods:
        print(f"\n=== {method} annual weekly-reset ({hours} h) ===", flush=True)
        env = PowerSystemEnv(run_id=f"annual_wr_{method}")
        try:
            if method == "b0":
                pol = RuleBasedController(env)
            elif method == "td3":
                pol = make_td3_policy(env, ROOT / args.td3_ckpt)
            elif method == "ghtd3":
                pol = make_ghtd3_policy(env, ROOT / args.ghtd3_ckpt, ROOT / args.ghtd3_config)
            else:
                raise ValueError(f"unknown method {method}")
            rows = per_window_stats(env, pol, hours)
            results["methods"][method] = summarize(method, rows)
            s = results["methods"][method]
            print(
                f"SUMMARY {method}: mean={s['reward_mean']:.2f}±{s['reward_std']:.2f} "
                f"soc={s['soc_pass']}/{s['n_windows']} ({100*s['soc_pass_rate']:.1f}%)",
                flush=True,
            )
        finally:
            env.close()

    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}", flush=True)

    # compact markdown
    md = ["# Annual weekly-reset evaluation (protocol A)", "", "| Method | Windows | Reward mean±std | SoC pass |", "|--------|---------|-----------------|----------|"]
    for method, s in results["methods"].items():
        md.append(
            f"| {method} | {s['n_windows']} | {s['reward_mean']:.2f}±{s['reward_std']:.2f} | "
            f"{s['soc_pass']}/{s['n_windows']} ({100*s['soc_pass_rate']:.1f}%) |"
        )
    md_path = out_path.with_suffix(".md")
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
