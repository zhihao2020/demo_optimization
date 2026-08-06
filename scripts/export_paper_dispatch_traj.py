#!/usr/bin/env python
"""Export multi-season closed-loop trajectories for paper dispatch figures.

Methods: B0 (rule), linprog (if controller available), PSO (replay θ or re-search),
TD3-scratch, Safe Market-GHTD3.

Season start times match eval_ghtd3_vs_td3 (weeks 0 / 13 / 26).
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
from optimization.pso_fmu import PSOConfig, ParametricPricePolicy, run_pso  # noqa: E402
from safety import GiveSafeController, load_givesafe_config  # noqa: E402
from training.evaluate_td3 import evaluate_policy  # noqa: E402
from training.ghtd3.agent import GHTD3Agent  # noqa: E402
from training.ghtd3.train import GHTD3PolicyWrapper, load_ghtd3_config  # noqa: E402
from training.hybrid_td3.algorithm import HybridTD3  # noqa: E402
from training.hybrid_td3.train import annual_episode_start_seconds  # noqa: E402

SEASONS = ("winter", "transition", "summer")
WEEK_IDX = {"winter": 0, "transition": 13, "summer": 26}

# PSO θ: re-searched on paper season starts (weeks 0/13/26); winter still early-fails under parametric policy
ARCHIVED_PSO_THETA: dict[str, list[float]] = {
    "winter": [0.4460785837683554, 0.8515864270482753, 0.6031202600921061, 0.9, 0.3, 0.5],
    "transition": [0.45539799067778075, 0.8932331724132613, 0.3, 0.0, 0.3, 1.0],
    "summer": [0.41850911006307856, 1.1702576640693108, 0.3465055919708009, 0.0, 0.3217454485873198, 0.9842988211565078],
}


def season_starts(env: PowerSystemEnv) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, week in WEEK_IDX.items():
        out[name] = float(annual_episode_start_seconds(env.config["fmu"], env.episode_steps, week))
    return out


def eval_to_csv(env: PowerSystemEnv, policy: Any, csv_path: Path, start: float) -> dict[str, Any]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    res = evaluate_policy(env, policy, csv_path, reset_options={"start_time": float(start)})
    return res


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


def try_linprog_policy(env: PowerSystemEnv):
    try:
        from optimization.rolling_linprog import RollingLinprogController

        return RollingLinprogController(env)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] linprog unavailable: {exc}", flush=True)
        return None


def summary_row(method: str, season: str, start: float, res: dict[str, Any], extra: dict | None = None) -> dict:
    row = {
        "method": method,
        "season": season,
        "start_time": start,
        "episode_reward": float(res.get("episode_reward") or 0.0),
        "terminal_soc_satisfied": bool(res.get("terminal_soc_satisfied")),
        "valid_steps": int(res.get("valid_steps") or 0),
        "weekly_raw_total_cost": res.get("weekly_raw_total_cost"),
        "economic_cashflow_total": res.get("economic_cashflow_total"),
    }
    metrics = res.get("metrics") or {}
    for k in (
        "battery_throughput_mwh",
        "caes_throughput_mwh",
        "thermal_generation_mwh",
        "curtailment_energy_mwh",
        "unserved_energy_mwh",
    ):
        if k in metrics:
            row[k] = metrics[k]
    if extra:
        row.update(extra)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Export paper multi-season dispatch trajectories")
    ap.add_argument("--out-dir", type=str, default="runs/paper_dispatch_traj")
    ap.add_argument("--ghtd3-ckpt", type=str, default="runs/ghtd3_abs_s1_35k/checkpoints/ghtd3.pt")
    ap.add_argument("--td3-ckpt", type=str, default="runs/td3_scratch_s1_35k/checkpoints/hybrid_givesafe_td3.pt")
    ap.add_argument("--ghtd3-config", type=str, default="src/config/ghtd3_config_abs.yaml")
    ap.add_argument("--pso-mode", choices=["replay", "search", "skip"], default="replay")
    ap.add_argument("--pso-iters", type=int, default=15)
    ap.add_argument("--pso-particles", type=int, default=10)
    ap.add_argument("--methods", type=str, default="b0,ghtd3,td3,pso,linprog")
    ap.add_argument("--seasons", type=str, default="winter,transition,summer")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = {m.strip() for m in args.methods.split(",") if m.strip()}
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]

    meta_env = PowerSystemEnv(run_id="paper_traj_meta")
    starts = season_starts(meta_env)
    meta_env.close()
    print("season starts:", starts, flush=True)

    rows: list[dict[str, Any]] = []
    pso_theta: dict[str, list[float]] = dict(ARCHIVED_PSO_THETA)

    for season in seasons:
        start = starts[season]
        print(f"\n=== {season} start={start} ===", flush=True)

        if "b0" in methods:
            env = PowerSystemEnv(run_id=f"paper_b0_{season}")
            pol = RuleBasedController(env)
            csv_path = out_dir / f"{season}_b0.csv"
            res = eval_to_csv(env, pol, csv_path, start)
            rows.append(summary_row("b0", season, start, res))
            print(f"  b0 reward={rows[-1]['episode_reward']:.2f} soc={rows[-1]['terminal_soc_satisfied']}", flush=True)
            env.close()

        if "linprog" in methods:
            env = PowerSystemEnv(run_id=f"paper_lp_{season}")
            pol = try_linprog_policy(env)
            if pol is not None:
                csv_path = out_dir / f"{season}_linprog.csv"
                res = eval_to_csv(env, pol, csv_path, start)
                rows.append(summary_row("linprog", season, start, res))
                print(
                    f"  linprog reward={rows[-1]['episode_reward']:.2f} soc={rows[-1]['terminal_soc_satisfied']}",
                    flush=True,
                )
            env.close()

        if "pso" in methods and args.pso_mode != "skip":
            if args.pso_mode == "search":
                cfg = PSOConfig(n_particles=args.pso_particles, n_iters=args.pso_iters, seed=0)
                print(f"  PSO search iters={cfg.n_iters} particles={cfg.n_particles}", flush=True)
                kpi = run_pso(start_time=start, cfg=cfg)
                theta = list(kpi.get("theta_best") or kpi.get("theta") or [])
                pso_theta[season] = [float(x) for x in theta]
                print(f"  PSO best reward≈{kpi.get('episode_reward')} theta={pso_theta[season]}", flush=True)
            theta = np.asarray(pso_theta[season], dtype=np.float64)
            env = PowerSystemEnv(run_id=f"paper_pso_{season}")
            pol = ParametricPricePolicy(env, theta)
            csv_path = out_dir / f"{season}_pso.csv"
            res = eval_to_csv(env, pol, csv_path, start)
            rows.append(summary_row("pso", season, start, res, extra={"theta": pso_theta[season]}))
            print(
                f"  pso reward={rows[-1]['episode_reward']:.2f} steps={rows[-1]['valid_steps']} "
                f"soc={rows[-1]['terminal_soc_satisfied']}",
                flush=True,
            )
            env.close()

        if "td3" in methods:
            ckpt = ROOT / args.td3_ckpt
            if not ckpt.is_file():
                print(f"  [skip] td3 ckpt missing: {ckpt}", flush=True)
            else:
                env = PowerSystemEnv(run_id=f"paper_td3_{season}")
                pol = make_td3_policy(env, ckpt)
                csv_path = out_dir / f"{season}_td3.csv"
                res = eval_to_csv(env, pol, csv_path, start)
                rows.append(summary_row("td3", season, start, res))
                print(
                    f"  td3 reward={rows[-1]['episode_reward']:.2f} soc={rows[-1]['terminal_soc_satisfied']}",
                    flush=True,
                )
                env.close()

        if "ghtd3" in methods:
            ckpt = ROOT / args.ghtd3_ckpt
            if not ckpt.is_file():
                print(f"  [skip] ghtd3 ckpt missing: {ckpt}", flush=True)
            else:
                env = PowerSystemEnv(run_id=f"paper_gh_{season}")
                cfg_path = ROOT / args.ghtd3_config if args.ghtd3_config else None
                pol = make_ghtd3_policy(env, ckpt, cfg_path)
                csv_path = out_dir / f"{season}_ghtd3.csv"
                res = eval_to_csv(env, pol, csv_path, start)
                rows.append(summary_row("ghtd3", season, start, res))
                print(
                    f"  ghtd3 reward={rows[-1]['episode_reward']:.2f} soc={rows[-1]['terminal_soc_satisfied']}",
                    flush=True,
                )
                env.close()

    summary = {
        "season_starts": starts,
        "ghtd3_ckpt": args.ghtd3_ckpt,
        "td3_ckpt": args.td3_ckpt,
        "pso_mode": args.pso_mode,
        "pso_theta": pso_theta,
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_dir / 'summary.json'} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
