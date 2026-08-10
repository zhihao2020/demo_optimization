"""Three-season eval for remote 840k HMSD + TD3 checkpoints (paper-aligned week indices)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config.paths import apply_process_cache_env  # noqa: E402

apply_process_cache_env()

from controllers.rule_based_controller import RuleBasedController  # noqa: E402
from envs.power_system_env import PowerSystemEnv  # noqa: E402
from safety import GiveSafeController, NoSafeActionFoundError, load_givesafe_config  # noqa: E402
from training.evaluate_td3 import evaluate_policy  # noqa: E402
from training.ghtd3.agent import GHTD3Agent  # noqa: E402
from training.ghtd3.train import GHTD3PolicyWrapper, load_ghtd3_config  # noqa: E402
from training.hybrid_td3.algorithm import HybridTD3  # noqa: E402
from training.hybrid_td3.train import annual_episode_start_seconds  # noqa: E402

REMOTE = ROOT / "runs" / "remote_840k"
OUT_DIR = REMOTE / "seasonal_eval"
SEASONS = ["winter", "transition", "summer"]
SEASON_WEEK = {"winter": 0, "transition": 13, "summer": 26}


def _start(env: PowerSystemEnv, season: str) -> float:
    return float(
        annual_episode_start_seconds(
            env.config["fmu"], env.episode_steps, SEASON_WEEK[season]
        )
    )


def _slim(res: dict) -> dict:
    terms = res.get("cost_terms") or {}
    metrics = res.get("metrics") or {}
    return {
        "episode_reward": res.get("episode_reward"),
        "terminal_soc_satisfied": bool(res.get("terminal_soc_satisfied")),
        "terminal_soc_l1": terms.get("terminal_soc_l1_error"),
        "economic_reward": terms.get("economic_reward"),
        "generalized_cashflow_delta": terms.get("generalized_cashflow_delta"),
        "carbon_cost_cny": terms.get("carbon_cost_cny"),
        "unserved_energy_mwh": terms.get("unserved_energy_mwh")
        or metrics.get("unserved_energy_mwh"),
        "battery_throughput_mwh": metrics.get("battery_throughput_mwh"),
        "caes_throughput_mwh": metrics.get("caes_throughput_mwh"),
        "thermal_generation_mwh": metrics.get("thermal_generation_mwh"),
    }


def eval_rule(start: float) -> dict:
    env = PowerSystemEnv(run_id=f"s840_rule_{int(start)}", forecast_enabled=True)
    pol = RuleBasedController(env)
    res = evaluate_policy(env, pol, None, reset_options={"start_time": start})
    env.close()
    return _slim(res)


def _fail(msg: str) -> dict:
    return {
        "episode_reward": float("nan"),
        "terminal_soc_satisfied": False,
        "terminal_soc_l1": None,
        "economic_reward": None,
        "generalized_cashflow_delta": None,
        "carbon_cost_cny": None,
        "unserved_energy_mwh": None,
        "battery_throughput_mwh": None,
        "caes_throughput_mwh": None,
        "thermal_generation_mwh": None,
        "error": msg,
    }


def eval_td3(ckpt: Path, start: float) -> dict:
    env = PowerSystemEnv(run_id=f"s840_td3_{ckpt.parent.name}_{int(start)}", forecast_enabled=True)
    try:
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

        res = evaluate_policy(env, Pol(), None, reset_options={"start_time": start})
        return _slim(res)
    except Exception as exc:  # noqa: BLE001 — keep seasonal matrix complete
        return _fail(f"{type(exc).__name__}: {exc}")
    finally:
        env.close()


def eval_hmsd(ckpt: Path, config_path: Path, start: float) -> dict:
    env = PowerSystemEnv(run_id=f"s840_hmsd_{ckpt.parent.name}_{int(start)}", forecast_enabled=True)
    try:
        dim = int(np.prod(env.observation_space.shape))
        cfg = dict(load_ghtd3_config(config_path).get("ghtd3") or {})
        cfg["execution_mode"] = "goal_conditioned"
        agent = GHTD3Agent(dim, cfg)
        agent.load(ckpt, strict=False)
        agent.execution_mode = "goal_conditioned"
        gs = load_givesafe_config(ROOT / "src/config/givesafe_config.yaml")
        ctrl = GiveSafeController(oracle=env.oracle, shadow=None, config=gs)
        pol = GHTD3PolicyWrapper(agent, env, ctrl, cfg)
        res = evaluate_policy(env, pol, None, reset_options={"start_time": start})
        return _slim(res)
    except Exception as exc:  # noqa: BLE001
        return _fail(f"{type(exc).__name__}: {exc}")
    finally:
        env.close()


def mean_std(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return m, math.sqrt(v)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env_meta = PowerSystemEnv(run_id="s840_meta", forecast_enabled=True)
    starts = {s: _start(env_meta, s) for s in SEASONS}
    env_meta.close()
    print("season starts (s):", starts, flush=True)

    # B0 once per season
    b0_by_season: dict[str, dict] = {}
    for season in SEASONS:
        print(f"[B0] {season}", flush=True)
        b0_by_season[season] = eval_rule(starts[season])
        print(b0_by_season[season], flush=True)

    rows: list[dict] = []
    for seed in (0, 1, 2):
        hmsd_dir = REMOTE / f"ghtd3_abs_s{seed}"
        td3_dir = REMOTE / f"td3_scratch_s{seed}"
        hmsd_ckpt = hmsd_dir / "checkpoints" / "ghtd3.pt"
        td3_ckpt = td3_dir / "checkpoints" / "hybrid_givesafe_td3.pt"
        if not td3_ckpt.is_file():
            # alternate names
            cands = list((td3_dir / "checkpoints").glob("*.pt")) if (td3_dir / "checkpoints").is_dir() else []
            td3_ckpt = cands[0] if cands else td3_ckpt
        hmsd_cfg = hmsd_dir / "config" / "ghtd3_config.yaml"
        if not hmsd_cfg.is_file():
            hmsd_cfg = ROOT / "src/config/ghtd3_config_abs.yaml"

        for season in SEASONS:
            start = starts[season]
            print(f"[seed {seed}] {season} HMSD", flush=True)
            gh = eval_hmsd(hmsd_ckpt, hmsd_cfg, start)
            print(f"[seed {seed}] {season} TD3", flush=True)
            td = eval_td3(td3_ckpt, start)
            def _rf(x):
                try:
                    v = float(x)
                    return v if math.isfinite(v) else float("nan")
                except Exception:
                    return float("nan")

            dlt = _rf(gh.get("episode_reward")) - _rf(td.get("episode_reward"))
            row = {
                "seed": seed,
                "season": season,
                "start_time": start,
                "b0": b0_by_season[season],
                "hmsd": gh,
                "td3": td,
                "delta_hmsd_minus_td3": dlt,
            }
            rows.append(row)
            print(
                f"  reward HMSD={_rf(gh.get('episode_reward')):.2f} "
                f"soc={gh.get('terminal_soc_satisfied')} err={gh.get('error')} | "
                f"TD3={_rf(td.get('episode_reward')):.2f} "
                f"soc={td.get('terminal_soc_satisfied')} err={td.get('error')} | "
                f"Δ={dlt:+.2f}",
                flush=True,
            )
            # incremental save
            (OUT_DIR / "rows_partial.json").write_text(
                json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    # aggregate mean over seeds for each season
    summary: dict = {"by_season": {}, "overall": {}}
    for season in SEASONS:
        sub = [r for r in rows if r["season"] == season]
        for method, key in (("hmsd", "hmsd"), ("td3", "td3"), ("b0", "b0")):
            rews = []
            for r in sub:
                try:
                    v = float(r[key]["episode_reward"])
                    if math.isfinite(v):
                        rews.append(v)
                except Exception:
                    pass
            soc = sum(1 for r in sub if r[key].get("terminal_soc_satisfied"))
            fails = sum(1 for r in sub if r[key].get("error"))
            if rews:
                m, sd = mean_std(rews)
            else:
                m, sd = float("nan"), float("nan")
            summary["by_season"].setdefault(season, {})[method] = {
                "reward_mean": m,
                "reward_std": sd,
                "soc_pass": f"{soc}/{len(sub)}",
                "n_valid_reward": len(rews),
                "n_error": fails,
            }
        hm = summary["by_season"][season]["hmsd"]["reward_mean"]
        tm = summary["by_season"][season]["td3"]["reward_mean"]
        summary["by_season"][season]["delta_hmsd_minus_td3"] = hm - tm

    # overall: average of three seasonal means (paper-style three-season mean)
    for method in ("hmsd", "td3", "b0"):
        seas_means = [
            summary["by_season"][s][method]["reward_mean"]
            for s in SEASONS
            if math.isfinite(summary["by_season"][s][method]["reward_mean"])
        ]
        m, sd = (mean_std(seas_means) if seas_means else (float("nan"), float("nan")))
        seed_avgs = []
        if method == "b0":
            seed_avgs = seas_means
        else:
            for seed in (0, 1, 2):
                rs = []
                for r in rows:
                    if r["seed"] != seed:
                        continue
                    try:
                        v = float(r[method]["episode_reward"])
                        if math.isfinite(v):
                            rs.append(v)
                    except Exception:
                        pass
                if rs:
                    seed_avgs.append(sum(rs) / len(rs))
        sm, ssd = (mean_std(seed_avgs) if seed_avgs else (float("nan"), float("nan")))
        summary["overall"][method] = {
            "three_season_mean_of_season_means": m,
            "seed_three_season_avg_mean": sm,
            "seed_three_season_avg_std": ssd,
        }

    out = {
        "protocol": {
            "seasons": SEASONS,
            "week_indices": SEASON_WEEK,
            "episode_hours": 168,
            "note": "Paper-aligned seasonal week indices via annual_episode_start_seconds",
        },
        "rows": rows,
        "summary": summary,
    }
    out_path = OUT_DIR / "seasonal_table.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", out_path, flush=True)

    # markdown brief
    md = ["# Remote 840k three-season evaluation", ""]
    md.append("| Season | B0 | TD3 mean±std (SoC) | HMSD mean±std (SoC) | Δ(HMSD−TD3) |")
    md.append("|--------|----|--------------------|---------------------|-------------|")
    for season in SEASONS:
        s = summary["by_season"][season]
        md.append(
            f"| {season} | {s['b0']['reward_mean']:.1f} | "
            f"{s['td3']['reward_mean']:.1f}±{s['td3']['reward_std']:.1f} ({s['td3']['soc_pass']}) | "
            f"{s['hmsd']['reward_mean']:.1f}±{s['hmsd']['reward_std']:.1f} ({s['hmsd']['soc_pass']}) | "
            f"{s['delta_hmsd_minus_td3']:+.1f} |"
        )
    md.append("")
    md.append("## Overall (mean of three seasonal means)")
    for method in ("b0", "td3", "hmsd"):
        o = summary["overall"][method]
        md.append(
            f"- **{method.upper()}**: season-mean average = {o['three_season_mean_of_season_means']:.2f}"
        )
    (OUT_DIR / "seasonal_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
