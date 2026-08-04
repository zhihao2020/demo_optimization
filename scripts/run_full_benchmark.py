#!/usr/bin/env python
"""全方法基准：B0/B1/LP/PSO/Hybrid/GHTD3 三季对比 + 相对原始运行变化。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from actions import CaesMode  # noqa: E402
from controllers.price_aware_rule import PriceAwareRuleController  # noqa: E402
from controllers.rule_based_controller import RuleBasedController  # noqa: E402
from envs.power_system_env import PowerSystemEnv  # noqa: E402
from optimization.metrics import extract_kpi_from_eval, relative_to_baseline  # noqa: E402
from optimization.pso_fmu import PSOConfig, run_pso  # noqa: E402
from optimization.rolling_lp import RollingLPController  # noqa: E402
from safety import GiveSafeController, NoSafeActionFoundError, load_givesafe_config  # noqa: E402
from training.evaluate_td3 import evaluate_policy  # noqa: E402
from training.ghtd3.agent import GHTD3Agent  # noqa: E402
from training.ghtd3.train import GHTD3PolicyWrapper, load_ghtd3_config  # noqa: E402
from training.hybrid_td3.algorithm import HybridTD3  # noqa: E402

SEASONS = {
    "winter": 0.0,
    "summer": 180 * 24 * 3600.0,
    "transition": 90 * 24 * 3600.0,
}


class HybridPolicy:
    def __init__(self, algo, env, ctrl):
        self.algo, self.env, self.ctrl = algo, env, ctrl

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
            return self.algo.select_action(obs, feas, deterministic=deterministic)

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
                "caes_mode": 1,
                "caes_magnitude": np.asarray([0.0], np.float32),
            }


def _fix_j(kpi: dict[str, Any], res: dict[str, Any]) -> dict[str, Any]:
    raw = res.get("weekly_raw_total_cost")
    if raw is not None:
        kpi["net_cashflow_j"] = -float(raw)
    return kpi


def eval_policy_method(
    name: str,
    make_pol: Callable,
    start_time: float,
    out_csv: Path | None,
) -> dict[str, Any]:
    env = PowerSystemEnv(run_id=f"bm_{name}", forecast_enabled=True)
    try:
        pol = make_pol(env)
        t0 = time.perf_counter()
        res = evaluate_policy(env, pol, output_csv=out_csv, reset_options={"start_time": start_time})
        wall = time.perf_counter() - t0
        kpi = extract_kpi_from_eval(res, wall_s=wall, fmu_steps=res.get("valid_steps"))
        kpi = _fix_j(kpi, res)
        kpi["method"] = name
        return kpi
    finally:
        env.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", type=str, default="winter,summer,transition")
    p.add_argument("--methods", type=str, default="b0,b1,lp,hybrid,ghtd3")
    p.add_argument("--pso-iters", type=int, default=15)
    p.add_argument("--pso-pop", type=int, default=10)
    p.add_argument(
        "--ghtd3-ckpt",
        default="runs/ghtd3_annual_soc_boost_40k_20260803/checkpoints/ghtd3.pt",
    )
    p.add_argument(
        "--hybrid-ckpt",
        default="runs/market_soc_ok_15k_20260803/checkpoints/hybrid_givesafe_td3.pt",
    )
    p.add_argument("--out-dir", default="runs/benchmark_full_20260804")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "trajectories").mkdir(exist_ok=True)

    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    gs_cfg = load_givesafe_config(ROOT / "src/config/givesafe_config.yaml")
    gh_cfg = dict(load_ghtd3_config().get("ghtd3") or {})

    hybrid_algo = None
    gh_agent = None
    if "hybrid" in methods:
        probe = PowerSystemEnv(run_id="probe", forecast_enabled=True)
        dim = int(np.prod(probe.observation_space.shape))
        probe.close()
        hybrid_algo = HybridTD3(obs_dim=dim, device="cpu")
        hybrid_algo.load(Path(args.hybrid_ckpt))
    if "ghtd3" in methods:
        probe = PowerSystemEnv(run_id="probe2", forecast_enabled=True)
        dim = int(np.prod(probe.observation_space.shape))
        probe.close()
        gh_agent = GHTD3Agent(dim, gh_cfg)
        gh_agent.load(args.ghtd3_ckpt)

    rows: list[dict[str, Any]] = []
    for season in seasons:
        start = SEASONS[season]
        b0_kpi = None
        for name in methods:
            print(f"=== {season} × {name} ===")
            csv_path = out / "trajectories" / f"{season}_{name}.csv"
            if name == "b0":
                kpi = eval_policy_method(
                    name, lambda e: RuleBasedController(e), start, csv_path
                )
            elif name == "b1":
                kpi = eval_policy_method(
                    name, lambda e: PriceAwareRuleController(e), start, csv_path
                )
            elif name == "lp":
                kpi = eval_policy_method(
                    name, lambda e: RollingLPController(e), start, csv_path
                )
            elif name == "hybrid":
                kpi = eval_policy_method(
                    name,
                    lambda e: HybridPolicy(
                        hybrid_algo,
                        e,
                        GiveSafeController(oracle=e.oracle, shadow=None, config=gs_cfg),
                    ),
                    start,
                    csv_path,
                )
            elif name == "ghtd3":
                kpi = eval_policy_method(
                    name,
                    lambda e: GHTD3PolicyWrapper(
                        gh_agent,
                        e,
                        GiveSafeController(oracle=e.oracle, shadow=None, config=gs_cfg),
                        gh_cfg,
                    ),
                    start,
                    csv_path,
                )
            elif name == "pso":
                kpi = run_pso(
                    start_time=start,
                    cfg=PSOConfig(
                        n_particles=args.pso_pop,
                        n_iters=args.pso_iters,
                        seed=0,
                    ),
                )
                kpi["method"] = "pso"
            else:
                raise ValueError(name)
            kpi["season"] = season
            kpi["start_time"] = start
            if name == "b0":
                b0_kpi = dict(kpi)
            if b0_kpi is not None and name != "b0":
                kpi["vs_b0"] = relative_to_baseline(kpi, b0_kpi)
            rows.append(kpi)
            print(
                f"  J={kpi.get('net_cashflow_j')} reward={kpi.get('episode_reward')} "
                f"soc={kpi.get('terminal_soc_satisfied')} curt={kpi.get('curtailment_mwh')}"
            )

    summary = {
        "seasons": {k: SEASONS[k] for k in seasons},
        "methods": methods,
        "rows": rows,
        "ghtd3_ckpt": args.ghtd3_ckpt,
        "hybrid_ckpt": args.hybrid_ckpt,
    }
    (out / "benchmark_table.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    lines = [
        "# 全方法基准对比（相对 B0 原始保守运行）",
        "",
        "| 季节 | 方法 | 净现金流 J | ΔJ vs B0 | 周 reward | SOC | 弃电 MWh | 火电 MWh | 电池吞吐 | CAES吞吐 | 墙钟s |",
        "|------|------|------------|----------|-----------|-----|----------|----------|----------|----------|-------|",
    ]
    for r in rows:
        vs = r.get("vs_b0") or {}
        lines.append(
            "| {season} | {method} | {j:.3e} | {dj} | {rew:.1f} | {soc} | {cu:.2f} | {th:.0f} | {bt:.0f} | {ct:.0f} | {w:.1f} |".format(
                season=r.get("season"),
                method=r.get("method"),
                j=float(r.get("net_cashflow_j") or 0),
                dj=("—" if vs.get("delta_j_vs_b0") is None else f"{vs['delta_j_vs_b0']:.3e}"),
                rew=float(r.get("episode_reward") or 0),
                soc="是" if r.get("terminal_soc_satisfied") else "否",
                cu=float(r.get("curtailment_mwh") or 0),
                th=float(r.get("thermal_mwh") or 0),
                bt=float(r.get("battery_throughput_mwh") or 0),
                ct=float(r.get("caes_throughput_mwh") or 0),
                w=float(r.get("wall_s") or r.get("wall_s_search") or 0),
            )
        )
    lines.append("")
    md = "\n".join(lines) + "\n"
    (out / "benchmark_table.md").write_text(md, encoding="utf-8")
    (ROOT / "docs" / "全方法基准对比结果.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
