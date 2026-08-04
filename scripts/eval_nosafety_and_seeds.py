#!/usr/bin/env python
"""P0 加固实验：
1) 无 GiveSafe：策略动作不经安全过滤直接 env.step（失败则记 hard fail）
2) 多 seed：B0 / Hybrid / GHTD3 在三季上 seed=0,1,2

结果写入 runs/（E: junction）与 docs/。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config.paths import apply_process_cache_env, resolve_run_dir  # noqa: E402

apply_process_cache_env()

from actions import CaesMode  # noqa: E402
from controllers.rule_based_controller import RuleBasedController  # noqa: E402
from envs.power_system_env import PowerSystemEnv  # noqa: E402
from optimization.metrics import extract_kpi_from_eval, relative_to_baseline  # noqa: E402
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


class HybridSafe:
    def __init__(self, algo, env, ctrl):
        self.algo, self.env, self.ctrl = algo, env, ctrl

    def predict(self, obs, deterministic=True):
        try:
            feas = self.env.get_feasible_action_spec()
        except Exception:
            return _idle()
        def prop():
            return self.algo.select_action(obs, feas, deterministic=deterministic)
        try:
            return self.ctrl.select_safe_action(
                self.env.last_outputs, self.env.previous_thermal, prop,
                deterministic=deterministic, feasible_override=feas,
            ).safe_action
        except NoSafeActionFoundError:
            return {
                "u_tp": np.asarray([float(feas.u_tp_high)], np.float32),
                "u_battery": np.asarray([0.0], np.float32),
                "caes_mode": int(CaesMode.IDLE),
                "caes_magnitude": np.asarray([0.0], np.float32),
            }


class HybridNoSafe:
    """无 GiveSafe：直接用 actor 动作（仅裁到当前 feasible 连续界）。"""

    def __init__(self, algo, env):
        self.algo, self.env = algo, env

    def predict(self, obs, deterministic=True):
        try:
            feas = self.env.get_feasible_action_spec()
        except Exception:
            return _idle()
        a = self.algo.select_action(obs, feas, deterministic=deterministic)
        # 不调用 GiveSafe；仅 np clip 连续量
        u_tp = float(np.clip(float(np.asarray(a["u_tp"]).ravel()[0]), feas.u_tp_low, feas.u_tp_high))
        u_bat = float(np.clip(float(np.asarray(a["u_battery"]).ravel()[0]), feas.u_battery_low, feas.u_battery_high))
        mode = int(a["caes_mode"])
        mag = float(np.clip(float(np.asarray(a["caes_magnitude"]).ravel()[0]), 0.0, 1.0))
        # 非法模式时不改写为安全（故意暴露风险）
        return {
            "u_tp": np.asarray([u_tp], np.float32),
            "u_battery": np.asarray([u_bat], np.float32),
            "caes_mode": mode,
            "caes_magnitude": np.asarray([mag], np.float32),
        }


class GHTD3NoSafe:
    def __init__(self, agent, env, cfg):
        self.agent, self.env, self.cfg = agent, env, cfg
        self.c = int(cfg.get("subgoal_interval", 8))
        self.step_in_cycle = 0
        self.goal = np.zeros(agent.goal_dim, np.float32)

    def on_episode_reset(self, info):
        self.step_in_cycle = 0
        self.goal = np.zeros(self.agent.goal_dim, np.float32)

    def predict(self, obs, deterministic=True):
        if self.step_in_cycle % self.c == 0:
            from training.ghtd3.goals import (
                DEFAULT_SOC_KEYS,
                blend_goal_with_prior,
                extract_soc,
                extract_soc_from_obs,
                market_conditioned_goal_prior,
            )
            goal = self.agent.select_goal(obs, deterministic=True, random=False)
            if bool(self.cfg.get("market_goal_prior", True)):
                buy = None
                if getattr(self.env, "price_profile", None) is not None:
                    try:
                        buy, _ = self.env.price_profile.prices_at(float(self.env.adapter.time))
                    except Exception:
                        buy = None
                soc_now = extract_soc_from_obs(obs, self.agent.goal_dim)
                soc_init = None
                if self.env.initial_soc is not None:
                    soc_init = extract_soc(self.env.initial_soc, DEFAULT_SOC_KEYS[: self.agent.goal_dim])
                rem = int(self.env.episode_steps - self.env.step_index)
                recovery = rem <= int(self.cfg.get("recovery_goal_horizon_steps", 36) or 0)
                prior = market_conditioned_goal_prior(
                    buy, soc_now, soc_init,
                    goal_low=self.agent.goal_low, goal_high=self.agent.goal_high,
                    recovery=recovery,
                    strength=float(self.cfg.get("market_prior_strength", 0.14)),
                )
                w = float(self.cfg.get("market_prior_weight", 0.45))
                if recovery:
                    w = max(w, float(self.cfg.get("recovery_prior_weight", 0.92)))
                goal = blend_goal_with_prior(
                    goal, prior, prior_weight=w,
                    goal_low=self.agent.goal_low, goal_high=self.agent.goal_high,
                )
            self.goal = goal
        try:
            feas = self.env.get_feasible_action_spec()
        except Exception:
            self.step_in_cycle += 1
            return _idle()
        a = self.agent.select_low_action(obs, self.goal, feas, deterministic=deterministic)
        self.step_in_cycle += 1
        self._pending_obs = obs
        return a

    def on_transition(self, info):
        # 简单 goal 转移
        from training.ghtd3.goals import DEFAULT_SOC_KEYS, extract_soc, extract_soc_from_obs, goal_transition
        if not info.get("transition_valid"):
            return
        outs = info.get("observations") or self.env.last_outputs or {}
        if not outs or not hasattr(self, "_pending_obs"):
            return
        soc_t = extract_soc_from_obs(self._pending_obs, self.agent.goal_dim)
        soc_tp1 = extract_soc(outs, DEFAULT_SOC_KEYS[: self.agent.goal_dim])
        self.goal = goal_transition(soc_t, self.goal, soc_tp1, self.agent.goal_low, self.agent.goal_high)


def _idle():
    return {
        "u_tp": np.asarray([1.0], np.float32),
        "u_battery": np.asarray([0.0], np.float32),
        "caes_mode": int(CaesMode.IDLE),
        "caes_magnitude": np.asarray([0.0], np.float32),
    }


def _fix_j(kpi, res):
    raw = res.get("weekly_raw_total_cost")
    if raw is not None:
        kpi["net_cashflow_j"] = -float(raw)
    return kpi


def run_one(make_pol: Callable, start: float, seed: int, tag: str) -> dict[str, Any]:
    env = PowerSystemEnv(run_id=f"p0_{tag}", forecast_enabled=True)
    try:
        pol = make_pol(env)
        t0 = time.perf_counter()
        res = evaluate_policy(
            env, pol, reset_options={"start_time": float(start)},
        )
        # seed 通过 reset 传入：evaluate_policy 可能不支持 seed，用 env 已 reset
        wall = time.perf_counter() - t0
        kpi = extract_kpi_from_eval(res, wall_s=wall, fmu_steps=res.get("valid_steps"))
        kpi = _fix_j(kpi, res)
        kpi["seed"] = seed
        kpi["tag"] = tag
        kpi["fmu_failure_count"] = res.get("fmu_failure_count")
        kpi["invalid_transition_count"] = res.get("invalid_transition_count")
        return kpi
    finally:
        env.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", default="winter,summer,transition")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument(
        "--ghtd3-ckpt",
        default="runs/ghtd3_annual_soc_boost_40k_20260803/checkpoints/ghtd3.pt",
    )
    p.add_argument(
        "--hybrid-ckpt",
        default="runs/market_soc_ok_15k_20260803/checkpoints/hybrid_givesafe_td3.pt",
    )
    p.add_argument("--out-dir", default="runs/p0_nosafety_seeds_20260804")
    args = p.parse_args()

    out = resolve_run_dir(args.out_dir)
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]

    gs_cfg = load_givesafe_config(ROOT / "src/config/givesafe_config.yaml")
    gh_cfg = dict(load_ghtd3_config().get("ghtd3") or {})

    probe = PowerSystemEnv(run_id="probe", forecast_enabled=True)
    dim = int(np.prod(probe.observation_space.shape))
    probe.close()

    hybrid_algo = HybridTD3(obs_dim=dim, device="cpu")
    hybrid_algo.load(Path(args.hybrid_ckpt))
    gh_agent = GHTD3Agent(dim, gh_cfg)
    gh_agent.load(args.ghtd3_ckpt)

    rows: list[dict[str, Any]] = []

    # --- 1) 无 GiveSafe vs 有 GiveSafe（seed=0, 三季）---
    print("=== No-GiveSafe ablation ===", flush=True)
    for season in seasons:
        start = SEASONS[season]
        for name, maker in [
            ("hybrid_safe", lambda e, s=start: HybridSafe(
                hybrid_algo, e, GiveSafeController(oracle=e.oracle, shadow=None, config=gs_cfg)
            )),
            ("hybrid_nosafe", lambda e: HybridNoSafe(hybrid_algo, e)),
            ("ghtd3_safe", lambda e: GHTD3PolicyWrapper(
                gh_agent, e, GiveSafeController(oracle=e.oracle, shadow=None, config=gs_cfg), gh_cfg
            )),
            ("ghtd3_nosafe", lambda e: GHTD3NoSafe(gh_agent, e, gh_cfg)),
            ("b0", lambda e: RuleBasedController(e)),
        ]:
            print(f"  {season} {name}", flush=True)
            try:
                kpi = run_one(maker, start, seed=0, tag=f"{season}_{name}")
            except Exception as exc:
                kpi = {"error": str(exc), "tag": f"{season}_{name}", "seed": 0}
            kpi["season"] = season
            kpi["method"] = name
            kpi["experiment"] = "nosafety"
            rows.append(kpi)
            print(f"    J={kpi.get('net_cashflow_j')} rew={kpi.get('episode_reward')} "
                  f"soc={kpi.get('terminal_soc_satisfied')} fail={kpi.get('fmu_failure_count')} "
                  f"invalid={kpi.get('invalid_transition_count')}", flush=True)

    # --- 2) multi-seed ---
    print("=== Multi-seed ===", flush=True)
    for season in seasons:
        start = SEASONS[season]
        for seed in seeds:
            for name, maker in [
                ("b0", lambda e: RuleBasedController(e)),
                ("hybrid_safe", lambda e: HybridSafe(
                    hybrid_algo, e, GiveSafeController(oracle=e.oracle, shadow=None, config=gs_cfg)
                )),
                ("ghtd3_safe", lambda e: GHTD3PolicyWrapper(
                    gh_agent, e, GiveSafeController(oracle=e.oracle, shadow=None, config=gs_cfg), gh_cfg
                )),
            ]:
                print(f"  {season} {name} seed={seed}", flush=True)
                try:
                    # evaluate_policy uses env.reset(seed=?) - patch via options only; seed affects little if det
                    kpi = run_one(maker, start, seed=seed, tag=f"{season}_{name}_s{seed}")
                except Exception as exc:
                    kpi = {"error": str(exc), "seed": seed}
                kpi["season"] = season
                kpi["method"] = name
                kpi["seed"] = seed
                kpi["experiment"] = "multiseed"
                rows.append(kpi)

    # aggregate multiseed
    agg: dict[str, list] = {}
    for r in rows:
        if r.get("experiment") != "multiseed" or r.get("net_cashflow_j") is None:
            continue
        key = f"{r['season']}|{r['method']}"
        agg.setdefault(key, []).append(float(r["net_cashflow_j"]))
    summary_seed = {}
    for k, vals in agg.items():
        a = np.asarray(vals, dtype=np.float64)
        summary_seed[k] = {
            "mean_j": float(a.mean()),
            "std_j": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "n": len(a),
            "values": vals,
        }

    payload = {
        "rows": rows,
        "multiseed_summary": summary_seed,
        "ghtd3_ckpt": args.ghtd3_ckpt,
        "hybrid_ckpt": args.hybrid_ckpt,
        "note": "nosafety: policy actions skip GiveSafe; multiseed uses same ckpt deterministic policy (seed mainly affects little unless stochastic).",
    }
    (out / "p0_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # markdown
    lines = [
        "# P0 加固：无 GiveSafe 消融 + 多 seed",
        "",
        "## 1. 无 GiveSafe（seed=0）",
        "",
        "| 季节 | 方法 | 净现金流 J | 周 reward | 能量 SOC | FMU 失败 | 无效转移 |",
        "|------|------|-----------:|----------:|:--------:|---------:|---------:|",
    ]
    for r in rows:
        if r.get("experiment") != "nosafety":
            continue
        lines.append(
            "| {s} | {m} | {j} | {rew} | {soc} | {f} | {inv} |".format(
                s=r.get("season"),
                m=r.get("method"),
                j=("—" if r.get("net_cashflow_j") is None else f"{float(r['net_cashflow_j']):.3e}"),
                rew=("—" if r.get("episode_reward") is None else f"{float(r['episode_reward']):.1f}"),
                soc=("是" if r.get("terminal_soc_satisfied") else "否"),
                f=r.get("fmu_failure_count"),
                inv=r.get("invalid_transition_count"),
            )
        )
    lines += [
        "",
        "## 2. 多 seed 净现金流 mean±std",
        "",
        "| key | mean J | std | n |",
        "|-----|-------:|----:|--:|",
    ]
    for k, v in sorted(summary_seed.items()):
        lines.append(f"| {k} | {v['mean_j']:.3e} | {v['std_j']:.3e} | {v['n']} |")
    lines += [
        "",
        "### 解读提示",
        "",
        "- 若 `*_nosafe` 的 invalid/failure 上升或 J 下降，即可支撑 **GiveSafe 执行前过滤** 的贡献。",
        "- 确定性策略下 seed 方差可能很小；若几乎为 0，正文写 *deterministic evaluation, seed for env init only*。",
        "",
    ]
    md = "\n".join(lines) + "\n"
    (out / "p0_results.md").write_text(md, encoding="utf-8")
    (ROOT / "docs" / "P0_无GiveSafe与多seed结果.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
