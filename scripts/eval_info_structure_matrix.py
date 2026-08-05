#!/usr/bin/env python
"""P1-1 信息结构矩阵：price × resource forecast × method × season。

不修改仓库内 env_config.yaml；使用临时 config_path。
物理真值与 FMU 不变；电价结算在 predicted 模式下用 price_realized。
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config.paths import apply_process_cache_env, resolve_run_dir  # noqa: E402

apply_process_cache_env()

from actions import CaesMode  # noqa: E402
from controllers.rule_based_controller import RuleBasedController  # noqa: E402
from envs.power_system_env import PowerSystemEnv  # noqa: E402
from optimization.metrics import extract_kpi_from_eval  # noqa: E402
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


class GiveSafeWrappedPolicy:
    def __init__(self, agent, env, controller):
        self.agent, self.env, self.controller = agent, env, controller

    def predict(self, obs, deterministic=True):
        try:
            feas = self.env.get_feasible_action_spec()
        except Exception:
            return {
                "u_tp": np.asarray([1.0], np.float32),
                "u_battery": np.asarray([0.0], np.float32),
                "caes_mode": int(CaesMode.IDLE),
                "caes_magnitude": np.asarray([0.0], np.float32),
            }

        def prop():
            return self.agent.select_action(obs, feas, deterministic=deterministic)

        try:
            return self.controller.select_safe_action(
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
                "caes_mode": int(CaesMode.IDLE),
                "caes_magnitude": np.asarray([0.0], np.float32),
            }


def _write_temp_env_config(price_mode: str) -> Path:
    src = ROOT / "src/config/env_config.yaml"
    cfg = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    market = dict(cfg.get("market") or {})
    market["available"] = True
    if price_mode == "perfect":
        market["price_path"] = "data/price_tou.csv"
        market["obs_price_path"] = None
    elif price_mode == "predicted":
        if not (ROOT / "data/price_realized.csv").is_file():
            raise FileNotFoundError("data/price_realized.csv missing; run price residual + bilstm")
        if not (ROOT / "data/price_predicted.csv").is_file():
            raise FileNotFoundError("data/price_predicted.csv missing; run train_price_bilstm")
        market["price_path"] = "data/price_realized.csv"
        market["obs_price_path"] = "data/price_predicted.csv"
    else:
        raise ValueError(price_mode)
    cfg["market"] = market
    # forecast.mode 由 PowerSystemEnv(forecast_mode=...) 覆盖，此处保持 perfect 底稿
    fc = dict(cfg.get("forecast") or {})
    fc["mode"] = "perfect"
    cfg["forecast"] = fc
    tmp = Path(tempfile.mkdtemp(prefix="info_struct_"))
    out = tmp / "env_config.yaml"
    out.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return out


def build_policy(method: str, env: PowerSystemEnv, args: argparse.Namespace) -> Any:
    gs_cfg = load_givesafe_config(ROOT / "src/config/givesafe_config.yaml")
    ctrl = GiveSafeController(oracle=env.oracle, shadow=None, config=gs_cfg)
    dim = int(np.prod(env.observation_space.shape))
    gamma = float(env.reward_calculator.config.get("gamma", 0.99))
    if method == "b0":
        return RuleBasedController(env)
    if method == "hybrid":
        agent = HybridTD3(obs_dim=dim, gamma=gamma)
        agent.load(Path(args.hybrid_ckpt))
        return GiveSafeWrappedPolicy(agent, env, ctrl)
    if method == "ghtd3":
        full_cfg = load_ghtd3_config(ROOT / "src/config/ghtd3_config.yaml")
        cfg = dict(full_cfg.get("ghtd3") or full_cfg)
        agent = GHTD3Agent(dim, cfg)
        agent.load(Path(args.ghtd3_ckpt))
        return GHTD3PolicyWrapper(agent, env, ctrl, cfg)
    raise ValueError(method)


def main() -> None:
    p = argparse.ArgumentParser(description="Information structure matrix evaluation")
    p.add_argument("--methods", type=str, default="ghtd3,hybrid")
    p.add_argument("--seasons", type=str, default="winter,summer,transition")
    p.add_argument("--price-modes", type=str, default="perfect,predicted")
    p.add_argument("--resource-modes", type=str, default="perfect,noisy,predicted")
    p.add_argument("--noise-sigma", type=float, default=0.10)
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument(
        "--hybrid-ckpt",
        default="runs/market_soc_ok_15k_20260803/checkpoints/hybrid_givesafe_td3.pt",
    )
    p.add_argument(
        "--ghtd3-ckpt",
        default="runs/ghtd3_annual_soc_boost_40k_20260803/checkpoints/ghtd3.pt",
    )
    p.add_argument("--out-dir", type=str, default="runs/info_structure_matrix")
    p.add_argument("--docs-md", type=str, default="docs/信息结构实验结果.md")
    args = p.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    price_modes = [m.strip() for m in args.price_modes.split(",") if m.strip()]
    resource_modes = [m.strip() for m in args.resource_modes.split(",") if m.strip()]
    out = resolve_run_dir(args.out_dir)
    rows: list[dict[str, Any]] = []

    for season in seasons:
        start = SEASONS[season]
        for price_mode in price_modes:
            cfg_path = _write_temp_env_config(price_mode)
            for resource_mode in resource_modes:
                if resource_mode == "predicted":
                    pred = ROOT / "data/resource_predicted/winds.csv"
                    if not pred.is_file():
                        print(f"[skip] resource predicted missing: {pred}", flush=True)
                        continue
                for method in methods:
                    tag = f"{season}|p={price_mode}|r={resource_mode}|{method}"
                    print(f"=== {tag} ===", flush=True)
                    sigma = None
                    if resource_mode == "noisy":
                        sigma = {
                            "wind": float(args.noise_sigma),
                            "irradiance": float(args.noise_sigma),
                            "ambient_temperature": 0.0,
                            "planned_load": float(args.noise_sigma) * 0.8,
                        }
                    env = PowerSystemEnv(
                        config_path=cfg_path,
                        run_id=f"info_{season}_{price_mode}_{resource_mode}_{method}",
                        forecast_enabled=True,
                        forecast_mode=resource_mode,
                        forecast_noise_seed=int(args.noise_seed),
                        forecast_noise_sigma=sigma,
                    )
                    try:
                        pol = build_policy(method, env, args)
                        t0 = time.perf_counter()
                        res = evaluate_policy(
                            env, pol, reset_options={"start_time": float(start)}
                        )
                        wall = time.perf_counter() - t0
                        kpi = extract_kpi_from_eval(
                            res, wall_s=wall, fmu_steps=res.get("valid_steps")
                        )
                        raw = res.get("weekly_raw_total_cost")
                        if raw is not None:
                            kpi["net_cashflow_j"] = -float(raw)
                        kpi.update(
                            {
                                "season": season,
                                "price_mode": price_mode,
                                "resource_mode": resource_mode,
                                "method": method,
                                "provider_mode": getattr(
                                    env.forecast_provider, "mode", None
                                ),
                            }
                        )
                        rows.append(kpi)
                        print(
                            f"  J={kpi.get('net_cashflow_j')} rew={kpi.get('episode_reward')} "
                            f"soc={kpi.get('terminal_soc_satisfied')}",
                            flush=True,
                        )
                    finally:
                        env.close()

    payload = {
        "rows": rows,
        "note": (
            "Observation information structure only. FMU physics unchanged. "
            "price predicted: settle realized + obs predicted. "
            "resource predicted: residual BiLSTM CSV. Main table remains perfect×perfect."
        ),
        "config": {
            "noise_sigma": args.noise_sigma,
            "hybrid_ckpt": args.hybrid_ckpt,
            "ghtd3_ckpt": args.ghtd3_ckpt,
            "price_modes": price_modes,
            "resource_modes": resource_modes,
            "methods": methods,
            "seasons": seasons,
        },
    }
    json_path = out / "matrix.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"wrote {json_path}", flush=True)

    # markdown table
    md_path = ROOT / args.docs_md
    lines = [
        "# 信息结构实验结果（P1-1）",
        "",
        "结算：电价 `predicted` 时用 `price_realized`；观测用 `price_predicted`。  ",
        "资源：`perfect` / `noisy` / residual-BiLSTM `predicted`。  ",
        "FMU 物理真值不变。主表默认仍为 perfect×perfect。",
        "",
        f"数据文件：`{json_path}`",
        "",
        "| Season | Price | Resource | Method | J | Reward | SOC | Thermal MWh |",
        "|--------|-------|----------|--------|--:|-------:|:---:|------------:|",
    ]
    for r in rows:
        j = r.get("net_cashflow_j")
        rew = r.get("episode_reward")
        th = r.get("thermal_mwh")
        lines.append(
            "| {season} | {pm} | {rm} | {m} | {j} | {rew} | {soc} | {th} |".format(
                season=r.get("season"),
                pm=r.get("price_mode"),
                rm=r.get("resource_mode"),
                m=r.get("method"),
                j=f"{float(j):.3e}" if j is not None else "—",
                rew=f"{float(rew):.1f}" if rew is not None else "—",
                soc="Y" if r.get("terminal_soc_satisfied") else "N",
                th=f"{float(th):.0f}" if th is not None else "—",
            )
        )
    # attach forecast quality if present
    res_json = ROOT / "data/forecast_models/resource_bilstm.json"
    price_json = ROOT / "data/forecast_models/price_bilstm.json"
    lines.extend(["", "## 预报质量（离线）", ""])
    if res_json.is_file():
        lines.append(f"- 资源 BiLSTM：见 `{res_json.as_posix()}`")
    if price_json.is_file():
        lines.append(f"- 电价 BiLSTM：见 `{price_json.as_posix()}`")
    lines.extend(
        [
            "",
            "## 论文表述",
            "",
            "> Main experiments use perfect day-ahead boundary forecasts. "
            "We further evaluate an information-structure matrix over price and resource "
            "forecast qualities (perfect / noisy / learned residual BiLSTM), with settlement "
            "and FMU physics always realized.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
