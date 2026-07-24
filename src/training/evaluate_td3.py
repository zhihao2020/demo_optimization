"""策略评估：支持 Hybrid Dict 动作与审计日志。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


def evaluate_policy(
    env,
    policy: Any,
    output_csv: Path | None = None,
    gamma: float = 0.99,
    *,
    reset_options: dict[str, Any] | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """评估单个时间窗口内的策略表现。

    Args:
        env: 电力系统环境(PowerSystemEnv) 实例。
        policy: 需实现 ``predict(obs, deterministic=...)`` 的策略对象。
        output_csv: 可选逐步轨迹 CSV 路径。
        gamma: 折扣因子，用于计算折扣回报。
        reset_options: 传给 ``env.reset(options=...)`` 的选项，如 ``start_time``。
        max_steps: 最大步数；用于年度尾窗不足一周时截断。

    Returns:
        含步数、奖励、成本分项、SOC、CAES 合规率等字段的字典。

    Raises:
        ValueError: ``max_steps`` 非正时抛出。
    """
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps 必须为正数")
    obs, info0 = env.reset(options=reset_options)
    if hasattr(policy, "on_episode_reset"):
        policy.on_episode_reset(info0)
    rows: list[dict[str, Any]] = []
    totals: dict[str, float] = {}
    metrics = {
        "curtailment_energy_mwh": 0.0,
        "unserved_energy_mwh": 0.0,
        "battery_throughput_mwh": 0.0,
        "caes_throughput_mwh": 0.0,
        "thermal_generation_mwh": 0.0,
        "max_thermal_ramp_mw": 0.0,
    }
    previous_thermal: float | None = None
    weekly_raw = 0.0
    weekly_reward = 0.0
    weekly_discounted = 0.0
    terminal_bonus = 0.0
    forbidden = 0
    invalid_transition = 0
    caes_segments: list[dict[str, Any]] = []
    caes_interruptions = 0
    while True:
        predicted = policy.predict(obs, deterministic=True)
        action = predicted[0] if isinstance(predicted, tuple) else predicted
        obs, reward, terminated, truncated, info = env.step(action)
        if hasattr(policy, "on_transition"):
            policy.on_transition(info)
        if info.get("failure_type") in ("StaticActionViolation", "ForbiddenModeViolation", "DynamicStateConstraintViolation"):
            forbidden += 1
        if not info.get("transition_valid", False):
            invalid_transition += 1
        terms = info.get("reward_terms") or {}
        weekly_raw += float(terms.get("raw_total_cost", 0.0))
        weekly_reward += float(reward)
        weekly_discounted += (gamma ** len(rows)) * float(reward)
        terminal_bonus += float(terms.get("terminal_soc_bonus", 0.0))
        if info.get("caes_min_run_completed_segment"):
            caes_segments.append(dict(info["caes_min_run_completed_segment"]))
        if info.get("caes_min_run_final_event"):
            caes_segments.append(dict(info["caes_min_run_final_event"]))
            caes_interruptions += 1
        event = (info.get("feasible_action_spec") or {}).get("caes_min_run_event")
        if event:
            caes_segments.append(dict(event))
            caes_interruptions += 1
        row = {
            "time": info.get("time"),
            "step": len(rows),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "fmu_status": info.get("fmu_status"),
            "fmu_error": info.get("fmu_error") or info.get("failure_reason"),
            "failure_type": info.get("failure_type"),
            "transition_valid": info.get("transition_valid"),
            "requested_u_tp": info.get("requested_u_tp"),
            "requested_u_battery": info.get("requested_u_battery"),
            "requested_caes_mode": info.get("requested_caes_mode"),
            "requested_caes_magnitude": info.get("requested_caes_magnitude"),
            "decoded_u_tp": info.get("decoded_u_tp"),
            "decoded_u_battery": info.get("decoded_u_battery"),
            "decoded_u_caes": info.get("decoded_u_caes"),
            **{f"rt_{k}": v for k, v in terms.items()},
            **{f"obs_{spec.name}": float(obs[i]) for i, spec in enumerate(env.observation_builder.specs)},
            **{k: info.get(k) for k in (
                "u_tp_dynamic_low", "u_tp_dynamic_high",
                "u_battery_dynamic_low", "u_battery_dynamic_high",
                "caes_discharge_allowed", "caes_idle_allowed", "caes_charge_allowed",
                "caes_locked_mode", "caes_locked_steps_completed", "caes_locked_steps_remaining",
            )},
        }
        rows.append(row)
        for key, value in terms.items():
            if key in (
                "cost_reference",
                "cost_reference_missing",
                "terminal_soc_tolerance",
                "terminal_soc_l1_error",
                "terminal_soc_l2_error",
                "terminal_soc_satisfied",
                "terminal_soc_bonus",
            ):
                totals[key] = float(value)  # 取最近一步（最终步覆盖）
            else:
                totals[key] = totals.get(key, 0.0) + float(value)
        current = info.get("last_valid_outputs") or env.last_outputs or {}
        dt_hours = env.config["fmu"]["decision_interval_seconds"] / 3600.0
        if current:
            metrics["curtailment_energy_mwh"] += float(current.get("p_curtailment", 0)) * 1e-6 * dt_hours
            metrics["unserved_energy_mwh"] += float(current.get("p_unserved", 0)) * 1e-6 * dt_hours
            metrics["battery_throughput_mwh"] += abs(float(current.get("p_battery", 0))) * 1e-6 * dt_hours
            metrics["caes_throughput_mwh"] += abs(float(current.get("p_caes", 0))) * 1e-6 * dt_hours
            metrics["thermal_generation_mwh"] += abs(float(current.get("p_thermal", 0))) * 1e-6 * dt_hours
            if previous_thermal is not None:
                metrics["max_thermal_ramp_mw"] = max(
                    metrics["max_thermal_ramp_mw"],
                    abs(float(current["p_thermal"]) - previous_thermal) * 1e-6,
                )
            previous_thermal = float(current["p_thermal"])
        if terminated or truncated:
            break
        if max_steps is not None and len(rows) >= max_steps:
            break
    if output_csv and rows:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    last = env.last_outputs or {}
    return {
        "steps": len(rows),
        "valid_steps": sum(1 for r in rows if r.get("transition_valid")),
        "episode_reward": weekly_reward,
        "weekly_episode_reward": weekly_reward,
        "weekly_discounted_return": weekly_discounted,
        "weekly_raw_total_cost": weekly_raw,
        "weekly_terminal_soc_bonus": float(totals.get("terminal_soc_bonus", terminal_bonus)),
        "terminal_soc_satisfied": bool(float((rows[-1].get("rt_terminal_soc_satisfied") if rows else 0) or 0)),
        "cost_terms": totals,
        "metrics": metrics,
        "terminal_soc": {name: float(last.get(name, float("nan"))) for name in ("battery_soc", "caes_gas_soc", "caes_hot_soc", "caes_cold_soc")},
        "initial_soc": info0.get("initial_soc"),
        "fmu_failure_count": sum(1 for r in rows if r.get("fmu_status") == "failure"),
        "forbidden_action_count": forbidden,
        "invalid_transition_count": invalid_transition,
        "action_violation_count": forbidden,
        "economic_cashflow_total": float(last.get("economic_cashflow_total", float("nan"))),
        "economic_cashflow_components": {
            name: float(last.get(f"economic_cashflow_{name}", float("nan")))
            for name in ("wind", "pv", "thermal", "battery", "caes", "load", "grid")
        },
        "caes_run_segments": caes_segments,
        "caes_min_run_interruption_count": caes_interruptions,
        "caes_min_run_compliance_rate": (
            1.0 if not caes_segments else sum(bool(item.get("completed")) for item in caes_segments) / len(caes_segments)
        ),
        "gamma": gamma,
    }


def evaluate_annual_policy(
    env,
    policy: Any,
    *,
    annual_horizon_hours: int,
    gamma: float = 0.99,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """按训练一致的周窗口滑动覆盖全年评估。

    Args:
        env: 电力系统环境(PowerSystemEnv) 实例。
        policy: 评估用策略，需支持 ``predict`` 及可选 ``on_episode_reset`` / ``on_transition``。
        annual_horizon_hours: 年度总小时数。
        gamma: 各窗口折扣因子。
        output_dir: 可选目录，每窗写入 ``window_XXXXh.csv``。

    Returns:
        全年汇总字典：总步数、总成本、各能量指标、违规计数等。

    Raises:
        ValueError: 年度小时数、决策间隔或 episode 长度配置非法时抛出。
    """
    if annual_horizon_hours <= 0:
        raise ValueError("annual_horizon_hours 必须为正数")
    step_hours = float(env.config["fmu"]["decision_interval_seconds"]) / 3600.0
    if step_hours <= 0 or not step_hours.is_integer():
        raise ValueError("年度评估要求整数小时决策间隔")
    episode_hours = int(env.episode_steps * step_hours)
    if episode_hours <= 0:
        raise ValueError("episode_steps 必须为正数")

    windows: list[dict[str, Any]] = []
    starts = list(range(0, annual_horizon_hours, episode_hours))
    for start_hour in tqdm(starts, desc="AnnualEval", unit="win", dynamic_ncols=True):
        hours = min(episode_hours, annual_horizon_hours - start_hour)
        output_csv = None
        if output_dir is not None:
            output_csv = output_dir / f"window_{start_hour:04d}h.csv"
        windows.append(
            evaluate_policy(
                env,
                policy,
                output_csv=output_csv,
                gamma=gamma,
                reset_options={"start_time": float(start_hour * 3600)},
                max_steps=hours,
            )
        )

    metric_names = tuple(windows[0]["metrics"]) if windows else ()
    return {
        "annual_horizon_hours": annual_horizon_hours,
        "windows": len(windows),
        "steps": sum(int(item["steps"]) for item in windows),
        "valid_steps": sum(int(item["valid_steps"]) for item in windows),
        "annual_raw_total_cost": sum(float(item["weekly_raw_total_cost"]) for item in windows),
        "annual_episode_reward": sum(float(item["weekly_episode_reward"]) for item in windows),
        "window_discounted_return_sum": sum(float(item["weekly_discounted_return"]) for item in windows),
        "metrics": {name: sum(float(item["metrics"][name]) for item in windows) for name in metric_names},
        "fmu_failure_count": sum(int(item["fmu_failure_count"]) for item in windows),
        "forbidden_action_count": sum(int(item["forbidden_action_count"]) for item in windows),
        "invalid_transition_count": sum(int(item["invalid_transition_count"]) for item in windows),
        "terminal_soc_satisfied_windows": sum(bool(item["terminal_soc_satisfied"]) for item in windows),
        "annual_economic_cashflow": sum(
            float(item["cost_terms"].get("economic_cashflow_delta", 0.0)) for item in windows
        ),
        "caes_min_run_interruption_count": sum(int(item["caes_min_run_interruption_count"]) for item in windows),
    }
