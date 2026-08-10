"""统一 KPI：从 evaluate_policy 结果抽取，并相对 B0 计算变化。"""

from __future__ import annotations

from typing import Any


def extract_kpi_from_eval(res: dict[str, Any], *, wall_s: float = 0.0, fmu_steps: int | None = None) -> dict[str, Any]:
    terms = res.get("cost_terms") or {}
    metrics = res.get("metrics") or {}
    # Primary: generalized cashflow sum (ΔJ_gen = CF - carbon - CUT - deg).
    # evaluate_policy accumulates per-step terms into cost_terms.
    j_gen = terms.get("generalized_cashflow_delta")
    j_cf = terms.get("economic_cashflow_delta") or terms.get("cashflow_delta")
    if j_gen is None:
        j_gen = j_cf if j_cf is not None else 0.0
    if j_cf is None:
        j_cf = j_gen
    return {
        "episode_reward": res.get("episode_reward"),
        "sum_delta_j_gen": float(j_gen),
        "net_cashflow_j": float(j_cf),
        "raw_total_cost": terms.get("raw_total_cost") or res.get("weekly_raw_total_cost"),
        "economic_reward": terms.get("economic_reward"),
        "market_buy_cost": terms.get("market_buy_cost"),
        "market_sell_revenue": terms.get("market_sell_revenue"),
        "market_energy_buy_mwh": terms.get("market_energy_buy_mwh"),
        "market_energy_sell_mwh": terms.get("market_energy_sell_mwh"),
        "curtailment_mwh": metrics.get("curtailment_energy_mwh"),
        "unserved_mwh": metrics.get("unserved_energy_mwh"),
        "thermal_mwh": metrics.get("thermal_generation_mwh"),
        "battery_throughput_mwh": metrics.get("battery_throughput_mwh"),
        "caes_throughput_mwh": metrics.get("caes_throughput_mwh"),
        "terminal_soc_satisfied": res.get("terminal_soc_satisfied"),
        "terminal_soc_l1": terms.get("terminal_soc_l1_error") or terms.get("terminal_soc_l1_energy"),
        "terminal_soc_l1_full": terms.get("terminal_soc_l1_full"),
        "terminal_soc": res.get("terminal_soc"),
        "fmu_failure_count": res.get("fmu_failure_count"),
        "invalid_transition_count": res.get("invalid_transition_count"),
        "wall_s": wall_s,
        "fmu_steps": fmu_steps if fmu_steps is not None else res.get("valid_steps"),
        "steps": res.get("steps"),
        "valid_steps": res.get("valid_steps"),
    }


def relative_to_baseline(kpi: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    def d(key: str) -> float | None:
        try:
            return float(kpi.get(key) or 0.0) - float(base.get(key) or 0.0)
        except Exception:
            return None

    thr_b = float(base.get("battery_throughput_mwh") or 0.0) + float(base.get("caes_throughput_mwh") or 0.0)
    thr_k = float(kpi.get("battery_throughput_mwh") or 0.0) + float(kpi.get("caes_throughput_mwh") or 0.0)
    th_b = float(base.get("thermal_mwh") or 0.0) or 1e-9
    return {
        "delta_j_vs_b0": d("net_cashflow_j"),
        "delta_curtailment_mwh": d("curtailment_mwh"),
        "delta_unserved_mwh": d("unserved_mwh"),
        "thermal_ratio_vs_b0": float(kpi.get("thermal_mwh") or 0.0) / th_b,
        "storage_throughput_ratio_vs_b0": thr_k / (thr_b + 1e-9),
        "delta_episode_reward": d("episode_reward"),
    }
