"""统一 KPI：从 evaluate_policy 结果抽取，并相对 B0 计算变化。"""

from __future__ import annotations

from typing import Any


def extract_kpi_from_eval(res: dict[str, Any], *, wall_s: float = 0.0, fmu_steps: int | None = None) -> dict[str, Any]:
    terms = res.get("cost_terms") or {}
    metrics = res.get("metrics") or {}
    # 净现金流：优先周内 economic_cashflow_delta 累计（evaluate 已在 cost_terms 中给出末步累计语义时需注意）
    # evaluate_policy 的 cost_terms 末行累计差分在 weekly 汇总中：
    j = float(terms.get("economic_cashflow_delta", 0.0))
    # 若为末步单点，使用 episode 级：部分版本把总和放在 economic 分量
    if abs(j) < 1e-12 and "economic_cashflow_components" in res:
        # fallback: 不使用
        pass
    return {
        "episode_reward": res.get("episode_reward"),
        "net_cashflow_j": j,
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
