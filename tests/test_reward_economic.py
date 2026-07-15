"""经济 reward 与终端 SOC 二元奖励。"""

import pytest

from envs.reward_calculator import RewardCalculator


def _cfg(**term_overrides):
    term = {
        "enabled": True,
        "mode": "binary_bonus",
        "bonus": 10.0,
        "tolerance": 0.05,
        "weights": {"battery_soc": 1.0, "caes_gas_soc": 1.0, "caes_hot_soc": 1.0, "caes_cold_soc": 1.0},
        "require_complete_episode": True,
        "require_no_failure": True,
    }
    term.update(term_overrides)
    return {
        "decision_interval_seconds": 3600,
        "episode_steps": 168,
        "buy_price_yuan_per_mwh": 600.0,
        "sell_price_yuan_per_mwh": 100.0,
        "thermal_a": 0.0,
        "thermal_b": 400.0,
        "thermal_c": 0.0,
        "curtailment_yuan_per_mwh": 0.0,
        "unserved_yuan_per_mwh": 0.0,
        "battery_throughput_yuan_per_mwh": 0.0,
        "caes_throughput_yuan_per_mwh": 0.0,
        "ramp_yuan_per_mw": 0.0,
        "cost_reference": {"value": 1000.0, "unit": "CNY_per_step", "source": "test"},
        "terminal_soc": term,
    }


def _out(**kw):
    d = {
        "p_grid": 1e6,
        "p_thermal": -50e6,
        "p_battery": 0,
        "p_caes": 0,
        "p_curtailment": 0,
        "p_unserved": 0,
        "battery_soc": 0.5,
        "caes_gas_soc": 0.85,
        "caes_hot_soc": 0.5,
        "caes_cold_soc": 0.5,
    }
    d.update(kw)
    return d


def test_reward_formula_no_failure_penalty():
    calc = RewardCalculator(_cfg())
    calc.reset(_out())
    r, t = calc.calculate(_out(), -50e6, is_final_step=False, episode_completed=False, no_failure=True)
    assert abs(r - (-t["raw_total_cost"] / 1000.0)) < 1e-9
    assert t["terminal_soc_bonus"] == 0.0
    assert "solver_failure" not in str(t).lower() or t.get("solver_failure_cost", 0) == 0


def test_terminal_bonus_only_on_complete_satisfied():
    calc = RewardCalculator(_cfg())
    calc.reset(_out())
    # 普通步
    _, t0 = calc.calculate(_out(), -50e6, is_final_step=False, episode_completed=False, no_failure=True, valid_episode_steps=10)
    assert t0["terminal_soc_bonus"] == 0.0
    # 完整且满足
    r, t = calc.calculate(_out(), -50e6, is_final_step=True, episode_completed=True, no_failure=True, valid_episode_steps=168)
    assert t["terminal_soc_bonus"] == 10.0
    assert abs(r - (-t["normalized_cost"] + 10.0)) < 1e-9
    # 完整但不满足
    _, t2 = calc.calculate(
        _out(battery_soc=0.1),
        -50e6,
        is_final_step=True,
        episode_completed=True,
        no_failure=True,
        valid_episode_steps=168,
    )
    assert t2["terminal_soc_bonus"] == 0.0
    # 提前终止
    _, t3 = calc.calculate(_out(), -50e6, is_final_step=True, episode_completed=False, no_failure=False, valid_episode_steps=50)
    assert t3["terminal_soc_bonus"] == 0.0


def test_binary_and_quadratic_not_both():
    # 互斥由配置 mode 保证；同一次 calculate 只走一个分支
    calc = RewardCalculator(_cfg(mode="quadratic_penalty", quadratic_weight=2.0, bonus=999))
    calc.reset(_out())
    _, t = calc.calculate(
        _out(battery_soc=0.4),
        -50e6,
        is_final_step=True,
        episode_completed=True,
        no_failure=True,
        valid_episode_steps=168,
    )
    assert t["terminal_soc_bonus"] <= 0.0  # penalty 记在 bonus 字段为负
    assert t["terminal_soc_bonus"] != 999


def test_cost_ref_positive_and_stable():
    cfg = _cfg()
    calc = RewardCalculator(cfg)
    calc.reset(_out())
    _, t1 = calc.calculate(_out(), -50e6, is_final_step=False, episode_completed=False, no_failure=True)
    _, t2 = calc.calculate(_out(p_grid=2e6), -50e6, is_final_step=False, episode_completed=False, no_failure=True)
    assert t1["cost_reference"] == t2["cost_reference"] == 1000.0
    assert t1["cost_reference"] > 0
