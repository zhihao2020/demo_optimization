"""Modelica 累计现金流 reward 与终端 SOC 奖励。"""

from envs.reward_calculator import RewardCalculator


def _cfg(**term_overrides):
    term = {"enabled": True, "mode": "binary_bonus", "bonus": 10.0, "tolerance": 0.05,
            "weights": {"battery_soc": 1.0, "caes_gas_soc": 1.0, "caes_hot_soc": 1.0, "caes_cold_soc": 1.0}}
    term.update(term_overrides)
    return {"episode_steps": 168, "cost_reference": {"value": 1000.0, "unit": "CNY_per_step", "source": "test"}, "terminal_soc": term}


def _out(**kw):
    d = {"battery_soc": .5, "caes_gas_soc": .85, "caes_hot_soc": .5, "caes_cold_soc": .5,
         "economic_cashflow_total": 100.0, "economic_cashflow_wind": 10.0, "economic_cashflow_pv": 10.0,
         "economic_cashflow_thermal": 10.0, "economic_cashflow_battery": 10.0, "economic_cashflow_caes": 10.0,
         "economic_cashflow_load": 10.0, "economic_cashflow_grid": 40.0}
    d.update(kw)
    return d


def test_reward_formula_uses_fmu_total_delta():
    calc = RewardCalculator(_cfg())
    calc.reset(_out())
    r, t = calc.calculate(_out(economic_cashflow_total=150.0), is_final_step=False, episode_completed=False, no_failure=True)
    assert r == 0.05
    assert t["normalized_cost"] == -0.05
    assert t["terminal_soc_bonus"] == 0.0


def test_terminal_bonus_only_on_complete_satisfied():
    calc = RewardCalculator(_cfg())
    calc.reset(_out())
    r, t = calc.calculate(_out(economic_cashflow_total=100.0), is_final_step=True, episode_completed=True, no_failure=True, valid_episode_steps=168)
    assert t["terminal_soc_bonus"] == 10.0
    assert r == 10.0


def test_quadratic_terminal_mode_is_exclusive():
    calc = RewardCalculator(_cfg(mode="quadratic_penalty", quadratic_weight=2.0, bonus=999))
    calc.reset(_out())
    _, t = calc.calculate(_out(battery_soc=.4), is_final_step=True, episode_completed=True, no_failure=True, valid_episode_steps=168)
    assert t["terminal_soc_bonus"] < 0.0
    assert t["terminal_soc_bonus"] != 999
