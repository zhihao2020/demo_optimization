"""终端 SOC 奖励专项。"""

from envs.reward_calculator import RewardCalculator


def _mk(mode="binary_bonus"):
    cfg = {
        "decision_interval_seconds": 3600,
        "episode_steps": 168,
        "buy_price_yuan_per_mwh": 0.0,
        "sell_price_yuan_per_mwh": 0.0,
        "thermal_a": 0.0,
        "thermal_b": 0.0,
        "thermal_c": 0.0,
        "curtailment_yuan_per_mwh": 0.0,
        "unserved_yuan_per_mwh": 0.0,
        "battery_throughput_yuan_per_mwh": 0.0,
        "caes_throughput_yuan_per_mwh": 0.0,
        "ramp_yuan_per_mw": 0.0,
        "cost_reference": {"value": 1.0},
        "terminal_soc": {
            "enabled": True,
            "mode": mode,
            "bonus": 7.0,
            "tolerance": 0.1,
            "quadratic_weight": 3.0,
            "weights": {"battery_soc": 1.0, "caes_gas_soc": 1.0, "caes_hot_soc": 1.0, "caes_cold_soc": 1.0},
            "require_complete_episode": True,
            "require_no_failure": True,
        },
    }
    calc = RewardCalculator(cfg)
    init = {
        "p_grid": 0,
        "p_thermal": -1e6,
        "p_battery": 0,
        "p_caes": 0,
        "p_curtailment": 0,
        "p_unserved": 0,
        "battery_soc": 0.5,
        "caes_gas_soc": 0.8,
        "caes_hot_soc": 0.4,
        "caes_cold_soc": 0.4,
    }
    calc.reset(init)
    return calc, init


def test_steps_0_to_166_bonus_zero():
    calc, init = _mk()
    for step in range(167):
        _, t = calc.calculate(init, -1e6, is_final_step=False, episode_completed=False, no_failure=True, valid_episode_steps=step + 1)
        assert t["terminal_soc_bonus"] == 0.0


def test_initial_soc_from_reset_used():
    calc, init = _mk()
    assert calc.initial_soc["battery_soc"] == 0.5
    # 四 SOC 联合：仅一个偏离超限则无 bonus（tol=0.1，四者各偏 0.03 => L1=0.12）
    out = dict(init)
    out.update({"battery_soc": 0.53, "caes_gas_soc": 0.83, "caes_hot_soc": 0.43, "caes_cold_soc": 0.43})
    _, t = calc.calculate(out, -1e6, is_final_step=True, episode_completed=True, no_failure=True, valid_episode_steps=168)
    assert abs(t["terminal_soc_l1_error"] - 0.12) < 1e-9
    assert t["terminal_soc_bonus"] == 0.0
