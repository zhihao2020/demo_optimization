"""RewardCalculator 测试：Modelica 累计现金流增量与 reset 基线。"""

from envs.reward_calculator import RewardCalculator


def _outputs(**overrides):
    """构造含经济现金流字段的输出快照。

    Args:
        **overrides: 覆盖默认字段。

    Returns:
        FMU 输出字典。
    """
    data = {
        "battery_soc": .5, "caes_gas_soc": .5, "caes_hot_soc": .5, "caes_cold_soc": .5,
        "economic_cashflow_total": 100.0,
        "economic_cashflow_wind": 10.0, "economic_cashflow_pv": 10.0,
        "economic_cashflow_thermal": 10.0, "economic_cashflow_battery": 10.0,
        "economic_cashflow_caes": 10.0, "economic_cashflow_load": 10.0,
        "economic_cashflow_grid": 40.0,
    }
    data.update(overrides)
    return data


def _calculator():
    """构造已 reset 的 RewardCalculator（C_ref=1000，终端 SOC 关闭）。

    Returns:
        已以默认输出 reset 的 RewardCalculator 实例。
    """
    calculator = RewardCalculator({
        "episode_steps": 168,
        "cost_reference": {"value": 1000.0},
        "terminal_soc": {"enabled": False},
    })
    calculator.reset(_outputs())
    return calculator


def test_reward_is_modelica_cashflow_delta_not_python_repricing():
    """验证 reward 仅反映 total 增量/C_ref，分项成本不参与 Python 重定价。"""
    current = _outputs(economic_cashflow_total=160.0, economic_cashflow_grid=100.0)
    reward, terms = _calculator().calculate(
        current, is_final_step=False, episode_completed=False, no_failure=True,
    )
    assert reward == 0.06
    assert terms["economic_cashflow_delta"] == 60.0
    assert terms["raw_total_cost"] == -60.0
    assert terms["raw_grid_cost"] == -60.0
    assert "raw_thermal_cost" in terms


def test_reset_baseline_makes_first_delta_relative_to_fmu_reset_value():
    """验证 reset 后每步 delta 相对上一时刻累计现金流，非绝对值。"""
    calculator = _calculator()
    _, first = calculator.calculate(_outputs(economic_cashflow_total=110.0), is_final_step=False, episode_completed=False, no_failure=True)
    _, second = calculator.calculate(_outputs(economic_cashflow_total=105.0), is_final_step=False, episode_completed=False, no_failure=True)
    assert first["economic_cashflow_delta"] == 10.0
    assert second["economic_cashflow_delta"] == -5.0


def test_soc_potential_shaping_telescopes_to_final_l1():
    calculator = RewardCalculator({
        "episode_steps": 3,
        "cost_reference": {"value": 1000.0},
        "terminal_soc": {
            "enabled": False,
            "shaping": {"enabled": True, "mode": "potential", "coef": 2.0},
            "weights": {
                "battery_soc": 1.0, "caes_gas_soc": 1.0,
                "caes_hot_soc": 1.0, "caes_cold_soc": 1.0,
            },
        },
    })
    calculator.reset(_outputs())
    # 只动 battery_soc：0.5 -> 0.6 -> 0.7  => L1: 0.1, 0.2
    r1, t1 = calculator.calculate(
        _outputs(battery_soc=0.6, economic_cashflow_total=100.0),
        is_final_step=False, episode_completed=False, no_failure=True,
    )
    r2, t2 = calculator.calculate(
        _outputs(battery_soc=0.7, economic_cashflow_total=100.0),
        is_final_step=True, episode_completed=True, no_failure=True,
    )
    assert abs(t1["soc_shaping_reward"] - 2.0 * (0.0 - 0.1)) < 1e-9
    assert abs(t2["soc_shaping_reward"] - 2.0 * (0.1 - 0.2)) < 1e-9
    # 望远镜和 ≈ -coef * L1_final
    assert abs(t1["soc_shaping_reward"] + t2["soc_shaping_reward"] - (-2.0 * 0.2)) < 1e-9
    assert abs(r1 - t1["soc_shaping_reward"]) < 1e-9  # 经济 delta=0
    assert abs(r2 - t2["soc_shaping_reward"]) < 1e-9


def test_terminal_fail_penalty_when_soc_not_recovered():
    calc = RewardCalculator({
        "episode_steps": 2,
        "cost_reference": {"value": 1000.0},
        "terminal_soc": {
            "enabled": True,
            "mode": "binary_bonus",
            "bonus": 10.0,
            "fail_penalty_l1": 20.0,
            "tolerance": 0.05,
            "weights": {
                "battery_soc": 1.0, "caes_gas_soc": 1.0,
                "caes_hot_soc": 1.0, "caes_cold_soc": 1.0,
            },
            "shaping": {"enabled": False},
        },
    })
    calc.reset(_outputs())
    calc.calculate(
        _outputs(battery_soc=0.6, economic_cashflow_total=100.0),
        is_final_step=False, episode_completed=False, no_failure=True,
        valid_episode_steps=1,
    )
    # L1 = |0.7-0.5|=0.2 > 0.05 → fail penalty -4.0
    r, t = calc.calculate(
        _outputs(battery_soc=0.7, economic_cashflow_total=100.0),
        is_final_step=True, episode_completed=True, no_failure=True,
        valid_episode_steps=2,
    )
    assert t["terminal_soc_satisfied"] == 0.0
    assert abs(t["terminal_soc_bonus"] - (-20.0 * 0.2)) < 1e-9
    assert abs(r - t["terminal_soc_bonus"]) < 1e-9


def test_soc_recovery_horizon_boosts_coef_near_end():
    calc = RewardCalculator({
        "episode_steps": 10,
        "cost_reference": {"value": 1000.0},
        "terminal_soc": {
            "enabled": False,
            "shaping": {
                "enabled": True,
                "mode": "potential",
                "coef": 2.0,
                "absolute_coef": 0.0,
                "recovery_horizon_steps": 4,
                "recovery_coef_scale": 3.0,
            },
            "weights": {
                "battery_soc": 1.0, "caes_gas_soc": 1.0,
                "caes_hot_soc": 1.0, "caes_cold_soc": 1.0,
            },
        },
    })
    calc.reset(_outputs())
    # early step: no boost
    _, t_early = calc.calculate(
        _outputs(battery_soc=0.6, economic_cashflow_total=100.0),
        is_final_step=False, episode_completed=False, no_failure=True,
        valid_episode_steps=2,
    )
    assert abs(t_early["soc_recovery_scale"] - 1.0) < 1e-9
    # near end remaining=1 (steps_done=9) → scale > 1
    for s in range(3, 10):
        _, t = calc.calculate(
            _outputs(battery_soc=0.5 + 0.01 * s, economic_cashflow_total=100.0),
            is_final_step=(s == 10), episode_completed=False, no_failure=True,
            valid_episode_steps=s,
        )
    assert t["soc_recovery_scale"] > 1.5


def test_cui_style_terminal_is_bonus_or_zero():
    """对齐崔文：过门给加分，不过门为 0，不按 L1 重罚。"""
    calc = RewardCalculator({
        "episode_steps": 2,
        "cost_reference": {"value": 1000.0},
        "terminal_soc": {
            "enabled": True,
            "mode": "binary_bonus",
            "bonus": 15.0,
            "fail_penalty_l1": 0.0,
            "tolerance": 0.06,
            "weights": {
                "battery_soc": 1.0, "caes_gas_soc": 1.0,
                "caes_hot_soc": 0.35, "caes_cold_soc": 0.35,
            },
            "shaping": {"enabled": False},
        },
    })
    calc.reset(_outputs())
    calc.calculate(
        _outputs(battery_soc=0.6, economic_cashflow_total=100.0),
        is_final_step=False, episode_completed=False, no_failure=True,
        valid_episode_steps=1,
    )
    _, t = calc.calculate(
        _outputs(battery_soc=0.7, economic_cashflow_total=100.0),
        is_final_step=True, episode_completed=True, no_failure=True,
        valid_episode_steps=2,
    )
    assert t["terminal_soc_satisfied"] == 0.0
    assert abs(t["terminal_soc_bonus"]) < 1e-9


def test_recovery_horizons_zero_disables_battery():
    from envs.power_system_env import recovery_horizons

    h, b = recovery_horizons({
        "soc_recovery_horizon": 40,
        "soc_recovery_battery_horizon": 0,
    })
    assert h == 40
    assert b == 0
    h2, b2 = recovery_horizons({"soc_recovery_horizon": 40})
    assert h2 == 40
    assert b2 == 56
