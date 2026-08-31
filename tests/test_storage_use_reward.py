"""storage_use is Cui Eq. (35) R^F and does not enter J^gen."""

from pathlib import Path

import yaml

from envs.reward_calculator import RewardCalculator

ROOT = Path(__file__).resolve().parents[1]


def _calc(**extra):
    cfg = {
        "episode_steps": 168,
        "cost_reference": {"value": 1000.0},
        "terminal_soc": {"enabled": False},
        "grid_contract": {"enabled": True, "p_lim_w": 2.0e8},
        "storage_use": {
            "enabled": True,
            "theta_w": 5.0e7,
            "rf_coef": 0.05,
            "active_w": 1.0e6,
        },
        **extra,
    }
    return RewardCalculator(cfg)


def test_small_deviation_bess_only_scores():
    # |p_disp|=20 MW < 50 MW; battery 20 MW; CAES idle
    r, terms = _calc()._storage_use_reward(
        {"p_grid": 0.0, "p_caes": 0.0, "p_battery": 2.0e7}
    )
    assert abs(r - 0.05) < 1e-9
    assert terms["storage_use_small"] == 1.0
    assert terms["storage_use_large"] == 0.0


def test_small_deviation_caes_scores_zero():
    r, terms = _calc()._storage_use_reward(
        {"p_grid": 0.0, "p_caes": 2.0e7, "p_battery": 0.0}
    )
    assert r == 0.0
    assert terms["storage_use_small"] == 0.0
    assert terms["storage_use_large"] == 0.0


def test_large_deviation_caes_scores():
    # |p_disp|=150 MW ≥ 50 MW; CAES on (battery optional)
    r, terms = _calc()._storage_use_reward(
        {"p_grid": 0.0, "p_caes": 1.5e8, "p_battery": 0.0}
    )
    assert abs(r - 0.05) < 1e-9
    assert terms["storage_use_large"] == 1.0
    assert terms["storage_use_small"] == 0.0


def test_large_deviation_bess_only_scores_zero():
    r, terms = _calc()._storage_use_reward(
        {"p_grid": 0.0, "p_caes": 0.0, "p_battery": 8.0e7}
    )
    assert r == 0.0
    assert terms["storage_use_large"] == 0.0
    assert terms["storage_use_small"] == 0.0


def test_yaml_rf_coef_matches_cui():
    cfg = yaml.safe_load((ROOT / "src" / "config" / "reward_config.yaml").read_text(encoding="utf-8"))
    assert float(cfg["storage_use"]["rf_coef"]) == 0.5
    assert cfg["storage_use"]["enabled"] is False


def test_storage_use_not_in_j_gen_terms_as_cash():
    calc = _calc()
    r, terms = calc._storage_use_reward(
        {"p_grid": 0.0, "p_caes": 1.5e8, "p_battery": 0.0}
    )
    assert "generalized_cashflow_delta" not in terms
    assert r > 0.0
    assert calc.config["storage_use"]["enabled"] is True
