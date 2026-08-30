"""FS-HSAC epsilon schedule (Cui 2024 Table 4, scaled to horizon)."""

from training.hybrid_common.explore import (
    CUI_DELTA_EPS,
    CUI_EPS_MIN,
    CUI_TRAIN_STEPS_REF,
    cui_eps_horizon,
    explore_epsilon,
    scaled_replay,
)


def test_explore_epsilon_cui_reference_horizon():
    # On Cui's 200k-step run, Δε=6e-6 hits 0.05 at (1-0.05)/6e-6.
    hit = int(round((1.0 - CUI_EPS_MIN) / CUI_DELTA_EPS))
    assert abs(explore_epsilon(0, CUI_TRAIN_STEPS_REF) - 1.0) < 1e-9
    assert abs(explore_epsilon(hit, CUI_TRAIN_STEPS_REF) - 0.05) < 1e-3
    assert abs(explore_epsilon(CUI_TRAIN_STEPS_REF, CUI_TRAIN_STEPS_REF) - 0.05) < 1e-9


def test_explore_epsilon_scaled_week_protocol():
    total = 840000
    horizon = cui_eps_horizon(total)
    assert abs(explore_epsilon(0, total) - 1.0) < 1e-9
    assert abs(explore_epsilon(horizon, total) - 0.05) < 1e-9
    assert abs(explore_epsilon(total, total) - 0.05) < 1e-9
    mid = explore_epsilon(horizon // 2, total)
    assert abs(mid - 0.525) < 1e-3


def test_scaled_replay_grows_with_horizon():
    assert scaled_replay(200_000) == 10_000
    assert scaled_replay(840_000) == 42_000
