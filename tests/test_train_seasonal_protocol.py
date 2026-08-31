"""Paper-protocol defaults: support-only CLI and no end-of-week u_caes rewrite."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from seasonal_cli import parse_args  # noqa: E402


def test_env_config_soc_recovery_horizon_defaults_to_zero():
    cfg = yaml.safe_load((ROOT / "src" / "config" / "env_config.yaml").read_text(encoding="utf-8"))
    market = cfg["market"]
    assert int(market["soc_recovery_horizon"] or 0) == 0
    assert int(market["soc_recovery_battery_horizon"] or 0) == 0


def test_support_and_no_feas_flags_are_aliases():
    support = parse_args(["--method", "fs_hsac", "--season", "winter", "--support"])
    no_feas = parse_args(["--method", "fs_hsac", "--season", "winter", "--no-feas"])
    full = parse_args(["--method", "fs_hsac", "--season", "winter"])
    sac = parse_args(["--method", "sac", "--season", "winter"])
    assert support.support_only is True
    assert no_feas.support_only is True
    assert full.support_only is False
    assert sac.support_only is False


def test_td3_cli_defaults_to_pc_hybrid():
    args = parse_args(["--method", "td3", "--season", "all"])
    assert args.ablation == "none"
    proj = parse_args(["--method", "td3", "--season", "winter", "--ablation", "projection"])
    static = parse_args(["--method", "td3", "--season", "summer", "--ablation", "static-support"])
    assert proj.ablation == "projection"
    assert static.ablation == "static-support"


def test_paper_cli_rule_stage_forecast():
    from seasonal_cli import STAGE_STEPS

    rule = parse_args(["--method", "rule", "--season", "winter"])
    assert rule.forecast_mode == "perfect"
    noisy = parse_args(["--method", "td3", "--season", "all", "--forecast-mode", "noisy", "--stage", "B"])
    assert noisy.forecast_mode == "noisy"
    assert noisy.stage == "B"
    assert STAGE_STEPS["B"] == 5000
    assert STAGE_STEPS["D"] == 400000


def test_paper_week_split_is_36_8_8_disjoint():
    sys.path.insert(0, str(ROOT / "src"))
    from training.episode_starts import TEST_WEEK_IDS, TRAIN_WEEK_IDS, VAL_WEEK_IDS
    from seasonal_cli import SEASON_WEEKS

    assert len(TRAIN_WEEK_IDS) == 36
    assert len(VAL_WEEK_IDS) == 8
    assert len(TEST_WEEK_IDS) == 8
    train, val, test = set(TRAIN_WEEK_IDS), set(VAL_WEEK_IDS), set(TEST_WEEK_IDS)
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)
    assert train | val | test == set(range(52))
    assert SEASON_WEEKS["all"]["train"] == list(TRAIN_WEEK_IDS)
    assert SEASON_WEEKS["all"]["val"] == list(VAL_WEEK_IDS)
    assert SEASON_WEEKS["all"]["test"] == list(TEST_WEEK_IDS)
    assert SEASON_WEEKS["winter"]["eval"] == SEASON_WEEKS["winter"]["test"][0]
