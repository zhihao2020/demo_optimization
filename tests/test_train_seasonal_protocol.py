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
