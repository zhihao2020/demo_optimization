"""BoundaryProvider 与边界配置契约测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.envs.boundary_provider import (
    BOUNDARY_FMU_NAMES,
    BoundaryDataError,
    BoundaryProvider,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def provider() -> BoundaryProvider:
    cfg = yaml.safe_load((ROOT / "src/config/env_config.yaml").read_text(encoding="utf-8"))
    return BoundaryProvider(
        ROOT,
        cfg["boundaries"],
        annual_horizon_hours=int(cfg["fmu"]["annual_horizon_hours"]),
        step_seconds=float(cfg["fmu"]["decision_interval_seconds"]),
    )


def test_boundary_keys_match_fmu_ports(provider: BoundaryProvider) -> None:
    b0 = provider.at_time(0.0)
    assert set(b0) == set(BOUNDARY_FMU_NAMES)


def test_boundary_matches_csv_start_values(provider: BoundaryProvider) -> None:
    b0 = provider.at_time(0.0)
    assert b0["v_wind_in"] == pytest.approx(1.45)
    assert b0["g_irradiance_in"] == pytest.approx(0.0)
    assert b0["t_air_in"] == pytest.approx(262.40)
    assert b0["p_load_plan_in"] == pytest.approx(214197861.9)


def test_boundary_hour_grid(provider: BoundaryProvider) -> None:
    b1 = provider.at_time(3600.0)
    assert b1["v_wind_in"] == pytest.approx(1.33)
    assert b1["p_load_plan_in"] == pytest.approx(203661340.2)


def test_boundary_rejects_off_grid_time(provider: BoundaryProvider) -> None:
    with pytest.raises(BoundaryDataError):
        provider.at_time(1800.0)


def test_boundaries_section_not_in_actions() -> None:
    """边界不得混入 actions 段。"""
    cfg = yaml.safe_load((ROOT / "src/config/env_config.yaml").read_text(encoding="utf-8"))
    action_names = {item["name"] for item in cfg["actions"]}
    assert action_names == {"u_tp", "u_battery", "u_caes"}
    assert "boundaries" in cfg
    for src in cfg["boundaries"]["sources"]:
        assert src["fmu_variable"] in BOUNDARY_FMU_NAMES
        assert src["fmu_variable"] not in action_names
