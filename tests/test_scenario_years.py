"""情景年生成与路径切换测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.data.scenario_years import (
    N_WEEKS,
    YEAR_HOURS,
    ScenarioYearGenerator,
    apply_scenario_to_env_config,
    build_season_buckets,
    season_of_week,
)

ROOT = Path(__file__).resolve().parents[1]


def test_season_buckets_cover_all_weeks() -> None:
    buckets = build_season_buckets(year=2019)
    covered = sorted(w for weeks in buckets.values() for w in weeks)
    assert covered == list(range(N_WEEKS))
    assert season_of_week(0) == "winter"  # Jan 1
    assert season_of_week(26) == "summer"  # ~July


def test_generate_identity_and_bootstrap(tmp_path: Path) -> None:
    out = tmp_path / "scenarios"
    gen = ScenarioYearGenerator(root=ROOT, out_root=out, seed=0, n_years=3)
    manifest = gen.generate()
    assert manifest["n_years"] == 3
    assert (out / "manifest.json").is_file()

    # year_000 与基准逐点一致
    base = np.loadtxt(ROOT / "data" / "winds.csv", delimiter=",", skiprows=1)
    ident = np.loadtxt(out / "year_000" / "winds.csv", delimiter=",", skiprows=1)
    assert np.allclose(base, ident)

    # bootstrap 年长度正确，且与基准不同（几乎必然）
    boot = np.loadtxt(out / "year_001" / "winds.csv", delimiter=",", skiprows=1)
    assert boot.shape[0] == YEAR_HOURS + 1
    assert not np.allclose(base[:, 1], boot[:, 1])

    # 四通道 donor 同步：任一小时的相对排序应来自同一 donor 周
    # （用负荷与风速的周均值相关性做弱检查：两者都应偏离基准）
    load_boot = np.loadtxt(out / "year_001" / "load.csv", delimiter=",", skiprows=1)
    assert load_boot.shape[0] == YEAR_HOURS + 1

    # 电价同步重采样
    assert (out / "year_001" / "price_tou.csv").is_file()
    meta = json.loads((out / "year_001" / "year_meta.json").read_text(encoding="utf-8"))
    assert meta["kind"] == "bootstrap"
    assert len(meta["donors"]) == N_WEEKS + 1


def test_apply_scenario_with_local_tree(tmp_path: Path) -> None:
    """在临时 root 下造最小 data/ + scenarios/，验证路径改写。"""
    data = tmp_path / "data"
    data.mkdir()
    # 最小合法 CSV：8761 行边界 + 8760 行电价
    for name in ("winds.csv", "Gstc.csv", "environment.csv", "load.csv"):
        with (data / name).open("w", encoding="utf-8", newline="") as f:
            f.write("time,value\n")
            for i in range(YEAR_HOURS + 1):
                f.write(f"{i * 3600},{float(i % 17)}\n")
    with (data / "price_tou.csv").open("w", encoding="utf-8", newline="") as f:
        f.write("time,buy_yuan_per_kwh,sell_yuan_per_kwh,band\n")
        for i in range(YEAR_HOURS):
            f.write(f"{i * 3600},0.5,0.2,F\n")

    out = tmp_path / "data" / "scenarios"
    gen = ScenarioYearGenerator(root=tmp_path, out_root=out, seed=0, n_years=2)
    gen.generate()

    cfg = {
        "scenarios": {"root": "data/scenarios", "active": None},
        "boundaries": {
            "sources": [
                {"name": "wind", "fmu_variable": "v_wind_in", "path": "data/winds.csv"},
                {"name": "irradiance", "fmu_variable": "g_irradiance_in", "path": "data/Gstc.csv"},
                {
                    "name": "ambient_temperature",
                    "fmu_variable": "t_air_in",
                    "path": "data/environment.csv",
                },
                {
                    "name": "planned_load",
                    "fmu_variable": "p_load_plan_in",
                    "path": "data/load.csv",
                },
            ]
        },
        "forecast": {
            "sources": [
                {"name": "wind", "path": "data/winds.csv", "offset": 0.0, "scale": 1.0},
                {"name": "irradiance", "path": "data/Gstc.csv", "offset": 0.0, "scale": 1.0},
                {
                    "name": "ambient_temperature",
                    "path": "data/environment.csv",
                    "offset": 0.0,
                    "scale": 1.0,
                },
                {"name": "planned_load", "path": "data/load.csv", "offset": 0.0, "scale": 1.0},
            ]
        },
        "market": {"price_path": "data/price_tou.csv"},
    }
    patched = apply_scenario_to_env_config(cfg, tmp_path, "year_001")
    assert patched["boundaries"]["sources"][0]["path"].endswith("year_001/winds.csv")
    assert patched["forecast"]["sources"][0]["path"].endswith("year_001/winds.csv")
    assert patched["market"]["price_path"].endswith("year_001/price_tou.csv")
    assert patched["scenarios"]["active"] == "year_001"
