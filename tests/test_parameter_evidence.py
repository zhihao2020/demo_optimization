"""Official carbon / price-parameter profile alignment."""

from __future__ import annotations

from pathlib import Path

import yaml

from envs.reward_calculator import RewardCalculator

ROOT = Path(__file__).resolve().parents[1]


def _load_reward_cfg() -> dict:
    with (ROOT / "src/config/reward_config.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_official_profile_id_and_main_carbon_values():
    cfg = _load_reward_cfg()
    assert cfg.get("parameter_profile_id") == "official-2024-ets-sd-grid-v1"
    carbon = cfg["carbon"]
    assert abs(float(carbon["price_cny_per_t"]) - 97.49) < 1e-9
    assert abs(float(carbon["beta_t_per_mwh"]) - 0.8049) < 1e-9
    assert abs(float(carbon["eta_grid_t_per_mwh"]) - 0.6191) < 1e-9
    assert abs(float(carbon["eta_thermal_t_per_mwh"]) - 0.85) < 1e-9
    assert carbon["mode"] == "intensity_benchmark"
    assert float(carbon["eta_thermal_t_per_mwh"]) != float(carbon["beta_t_per_mwh"])
    sens = carbon.get("sensitivity_cny_per_t") or []
    assert 97.49 in sens and 80.0 in sens and 86.4 in sens


def test_curtailment_cross_checks_ghtd3_order():
    cfg = _load_reward_cfg()
    cut = cfg["curtailment"]
    assert abs(float(cut["nu_curt_cny_per_mwh"]) - 300.0) < 1e-9
    # GHTD3 0.041 USD/kWh * 7.2 ≈ 295.2
    assert abs(0.041 * 7.2 * 1000.0 - 295.2) < 1e-6


def test_caes_startup_scale_modes():
    base = {
        "c_su_usd_ref": 3.42,
        "p_ref_w": 8.0e5,
        "p_cap_w": 1.5e8,
        "usd_cny": 7.2,
    }
    linear = RewardCalculator.caes_startup_unit_cny({**base, "scale_mode": "linear_capacity"})
    none = RewardCalculator.caes_startup_unit_cny({**base, "scale_mode": "none"})
    sqrt = RewardCalculator.caes_startup_unit_cny({**base, "scale_mode": "sqrt_capacity"})
    assert abs(linear - 3.42 * (1.5e8 / 8.0e5) * 7.2) < 1e-6
    assert abs(none - 3.42 * 7.2) < 1e-6
    assert none < sqrt < linear


def test_parameter_evidence_doc_mentions_official_values():
    text = (ROOT / "docs/parameter_evidence.md").read_text(encoding="utf-8")
    for token in ("97.49", "0.8049", "0.6191", "official-2024-ets-sd-grid-v1", "legacy-2022"):
        assert token in text


def test_main_tex_cites_official_carbon_sources():
    tex = (ROOT / "Paper/main.tex").read_text(encoding="utf-8")
    assert "97.49" in tex
    assert "0.6191" in tex
    assert "0.8049" in tex
    assert "mee2025etsnews" in tex or "mee2025etsreport" in tex
    assert "mee2024quota" in tex
    assert "mee2023ef" in tex
    assert "monthly" in tex.lower() and "agency-purchase" in tex.lower()
    # must not still claim 0.5703 as the main grid factor
    assert "0.5703 / 0.82" not in tex


def test_bib_has_official_entries():
    bib = (ROOT / "Paper/references.bib").read_text(encoding="utf-8")
    for key in (
        "mee2025etsnews",
        "mee2025etsreport",
        "mee2024quota",
        "mee2023ef",
        "worldbank2024usdnyc",
        "schroeder2015voll",
        "shandong2023tou",
        "ndrc2023td",
    ):
        assert key in bib


def test_tou_readme_does_not_claim_official_absolute_tariff():
    text = (ROOT / "data/price_tou_README.md").read_text(encoding="utf-8")
    assert "构造" in text or "constructive" in text.lower()
    assert "实收电价" in text or "settlement" in text.lower()


def test_prepare_run_dir_stamps_official_profile(tmp_path):
    from training.hybrid_common.eval_and_save import parameter_profile_fields, prepare_run_dir

    prepare_run_dir(tmp_path, ROOT)
    snap = tmp_path / "config" / "reward_config.yaml"
    assert snap.is_file()
    cfg = yaml.safe_load(snap.read_text(encoding="utf-8"))
    assert cfg["parameter_profile_id"] == "official-2024-ets-sd-grid-v1"
    fields = parameter_profile_fields(tmp_path)
    assert fields["parameter_profile_id"] == "official-2024-ets-sd-grid-v1"
    assert abs(float(fields["carbon_price_cny_per_t"]) - 97.49) < 1e-9
    assert abs(float(fields["carbon_beta_t_per_mwh"]) - 0.8049) < 1e-9
    assert abs(float(fields["carbon_eta_grid_t_per_mwh"]) - 0.6191) < 1e-9
    assert fields["carbon_price_source"]
    assert "2024" in str(fields["carbon_price_source"])
    assert "Shandong" in str(fields["carbon_eta_grid_source"]) or "0.6191" in str(
        fields["carbon_eta_grid_source"]
    )


def test_legacy_run_snapshots_are_not_the_src_config():
    """Guard: active src config must not silently equal the legacy proxy book."""
    cfg = _load_reward_cfg()
    legacy = (
        abs(float(cfg["carbon"]["price_cny_per_t"]) - 80.0) < 1e-9
        and abs(float(cfg["carbon"]["eta_grid_t_per_mwh"]) - 0.5703) < 1e-9
        and abs(float(cfg["carbon"]["beta_t_per_mwh"]) - 0.82) < 1e-9
    )
    assert not legacy


def test_intensity_benchmark_settlement_identity_official_prices():
    """End-of-episode thermal position settles at −π·Q; grid import is step tax."""
    cfg = _load_reward_cfg()
    cfg = dict(cfg)
    cfg["curtailment"] = {**(cfg.get("curtailment") or {}), "enabled": False}
    cfg["battery_degradation"] = {**(cfg.get("battery_degradation") or {}), "enabled": False}
    cfg["caes_startup"] = {**(cfg.get("caes_startup") or {}), "enabled": False}
    cfg["grid_contract"] = {**(cfg.get("grid_contract") or {}), "enabled": False}
    cfg["terminal_soc"] = {"enabled": False}
    rc = RewardCalculator(cfg, require_complete=False)
    cash = {
        "economic_cashflow_total": 0.0,
        "economic_cashflow_wind": 0.0,
        "economic_cashflow_pv": 0.0,
        "economic_cashflow_thermal": 0.0,
        "economic_cashflow_battery": 0.0,
        "economic_cashflow_caes": 0.0,
        "economic_cashflow_load": 0.0,
        "economic_cashflow_grid": 0.0,
        "p_thermal": 1.0e6,
        "p_grid": 1.0e6,
        "p_battery": 0.0,
        "p_caes": 0.0,
        "p_curtailment": 0.0,
        "p_unserved": 0.0,
        "battery_soc": 0.5,
        "caes_gas_soc": 0.5,
        "caes_hot_soc": 0.5,
        "caes_cold_soc": 0.5,
    }
    rc.reset(cash)
    _, terms = rc.calculate(
        cash,
        is_final_step=True,
        episode_completed=True,
        no_failure=True,
        decision_interval_hours=1.0,
    )
    pi = 97.49
    beta = 0.8049
    eta_th = 0.85
    eta_g = 0.6191
    e_th = 1.0
    q = beta * e_th - eta_th * e_th
    expected_settlement = -pi * q
    expected_grid = pi * eta_g * 1.0
    assert abs(float(terms["carbon_settlement_cny"]) - expected_settlement) < 1e-6
    assert abs(float(terms["carbon_grid_step_cny"]) - expected_grid) < 1e-6
    assert abs(float(terms["carbon_cost_cny"]) - (expected_settlement + expected_grid)) < 1e-6
