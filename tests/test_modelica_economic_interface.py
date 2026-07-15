from pathlib import Path


def test_modelica_exports_total_and_all_cumulative_cashflow_components():
    root = Path(__file__).resolve().parents[1]
    model = (root / "resources/Example/TypicalScene/PowerSystem_8760h.mo").read_text(encoding="utf-8")
    bus = (root / "resources/TypicalScenarios.mo").read_text(encoding="utf-8")
    for name in ("total", "wind", "pv", "thermal", "battery", "caes", "load", "grid"):
        assert f"economic_cashflow_{name}" in model
    for component in ("PV", "WT", "TP", "BT", "CAES", "Eload", "Grid"):
        assert f"der(Income_{component})" in bus
    assert "economic_cashflow_total = bus.OPT_goal" in model
