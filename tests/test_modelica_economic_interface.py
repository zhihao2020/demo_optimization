"""Modelica 经济接口测试：验证顶层现金流变量与 TypicalScenarios 总线对齐。"""

from pathlib import Path


def test_modelica_exports_total_and_all_cumulative_cashflow_components():
    """验证 PowerSystem_8760h 导出全部 economic_cashflow_* 且总线含各组件 der(Income_*)。 """
    root = Path(__file__).resolve().parents[1]
    model = (root / "resources/Example/TypicalScene/PowerSystem_8760h.mo").read_text(encoding="utf-8")
    bus = (root / "resources/TypicalScenarios.mo").read_text(encoding="utf-8")
    for name in ("total", "wind", "pv", "thermal", "battery", "caes", "load", "grid"):
        assert f"economic_cashflow_{name}" in model
    for component in ("PV", "WT", "TP", "BT", "CAES", "Eload", "Grid"):
        assert f"der(Income_{component})" in bus
    assert "economic_cashflow_total = bus.OPT_goal" in model
