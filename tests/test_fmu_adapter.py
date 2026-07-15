from pathlib import Path
import yaml
import pytest

from fmpy import read_model_description

from fmu.variable_registry import build_registry


def test_registry_uses_actual_fmu_inputs_and_outputs():
    config = yaml.safe_load(Path("src/config/env_config.yaml").read_text(encoding="utf-8"))
    names = {item.name for item in read_model_description(config["fmu"]["path"]).modelVariables}
    required_economics = {item["name"] for item in config["economics"]}
    if not required_economics <= names:
        pytest.skip("本地 FMU 二进制尚未按新增经济接口重新导出")
    registry = build_registry(Path(config["fmu"]["path"]), config)
    assert registry.action_names == ("u_tp", "u_battery", "u_caes")
    assert registry.outputs["p_grid"].unit == "W"
    assert registry.economics["economic_cashflow_total"].unit == "CNY"
