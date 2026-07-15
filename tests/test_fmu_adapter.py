from pathlib import Path
import yaml

from fmu.variable_registry import build_registry


def test_registry_uses_actual_fmu_inputs_and_outputs():
    config = yaml.safe_load(Path("src/config/env_config.yaml").read_text(encoding="utf-8"))
    registry = build_registry(Path(config["fmu"]["path"]), config)
    assert registry.action_names == ("u_tp", "u_battery", "u_caes")
    assert registry.outputs["p_grid"].unit == "W"
