"""打印仓库内默认 FMU 的 FMI 版本与 input/output 清单（不仿真）。"""

from __future__ import annotations

from pathlib import Path
from fmpy import read_model_description

p = Path("data/TypicalScensrio_Example_TypicalScene_PowerSystem_8760h.fmu")
md = read_model_description(str(p))
print({"fmi_version": md.fmiVersion, "co_simulation": md.coSimulation is not None,
       "default_step_seconds": md.defaultExperiment.stepSize, "stop_time_seconds": md.defaultExperiment.stopTime})
for item in md.modelVariables:
    if item.causality in ("input", "output"):
        print(item.causality, item.name, getattr(item, "unit", None), getattr(item, "start", None), item.description)
