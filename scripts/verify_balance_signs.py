"""实际 FMU 下的三类供需尝试；不把电网自动平衡后的零残差误写为缺供/弃电。"""
from pathlib import Path
import json

from fmu.session import FmuSession

fmu_path = Path("data/TypicalScensrio_Example_TypicalScene_PowerSystem_8760h.fmu")
cases = {
    "generation_surplus_attempt": {"u_tp": 1.0, "u_battery": 1.0, "u_caes": 1.0},
    "generation_deficit_attempt": {"u_tp": 1.0 / 3.0, "u_battery": -1.0, "u_caes": -1.0},
    "near_balance_attempt": {"u_tp": 1.0, "u_battery": 0.0, "u_caes": 0.0},
}
records = {}
with FmuSession(fmu_path) as session:
    for name, action in cases.items():
        session.reset()
        output = session.step(action)
        records[name] = {key: output[key] for key in ("p_curtailment", "p_unserved", "p_grid", "p_thermal", "p_battery", "p_caes")}
print(json.dumps(records, ensure_ascii=False, indent=2))
