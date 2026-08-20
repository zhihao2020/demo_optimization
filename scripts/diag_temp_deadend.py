"""诊断：气罐温度死角处 mode_mask 与逐动作校验为何不一致。

复现 runs/coldfix_stress_fixed_s0_8760h 第 268 步被拒的状态。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from actions.caes_u import u_from_mode_mag
from actions.feasibility_oracle import FeasibilityOracle
from actions.types import CaesMode, PhysicalFmuAction

STATE = {
    "battery_soc": 0.800972,
    "caes_gas_soc": 0.781942,
    "caes_hot_soc": 0.523851,
    "caes_cold_soc": 0.605186,
    "caes_gas_pressure": 7819420.0,
    "caes_gas_temperature": 254.206253,
    "caes_hot_temperature": 458.042877,
    "caes_cold_temperature": 302.451141,
    "p_thermal": -55840364.0,
    "p_battery": 0.0,
    "p_caes": 0.0,
    "p_grid": 0.0,
}


def main() -> None:
    o = FeasibilityOracle()
    print("u_from_mode_mag(DISCHARGE, mag):")
    for mag in (0.0, 0.5, 1.0):
        print(f"  mag={mag} -> u={u_from_mode_mag(CaesMode.DISCHARGE, mag)}")
    print("\nbase mask:", o._caes_mode_mask_base(STATE))
    print("full mask:", o._caes_mode_mask(STATE))
    print("\n逐动作一步预测：")
    for u in (-0.33, -0.5, -0.6976, -0.9, -1.0):
        pred = o.predict_next_state(STATE, PhysicalFmuAction(1.0, 0.0, u))
        ok, why = o.post_step_hard_ok(pred, use_safe=False)
        print(
            f"  u={u:+.4f}  pred_T_gas={pred['caes_gas_temperature']:.3f}  "
            f"ok={ok}  {why or ''}"
        )
    mask, intervals = o._caes_mask_and_intervals(STATE)
    print("\n安全幅值子区间:", intervals)
    print("magnitude caps:", o._caes_magnitude_caps(intervals))
    feas = o.compute(STATE, STATE["p_thermal"])
    print("\n可行集导出:")
    for k in (
        "caes_discharge_allowed",
        "caes_charge_allowed",
        "u_caes_discharge_low",
        "u_caes_discharge_high",
        "u_caes_charge_low",
        "u_caes_charge_high",
    ):
        print(f"  {k} = {feas.as_dict()[k]}")
    print("\n逐动作预检（check_action_executable）:")
    for u in (-1.0, -0.6976, -0.33, 0.86, 1.0):
        ok, why = o.check_action_executable(
            PhysicalFmuAction(0.5, 0.0, u), STATE, feas, STATE["p_thermal"]
        )
        print(f"  u={u:+.4f}  ok={ok}  {why or ''}")


if __name__ == "__main__":
    main()
