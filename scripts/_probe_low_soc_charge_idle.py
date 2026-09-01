"""Directed low-SOC charge→idle matrix on 0831. No training, no Oracle change.

Tries internal Mdot names. 0831 modelDescription has 95 vars and no independent
Mdot_c1 / coldtank.port_*.m_flow; mflow_*.x3 share t_air_in VR.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config.paths import apply_process_cache_env, file_sha256  # noqa: E402
from fmu.session import DEFAULT_OUTPUTS, FmuSession, describe_fmu  # noqa: E402


MDOT_CANDIDATES = (
    "Mdot_c1",
    "Mdot_c2",
    "compressedAirEnergyStorage.Mdot_c1",
    "compressedAirEnergyStorage.Mdot_c2",
    "compressedAirEnergyStorage.coldtank.port_a.m_flow",
    "compressedAirEnergyStorage.coldtank.port_b.m_flow",
    "compressedAirEnergyStorage.coldtank.m",
    "compressedAirEnergyStorage.mflow_coldtank.x3",
    "compressedAirEnergyStorage.mflow_hottank.x3",
    "compressedAirEnergyStorage.mflow_gasTank.x3",
)
MAP_CANDIDATES = (
    "u_caes",
    "compressedAirEnergyStorage.u_dispatch",
    "caes_cold_soc",
    "compressedAirEnergyStorage.coldtank.SOC",
    "p_caes",
    "compressedAirEnergyStorage.PBS.P_act",
)


def raw_step(sess: FmuSession, u: float) -> dict:
    sess.set_inputs({"u_tp": 1.0, "u_battery": 0.0, "u_caes": float(u)})
    rb = dict(sess.last_input_readback)
    sess._fmu.doStep(currentCommunicationPoint=sess.time, communicationStepSize=3600.0)
    sess.time += 3600.0
    values = sess._fmu.getFloat64(sess._read_vrs)
    out = dict(zip(sess.outputs, (float(v) for v in values)))
    out["_u_rb"] = rb.get("u_caes")
    for name in MDOT_CANDIDATES + MAP_CANDIDATES:
        out[name] = sess.try_get(name)
    return out


def internals(sess: FmuSession, out: dict) -> dict:
    return {
        "u_caes_readback": out.get("_u_rb"),
        "u_dispatch": out.get("compressedAirEnergyStorage.u_dispatch"),
        "cold_soc": out.get("caes_cold_soc"),
        "coldtank_SOC": out.get("compressedAirEnergyStorage.coldtank.SOC"),
        "p_caes": out.get("p_caes"),
        "P_act": out.get("compressedAirEnergyStorage.PBS.P_act"),
        "Mdot_c1": out.get("Mdot_c1") or out.get("compressedAirEnergyStorage.Mdot_c1"),
        "Mdot_c2": out.get("Mdot_c2") or out.get("compressedAirEnergyStorage.Mdot_c2"),
        "port_a_m_flow": out.get("compressedAirEnergyStorage.coldtank.port_a.m_flow"),
        "port_b_m_flow": out.get("compressedAirEnergyStorage.coldtank.port_b.m_flow"),
        "coldtank_m": out.get("compressedAirEnergyStorage.coldtank.m"),
        "mflow_coldtank_x3": out.get("compressedAirEnergyStorage.mflow_coldtank.x3"),
        "vr_mflow_coldtank": sess.value_reference("compressedAirEnergyStorage.mflow_coldtank.x3"),
        "vr_t_air_in": sess.value_reference("t_air_in"),
    }


def drive_to_cold(sess: FmuSession, target: float, cap: int = 400) -> dict:
    out = {
        k: sess.try_get(k) if k not in sess.outputs else None
        for k in ()
    }
    values = sess._fmu.getFloat64(sess._read_vrs)
    out = dict(zip(sess.outputs, (float(v) for v in values)))
    h = 0
    while float(out["caes_cold_soc"]) > target + 0.005 and h < cap:
        gas = float(out["caes_gas_soc"])
        u = 0.93 if gas < 0.94 else -0.50
        out = raw_step(sess, u)
        h += 1
    out["_drive_h"] = h
    return out


def main() -> int:
    apply_process_cache_env()
    fmu = ROOT / "data" / "TypicalScensrio_Example_TypicalScene_PowerSystem_8760h.fmu"
    ident = describe_fmu(fmu)
    print("IDENT", json.dumps(ident, indent=2))
    extra = (
        "compressedAirEnergyStorage.coldtank.SOC",
        "compressedAirEnergyStorage.PBS.P_act",
    )
    sess = FmuSession(
        fmu,
        step_size=3600.0,
        outputs=tuple(DEFAULT_OUTPUTS) + extra,
        require_boundaries=False,
    )
    print("VR_ALIASES")
    seen: dict[int, list[str]] = {}
    for name, vr in sess._vrs.items():
        seen.setdefault(int(vr), []).append(name)
    for vr, names in sorted(seen.items()):
        if len(names) > 1:
            print(" ", vr, names)

    colds = (0.25, 0.20, 0.15, 0.12, 0.10)
    us = (0.86, 0.93, 1.0)
    lengths = (1, 2, 4, 8, 16)
    rows = []
    for cold_t in colds:
        for u_chg in us:
            for length in lengths:
                sess.reset(0.0)
                driven = drive_to_cold(sess, cold_t)
                if float(driven["caes_cold_soc"]) > cold_t + 0.02:
                    rows.append(
                        {
                            "target_cold": cold_t,
                            "u_chg": u_chg,
                            "L": length,
                            "reached": False,
                            "cold": driven["caes_cold_soc"],
                            "drive_h": driven.get("_drive_h"),
                        }
                    )
                    continue
                last = driven
                charged = 0
                for _ in range(length):
                    if float(last["caes_gas_soc"]) > 0.97:
                        break
                    last = raw_step(sess, u_chg)
                    charged += 1
                before = dict(last)
                after = raw_step(sess, 0.0)
                rec = {
                    "target_cold": cold_t,
                    "u_chg": u_chg,
                    "L_requested": length,
                    "L_actual": charged,
                    "reached": True,
                    "drive_h": driven.get("_drive_h"),
                    "before": internals(sess, before),
                    "after": internals(sess, after),
                    "d_cold": float(after["caes_cold_soc"]) - float(before["caes_cold_soc"]),
                    "p_caes_before": before["p_caes"],
                    "p_caes_after": after["p_caes"],
                    "u_rb_idle": after.get("_u_rb"),
                    "map_ok": before.get("caes_cold_soc")
                    == before.get("compressedAirEnergyStorage.coldtank.SOC"),
                }
                rows.append(rec)
                print(
                    f"cold~{before['caes_cold_soc']:.3f} u={u_chg} L={charged} "
                    f"d_cold={rec['d_cold']:+.4f} u_rb={after.get('_u_rb')} "
                    f"Mdot_c2={rec['after']['Mdot_c2']} "
                    f"mflow_x3={rec['after']['mflow_coldtank_x3']}"
                )
    sess.reset(0.0)
    floor = drive_to_cold(sess, 0.10, cap=400)
    for _ in range(8):
        if float(floor["caes_gas_soc"]) > 0.97:
            break
        floor = raw_step(sess, 0.93)
    before = dict(floor)
    after = raw_step(sess, 0.0)
    floor_idle = {
        "cold_floor_after_400h": float(before["caes_cold_soc"]),
        "gas": float(before["caes_gas_soc"]),
        "d_cold_idle": float(after["caes_cold_soc"]) - float(before["caes_cold_soc"]),
        "u_rb": after.get("_u_rb"),
        "before": internals(sess, before),
        "after": internals(sess, after),
    }
    print("FLOOR_IDLE", json.dumps(floor_idle, indent=2, default=str))

    out_path = ROOT / "docs" / "_charge_idle_probe.json"
    payload = {
        "ident": ident,
        "sha256": file_sha256(fmu),
        "note": "mflow_*.x3 share t_air_in VR; not mass flow. Mdot_c1/c2 absent.",
        "n": len(rows),
        "rows": rows,
        "floor_idle": floor_idle,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    drops = [r["d_cold"] for r in rows if r.get("reached") and "d_cold" in r]
    print("WROTE", out_path, "n", len(rows), "drops_min", min(drops) if drops else None, "max", max(drops) if drops else None)
    sess.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
