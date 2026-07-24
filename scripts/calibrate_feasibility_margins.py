"""从探针结果回写 feasibility_margins.yaml 中的 measured residual P99。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main(summary_path: str = "runs/feasibility_probe/summary.json") -> None:
    """从探针 summary 回写 feasibility_margins.yaml 的 measured P99 残差。

    Args:
        summary_path: feasibility_probe 输出的 summary.json 相对路径。

    Raises:
        FileNotFoundError: summary 或 margins 配置文件不存在。
        json.JSONDecodeError: summary 非合法 JSON。
    """
    summary = json.loads((ROOT / summary_path).read_text(encoding="utf-8"))
    stats = summary.get("residual_stats_all") or {}
    by_mode = summary.get("residual_stats_by_mode") or {}
    margins_path = ROOT / "src" / "config" / "feasibility_margins.yaml"
    with margins_path.open(encoding="utf-8") as f:
        marg = yaml.safe_load(f)

    bat = marg.setdefault("battery", {})
    if "battery_soc" in stats:
        # 危险方向：充电时正 residual；放电时负 residual 取绝对值
        bat["residual_p99_charge_high"] = max(float(bat.get("residual_p99_charge_high", 0)), float(stats["battery_soc"]["p99"]))
        bat["residual_p99_discharge_low"] = max(
            float(bat.get("residual_p99_discharge_low", 0)),
            abs(float(stats["battery_soc"]["min"])),
        )
        bat["measured_from"] = summary_path

    caes = marg.setdefault("caes", {})
    charge_stats = by_mode.get("2") or {}
    discharge_stats = by_mode.get("0") or {}
    chg = caes.setdefault("charge", {})
    dis = caes.setdefault("discharge", {})
    if "caes_gas_soc" in charge_stats:
        chg["residual_p99_gas_high"] = max(float(chg.get("residual_p99_gas_high", 0)), float(charge_stats["caes_gas_soc"]["p99"]))
    if "caes_gas_soc" in discharge_stats:
        dis["residual_p99_gas_low"] = max(
            float(dis.get("residual_p99_gas_low", 0)),
            abs(float(discharge_stats["caes_gas_soc"]["min"])),
        )
    for key, dest, field in (
        ("caes_hot_soc", chg, "residual_p99_hot_high"),
        ("caes_cold_soc", chg, "residual_p99_cold_high"),
        ("caes_gas_pressure", chg, "residual_p99_pressure_high"),
    ):
        if key in charge_stats:
            dest[field] = max(float(dest.get(field, 0)), float(charge_stats[key]["p99"]))
    for key, dest, field in (
        ("caes_hot_soc", dis, "residual_p99_hot_low"),
        ("caes_cold_soc", dis, "residual_p99_cold_low"),
        ("caes_gas_pressure", dis, "residual_p99_pressure_low"),
    ):
        if key in discharge_stats:
            dest[field] = max(float(dest.get(field, 0)), abs(float(discharge_stats[key]["min"])))

    marg["oracle_version"] = "d5.2-probe-calibrated"
    with margins_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(marg, f, allow_unicode=True, sort_keys=False)
    print(f"updated {margins_path} oracle_version={marg['oracle_version']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/feasibility_probe/summary.json")
