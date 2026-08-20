#!/usr/bin/env python
"""新 FMU 边界注入回归：喂 CSV / *_ref，校验内嵌表与物理驱动一致。

验收判据：
1. 每步 ``*_ref`` 与 ``data/*.csv`` 同源值一致（内嵌表仍在跑）；
2. CSV 驱动与「读 ref 再回写」两条轨迹的物理量逐点一致；
3. 短窗与可选全年均可跑通，无 FMU 失败。

用法::

    python scripts/regress_boundary_injection.py --hours 168
    python scripts/regress_boundary_injection.py --hours 8760
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config.paths import apply_process_cache_env, resolve_fmu_path  # noqa: E402

apply_process_cache_env()

from envs.boundary_provider import BoundaryProvider  # noqa: E402
from fmu.session import (  # noqa: E402
    BOUNDARY_REF_OUTPUTS,
    DEFAULT_OUTPUTS,
    FmuSession,
)

PHYS_KEYS = (
    "battery_soc",
    "caes_gas_soc",
    "caes_hot_soc",
    "caes_cold_soc",
    "p_thermal",
    "p_battery",
    "p_caes",
    "p_grid",
    "p_wind_available",
    "p_pv_available",
    "p_load_actual",
    "caes_gas_pressure",
    "caes_gas_temperature",
)

REF_TO_INPUT = {
    "v_wind_ref": "v_wind_in",
    "g_irradiance_ref": "g_irradiance_in",
    "t_air_ref": "t_air_in",
    "p_load_plan_ref": "p_load_plan_in",
}

IDLE_ACTION = {"u_tp": 1.0, "u_battery": 0.0, "u_caes": 0.0}


def make_session() -> FmuSession:
    cfg = yaml.safe_load((ROOT / "src/config/env_config.yaml").read_text(encoding="utf-8"))
    fmu_path = resolve_fmu_path(ROOT / cfg["fmu"]["path"])
    return FmuSession(
        fmu_path,
        step_size=float(cfg["fmu"]["communication_step_seconds"]),
        outputs=DEFAULT_OUTPUTS + BOUNDARY_REF_OUTPUTS,
        require_boundaries=True,
    )


def make_provider() -> BoundaryProvider:
    cfg = yaml.safe_load((ROOT / "src/config/env_config.yaml").read_text(encoding="utf-8"))
    return BoundaryProvider(
        ROOT,
        cfg["boundaries"],
        annual_horizon_hours=int(cfg["fmu"]["annual_horizon_hours"]),
        step_seconds=float(cfg["fmu"]["decision_interval_seconds"]),
    )


def run_csv_driven(hours: int, provider: BoundaryProvider) -> list[dict[str, float]]:
    """用 CSV 真值驱动边界。"""
    session = make_session()
    try:
        rows: list[dict[str, float]] = []
        out = session.reset(0.0, boundaries=provider.at_time(0.0))
        rows.append({"time": 0.0, **out, **{f"in_{k}": v for k, v in provider.at_time(0.0).items()}})
        for _ in range(hours):
            t = float(session.time)
            boundaries = provider.at_time(t)
            out = session.step(IDLE_ACTION, boundaries=boundaries)
            rows.append(
                {
                    "time": float(session.time),
                    **out,
                    **{f"in_{k}": v for k, v in boundaries.items()},
                }
            )
        return rows
    finally:
        session.close()


def run_ref_feedback(hours: int) -> list[dict[str, float]]:
    """读内嵌表 *_ref，回写为边界输入（不依赖 CSV）。"""
    session = make_session()
    try:
        rows: list[dict[str, float]] = []
        # 初始化：先用默认边界进 init，读出 t=0 的 ref，再重新 reset 用 ref
        probe = session.reset(0.0)
        b0 = {dst: float(probe[src]) for src, dst in REF_TO_INPUT.items()}
        session.close()
        session = make_session()
        out = session.reset(0.0, boundaries=b0)
        rows.append({"time": 0.0, **out, **{f"in_{k}": v for k, v in b0.items()}})
        for _ in range(hours):
            # 当前通信点的 ref 即本步应写入的边界（ConstantSegments）
            boundaries = {dst: float(out[src]) for src, dst in REF_TO_INPUT.items()}
            out = session.step(IDLE_ACTION, boundaries=boundaries)
            rows.append(
                {
                    "time": float(session.time),
                    **out,
                    **{f"in_{k}": v for k, v in boundaries.items()},
                }
            )
        return rows
    finally:
        session.close()


def max_abs_diff(a: list[dict], b: list[dict], keys: tuple[str, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in keys:
        diffs = [abs(float(a[i][key]) - float(b[i][key])) for i in range(len(a))]
        out[key] = float(max(diffs)) if diffs else 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=168)
    parser.add_argument("--out-dir", type=str, default="runs/boundary_regress")
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    provider = make_provider()

    print(f"=== CSV-driven {args.hours}h ===", flush=True)
    csv_rows = run_csv_driven(args.hours, provider)
    print(f"=== ref-feedback {args.hours}h ===", flush=True)
    ref_rows = run_ref_feedback(args.hours)

    assert len(csv_rows) == len(ref_rows) == args.hours + 1

    # 判据 1：CSV 写入值 vs 内嵌表 *_ref（在 CSV 驱动轨迹上）
    ref_vs_csv: dict[str, float] = {}
    for ref_name, in_name in REF_TO_INPUT.items():
        diffs = []
        for row in csv_rows[:-1]:  # 最后一行是步后时刻，边界对应已推进
            # 行内存的是「本步写入的 in_*」与「步后读到的 *_ref」；
            # ConstantSegments 下步后 ref 是下一小时表值，不能直接比。
            # 改用：步前写入的 in 应等于该通信点 CSV；用 provider 重算。
            pass
        # 直接：每个通信点，写入值 == provider == 应用 ref 前的表值
        # 在 CSV 轨迹里，in_* 就是 provider；与同通信点重新读表比：
        ref_vs_csv[ref_name] = 0.0
    # 更干净：逐步对比「写入的边界」与「同一步开始时的 *_ref」
    # 需要在 step 前读 ref。为此重跑短诊断。
    session = make_session()
    try:
        provider_b = provider.at_time(0.0)
        out = session.reset(0.0, boundaries=provider_b)
        max_ref_err = {k: 0.0 for k in REF_TO_INPUT}
        for step in range(args.hours):
            t = float(session.time)
            boundaries = provider.at_time(t)
            for ref_name, in_name in REF_TO_INPUT.items():
                err = abs(float(out[ref_name]) - float(boundaries[in_name]))
                max_ref_err[ref_name] = max(max_ref_err[ref_name], err)
            out = session.step(IDLE_ACTION, boundaries=boundaries)
    finally:
        session.close()

    phys_err = max_abs_diff(csv_rows, ref_rows, PHYS_KEYS)
    summary = {
        "hours": args.hours,
        "max_ref_minus_csv": max_ref_err,
        "max_phys_csv_vs_reffeedback": phys_err,
        "ref_pass": all(
            e <= args.atol or e <= args.rtol * 1.0 for e in max_ref_err.values()
        ),
        "phys_pass": all(
            np.isfinite(e) and (e <= args.atol or e <= args.rtol * max(1.0, abs(e)))
            for e in phys_err.values()
        ),
    }
    # 物理量：用相对+绝对混合判据
    phys_fail = []
    for key, err in phys_err.items():
        scale = max(
            abs(float(csv_rows[0][key])),
            abs(float(csv_rows[-1][key])),
            1.0,
        )
        if err > args.atol + args.rtol * scale:
            phys_fail.append((key, err, scale))
    summary["phys_pass"] = len(phys_fail) == 0
    summary["phys_failures"] = [
        {"key": k, "max_abs_err": e, "scale": s} for k, e, s in phys_fail
    ]
    summary["pass"] = bool(summary["ref_pass"] and summary["phys_pass"])

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"[{'PASS' if summary['pass'] else 'FAIL'}] wrote {out_dir / 'summary.json'}", flush=True)
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
