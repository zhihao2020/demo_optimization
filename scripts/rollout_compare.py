"""将 compare/ 开环方案接入 FMU 回放。

校验失败（含 CAES 禁区）时非零退出，不写假装成功的 summary。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from controllers.compare_schedule import SchemeName, build_plan_for_scheme
from fmu.session import FmuSession
from fmu.types import DispatchPlan, SimulationResult


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="compare 开环调度 → FMU rollout")
    parser.add_argument(
        "--scheme",
        choices=("output1", "output2", "both"),
        default="both",
        help="要回放的 compare 方案",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=168,
        help="回放小时数（默认 168；全年 8760）",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/compare"),
        help="输出目录",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/config/env_config.yaml"),
        help="环境配置（读取 FMU 路径与步长）",
    )
    return parser.parse_args()


def _load_fmu_settings(config_path: Path) -> tuple[Path, float]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fmu_cfg = cfg["fmu"]
    fmu_path = Path(fmu_cfg["path"])
    if not fmu_path.is_absolute():
        fmu_path = ROOT / fmu_path
    step = float(fmu_cfg.get("communication_step_seconds", 3600))
    return fmu_path, step


def _write_trajectory(
    path: Path,
    plan: DispatchPlan,
    result: SimulationResult,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    var_names = list(result.variables.keys())
    hours_done = int(result.metadata.get("hours_done", 0))
    # 初始时刻 + 每步后：长度 hours_done+1；动作对齐到步进后的行（index>=1）
    fieldnames = [
        "time",
        "step",
        "u_tp",
        "u_battery",
        "u_caes",
        *var_names,
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        n_rows = len(result.time)
        for i in range(n_rows):
            row = {
                "time": float(result.time[i]),
                "step": i,
                "u_tp": "" if i == 0 or i > hours_done else float(plan.u_tp[i - 1]),
                "u_battery": "" if i == 0 or i > hours_done else float(plan.u_battery[i - 1]),
                "u_caes": "" if i == 0 or i > hours_done else float(plan.u_caes[i - 1]),
            }
            for name in var_names:
                arr = result.variables[name]
                row[name] = float(arr[i]) if i < len(arr) else ""
            writer.writerow(row)


def _scheme_summary(
    scheme: str,
    plan: DispatchPlan,
    result: SimulationResult,
    traj_path: Path,
) -> dict:
    meta = dict(result.metadata)
    summary: dict = {
        "scheme": scheme,
        "hours_requested": int(len(plan.u_tp)),
        "hours_done": int(meta.get("hours_done", 0)),
        "simulation_failed": bool(meta.get("simulation_failed", False)),
        "error": meta.get("error"),
        "trajectory_csv": str(traj_path).replace("\\", "/"),
    }
    total = result.get("economic_cashflow_total")
    if total is not None and len(total) > 0:
        summary["economic_cashflow_total_final"] = float(total[-1])
        if len(total) > 1:
            summary["economic_cashflow_total_delta"] = float(total[-1] - total[0])
    return summary


def main() -> int:
    args = _parse_args()
    if args.hours <= 0:
        print(f"错误: --hours 必须为正，得到 {args.hours}", file=sys.stderr)
        return 2

    schemes: list[SchemeName]
    if args.scheme == "both":
        schemes = ["output1", "output2"]
    else:
        schemes = [args.scheme]  # type: ignore[list-item]

    fmu_path, step_seconds = _load_fmu_settings(args.config)
    if not fmu_path.is_file():
        print(f"错误: FMU 不存在: {fmu_path}", file=sys.stderr)
        return 2

    run_dir = args.run_dir
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    (run_dir / "trajectories").mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    exit_code = 0

    # 先校验全部 scheme；任一非法则整体失败、不写成功 summary
    plans: dict[str, tuple] = {}
    try:
        for scheme in schemes:
            plans[scheme] = build_plan_for_scheme(
                scheme, hours=args.hours, step_seconds=step_seconds
            )
    except ValueError as exc:
        print(f"校验失败（compare 规则不合法，未启动 FMU）: {exc}", file=sys.stderr)
        fail_path = run_dir / "validation_error.json"
        fail_path.write_text(
            json.dumps(
                {"ok": False, "error": str(exc), "schemes": schemes, "hours": args.hours},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return 1

    with FmuSession(fmu_path, step_size=step_seconds) as session:
        for scheme in schemes:
            plan, plan_meta = plans[scheme]
            result = session.rollout(plan, horizon_hours=args.hours, start_time=0.0)
            traj_path = run_dir / "trajectories" / f"{scheme}.csv"
            _write_trajectory(traj_path, plan, result)
            summary = _scheme_summary(scheme, plan, result, traj_path)
            summary.update(plan_meta)
            results[scheme] = summary
            if summary.get("simulation_failed"):
                exit_code = 1

    summary_path = run_dir / "summary.json"
    payload = {"ok": exit_code == 0, "schemes": results}
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
