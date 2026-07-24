#!/usr/bin/env python
"""从已有训练 run 目录生成可读策略报告。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.report_policy_run import generate_policy_report


def main() -> int:
    p = argparse.ArgumentParser(description="生成策略评估报告 (report.md + 图)")
    p.add_argument("--run-dir", type=Path, required=True, help="如 runs/givesafe_td3_smoke")
    args = p.parse_args()
    run_dir = args.run_dir
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    path = generate_policy_report(run_dir)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
