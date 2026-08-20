#!/usr/bin/env python
"""FS-HSAC smoke: unit gates + optional short FMU roll.

Usage:
  python logs/_smoke_fs_hsac.py
  python logs/_smoke_fs_hsac.py --fmu-steps 50
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_pytest() -> None:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_fs_hsac_support.py",
        "tests/test_fs_hsac_algorithm.py",
        "tests/test_fs_hsac_replay.py",
        "tests/test_fs_hsac_feasibility.py",
        "-q",
    ]
    env = dict(**{**subprocess.os.environ, "PYTHONPATH": str(ROOT / "src")})
    print("RUN", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT, env=env)


def run_fmu_smoke(steps: int) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from training.fs_hsac.train import run_fs_hsac_training

    out = run_fs_hsac_training(
        total_valid_steps=int(steps),
        run_dir=ROOT / "runs" / "fs_hsac_smoke",
        seed=0,
        learning_starts=min(32, steps),
        batch_size=32,
        enable_shadow=False,
        use_feasibility_penalty=True,
    )
    print("FMU_SMOKE", out.get("status"), "valid_steps", out.get("valid_steps"), flush=True)
    if out.get("status") not in ("completed", "running"):
        raise SystemExit(f"FMU smoke failed: {out.get('status')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmu-steps", type=int, default=0, help="If >0, run short FMU training")
    args = ap.parse_args()
    run_pytest()
    print("UNIT_GATES_OK", flush=True)
    if args.fmu_steps > 0:
        run_fmu_smoke(args.fmu_steps)
    print("ALL_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
