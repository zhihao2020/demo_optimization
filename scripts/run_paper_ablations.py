#!/usr/bin/env python
"""Launch HMSD paper ablations (local or print commands).

Examples:
  python scripts/run_paper_ablations.py --dry-run
  python scripts/run_paper_ablations.py --season winter --episodes 200 --seed 0
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ABLATIONS = [
    ("full", ROOT / "src/config/ghtd3_config.yaml"),
    ("no_her", ROOT / "src/config/ablation/ghtd3_no_her.yaml"),
    ("no_reject_learn", ROOT / "src/config/ablation/ghtd3_no_reject_learn.yaml"),
]


def main() -> None:
    p = argparse.ArgumentParser(description="HMSD paper ablations")
    p.add_argument("--season", default="winter", choices=["winter", "transition", "summer"])
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--include-flat-td3", action="store_true", default=True)
    args = p.parse_args()

    py = sys.executable
    script = ROOT / "scripts" / "train_seasonal.py"
    cmds: list[list[str]] = []
    for name, cfg in ABLATIONS:
        run_dir = ROOT / "runs" / "ablation" / f"{name}_{args.season}_s{args.seed}"
        cmds.append(
            [
                py,
                str(script),
                "--method",
                "hmsd",
                "--season",
                args.season,
                "--episodes",
                str(args.episodes),
                "--seed",
                str(args.seed),
                "--config",
                str(cfg),
                "--run-dir",
                str(run_dir),
            ]
        )
    if args.include_flat_td3:
        run_dir = ROOT / "runs" / "ablation" / f"flat_td3_{args.season}_s{args.seed}"
        cmds.append(
            [
                py,
                str(script),
                "--method",
                "td3",
                "--season",
                args.season,
                "--episodes",
                str(args.episodes),
                "--seed",
                str(args.seed),
                "--run-dir",
                str(run_dir),
            ]
        )

    for c in cmds:
        print(" ".join(c), flush=True)
        if not args.dry_run:
            subprocess.check_call(c, cwd=str(ROOT))


if __name__ == "__main__":
    main()
