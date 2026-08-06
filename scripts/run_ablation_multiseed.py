#!/usr/bin/env python
"""Matched-budget multi-seed ablations for MSGP / MS-HER / F-MLE.

Default: 15k steps, seeds 0,1,2; variants full + no prior/her/fmle.
Launches local sequential training unless --dry-run.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

VARIANTS = [
    ("full", "src/config/ghtd3_config_abs.yaml"),
    ("noprior", "src/config/ghtd3_config_abs_noprior.yaml"),
    ("noher", "src/config/ghtd3_config_abs_noher.yaml"),
    ("nofmle", "src/config/ghtd3_config_abs_nofmle.yaml"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", type=str, default=None, help="comma variants e.g. full,noher")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    only = {x.strip() for x in args.only.split(",")} if args.only else None

    jobs = []
    for tag, cfg in VARIANTS:
        if only is not None and tag not in only:
            continue
        for seed in seeds:
            run_dir = ROOT / "runs" / f"ghtd3_abs_abl_{tag}_s{seed}_{args.steps // 1000}k"
            # alias: existing folders use nofMLE naming without full
            if tag == "full":
                run_dir = ROOT / "runs" / f"ghtd3_abs_abl_full_s{seed}_{args.steps // 1000}k"
            jobs.append((tag, cfg, seed, run_dir))

    print(f"{len(jobs)} jobs", flush=True)
    for tag, cfg, seed, run_dir in jobs:
        ckpt = run_dir / "checkpoints" / "ghtd3.pt"
        if ckpt.is_file():
            print(f"[skip] {run_dir.name}", flush=True)
            continue
        cmd = [
            args.python,
            str(ROOT / "scripts" / "train_ghtd3.py"),
            "--mode",
            "custom",
            "--steps",
            str(args.steps),
            "--seed",
            str(seed),
            "--run-dir",
            str(run_dir),
            "--config",
            str(ROOT / cfg),
        ]
        print("RUN", " ".join(cmd), flush=True)
        if args.dry_run:
            continue
        subprocess.check_call(cmd, cwd=str(ROOT))
        # paired eval vs td3 same seed if exists
        td3 = ROOT / f"runs/td3_scratch_s{seed}_35k/checkpoints/hybrid_givesafe_td3.pt"
        if not td3.is_file():
            td3 = ROOT / f"runs/td3_scratch_s0_35k/checkpoints/hybrid_givesafe_td3.pt"
        gpath = run_dir / "checkpoints" / "ghtd3.pt"
        if td3.is_file() and gpath.is_file():
            out = run_dir / "vs_td3.json"
            ecmd = [
                args.python,
                str(ROOT / "scripts" / "eval_ghtd3_vs_td3.py"),
                "--ghtd3",
                str(gpath),
                "--td3",
                str(td3),
                "--config",
                str(ROOT / cfg),
                "--out",
                str(out),
            ]
            print("EVAL", " ".join(ecmd), flush=True)
            subprocess.check_call(ecmd, cwd=str(ROOT))

    print("done")


if __name__ == "__main__":
    main()
