#!/usr/bin/env python
"""Train Safe Market-GHTD3 sensitivity grid over c and alpha_end (seed 0).

Each run uses 15k valid steps by default. After all runs (or using existing),
writes runs/ghtd3_sens_summary.json for plotting.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (tag, config relative path, kind, value)
GRID = [
    ("c4", "src/config/ghtd3_config_abs_sens_c4.yaml", "c", 4),
    ("c8", "src/config/ghtd3_config_abs.yaml", "c", 8),
    ("c12", "src/config/ghtd3_config_abs_sens_c12.yaml", "c", 12),
    ("c24", "src/config/ghtd3_config_abs_sens_c24.yaml", "c", 24),
    ("a010", "src/config/ghtd3_config_abs_sens_a010.yaml", "alpha", 0.10),
    ("a022", "src/config/ghtd3_config_abs.yaml", "alpha", 0.22),
    ("a035", "src/config/ghtd3_config_abs_sens_a035.yaml", "alpha", 0.35),
    ("a050", "src/config/ghtd3_config_abs_sens_a050.yaml", "alpha", 0.50),
    ("a070", "src/config/ghtd3_config_abs_sens_a070.yaml", "alpha", 0.70),
]


def train_one(tag: str, config: str, steps: int, seed: int, python: str) -> Path:
    run_dir = ROOT / "runs" / f"ghtd3_sens_{tag}_s{seed}_{steps // 1000}k"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "checkpoints" / "ghtd3.pt"
    if ckpt.is_file():
        print(f"[skip train] exists {ckpt}", flush=True)
        return run_dir
    cmd = [
        python,
        str(ROOT / "scripts" / "train_ghtd3.py"),
        "--mode",
        "custom",
        "--steps",
        str(steps),
        "--seed",
        str(seed),
        "--run-dir",
        str(run_dir),
        "--config",
        str(ROOT / config),
    ]
    print("RUN", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))
    return run_dir


def eval_three_seasons(run_dir: Path, config: str, python: str) -> list[dict]:
    """Call eval_ghtd3_vs_td3 if TD3 scratch exists; else skip TD3 and only eval GHTD3 via export-like path."""
    gh = run_dir / "checkpoints" / "ghtd3.pt"
    td3 = ROOT / "runs" / "td3_scratch_s0_35k" / "checkpoints" / "hybrid_givesafe_td3.pt"
    out = run_dir / "vs_td3.json"
    if not gh.is_file():
        return []
    if not td3.is_file():
        print(f"[warn] no td3 ckpt for pairing: {td3}", flush=True)
        return []
    cmd = [
        python,
        str(ROOT / "scripts" / "eval_ghtd3_vs_td3.py"),
        "--ghtd3",
        str(gh),
        "--td3",
        str(td3),
        "--config",
        str(ROOT / config),
        "--out",
        str(out),
    ]
    print("EVAL", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))
    return json.loads(out.read_text(encoding="utf-8"))


def aggregate(results: list[dict]) -> dict:
    """Build plot-ready summary from completed runs."""
    by_c: dict[float, list[float]] = {}
    by_a: dict[float, list[float]] = {}
    meta = []
    for r in results:
        rews = [float(row.get("ghtd3_reward") or 0.0) for row in r["rows"]]
        mean_r = float(sum(rews) / max(len(rews), 1))
        meta.append({**r["spec"], "reward_mean": mean_r, "rewards": rews, "run_dir": r["run_dir"]})
        if r["spec"]["kind"] == "c":
            by_c.setdefault(float(r["spec"]["value"]), []).append(mean_r)
        else:
            by_a.setdefault(float(r["spec"]["value"]), []).append(mean_r)

    def pack(d: dict[float, list[float]]):
        out = []
        for v in sorted(d):
            ys = d[v]
            out.append(
                {
                    "value": v,
                    "reward_mean": float(sum(ys) / len(ys)),
                    "reward_std": float(pd_std(ys)),
                }
            )
        return out

    return {"c": pack(by_c), "alpha": pack(by_a), "runs": meta}


def pd_std(ys: list[float]) -> float:
    if len(ys) < 2:
        return 0.0
    m = sum(ys) / len(ys)
    return (sum((y - m) ** 2 for y in ys) / (len(ys) - 1)) ** 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--only", type=str, default=None, help="comma tags e.g. c4,c12,a035")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    only = {x.strip() for x in args.only.split(",")} if args.only else None
    results = []
    for tag, config, kind, value in GRID:
        if only is not None and tag not in only:
            continue
        # skip duplicate training of abs yaml for both c8 and a022 — train once as c8
        if tag == "a022":
            run_dir = ROOT / "runs" / f"ghtd3_sens_c8_s{args.seed}_{args.steps // 1000}k"
        else:
            run_dir = ROOT / "runs" / f"ghtd3_sens_{tag}_s{args.seed}_{args.steps // 1000}k"
        if args.train and tag != "a022":
            train_one(tag, config, args.steps, args.seed, args.python)
        rows = []
        vs = run_dir / "vs_td3.json"
        if args.eval or (not vs.is_file() and (run_dir / "checkpoints" / "ghtd3.pt").is_file()):
            if args.eval or args.train:
                rows = eval_three_seasons(run_dir, config, args.python)
        if vs.is_file():
            rows = json.loads(vs.read_text(encoding="utf-8"))
        if rows:
            results.append(
                {
                    "spec": {"tag": tag, "kind": kind, "value": value, "config": config},
                    "rows": rows,
                    "run_dir": str(run_dir),
                }
            )

    summary = aggregate(results)
    out = ROOT / "runs" / "ghtd3_sens_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", out, "c_pts", len(summary["c"]), "a_pts", len(summary["alpha"]))


if __name__ == "__main__":
    main()
