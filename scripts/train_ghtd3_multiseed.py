#!/usr/bin/env python
"""多 seed 训练 TEA stable 并三季 eval + 汇总 mean±std。"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="src/config/ghtd3_config_tea_stable.yaml")
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--run-prefix", default="runs/ghtd3_tea_stable")
    p.add_argument(
        "--hybrid",
        default="runs/market_bc_rl_60k_20260803/checkpoints/hybrid_givesafe_td3.pt",
    )
    p.add_argument("--out", default="runs/ghtd3_tea_stable/multi_seed_summary.json")
    p.add_argument("--skip-train", action="store_true")
    args = p.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    rows_by_seed = {}

    for seed in seeds:
        run_dir = f"{args.run_prefix}_s{seed}_{args.steps // 1000}k"
        ckpt = ROOT / run_dir / "checkpoints" / "ghtd3.pt"
        if not args.skip_train and not ckpt.is_file():
            cmd = [
                sys.executable,
                "scripts/train_ghtd3.py",
                "--mode",
                "custom",
                "--steps",
                str(args.steps),
                "--seed",
                str(seed),
                "--run-dir",
                run_dir,
                "--config",
                args.config,
            ]
            print("+", " ".join(cmd), flush=True)
            rc = subprocess.call(cmd, cwd=str(ROOT))
            if rc != 0:
                print("train failed", seed, rc)
                continue
        vs = ROOT / run_dir / "vs_hybrid.json"
        if ckpt.is_file():
            cmd = [
                sys.executable,
                "scripts/eval_ghtd3_vs_hybrid.py",
                "--ghtd3",
                str(ckpt),
                "--hybrid",
                args.hybrid,
                "--out",
                str(vs),
            ]
            print("+", " ".join(cmd), flush=True)
            subprocess.call(cmd, cwd=str(ROOT))
        if vs.is_file():
            rows_by_seed[seed] = json.loads(vs.read_text(encoding="utf-8"))

    # multi-seed stats
    seasons = ("winter", "transition", "summer")
    multi = {}
    for s in seasons:
        rews, socs, deltas = [], [], []
        hy = None
        for seed, rows in rows_by_seed.items():
            r = next(x for x in rows if x["season"] == s)
            rews.append(float(r["ghtd3_reward"]))
            socs.append(bool(r.get("ghtd3_soc")))
            deltas.append(float(r["delta_vs_hybrid"]))
            hy = float(r["hybrid_reward"])
        if not rews:
            continue
        multi[s] = {
            "mean": float(np.mean(rews)),
            "std": float(np.std(rews, ddof=1) if len(rews) > 1 else 0.0),
            "hybrid": hy,
            "delta_mean": float(np.mean(deltas)),
            "delta_std": float(np.std(deltas, ddof=1) if len(deltas) > 1 else 0.0),
            "soc_pass": int(sum(socs)),
            "n": len(rews),
            "wins": int(sum(1 for d in deltas if d > 0)),
        }
    mean_delta = float(np.mean([multi[s]["delta_mean"] for s in multi])) if multi else 0.0
    all_soc = all(multi[s]["soc_pass"] == multi[s]["n"] for s in multi) if multi else False
    all_mean_ge = all(multi[s]["delta_mean"] >= 0 for s in multi) if multi else False
    verdict = (
        "PASS_Q1_EVIDENCE_CORE"
        if all_mean_ge and all_soc and mean_delta >= 1.0
        else ("PARTIAL" if mean_delta > 0 or any(multi[s]["wins"] >= 2 for s in multi) else "FAIL")
    )
    out = {
        "seeds": list(rows_by_seed.keys()),
        "by_seed": {
            str(k): {r["season"]: {"reward": r["ghtd3_reward"], "delta": r["delta_vs_hybrid"], "soc": r["ghtd3_soc"]} for r in v}
            for k, v in rows_by_seed.items()
        },
        "multi_seed": multi,
        "mean_delta_all": mean_delta,
        "all_seasons_mean_ge_hybrid": all_mean_ge,
        "all_soc_pass": all_soc,
        "verdict": verdict,
    }
    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    # markdown
    lines = [
        "# TEA Stable multi-seed",
        "",
        f"Verdict: **{verdict}**  mean Δ={mean_delta:+.3f}  all_mean≥Hybrid={all_mean_ge}  all_SOC={all_soc}",
        "",
        "| Season | mean±std | Hybrid | Δ mean±std | SOC pass | wins |",
        "|--------|----------|--------|------------|----------|------|",
    ]
    for s in seasons:
        if s not in multi:
            continue
        m = multi[s]
        lines.append(
            f"| {s} | {m['mean']:.2f}±{m['std']:.2f} | {m['hybrid']:.2f} | "
            f"{m['delta_mean']:+.2f}±{m['delta_std']:.2f} | {m['soc_pass']}/{m['n']} | {m['wins']}/{m['n']} |"
        )
    lines.append("")
    lines.append("## Per seed")
    for seed, rows in rows_by_seed.items():
        parts = [f"s{seed}"]
        for r in rows:
            parts.append(f"{r['season']}:{r['ghtd3_reward']:.2f}({r['delta_vs_hybrid']:+.2f},SOC={r['ghtd3_soc']})")
        lines.append("- " + " | ".join(parts))
    md = outp.with_suffix(".md")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md.read_text(encoding="utf-8"))
    print("wrote", outp)


if __name__ == "__main__":
    main()
