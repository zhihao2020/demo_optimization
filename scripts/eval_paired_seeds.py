#!/usr/bin/env python
"""分 seed 配对评估：TD3-scratch_si vs GHTD3-abs_si 三季 + TD3 全 seed 表。

输出：
  runs/td3_scratch_s{i}_35k/season_eval.json
  runs/ghtd3_abs_s{i}_35k/vs_td3_paired.json
  runs/ghtd3_abs/multi_seed_summary_paired.json
  runs/td3_scratch/season_multi_seed.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from config.paths import apply_process_cache_env  # noqa: E402

apply_process_cache_env()

from eval_ghtd3_vs_td3 import (  # noqa: E402
    _start_for_season,
    eval_ghtd3,
    eval_rule,
    eval_td3,
)
from envs.power_system_env import PowerSystemEnv  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--steps-tag", default="35k", help="run dir suffix e.g. 35k")
    p.add_argument("--td3-prefix", default="runs/td3_scratch")
    p.add_argument("--ghtd3-prefix", default="runs/ghtd3_abs")
    p.add_argument("--ghtd3-config", default=None)
    args = p.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    seasons = ["winter", "transition", "summer"]

    env_tmp = PowerSystemEnv(run_id="paired_meta")
    starts = {name: _start_for_season(env_tmp, name, i) for i, name in enumerate(seasons)}
    env_tmp.close()

    # B0 once
    b0 = {name: eval_rule(starts[name]) for name in seasons}

    td3_by_seed: dict[int, list[dict]] = {}
    for seed in seeds:
        td3_ckpt = Path(f"{args.td3_prefix}_s{seed}_{args.steps_tag}/checkpoints/hybrid_givesafe_td3.pt")
        if not td3_ckpt.is_file():
            print(f"[skip] missing td3 {td3_ckpt}", flush=True)
            continue
        rows = []
        for name in seasons:
            start = starts[name]
            print(f"TD3 s{seed} {name} ...", flush=True)
            td = eval_td3(td3_ckpt, start)
            row = {
                "seed": seed,
                "season": name,
                "start_time": start,
                "b0_reward": b0[name].get("episode_reward"),
                "b0_soc": b0[name].get("terminal_soc_satisfied"),
                "td3_reward": td.get("episode_reward"),
                "td3_soc": td.get("terminal_soc_satisfied"),
                "td3_caes_mwh": td.get("caes_throughput_mwh"),
                "td3_bat_mwh": td.get("battery_throughput_mwh"),
                "td3_thermal_mwh": td.get("thermal_generation_mwh"),
            }
            rows.append(row)
            print(row, flush=True)
        td3_by_seed[seed] = rows
        outp = Path(f"{args.td3_prefix}_s{seed}_{args.steps_tag}/season_eval.json")
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print("wrote", outp, flush=True)

    # TD3 multi-seed rollup
    td3_multi = {}
    for name in seasons:
        rews, socs = [], []
        for seed, rows in td3_by_seed.items():
            r = next(x for x in rows if x["season"] == name)
            rews.append(float(r["td3_reward"] or 0))
            socs.append(bool(r.get("td3_soc")))
        if rews:
            td3_multi[name] = {
                "mean": float(np.mean(rews)),
                "std": float(np.std(rews, ddof=1) if len(rews) > 1 else 0.0),
                "soc_pass": int(sum(socs)),
                "n": len(rews),
                "per_seed": rews,
            }
    td3_sum_path = Path("runs/td3_scratch/season_multi_seed.json")
    td3_sum_path.parent.mkdir(parents=True, exist_ok=True)
    td3_sum_path.write_text(
        json.dumps({"seeds": list(td3_by_seed.keys()), "multi_seed": td3_multi}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("wrote", td3_sum_path, flush=True)

    # Paired GHTD3 vs same-seed TD3
    paired_by_seed: dict[int, list[dict]] = {}
    cfg = Path(args.ghtd3_config) if args.ghtd3_config else None
    for seed in seeds:
        gh_ckpt = Path(f"{args.ghtd3_prefix}_s{seed}_{args.steps_tag}/checkpoints/ghtd3.pt")
        td3_ckpt = Path(f"{args.td3_prefix}_s{seed}_{args.steps_tag}/checkpoints/hybrid_givesafe_td3.pt")
        if not gh_ckpt.is_file() or not td3_ckpt.is_file():
            print(f"[skip] pair s{seed}", flush=True)
            continue
        # prefer run-local config
        local_cfg = Path(f"{args.ghtd3_prefix}_s{seed}_{args.steps_tag}/config/ghtd3_config.yaml")
        use_cfg = local_cfg if local_cfg.is_file() else cfg
        rows = []
        for name in seasons:
            start = starts[name]
            print(f"PAIR s{seed} {name} ...", flush=True)
            td = eval_td3(td3_ckpt, start)
            gh = eval_ghtd3(gh_ckpt, start, config_path=use_cfg)
            td_r = float(td.get("episode_reward") or 0)
            gh_r = float(gh.get("episode_reward") or 0)
            row = {
                "seed": seed,
                "season": name,
                "start_time": start,
                "b0_reward": b0[name].get("episode_reward"),
                "b0_soc": b0[name].get("terminal_soc_satisfied"),
                "td3_reward": td.get("episode_reward"),
                "td3_soc": td.get("terminal_soc_satisfied"),
                "td3_caes_mwh": td.get("caes_throughput_mwh"),
                "ghtd3_reward": gh.get("episode_reward"),
                "ghtd3_soc": gh.get("terminal_soc_satisfied"),
                "ghtd3_caes_mwh": gh.get("caes_throughput_mwh"),
                "ghtd3_bat_mwh": gh.get("battery_throughput_mwh"),
                "delta_vs_td3": gh_r - td_r,
                "delta_pct_vs_td3": (100.0 * (gh_r - td_r) / abs(td_r)) if abs(td_r) > 1e-9 else None,
            }
            rows.append(row)
            print(row, flush=True)
        paired_by_seed[seed] = rows
        outp = Path(f"{args.ghtd3_prefix}_s{seed}_{args.steps_tag}/vs_td3_paired.json")
        outp.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print("wrote", outp, flush=True)

    multi = {}
    for name in seasons:
        rews, deltas, socs, td_rews, td_socs = [], [], [], [], []
        for seed, rows in paired_by_seed.items():
            r = next(x for x in rows if x["season"] == name)
            rews.append(float(r["ghtd3_reward"] or 0))
            deltas.append(float(r["delta_vs_td3"] or 0))
            socs.append(bool(r.get("ghtd3_soc")))
            td_rews.append(float(r["td3_reward"] or 0))
            td_socs.append(bool(r.get("td3_soc")))
        if not rews:
            continue
        multi[name] = {
            "ghtd3_mean": float(np.mean(rews)),
            "ghtd3_std": float(np.std(rews, ddof=1) if len(rews) > 1 else 0.0),
            "td3_mean": float(np.mean(td_rews)),
            "td3_std": float(np.std(td_rews, ddof=1) if len(td_rews) > 1 else 0.0),
            "delta_mean": float(np.mean(deltas)),
            "delta_std": float(np.std(deltas, ddof=1) if len(deltas) > 1 else 0.0),
            "ghtd3_soc_pass": int(sum(socs)),
            "td3_soc_pass": int(sum(td_socs)),
            "n": len(rews),
            "wins": int(sum(1 for d in deltas if d > 0)),
        }
    out = {
        "seeds": list(paired_by_seed.keys()),
        "pairing": "same_seed_td3_vs_ghtd3",
        "multi_seed": multi,
        "mean_delta_all": float(np.mean([multi[s]["delta_mean"] for s in multi])) if multi else None,
        "all_ghtd3_soc_pass": all(multi[s]["ghtd3_soc_pass"] == multi[s]["n"] for s in multi) if multi else False,
        "all_td3_soc_pass": all(multi[s]["td3_soc_pass"] == multi[s]["n"] for s in multi) if multi else False,
    }
    op = Path("runs/ghtd3_abs/multi_seed_summary_paired.json")
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", op, flush=True)
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
