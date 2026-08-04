#!/usr/bin/env python
"""Safe Market-GHTD3：主实验 50k+全年 + 消融表。

用法：
  python scripts/run_ghtd3_ablations.py --stage main
  python scripts/run_ghtd3_ablations.py --stage ablations
  python scripts/run_ghtd3_ablations.py --stage all
  python scripts/run_ghtd3_ablations.py --stage summarize
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.ghtd3.train import run_ghtd3_training  # noqa: E402


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def _make_variant_config(name: str, base: dict[str, Any], overrides: dict[str, Any]) -> Path:
    cfg = deepcopy(base)
    gh = dict(cfg.get("ghtd3") or {})
    gh.update(overrides)
    cfg["ghtd3"] = gh
    out = ROOT / "runs" / "ghtd3_ablation_configs" / f"{name}.yaml"
    _write_yaml(out, cfg)
    return out


def _pick(res: dict[str, Any]) -> dict[str, Any]:
    ev = res.get("eval") or {}
    terms = ev.get("cost_terms") or {}
    ann = res.get("annual_eval") or {}
    return {
        "status": res.get("status"),
        "valid_steps": res.get("valid_steps"),
        "episode_reward": ev.get("episode_reward"),
        "terminal_soc_satisfied": ev.get("terminal_soc_satisfied"),
        "terminal_soc_l1": terms.get("terminal_soc_l1_error"),
        "thermal_mwh": (ev.get("metrics") or {}).get("thermal_generation_mwh"),
        "economic_reward": terms.get("economic_reward"),
        "rule_reward": (res.get("rule") or {}).get("episode_reward"),
        "price_rule_reward": (res.get("price_rule") or {}).get("episode_reward"),
        "annual_episode_reward": ann.get("annual_episode_reward"),
        "annual_mean_reward": (
            (float(ann["annual_episode_reward"]) / max(int(ann.get("windows") or 1), 1))
            if ann.get("annual_episode_reward") is not None
            else None
        ),
        "annual_soc_pass": ann.get("terminal_soc_satisfied_windows"),
        "annual_windows": ann.get("windows") or ann.get("n_windows"),
        "annual_economic_cashflow": ann.get("annual_economic_cashflow"),
        "innovations": res.get("innovations"),
        "run_dir": res.get("run_dir"),
    }


def run_main(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = ROOT / "runs" / "ghtd3_market_50k_annual_20260803"
    resume = args.resume or str(ROOT / "runs/ghtd3_market_curriculum_20k_20260803/checkpoints/ghtd3.pt")
    if not Path(resume).exists():
        resume = None
    print(f"[main] steps={args.main_steps} resume={resume} annual=True -> {run_dir}")
    res = run_ghtd3_training(
        total_valid_steps=int(args.main_steps),
        run_dir=run_dir,
        seed=int(args.seed),
        annual_evaluation=True,
        resume_from=resume,
        skip_bc=bool(resume) or bool(args.skip_bc),
    )
    res["run_dir"] = str(run_dir)
    summary = _pick(res)
    (run_dir / "ablation_row.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def run_ablations(args: argparse.Namespace) -> list[dict[str, Any]]:
    base = _load_yaml(ROOT / "src/config/ghtd3_config.yaml")
    variants: list[tuple[str, dict[str, Any], str]] = [
        (
            "full",
            {},
            "完整 Safe Market-GHTD3",
        ),
        (
            "no_market_prior",
            {"market_goal_prior": False},
            "去掉市场 goal 先验",
        ),
        (
            "no_recovery_goal",
            {"recovery_goal_horizon_steps": 0, "recovery_prior_weight": 0.0},
            "去掉上层回收 goal（环境硬回收仍在）",
        ),
        (
            "no_bc",
            {"bc_pretrain": False},
            "去掉分层 BC",
        ),
        (
            "gamma_not_c",
            # agent 用 subgoal_interval=1 近似 γ^1；仍每 8 步换 goal
            {"subgoal_interval": 1, "note_ablation": "gamma_not_c via c=1"},
            "SMDP 折扣退化为 γ（c=1）",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for name, overrides, desc in variants:
        cfg_path = _make_variant_config(name, base, overrides)
        run_dir = ROOT / "runs" / f"ghtd3_ablation_{name}_{args.ablation_steps}"
        print(f"[ablation] {name}: {desc} -> {run_dir}")
        # gamma_not_c 改变 c 后结构不同，不 resume；其余可从 curriculum 冷启动或从头
        resume = None
        skip_bc = name == "no_bc"
        if name not in ("no_bc", "gamma_not_c") and args.resume_ablations:
            cand = ROOT / "runs/ghtd3_market_curriculum_20k_20260803/checkpoints/ghtd3.pt"
            if cand.exists() and name == "full":
                resume = str(cand)
                skip_bc = True
        res = run_ghtd3_training(
            total_valid_steps=int(args.ablation_steps),
            run_dir=run_dir,
            seed=int(args.seed),
            config_path=cfg_path,
            annual_evaluation=False,
            resume_from=resume,
            skip_bc=skip_bc,
        )
        res["run_dir"] = str(run_dir)
        row = _pick(res)
        row["name"] = name
        row["description"] = desc
        row["overrides"] = overrides
        (run_dir / "ablation_row.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2, default=str))
    out = ROOT / "runs" / "ghtd3_ablation_table.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    _write_markdown_table(rows, ROOT / "docs" / "GHTD3消融实验结果.md")
    return rows


def _write_markdown_table(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Safe Market-GHTD3 消融实验结果",
        "",
        "| 变体 | 周 reward | SOC 过关 | L1 | 火电 MWh | 说明 |",
        "|------|-----------|----------|-----|----------|------|",
    ]
    for r in rows:
        lines.append(
            "| {name} | {rew} | {soc} | {l1} | {th} | {desc} |".format(
                name=r.get("name"),
                rew=_fmt(r.get("episode_reward")),
                soc="是" if r.get("terminal_soc_satisfied") else "否",
                l1=_fmt(r.get("terminal_soc_l1")),
                th=_fmt(r.get("thermal_mwh"), 1),
                desc=r.get("description") or "",
            )
        )
    lines.extend(
        [
            "",
            "## 主实验（50k + 全年）",
            "",
            "见 `runs/ghtd3_market_50k_annual_20260803/summary.json`。",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(x: Any, nd: int = 2) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "—"


def summarize() -> dict[str, Any]:
    table_path = ROOT / "runs" / "ghtd3_ablation_table.json"
    main_path = ROOT / "runs" / "ghtd3_market_50k_annual_20260803" / "ablation_row.json"
    hybrid_path = ROOT / "runs" / "market_soc_ok_15k_20260803" / "summary.json"
    out: dict[str, Any] = {}
    if table_path.exists():
        out["ablations"] = json.loads(table_path.read_text(encoding="utf-8"))
    if main_path.exists():
        out["main"] = json.loads(main_path.read_text(encoding="utf-8"))
    elif (ROOT / "runs/ghtd3_market_50k_annual_20260803/summary.json").exists():
        s = json.loads((ROOT / "runs/ghtd3_market_50k_annual_20260803/summary.json").read_text(encoding="utf-8"))
        out["main"] = _pick(s)
    if hybrid_path.exists():
        s = json.loads(hybrid_path.read_text(encoding="utf-8"))
        ev = s.get("eval") or {}
        ann = s.get("annual_eval") or {}
        out["hybrid_sota"] = {
            "episode_reward": ev.get("episode_reward"),
            "terminal_soc_satisfied": ev.get("terminal_soc_satisfied"),
            "annual_soc_pass": ann.get("terminal_soc_satisfied_windows"),
            "annual_reward": ann.get("mean_episode_reward") or ann.get("total_episode_reward"),
        }
    out_path = ROOT / "runs" / "ghtd3_experiment_summary.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["main", "ablations", "all", "summarize"], default="all")
    p.add_argument("--main-steps", type=int, default=50000)
    p.add_argument("--ablation-steps", type=int, default=15000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--skip-bc", action="store_true")
    p.add_argument("--resume-ablations", action="store_true", help="full 消融可 resume curriculum")
    args = p.parse_args()

    if args.stage in ("main", "all"):
        main_row = run_main(args)
        print("[main done]", json.dumps(main_row, ensure_ascii=False, indent=2, default=str))
    if args.stage in ("ablations", "all"):
        rows = run_ablations(args)
        print(f"[ablations done] n={len(rows)}")
    if args.stage in ("summarize", "all"):
        summarize()


if __name__ == "__main__":
    main()
