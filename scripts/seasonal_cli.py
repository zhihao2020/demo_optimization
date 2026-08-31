"""Stdlib-only CLI for the seasonal suite (safe to import in unit tests)."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Formal paper split: 9 train / 2 val / 2 test per quarter (36/8/8). Eval week = first TEST week.
# Keep winter/transition/summer keys so existing launchers still parse.
SEASON_WEEKS = {
    "winter": {"train": list(range(0, 9)), "val": [9, 10], "test": [11, 12], "eval": 11},
    "transition": {"train": list(range(13, 22)), "val": [22, 23], "test": [24, 25], "eval": 24},
    "summer": {"train": list(range(26, 35)), "val": [35, 36], "test": [37, 38], "eval": 37},
    "autumn": {"train": list(range(39, 48)), "val": [48, 49], "test": [50, 51], "eval": 50},
    "all": {
        "train": list(range(0, 9))
        + list(range(13, 22))
        + list(range(26, 35))
        + list(range(39, 48)),
        "val": [9, 10, 22, 23, 35, 36, 48, 49],
        "test": [11, 12, 24, 25, 37, 38, 50, 51],
        "eval": 11,
    },
}
EPISODE_HOURS = 168
RL_METHODS = ("hmsd", "td3", "sac", "fs_hsac")
ALL_METHODS = ("hmsd", "td3", "sac", "fs_hsac", "pso", "linprog", "milp", "rule")
TD3_ABLATIONS = ("none", "projection", "static-support")
FORECAST_MODES = ("perfect", "noisy")
# Physical-step budgets from 检查.txt Stage A–D. Stage A is support-only (no FMU train).
STAGE_STEPS = {"A": 0, "B": 5000, "C": 30000, "D": 400000}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fair seasonal suite. Paper mainline: --method td3 (PC-HybridTD3)."
    )
    p.add_argument("--method", choices=list(ALL_METHODS), required=True)
    p.add_argument("--season", choices=list(SEASON_WEEKS.keys()), required=True)
    p.add_argument(
        "--episodes",
        type=int,
        default=5000,
        help="RL E_max; steps=E*168 (ignored for pso/linprog/milp)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--config", type=str, default=str(ROOT / "src/config/ghtd3_config.yaml"))
    p.add_argument("--train-weeks", type=str, default=None)
    p.add_argument("--val-weeks", type=str, default=None)
    p.add_argument("--test-weeks", type=str, default=None)
    p.add_argument("--eval-week", type=int, default=None, help="Override; default = first TEST week")
    p.add_argument("--single-week", action="store_true")
    p.add_argument(
        "--ablation",
        choices=list(TD3_ABLATIONS),
        default="none",
        help="TD3 only: none=PC-HybridTD3; projection=box+clamp; static-support=hybrid on static bands",
    )
    p.add_argument(
        "--lock-caes",
        action="store_true",
        help="Story A counterfactual: force u_caes=0 (idle) for train+eval",
    )
    p.add_argument(
        "--support",
        action="store_true",
        help="Archive FS-HSAC: disable residual C_ψ (same as --no-feas). Not paper mainline.",
    )
    p.add_argument(
        "--no-feas",
        action="store_true",
        dest="no_feas",
        help="Alias of --support (archive FS-HSAC only).",
    )
    p.add_argument("--pso-iters", type=int, default=25)
    p.add_argument("--pso-particles", type=int, default=12)
    p.add_argument(
        "--forecast-mode",
        choices=list(FORECAST_MODES),
        default="perfect",
        help="24 h forecast: perfect (main table) or noisy (σ_w=10%%, σ_s=10%%, σ_L=8%%)",
    )
    p.add_argument(
        "--stage",
        choices=("A", "B", "C", "D"),
        default=None,
        help="PC-HybridTD3 staged budget: A=support-only, B=5k, C=30k, D=400k physical steps",
    )
    p.add_argument(
        "--annual-eval",
        action="store_true",
        help="After RL train, run 8760 h deployment eval (not a TEST-week score)",
    )
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    args.support_only = bool(args.support or args.no_feas)
    return args
