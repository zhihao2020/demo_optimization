"""Stdlib-only CLI for the seasonal suite (safe to import in unit tests)."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SEASON_WEEKS = {
    "winter": {"train": [0, 1, 2, 3, 4], "eval": 5},
    "transition": {"train": [13, 14, 15, 16, 17], "eval": 18},
    "summer": {"train": [26, 27, 28, 29, 30], "eval": 31},
}
EPISODE_HOURS = 168
RL_METHODS = ("hmsd", "td3", "sac", "fs_hsac")
ALL_METHODS = ("hmsd", "td3", "sac", "fs_hsac", "pso", "linprog", "milp")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fair seasonal suite (RL + baselines)")
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
    p.add_argument("--eval-week", type=int, default=None)
    p.add_argument("--single-week", action="store_true")
    p.add_argument(
        "--lock-caes",
        action="store_true",
        help="Story A counterfactual: force u_caes=0 (idle) for train+eval",
    )
    p.add_argument(
        "--support",
        action="store_true",
        help=(
            "Paper mainline FS-HSAC: disable residual C_ψ (same as --no-feas / "
            "FS_HSAC_NO_FEAS=1). Default off so bare --method fs_hsac is full FS-HSAC."
        ),
    )
    p.add_argument(
        "--no-feas",
        action="store_true",
        dest="no_feas",
        help="Alias of --support: use_feasibility_penalty=False / FS_HSAC_NO_FEAS=1.",
    )
    p.add_argument("--pso-iters", type=int, default=25)
    p.add_argument("--pso-particles", type=int, default=12)
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    args.support_only = bool(args.support or args.no_feas)
    return args
