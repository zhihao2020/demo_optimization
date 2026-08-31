"""Episode start times for seasonal / multi-week training and matching eval.

Formal paper split (重构.txt): 52 complete weeks, stratified 9/2/2 per quarter
→ 36 train / 8 validation / 8 test. Training starts only from TRAIN_WEEK_IDS.
Checkpoint selection uses VAL; reported tables use TEST. A full 8760 h rollout
is deployment evaluation, not a held-out test.
"""

from __future__ import annotations

import os
from typing import Any

# Week index i starts at hour 168*i on the 8760 h calendar (week 51 is last full week).
QUARTER_WEEK_BLOCKS: dict[str, tuple[int, ...]] = {
    "winter": tuple(range(0, 13)),
    "transition": tuple(range(13, 26)),
    "summer": tuple(range(26, 39)),
    "autumn": tuple(range(39, 52)),
}


def _split_block(weeks: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if len(weeks) != 13:
        raise ValueError(f"quarter block must have 13 weeks, got {len(weeks)}")
    return weeks[:9], weeks[9:11], weeks[11:13]


SEASON_WEEK_SPLIT: dict[str, dict[str, tuple[int, ...]]] = {}
_train: list[int] = []
_val: list[int] = []
_test: list[int] = []
for _name, _block in QUARTER_WEEK_BLOCKS.items():
    _tr, _va, _te = _split_block(_block)
    SEASON_WEEK_SPLIT[_name] = {"train": _tr, "val": _va, "test": _te}
    _train.extend(_tr)
    _val.extend(_va)
    _test.extend(_te)

TRAIN_WEEK_IDS: tuple[int, ...] = tuple(_train)
VAL_WEEK_IDS: tuple[int, ...] = tuple(_val)
TEST_WEEK_IDS: tuple[int, ...] = tuple(_test)
SEASON_WEEK_SPLIT["all"] = {
    "train": TRAIN_WEEK_IDS,
    "val": VAL_WEEK_IDS,
    "test": TEST_WEEK_IDS,
}


def _parse_float_list(raw: str) -> list[float]:
    out: list[float] = []
    for part in raw.split(","):
        s = part.strip()
        if s:
            out.append(float(s))
    return out


def training_start_seconds(
    fmu_cfg: dict[str, Any],
    episode_steps: int,
    episode_index: int,
    *,
    annual_episode_start_seconds,
) -> float:
    """Resolve training episode start.

    Priority:
    1. OPTIMAL_DEMO_TRAIN_WEEK_STARTS — comma-separated seconds, round-robin by episode_index
    2. OPTIMAL_DEMO_FORCE_EPISODE_START — single fixed week (debug)
    3. annual window from episode_index
    """
    pool = os.environ.get("OPTIMAL_DEMO_TRAIN_WEEK_STARTS", "").strip()
    if pool:
        starts = _parse_float_list(pool)
        if starts:
            return starts[int(episode_index) % len(starts)]
    force = os.environ.get("OPTIMAL_DEMO_FORCE_EPISODE_START", "").strip()
    if force:
        return float(force)
    return float(annual_episode_start_seconds(fmu_cfg, episode_steps, episode_index))


def eval_start_seconds(fmu_cfg: dict[str, Any] | None = None, *, formal: bool | None = None) -> float:
    """Resolve evaluation start.

    Formal paper protocol requires an explicit eval/test week. Silent fallback
    onto a training week is a configuration error.
    """
    if formal is None:
        formal = os.environ.get("OPTIMAL_DEMO_FORMAL_SPLIT", "").strip() in {"1", "true", "True", "yes"}
    raw_eval = os.environ.get("OPTIMAL_DEMO_EVAL_EPISODE_START", "").strip()
    if raw_eval:
        return float(raw_eval)
    if formal:
        raise ValueError(
            "formal split: set OPTIMAL_DEMO_EVAL_EPISODE_START to a TEST week; "
            "do not fall back to a training week"
        )
    force = os.environ.get("OPTIMAL_DEMO_FORCE_EPISODE_START", "").strip()
    if force:
        return float(force)
    pool = os.environ.get("OPTIMAL_DEMO_TRAIN_WEEK_STARTS", "").strip()
    if pool:
        starts = _parse_float_list(pool)
        if starts:
            return starts[0]
    if fmu_cfg is not None:
        return float(fmu_cfg.get("start_time_seconds", 0.0) or 0.0)
    return 0.0
