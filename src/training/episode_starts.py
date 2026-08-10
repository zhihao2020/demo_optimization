"""Episode start times for seasonal / multi-week training and matching eval."""

from __future__ import annotations

import os
from typing import Any


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


def eval_start_seconds(fmu_cfg: dict[str, Any] | None = None) -> float:
    """Resolve evaluation start. Must not silently fall back when protocol sets training weeks.

    Priority:
    1. OPTIMAL_DEMO_EVAL_EPISODE_START
    2. OPTIMAL_DEMO_FORCE_EPISODE_START
    3. first of OPTIMAL_DEMO_TRAIN_WEEK_STARTS
    4. fmu start_time_seconds (default 0)
    """
    for key in ("OPTIMAL_DEMO_EVAL_EPISODE_START", "OPTIMAL_DEMO_FORCE_EPISODE_START"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return float(raw)
    pool = os.environ.get("OPTIMAL_DEMO_TRAIN_WEEK_STARTS", "").strip()
    if pool:
        starts = _parse_float_list(pool)
        if starts:
            return starts[0]
    if fmu_cfg is not None:
        return float(fmu_cfg.get("start_time_seconds", 0.0) or 0.0)
    return 0.0
