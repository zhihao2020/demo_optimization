#!/usr/bin/env python
"""Delete outdated obs=163 paper/RL artifacts. Keep only seasonal_v1.

Dry-run by default.

Usage:
  python scripts/clean_stale_runs.py
  python scripts/clean_stale_runs.py --apply
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

KEEP_DIRS = frozenset({"seasonal_v1"})
KEEP_FILES = frozenset({"README.md"})


def _iter_targets() -> list[Path]:
    out: list[Path] = []
    if not RUNS.is_dir():
        return out
    for p in RUNS.iterdir():
        if p.is_dir() and p.name in KEEP_DIRS:
            continue
        if p.is_file() and p.name in KEEP_FILES:
            continue
        out.append(p)
    for name in (".pytest_cache",):
        q = ROOT / name
        if q.exists():
            out.append(q)
    for q in ROOT.rglob("__pycache__"):
        if ".venv" in q.parts or "node_modules" in q.parts:
            continue
        if "seasonal_v1" in q.parts:
            continue
        out.append(q)
    return sorted(out)


def _nbytes(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    n = 0
    for f in p.rglob("*"):
        if f.is_file():
            n += f.stat().st_size
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    targets = _iter_targets()
    total = sum(_nbytes(p) for p in targets)
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"{mode}: {len(targets)} paths (~{total / 1e6:.1f} MB); keep {sorted(KEEP_DIRS)}")
    for p in targets:
        rel = p.relative_to(ROOT).as_posix()
        print(f"  {rel}")
        if not args.apply:
            continue
        if p.is_file():
            p.unlink(missing_ok=True)
        elif p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
    print("done" if args.apply else "re-run with --apply to delete")


if __name__ == "__main__":
    main()
