#!/usr/bin/env python
"""生成季节分桶整周 bootstrap 情景年。

默认 K=10：year_000 基准恒等拷贝 + year_001..009 bootstrap。

用法::

    python scripts/generate_scenario_years.py
    python scripts/generate_scenario_years.py --n-years 10 --seed 0
    python scripts/generate_scenario_years.py --n-years 5 --out-dir data/scenarios
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.scenario_years import ScenarioYearGenerator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-years", type=int, default=10, help="情景年总数（含基准）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="data/scenarios")
    parser.add_argument(
        "--calendar-year",
        type=int,
        default=2019,
        help="用于映射周→气象季节的非闰年日历（仅影响季节分桶）",
    )
    args = parser.parse_args()

    gen = ScenarioYearGenerator(
        root=ROOT,
        out_root=ROOT / args.out_dir,
        seed=int(args.seed),
        n_years=int(args.n_years),
        calendar_year=int(args.calendar_year),
    )
    manifest = gen.generate()
    print(json.dumps({
        "out_root": manifest["out_root"],
        "n_years": manifest["n_years"],
        "seed": manifest["seed"],
        "season_buckets": {k: len(v) for k, v in manifest["season_buckets"].items()},
        "years": [y["id"] for y in manifest["years"]],
    }, indent=2, ensure_ascii=False))
    print(f"wrote {ROOT / args.out_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
