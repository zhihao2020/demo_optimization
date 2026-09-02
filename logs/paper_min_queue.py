"""Paper-min holdout queue: Rule/MILP then PC-HybridTD3 on weeks 12/25/38/51."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"D:\xuzh\demo_optimization")
PY = ROOT / ".venv" / "Scripts" / "python.exe"
SCRIPT = ROOT / "scripts" / "eval_final_holdout.py"
STATE = ROOT / "logs" / "paper_min_queue_state.json"
LOG = ROOT / "logs" / "paper_min_queue_inner.log"
CKPT = ROOT / "runs" / "seasonal" / "all"
WEEKS = (12, 25, 38, 51)


def log(msg: str) -> None:
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + msg
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def jobs() -> list[dict]:
    out = []
    for week in WEEKS:
        out.append(
            {
                "name": f"rule_w{week}",
                "args": ["--method", "rule", "--week", str(week), "--seed", "0",
                         "--run-dir", str(ROOT / "runs" / "paper_min" / f"rule_w{week}")],
            }
        )
    for week in WEEKS:
        out.append(
            {
                "name": f"milp_w{week}",
                "args": ["--method", "milp", "--week", str(week), "--seed", "0",
                         "--run-dir", str(ROOT / "runs" / "paper_min" / f"milp_w{week}")],
            }
        )
    for seed in (0, 1, 2):
        ckpt = CKPT / f"pc_hybrid_td3_0903_s{seed}" / "checkpoints" / "hybrid_givesafe_td3.pt"
        for week in WEEKS:
            out.append(
                {
                    "name": f"pc_s{seed}_w{week}",
                    "args": [
                        "--method", "td3", "--week", str(week), "--seed", str(seed),
                        "--ckpt", str(ckpt),
                        "--run-dir", str(ROOT / "runs" / "paper_min" / f"pc_s{seed}_w{week}"),
                    ],
                }
            )
    return out


def main() -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    done = []
    if STATE.is_file():
        try:
            done = list(json.loads(STATE.read_text(encoding="utf-8")).get("done") or [])
        except Exception:
            done = []
    all_jobs = jobs()
    log(f"PAPER_MIN start n={len(all_jobs)} already={len(done)}")
    failed = None
    for job in all_jobs:
        if job["name"] in done:
            log(f"SKIP {job['name']}")
            continue
        log(f"START {job['name']}")
        cmd = [str(PY), str(SCRIPT), *job["args"]]
        job_log = ROOT / "logs" / f"paper_min_{job['name']}.log"
        with job_log.open("w", encoding="utf-8") as lf:
            proc = subprocess.run(cmd, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT)
        code = int(proc.returncode)
        log(f"END {job['name']} code={code}")
        if code != 0:
            failed = job["name"]
            STATE.write_text(json.dumps({"done": done, "failed": failed}, indent=2), encoding="utf-8")
            log(f"STOP failed={failed}")
            sys.exit(2)
        done.append(job["name"])
        STATE.write_text(json.dumps({"done": done, "failed": None}, indent=2), encoding="utf-8")
    log("PAPER_MIN_DONE")


if __name__ == "__main__":
    main()
