"""Sequential PC-HybridTD3 queue on the training box. Stop on Stage B gate failure."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"D:\xuzh\demo_optimization")
PY = ROOT / ".venv" / "Scripts" / "python.exe"
LOG = ROOT / "logs" / "pc_hybrid_queue.log"
STATE = ROOT / "logs" / "pc_hybrid_queue_state.json"
SCRIPT = ROOT / "scripts" / "train_seasonal.py"

JOBS = [
    {
        "name": "stageA_s0",
        "args": ["--method", "td3", "--season", "all", "--stage", "A", "--seed", "0"],
        "run_dir": str(ROOT / "runs" / "seasonal_tou2026" / "all" / "pc_hybrid_td3_stageA_s0"),
        "gate": "stage_a",
    },
    {
        "name": "stageB_s0",
        "args": ["--method", "td3", "--season", "all", "--stage", "B", "--seed", "0"],
        "run_dir": str(ROOT / "runs" / "seasonal_tou2026" / "all" / "pc_hybrid_td3_stageB_d53_s0"),
        "gate": "stage_b",
    },
    {
        "name": "stageC_s0",
        "args": ["--method", "td3", "--season", "all", "--stage", "C", "--seed", "0"],
        "run_dir": str(ROOT / "runs" / "seasonal_tou2026" / "all" / "pc_hybrid_td3_stageC_d53ep_s0"),
        "gate": "stage_c",
    },
    {
        "name": "stageD_s0",
        "args": ["--method", "td3", "--season", "all", "--stage", "D", "--seed", "0"],
        "run_dir": str(ROOT / "runs" / "seasonal_tou2026" / "all" / "pc_hybrid_td3_s0"),
        "gate": "stage_d",
    },
    {
        "name": "stageD_s1",
        "args": ["--method", "td3", "--season", "all", "--stage", "D", "--seed", "1"],
        "run_dir": str(ROOT / "runs" / "seasonal_tou2026" / "all" / "pc_hybrid_td3_s1"),
        "gate": "stage_d",
    },
    {
        "name": "stageD_s2",
        "args": ["--method", "td3", "--season", "all", "--stage", "D", "--seed", "2"],
        "run_dir": str(ROOT / "runs" / "seasonal_tou2026" / "all" / "pc_hybrid_td3_s2"),
        "gate": "stage_d",
    },
    {
        "name": "proj_D_s0",
        "args": [
            "--method", "td3", "--ablation", "projection",
            "--season", "all", "--stage", "D", "--seed", "0",
        ],
        "run_dir": str(ROOT / "runs" / "seasonal_tou2026" / "all" / "td3_proj_s0"),
        "gate": "stage_d",
    },
    {
        "name": "rule_winter_test",
        "args": ["--method", "rule", "--season", "winter", "--seed", "0"],
        "run_dir": str(ROOT / "runs" / "seasonal_tou2026" / "winter" / "rule_s0"),
        "gate": None,
    },
    {
        "name": "milp_winter_test",
        "args": ["--method", "milp", "--season", "winter", "--seed", "0"],
        "run_dir": str(ROOT / "runs" / "seasonal_tou2026" / "winter" / "milp_s0"),
        "gate": None,
    },
]


def log(msg: str) -> None:
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + msg
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_state(payload: dict) -> None:
    STATE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_result(run_dir: str) -> dict:
    p = Path(run_dir) / "train_result.json"
    if not p.is_file():
        p = Path(run_dir) / "summary.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def job_ok(job: dict, code: int, result: dict) -> bool:
    status = str(result.get("status") or "")
    if job.get("gate") == "stage_a":
        return code == 0 and status == "completed" and int(result.get("illegal_caes_mode") or 0) == 0
    if job.get("gate") == "stage_b":
        # Interaction pass is enough to continue diagnostic C. Greedy fail → partial_pass.
        if code != 0:
            return False
        if result.get("stage_b_interaction") == "failed":
            return False
        return status in {"completed", "partial_pass", "blocked_formal_gates_post"} or bool(result)
    if job.get("gate") == "stage_c":
        # Training crash stops the queue. Failed formal gates skip Stage D, do not abort baselines.
        if code != 0:
            return False
        return status in {"completed", "partial_pass", "blocked_formal_gates_post"} or bool(result)
    if job.get("gate") == "stage_d":
        return code == 0
    return code == 0


def run_job(job: dict) -> int:
    run_dir = Path(job["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "trajectories").mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "logs" / f"pc_hybrid_{job['name']}.log"
    err_path = ROOT / "logs" / f"pc_hybrid_{job['name']}.log.err"
    cmd = [str(PY), str(SCRIPT), *job["args"], "--run-dir", str(run_dir)]
    log("START " + job["name"] + " " + " ".join(cmd[2:]))
    with log_path.open("w", encoding="utf-8") as out, err_path.open("w", encoding="utf-8") as err:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=out, stderr=err)
    return int(proc.returncode)


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log("QUEUE start n=%d" % len(JOBS))
    state = {"done": [], "failed": None, "stage_c_ok": False}
    for job in JOBS:
        if job.get("gate") == "stage_d" and not state.get("stage_c_ok"):
            log("SKIP " + job["name"] + " (Stage D blocked until Stage C formal gates)")
            continue
        code = run_job(job)
        result = read_result(job["run_dir"])
        status = result.get("status")
        log(
            "END %s code=%s status=%s greedy=%s stage_c_passed=%s"
            % (job["name"], code, status, result.get("greedy_eval"), result.get("stage_c_passed"))
        )
        if job.get("gate") == "stage_c":
            state["stage_c_ok"] = bool(result.get("stage_c_passed"))
            log("STAGE_C_GATES passed=%s %s" % (state["stage_c_ok"], result.get("stage_c_gates")))
        if result.get("greedy_eval") == "passed":
            state["greedy_ok"] = True
        if not job_ok(job, code, result):
            state["failed"] = {"name": job["name"], "code": code, "status": status}
            save_state(state)
            log("QUEUE_STOP " + job["name"])
            return 2
        state["done"].append(job["name"])
        save_state(state)
    log("QUEUE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
