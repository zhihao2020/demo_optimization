#!/usr/bin/env python
"""Sync paper-min seasonal_tou2026 jobs to 172.16.1.80 and queue them.

Live pair + classical + mechanism ablation on monthly 110 kV TOU:
  fs_hsac --support x 3 seasons
  --method sac (sac_param) x 3
  --method milp x 3
  fs_hsac --support --lock-caes x 3

CPU, MAX_LIVE=3 (same envelope as the completed jfix wave).
  python scripts/remote_run_paper_min.py            # sync + remote pytest
  python scripts/remote_run_paper_min.py --launch   # then start the queue
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import paramiko

HOST = "172.16.1.80"
USER = "dell"
PASSWORD = "TR@SZ"
REMOTE = r"D:\xuzh\demo_optimization"
LOCAL = Path(__file__).resolve().parents[1]
PY = REMOTE + r"\.venv\Scripts\python.exe"
MAX_LIVE = 3
SEASONS = ("winter", "transition", "summer")
EPISODES = 5000

SYNCS = [
    "scripts/train_seasonal.py",
    "scripts/seasonal_cli.py",
    "src/training/fs_hsac/__init__.py",
    "src/training/fs_hsac/action_support.py",
    "src/training/fs_hsac/actor.py",
    "src/training/fs_hsac/algorithm.py",
    "src/training/fs_hsac/collector.py",
    "src/training/fs_hsac/compute.py",
    "src/training/fs_hsac/critic.py",
    "src/training/fs_hsac/feasibility.py",
    "src/training/fs_hsac/train.py",
    "src/replay/fs_hsac_replay.py",
    "src/training/hybrid_sac/__init__.py",
    "src/training/hybrid_sac/algorithm.py",
    "src/training/hybrid_sac/train.py",
    "src/training/hybrid_common/eval_and_save.py",
    "src/training/hybrid_common/explore.py",
    "src/training/hybrid_common/policy_wrapper.py",
    "src/training/hybrid_common/param_caes.py",
    "src/training/hybrid_common/stochastic_actor.py",
    "src/training/episode_starts.py",
    "src/optimization/rolling_linprog.py",
    "src/optimization/rolling_milp.py",
    "src/optimization/metrics.py",
    "src/actions/caes_min_run.py",
    "src/envs/power_system_env.py",
    "src/envs/reward_calculator.py",
    "src/config/env_config.yaml",
    "src/config/givesafe_config.yaml",
    "src/config/reward_config.yaml",
    "data/price_tou.csv",
    "data/price_tou_monthly_official.csv",
    "data/price_tou_meta.json",
    "tests/test_fs_hsac_support.py",
    "tests/test_fs_hsac_algorithm.py",
    "tests/test_fs_hsac_replay.py",
    "tests/test_fs_hsac_feasibility.py",
    "tests/test_fs_hsac_explore.py",
    "tests/test_train_seasonal_protocol.py",
    "tests/test_caes_lock.py",
    "tests/test_caes_min_run.py",
    "tests/test_storage_use_reward.py",
    "tests/test_storage_cash_exclusion.py",
    "tests/test_rolling_milp.py",
    "tests/test_hybrid_parameterized_actor.py",
]

REMOTE_PYTEST = [
    "tests/test_fs_hsac_support.py",
    "tests/test_fs_hsac_algorithm.py",
    "tests/test_fs_hsac_replay.py",
    "tests/test_fs_hsac_feasibility.py",
    "tests/test_fs_hsac_explore.py",
    "tests/test_train_seasonal_protocol.py",
    "tests/test_caes_lock.py",
    "tests/test_storage_use_reward.py",
    "tests/test_storage_cash_exclusion.py",
    "tests/test_rolling_milp.py",
    "tests/test_hybrid_parameterized_actor.py",
]


def run(t, cmd, timeout=90):
    ch = t.open_session()
    ch.set_combine_stderr(True)
    ch.exec_command(cmd)
    buf = b""
    end = time.time() + timeout
    while time.time() < end:
        if ch.recv_ready():
            buf += ch.recv(65536)
        elif ch.exit_status_ready():
            while ch.recv_ready():
                buf += ch.recv(65536)
            break
        else:
            time.sleep(0.05)
    return buf.decode("utf-8", "replace")


def sftp_mkdirs(sftp, remote: str) -> None:
    remote = remote.replace("/", "\\")
    parts = [p for p in remote.split("\\") if p]
    cur = parts[0] + "\\"
    for p in parts[1:]:
        cur = cur.rstrip("\\") + "\\" + p
        try:
            sftp.stat(cur)
        except OSError:
            try:
                sftp.mkdir(cur)
            except OSError:
                pass


def jobs() -> list[dict]:
    out: list[dict] = []
    # Fast classical first so they clear the live slots.
    for season in SEASONS:
        out.append(
            {
                "name": f"tou2026_{season}_milp_s0",
                "tag": "milp",
                "season": season,
                "method_args": "--method milp",
                "no_feas": False,
                "lock": False,
                "rl": False,
            }
        )
    for season in SEASONS:
        out.append(
            {
                "name": f"tou2026_{season}_fs_hsac_support_s0",
                "tag": "fs_hsac_support",
                "season": season,
                "method_args": "--method fs_hsac --support",
                "no_feas": True,
                "lock": False,
                "rl": True,
            }
        )
    for season in SEASONS:
        out.append(
            {
                "name": f"tou2026_{season}_sac_param_s0",
                "tag": "sac_param",
                "season": season,
                "method_args": "--method sac",
                "no_feas": False,
                "lock": False,
                "rl": True,
            }
        )
    for season in SEASONS:
        out.append(
            {
                "name": f"tou2026_{season}_fs_hsac_support_lockcaes_s0",
                "tag": "fs_hsac_support_lockcaes",
                "season": season,
                "method_args": "--method fs_hsac --support --lock-caes",
                "no_feas": True,
                "lock": True,
                "rl": True,
            }
        )
    return out


QUEUE_PY = r'''
import json, subprocess, time
from pathlib import Path

ROOT = r"D:\xuzh\demo_optimization"
MAX_LIVE = 3
QUEUE = json.loads(Path(ROOT, "logs", "paper_min_queue.json").read_text(encoding="utf-8"))
state_path = Path(ROOT, "logs", "paper_min_queue_state.json")

def load_started():
    if state_path.is_file():
        try:
            return set(json.loads(state_path.read_text(encoding="utf-8")).get("started", []))
        except Exception:
            return set()
    return set()

def save_started(started):
    state_path.write_text(json.dumps({"started": sorted(started)}, indent=2), encoding="utf-8")

def live_jobs():
    out = subprocess.check_output(
        "wmic process where name='python.exe' get CommandLine /FORMAT:LIST",
        shell=True, text=True, errors="replace",
    )
    dirs = set()
    for ln in out.splitlines():
        low = ln.lower()
        if "train_seasonal.py" not in low:
            continue
        if "seasonal_tou2026" not in low:
            continue
        if "--run-dir" not in low:
            continue
        part = ln.split("--run-dir", 1)[1].strip().strip('"').strip()
        part = part.split(" --")[0].strip().strip('"')
        dirs.add(part.lower())
    return dirs

def start_bat(bat):
    cmdline = 'cmd.exe /c call "%s"' % bat
    cmd = 'wmic process call create "%s"' % cmdline.replace('"', '\\"')
    subprocess.check_call(cmd, shell=True)

def log(msg):
    print(msg, flush=True)

started = load_started()
for j in QUEUE:
    rd = str(j.get("run_dir", "")).lower()
    if rd and rd in live_jobs():
        started.add(j["name"])
save_started(started)
log("PAPER_MIN_QUEUE start n=%d max=%d started=%d" % (len(QUEUE), MAX_LIVE, len(started)))

while True:
    pending = [j for j in QUEUE if j["name"] not in started]
    live = live_jobs()
    n_live = len(live)
    log("live=%d pending=%d started=%d" % (n_live, len(pending), len(started)))
    if not pending and n_live == 0:
        log("QUEUE_DONE")
        break
    if not pending:
        ours = {str(j.get("run_dir", "")).lower() for j in QUEUE}
        if not (live & ours):
            log("QUEUE_DONE_OUR_JOBS")
            break
        time.sleep(45)
        continue
    if n_live < MAX_LIVE:
        job = pending[0]
        try:
            start_bat(job["bat"])
            started.add(job["name"])
            save_started(started)
            log("STARTED " + job["name"])
        except Exception as exc:
            log("FAIL " + job["name"] + " " + str(exc))
        time.sleep(12)
        continue
    time.sleep(45)
'''


def bat_body(job: dict) -> str:
    name = job["name"]
    season = job["season"]
    tag = job["tag"]
    run_abs = rf"{REMOTE}\runs\seasonal_tou2026\{season}\{tag}_s0"
    log = rf"{REMOTE}\logs\{name}.log"
    err = log + ".err"
    feas = "set FS_HSAC_NO_FEAS=1" if job["no_feas"] else "set FS_HSAC_NO_FEAS="
    lock = "set OPTIMAL_DEMO_LOCK_CAES=1" if job["lock"] else "set OPTIMAL_DEMO_LOCK_CAES="
    ep = f"--episodes {EPISODES} " if job["rl"] else ""
    return f"""@echo off
cd /d {REMOTE}
set PYTHONUNBUFFERED=1
set PYTHONPATH={REMOTE}\\src
set OPTIMAL_DEMO_CACHE={REMOTE}_cache
set OPTIMAL_DEMO_JOB_ID={name}
set OPTIMAL_DEMO_TMP={REMOTE}_cache\\tmp\\{name}
set OPTIMAL_DEMO_FMU_ISOLATE=1
set CUDA_VISIBLE_DEVICES=-1
set OPTIMAL_DEMO_DEVICE=cpu
set OPTIMAL_DEMO_TORCH_THREADS=4
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set OPENBLAS_NUM_THREADS=4
set NUMEXPR_NUM_THREADS=1
{feas}
{lock}
if not exist "%OPTIMAL_DEMO_TMP%" mkdir "%OPTIMAL_DEMO_TMP%"
if not exist "{run_abs}" mkdir "{run_abs}"
echo START %DATE% %TIME% > "{log}"
"{PY}" "{REMOTE}\\scripts\\train_seasonal.py" {job["method_args"]} --season {season} {ep}--seed 0 --run-dir "{run_abs}" >> "{log}" 2>> "{err}"
echo EXIT %ERRORLEVEL% %DATE% %TIME% >> "{log}"
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--launch", action="store_true")
    p.add_argument("--skip-pytest", action="store_true")
    p.add_argument(
        "--kill-live",
        action="store_true",
        help="kill train_seasonal / paper_min_queue before launch",
    )
    args = p.parse_args()

    t = paramiko.Transport((HOST, 22))
    t.banner_timeout = 60
    t.auth_timeout = 60
    t.start_client(timeout=60)
    t.auth_password(USER, PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(t)

    print("=== GPU ===")
    print(
        run(
            t,
            "cmd /c nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader",
        ).strip()
    )
    live = run(
        t,
        r'cmd /c wmic process where "name=\"python.exe\"" get CommandLine /FORMAT:LIST',
        60,
    )
    n_train = sum(1 for ln in live.splitlines() if "train_seasonal" in ln.lower())
    print("live_train_seasonal_lines", n_train)

    for rel in SYNCS:
        src = LOCAL / rel
        if not src.is_file():
            raise FileNotFoundError(src)
        rpath = REMOTE + "\\" + rel.replace("/", "\\")
        sftp_mkdirs(sftp, str(Path(rpath).parent))
        sftp.put(str(src), rpath)
        print("PUT", rel)

    if not args.skip_pytest:
        tests = " ".join(rf"{REMOTE}\{x.replace('/', '\\')}" for x in REMOTE_PYTEST)
        cmd = (
            f'cmd /c "cd /d {REMOTE} && set PYTHONPATH={REMOTE}\\src;{REMOTE}\\scripts'
            f' && {PY} -m pytest {tests} -q --tb=line"'
        )
        print("=== remote pytest ===")
        out = run(t, cmd, timeout=180)
        print(out[-3000:] if len(out) > 3000 else out)
        low = out.lower()
        if "error collecting" in low or "failed" in low or "passed" not in low:
            raise SystemExit("remote pytest failed; not launching")

    queue_meta = []
    for job in jobs():
        season = job["season"]
        tag = job["tag"]
        run_abs = rf"{REMOTE}\runs\seasonal_tou2026\{season}\{tag}_s0"
        bat = rf"{REMOTE}\logs\run_{job['name']}.bat"
        sftp_mkdirs(sftp, str(Path(bat).parent))
        sftp_mkdirs(sftp, run_abs)
        with sftp.file(bat, "wb") as f:
            f.write(bat_body(job).replace("\n", "\r\n").encode("gbk", errors="replace"))
        queue_meta.append(
            {"name": job["name"], "bat": bat, "run_dir": run_abs, "season": season, "tag": tag}
        )
        print("BAT", job["name"])

    qpath = rf"{REMOTE}\logs\paper_min_queue.json"
    with sftp.file(qpath, "w") as f:
        f.write(json.dumps(queue_meta, indent=2))
    qpy = rf"{REMOTE}\logs\paper_min_queue.py"
    with sftp.file(qpy, "w") as f:
        f.write(QUEUE_PY)
    start_bat = rf"{REMOTE}\logs\start_paper_min_queue.bat"
    start_body = f"""@echo off
cd /d {REMOTE}
set PYTHONUNBUFFERED=1
"{PY}" "{qpy}" >> "{REMOTE}\\logs\\paper_min_queue.log" 2>&1
"""
    with sftp.file(start_bat, "wb") as f:
        f.write(start_body.replace("\n", "\r\n").encode("gbk", errors="replace"))

    if not args.launch:
        print("sync+pytest done; pass --launch to start the queue")
        sftp.close()
        t.close()
        return

    if args.kill_live or n_train:
        print("=== kill live train_seasonal / paper_min_queue ===")
        print(run(t, r"""cmd /c wmic process where "CommandLine like '%paper_min_queue%'" call terminate"""))
        time.sleep(2)
        print(run(t, r"""cmd /c wmic process where "CommandLine like '%train_seasonal.py%'" call terminate"""))
        time.sleep(3)
        live2 = run(
            t,
            r'cmd /c wmic process where "name=\"python.exe\"" get ProcessId,CommandLine /FORMAT:LIST',
            80,
        )
        pids = []
        cur_pid, cur_cmd = None, ""
        for ln in live2.splitlines():
            if ln.startswith("ProcessId="):
                cur_pid = ln.split("=", 1)[-1].strip()
            elif ln.startswith("CommandLine="):
                cur_cmd = ln.split("=", 1)[-1]
                low = cur_cmd.lower()
                if cur_pid and ("train_seasonal" in low or "paper_min_queue" in low):
                    pids.append(cur_pid)
                cur_pid, cur_cmd = None, ""
        if pids:
            print("FORCE", pids)
            print(run(t, "cmd /c " + " & ".join(f"taskkill /F /PID {p}" for p in pids)))
            time.sleep(2)

    done = []
    for job in jobs():
        logp = rf"{REMOTE}\logs\{job['name']}.log"
        tail = run(
            t,
            "powershell -NoProfile -Command "
            f"\"if (Test-Path '{logp}') {{ Get-Content '{logp}' -Tail 4 }} else {{ '' }}\"",
            20,
        )
        if "EXIT 0" in tail and job["tag"] == "milp":
            done.append(job["name"])
    state_path = rf"{REMOTE}\logs\paper_min_queue_state.json"
    with sftp.file(state_path, "w") as f:
        f.write(json.dumps({"started": sorted(set(done))}, indent=2))
    print("seed_started", done)
    # old queue log is append-only; rotate so the new run is readable
    run(
        t,
        "powershell -NoProfile -Command "
        f"\"if (Test-Path '{REMOTE}\\logs\\paper_min_queue.log') {{ "
        f"Move-Item -Force '{REMOTE}\\logs\\paper_min_queue.log' "
        f"'{REMOTE}\\logs\\paper_min_queue.log.bak' }}\"",
        20,
    )

    wmic = 'cmd /c wmic process call create "cmd.exe /c ' + start_bat + '","' + REMOTE + '"'
    print("START queue")
    print(run(t, wmic, 30)[:500])
    time.sleep(8)
    print("=== queue log ===")
    print(
        run(
            t,
            "powershell -NoProfile -Command "
            f"\"if (Test-Path '{REMOTE}\\logs\\paper_min_queue.log') {{ Get-Content '{REMOTE}\\logs\\paper_min_queue.log' -Tail 12 }} else {{ 'NOQUEUE' }}\"",
        ).strip()
    )
    print("=== live after start ===")
    live2 = run(
        t,
        r'cmd /c wmic process where "name=\"python.exe\"" get CommandLine /FORMAT:LIST',
        90,
    )
    for ln in live2.splitlines():
        low = ln.lower()
        if "train_seasonal" in low or "paper_min_queue" in low:
            print(ln[:360])
    sftp.close()
    t.close()


if __name__ == "__main__":
    main()
