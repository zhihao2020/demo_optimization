#!/usr/bin/env python
"""Sync FS-HSAC code to remote and launch seasonal_v1 training (no clean).

Does NOT terminate existing *_param_s0 jobs. Shares MAX_LIVE with all
seasonal_v1 train_seasonal processes.

Usage:
  python scripts/remote_fs_hsac.py --sync-only
  python scripts/remote_fs_hsac.py --episodes 5000 --max-live 3
  python scripts/remote_fs_hsac.py --include-support --episodes 5000
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
CACHE = r"D:\xuzh\demo_optimization_cache"
PY = r"D:\xuzh\demo_optimization\.venv\Scripts\python.exe"
LOCAL = Path(__file__).resolve().parents[1]

SYNC_GLOBS = [
    "src/actions/**/*.py",
    "src/config/**/*",
    "src/controllers/**/*.py",
    "src/data/**/*.py",
    "src/envs/**/*.py",
    "src/fmu/**/*.py",
    "src/market/**/*.py",
    "src/optimization/**/*.py",
    "src/replay/**/*.py",
    "src/safety/**/*.py",
    "src/training/**/*.py",
    "src/utils/**/*.py",
    "scripts/train_seasonal.py",
    "logs/_smoke_fs_hsac.py",
    "tests/test_fs_hsac_*.py",
]
SKIP_DIR = {"__pycache__", ".git", ".pytest_cache"}
SKIP_SUF = {".pyc", ".pyo"}

FORCE_RELS = [
    "scripts/train_seasonal.py",
    "src/replay/fs_hsac_replay.py",
    "src/training/fs_hsac/__init__.py",
    "src/training/fs_hsac/action_support.py",
    "src/training/fs_hsac/actor.py",
    "src/training/fs_hsac/critic.py",
    "src/training/fs_hsac/algorithm.py",
    "src/training/fs_hsac/feasibility.py",
    "src/training/fs_hsac/collector.py",
    "src/training/fs_hsac/train.py",
]

QUEUE_PY = r'''
import json, subprocess, time
from pathlib import Path

ROOT = r"D:\xuzh\demo_optimization"
MAX_LIVE = __MAX__
QUEUE = json.loads(Path(ROOT, "logs", "fair_queue_fs_hsac.json").read_text(encoding="utf-8"))
state_path = Path(ROOT, "logs", "fair_queue_fs_hsac_state.json")
logp = Path(ROOT, "logs", "fair_queue_fs_hsac.log")

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
        if "train_seasonal.py" not in ln.lower():
            continue
        if "seasonal_v1" not in ln.lower():
            continue
        if "--run-dir" not in ln.lower():
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
    with logp.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")

started = load_started()
for j in QUEUE:
    rd = str(j.get("run_dir", "")).lower()
    if rd and rd in live_jobs():
        started.add(j["name"])
save_started(started)
log("FS_HSAC_QUEUE start n=%d max=%d started=%d" % (len(QUEUE), MAX_LIVE, len(started)))

while True:
    pending = [j for j in QUEUE if j["name"] not in started]
    live = live_jobs()
    n_live = len(live)
    log("live=%d pending=%d started=%d" % (n_live, len(pending), len(started)))
    if not pending and n_live == 0:
        log("QUEUE_DONE")
        break
    # If only non-queue seasonal jobs remain and our queue is done, exit.
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


def connect():
    t = paramiko.Transport((HOST, 22))
    t.banner_timeout = 60
    t.auth_timeout = 60
    t.start_client(timeout=60)
    t.auth_password(USER, PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(t)
    assert sftp is not None
    return t, sftp


def run(t, cmd, timeout=120):
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


def safe_print(msg) -> None:
    text = str(msg)
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"), flush=True)


def mkdirs(sftp, remote: str) -> None:
    parts = [x for x in remote.replace("/", "\\").split("\\") if x]
    cur = parts[0]
    for part in parts[1:]:
        cur = cur + "\\" + part
        try:
            sftp.stat(cur)
        except OSError:
            try:
                sftp.mkdir(cur)
            except OSError:
                pass


def rpath(*parts: str) -> str:
    p = REMOTE
    for part in parts:
        p = p.rstrip("\\") + "\\" + part.replace("/", "\\")
    return p


def collect_files():
    out = []
    seen = set()
    for pat in SYNC_GLOBS:
        for fp in LOCAL.glob(pat):
            if not fp.is_file():
                continue
            if any(x in SKIP_DIR for x in fp.parts):
                continue
            if fp.suffix.lower() in SKIP_SUF:
                continue
            rel = fp.relative_to(LOCAL).as_posix()
            if rel in seen:
                continue
            seen.add(rel)
            out.append((fp, rel))
    return out


def sync(sftp) -> dict:
    files = collect_files()
    n_ok = n_skip = n_fail = 0
    for fp, rel in files:
        rp = rpath(*rel.split("/"))
        try:
            try:
                st = sftp.stat(rp)
                if int(st.st_size) == int(fp.stat().st_size):
                    n_skip += 1
                    continue
            except OSError:
                pass
            mkdirs(sftp, str(Path(rp).parent))
            sftp.put(str(fp), rp, confirm=False)
            n_ok += 1
            if n_ok % 25 == 0:
                print(f"  put {n_ok}", flush=True)
        except Exception as exc:
            n_fail += 1
            print(f"  FAIL {rel}: {exc}", flush=True)
    return {"uploaded": n_ok, "skipped": n_skip, "failed": n_fail, "total": len(files)}


def force_put(sftp, rel: str) -> None:
    lp = LOCAL / rel
    if not lp.is_file():
        print(f"  MISS local {rel}", flush=True)
        return
    rp = rpath(*rel.split("/"))
    mkdirs(sftp, str(Path(rp).parent))
    sftp.put(str(lp), rp, confirm=False)
    print(f"force {rel} size={lp.stat().st_size}", flush=True)


def build_jobs(
    episodes: int,
    seasons: list[str],
    *,
    include_support: bool,
    smoke_episodes: int,
) -> list[dict]:
    jobs: list[dict] = []
    # Optional short transition support smoke first (separate run dir).
    if smoke_episodes > 0:
        name = "transition_fs_hsac_support_smoke_s0"
        jobs.append(
            {
                "name": name,
                "season": "transition",
                "method": "fs_hsac",
                "seed": 0,
                "episodes": smoke_episodes,
                "no_feas": True,
                "job_id": f"seasonal_{name}",
                "run_dir": rf"{REMOTE}\runs\seasonal_v1\transition\fs_hsac_support_smoke_s0",
                "log": rf"{REMOTE}\logs\seasonal_v1_{name}.log",
                "bat": rf"{REMOTE}\logs\run_v1_{name}.bat",
            }
        )
    if include_support:
        for season in seasons:
            name = f"{season}_fs_hsac_support_s0"
            jobs.append(
                {
                    "name": name,
                    "season": season,
                    "method": "fs_hsac",
                    "seed": 0,
                    "episodes": episodes,
                    "no_feas": True,
                    "job_id": f"seasonal_{name}",
                    "run_dir": rf"{REMOTE}\runs\seasonal_v1\{season}\fs_hsac_support_s0",
                    "log": rf"{REMOTE}\logs\seasonal_v1_{name}.log",
                    "bat": rf"{REMOTE}\logs\run_v1_{name}.bat",
                }
            )
    for season in seasons:
        name = f"{season}_fs_hsac_s0"
        jobs.append(
            {
                "name": name,
                "season": season,
                "method": "fs_hsac",
                "seed": 0,
                "episodes": episodes,
                "no_feas": False,
                "job_id": f"seasonal_{name}",
                "run_dir": rf"{REMOTE}\runs\seasonal_v1\{season}\fs_hsac_s0",
                "log": rf"{REMOTE}\logs\seasonal_v1_{name}.log",
                "bat": rf"{REMOTE}\logs\run_v1_{name}.bat",
            }
        )
    return jobs


def make_bat(job: dict) -> str:
    no_feas = "set FS_HSAC_NO_FEAS=1\r\n" if job.get("no_feas") else "set FS_HSAC_NO_FEAS=\r\n"
    return (
        f"@echo off\r\n"
        f"cd /d {REMOTE}\r\n"
        f"set PYTHONUNBUFFERED=1\r\n"
        f"set PYTHONPATH={REMOTE}\\src\r\n"
        f"set OPTIMAL_DEMO_CACHE={CACHE}\r\n"
        f"set OPTIMAL_DEMO_JOB_ID={job['job_id']}\r\n"
        f"set OPTIMAL_DEMO_TMP={CACHE}\\tmp\\{job['job_id']}\r\n"
        f"set OPTIMAL_DEMO_FMU_ISOLATE=1\r\n"
        f"{no_feas}"
        f"if not exist \"%OPTIMAL_DEMO_TMP%\" mkdir \"%OPTIMAL_DEMO_TMP%\"\r\n"
        f"if not exist \"{job['run_dir']}\" mkdir \"{job['run_dir']}\"\r\n"
        f"echo START %DATE% %TIME% > \"{job['log']}\"\r\n"
        f"\"{PY}\" \"{REMOTE}\\scripts\\train_seasonal.py\" "
        f"--method {job['method']} --season {job['season']} "
        f"--episodes {job['episodes']} --seed {job['seed']} "
        f"--run-dir \"{job['run_dir']}\" "
        f">> \"{job['log']}\" 2>> \"{job['log']}.err\"\r\n"
        f"echo EXIT %ERRORLEVEL% %DATE% %TIME% >> \"{job['log']}\"\r\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Deploy FS-HSAC and launch remote seasonal jobs")
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--seasons", type=str, default="winter,transition,summer")
    ap.add_argument("--max-live", type=int, default=3, help="total seasonal_v1 train_seasonal cap")
    ap.add_argument("--include-support", action="store_true", help="also queue FS-HSAC-support ablations")
    ap.add_argument(
        "--smoke-episodes",
        type=int,
        default=20,
        help="prepend transition support smoke (0 to skip)",
    )
    ap.add_argument("--sync-only", action="store_true")
    ap.add_argument("--launch-only", action="store_true")
    args = ap.parse_args()

    seasons = [x.strip() for x in args.seasons.split(",") if x.strip()]
    jobs = build_jobs(
        args.episodes,
        seasons,
        include_support=bool(args.include_support),
        smoke_episodes=int(args.smoke_episodes),
    )
    print(
        f"jobs={len(jobs)} episodes={args.episodes} max_live={args.max_live} "
        f"support={args.include_support} smoke={args.smoke_episodes}",
        flush=True,
    )
    for j in jobs:
        tag = "NO_FEAS" if j.get("no_feas") else "FULL"
        print(f"  - {j['name']} ep={j['episodes']} [{tag}]", flush=True)

    t, sftp = connect()
    try:
        sftp.stat(REMOTE)
    except OSError as exc:
        raise SystemExit(f"remote missing: {exc}") from exc

    if not args.launch_only:
        print("=== sync (no clean) ===", flush=True)
        print(sync(sftp), flush=True)
        print("=== force key FS-HSAC files ===", flush=True)
        for rel in FORCE_RELS:
            force_put(sftp, rel)

    if args.sync_only:
        sftp.close()
        t.close()
        print("sync-only done", flush=True)
        return

    # remote smoke: import + unit tests (no FMU required for pytest subset)
    smoke = (
        "import sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, r'{REMOTE}\\src')\n"
        "from training.fs_hsac.algorithm import FSHSAC, ALGORITHM_VERSION\n"
        "from training.fs_hsac.actor import FSHSACActor\n"
        "from training.fs_hsac.critic import FSHSACCritic\n"
        "from replay.fs_hsac_replay import FSHSACReplayBuffer\n"
        "assert ALGORITHM_VERSION == 'fs_hsac_v2'\n"
        "a = FSHSAC(obs_dim=16, device='cpu')\n"
        "print('SMOKE_OK', ALGORITHM_VERSION, 'actor', type(a.actor).__name__)\n"
    )
    local_s = LOCAL / "logs" / "_fs_hsac_remote_smoke.py"
    local_s.parent.mkdir(parents=True, exist_ok=True)
    local_s.write_text(smoke, encoding="utf-8")
    sftp.put(str(local_s), rpath("logs", "_fs_hsac_remote_smoke.py"), confirm=False)
    smoke_out = run(
        t,
        f'powershell -NoProfile -Command "Set-Location \'{REMOTE}\'; '
        f'$env:PYTHONPATH=\'{REMOTE}\\src\'; & \'{PY}\' \'{rpath("logs", "_fs_hsac_remote_smoke.py")}\'"',
        timeout=180,
    )
    safe_print("smoke: " + smoke_out.strip()[:800])
    if "SMOKE_OK" not in smoke_out:
        raise SystemExit("remote FS-HSAC smoke failed; abort launch")

    # live snapshot (do not kill)
    safe_print("=== current train_seasonal (leave running) ===")
    safe_print(
        run(
            t,
            "wmic process where \"CommandLine like '%train_seasonal%'\" get ProcessId,CommandLine /FORMAT:LIST",
            timeout=60,
        )[:2000]
    )

    for job in jobs:
        local_b = LOCAL / "logs" / f"run_v1_{job['name']}.bat"
        local_b.write_text(make_bat(job), encoding="ascii")
        sftp.put(str(local_b), job["bat"], confirm=False)
    print(f"uploaded {len(jobs)} bats", flush=True)

    qj = [{"name": j["name"], "bat": j["bat"], "run_dir": j["run_dir"]} for j in jobs]
    local_q = LOCAL / "logs" / "fair_queue_fs_hsac.json"
    local_q.write_text(json.dumps(qj, indent=2), encoding="utf-8")
    sftp.put(str(local_q), rpath("logs", "fair_queue_fs_hsac.json"), confirm=False)

    qpy = QUEUE_PY.replace("__MAX__", str(int(args.max_live)))
    local_py = LOCAL / "logs" / "fair_queue_fs_hsac.py"
    local_py.write_text(qpy, encoding="utf-8")
    sftp.put(str(local_py), rpath("logs", "fair_queue_fs_hsac.py"), confirm=False)
    try:
        sftp.remove(rpath("logs", "fair_queue_fs_hsac_state.json"))
    except OSError:
        pass
    try:
        sftp.remove(rpath("logs", "fair_queue_fs_hsac.log"))
    except OSError:
        pass

    start_bat = (
        f"@echo off\r\ncd /d {REMOTE}\r\n"
        f"set OPTIMAL_DEMO_CACHE={CACHE}\r\n"
        f"\"{PY}\" \"{REMOTE}\\logs\\fair_queue_fs_hsac.py\"\r\n"
    )
    local_sb = LOCAL / "logs" / "start_fair_queue_fs_hsac.bat"
    local_sb.write_text(start_bat, encoding="ascii")
    sftp.put(str(local_sb), rpath("logs", "start_fair_queue_fs_hsac.bat"), confirm=False)

    out = run(
        t,
        f'wmic process call create "cmd.exe /c call \\"{rpath("logs", "start_fair_queue_fs_hsac.bat")}\\""',
        timeout=40,
    )
    safe_print(out.strip()[:400])

    time.sleep(25)
    safe_print("=== fair_queue_fs_hsac.log ===")
    safe_print(
        run(
            t,
            f'powershell -NoProfile -Command "if(Test-Path \'{REMOTE}\\logs\\fair_queue_fs_hsac.log\')'
            f'{{ Get-Content \'{REMOTE}\\logs\\fair_queue_fs_hsac.log\' -Tail 20 }} else {{ \'NOLOG\' }}"',
            timeout=40,
        ).strip()
    )
    safe_print("=== smoke / first job log tail ===")
    first = jobs[0]["name"]
    safe_print(
        run(
            t,
            f'powershell -NoProfile -Command "if(Test-Path \'{REMOTE}\\logs\\seasonal_v1_{first}.log\')'
            f'{{ Get-Content \'{REMOTE}\\logs\\seasonal_v1_{first}.log\' -Tail 25 }} else {{ \'NOLOG\' }}"',
            timeout=40,
        ).strip()[:1500]
    )
    safe_print(
        run(
            t,
            f'powershell -NoProfile -Command "if(Test-Path \'{REMOTE}\\logs\\seasonal_v1_{first}.log.err\')'
            f'{{ Get-Content \'{REMOTE}\\logs\\seasonal_v1_{first}.log.err\' -Tail 30 }} else {{ \'NOERR\' }}"',
            timeout=40,
        ).strip()[:1500]
    )

    man = LOCAL / "logs" / "remote_fs_hsac_manifest.json"
    man.write_text(
        json.dumps(
            {
                "host": HOST,
                "episodes": args.episodes,
                "seasons": seasons,
                "max_live": args.max_live,
                "include_support": args.include_support,
                "smoke_episodes": args.smoke_episodes,
                "n_jobs": len(jobs),
                "jobs": [j["name"] for j in jobs],
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "note": "did not clean seasonal_v1; did not kill existing param jobs",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("manifest", man, flush=True)
    sftp.close()
    t.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
