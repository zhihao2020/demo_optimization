#!/usr/bin/env python
"""Clean remote protocol-v0 mess, sync fair-suite code, launch full seasonal matrix.

Matrix (default):
  seasons × {hmsd,td3,sac} × seeds 0..2  +  seasons × {pso,linprog} × seed 0
Queue: max concurrent train_seasonal jobs (default 4), FMU isolate per job_id.

Usage:
  python scripts/remote_fair_suite.py --sync-only
  python scripts/remote_fair_suite.py --episodes 5000
  python scripts/remote_fair_suite.py --episodes 200 --max-live 4   # smoke
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

# Lean sync — training + opt paths only
SYNC_GLOBS = [
    "src/actions/**/*.py",
    "src/config/**/*",
    "src/controllers/**/*.py",
    "src/envs/**/*.py",
    "src/fmu/**/*.py",
    "src/market/**/*.py",
    "src/optimization/**/*.py",
    "src/replay/**/*.py",
    "src/safety/**/*.py",
    "src/training/**/*.py",
    "src/utils/**/*.py",
    "scripts/train_seasonal.py",
    "scripts/eval_seasonal_fair.py",
    "scripts/train_hybrid_sac.py",
    "scripts/train_ghtd3.py",
    "docs/cui_seasonal_min_protocol.md",
]
SKIP_DIR = {"__pycache__", ".git", ".pytest_cache"}
SKIP_SUF = {".pyc", ".pyo"}


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
    rp = rpath(*rel.split("/"))
    mkdirs(sftp, str(Path(rp).parent))
    sftp.put(str(lp), rp, confirm=False)
    print("force", rel, lp.stat().st_size, flush=True)


def build_jobs(episodes: int, seeds: list[int], seasons: list[str], methods: list[str]) -> list[dict]:
    jobs = []
    for season in seasons:
        for method in methods:
            mseeds = seeds if method in ("hmsd", "td3", "sac", "pso") else [0]
            for seed in mseeds:
                name = f"{season}_{method}_s{seed}"
                jobs.append(
                    {
                        "name": name,
                        "season": season,
                        "method": method,
                        "seed": seed,
                        "episodes": episodes,
                        "job_id": f"seasonal_{name}",
                        "run_dir": rf"{REMOTE}\runs\seasonal_v1\{season}\{method}_s{seed}",
                        "log": rf"{REMOTE}\logs\seasonal_v1_{name}.log",
                        "bat": rf"{REMOTE}\logs\run_v1_{name}.bat",
                    }
                )
    return jobs


def make_bat(job: dict) -> str:
    ep = job["episodes"]
    extra = ""
    if job["method"] == "pso":
        extra = " --pso-iters 20 --pso-particles 10"
    return (
        f"@echo off\r\n"
        f"cd /d {REMOTE}\r\n"
        f"set PYTHONUNBUFFERED=1\r\n"
        f"set PYTHONPATH={REMOTE}\\src\r\n"
        f"set OPTIMAL_DEMO_CACHE={CACHE}\r\n"
        f"set OPTIMAL_DEMO_JOB_ID={job['job_id']}\r\n"
        f"set OPTIMAL_DEMO_TMP={CACHE}\\tmp\\{job['job_id']}\r\n"
        f"set OPTIMAL_DEMO_FMU_ISOLATE=1\r\n"
        f"if not exist \"%OPTIMAL_DEMO_TMP%\" mkdir \"%OPTIMAL_DEMO_TMP%\"\r\n"
        f"if not exist \"{job['run_dir']}\" mkdir \"{job['run_dir']}\"\r\n"
        f"echo START %DATE% %TIME% > \"{job['log']}\"\r\n"
        f"\"{PY}\" \"{REMOTE}\\scripts\\train_seasonal.py\" "
        f"--method {job['method']} --season {job['season']} "
        f"--episodes {ep} --seed {job['seed']} "
        f"--run-dir \"{job['run_dir']}\"{extra} "
        f">> \"{job['log']}\" 2>> \"{job['log']}.err\"\r\n"
        f"echo EXIT %ERRORLEVEL% %DATE% %TIME% >> \"{job['log']}\"\r\n"
    )


QUEUE_PY = r'''
import json, subprocess, time
from pathlib import Path

ROOT = r"D:\xuzh\demo_optimization"
MAX_LIVE = __MAX__
QUEUE = json.loads(Path(ROOT, "logs", "fair_queue.json").read_text(encoding="utf-8"))
state_path = Path(ROOT, "logs", "fair_queue_state.json")
logp = Path(ROOT, "logs", "fair_queue.log")

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
log("FAIR_QUEUE start n=%d max=%d started=%d" % (len(QUEUE), MAX_LIVE, len(started)))

while True:
    pending = [j for j in QUEUE if j["name"] not in started]
    live = live_jobs()
    n_live = len(live)
    log("live=%d pending=%d started=%d" % (n_live, len(pending), len(started)))
    if not pending and n_live == 0:
        log("QUEUE_DONE")
        break
    if pending and n_live < MAX_LIVE:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--seasons", type=str, default="winter,transition,summer")
    ap.add_argument("--methods", type=str, default="hmsd,td3,sac,pso,linprog")
    ap.add_argument("--max-live", type=int, default=4)
    ap.add_argument("--sync-only", action="store_true")
    ap.add_argument("--launch-only", action="store_true")
    ap.add_argument("--no-clean", action="store_true")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    seasons = [x.strip() for x in args.seasons.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    jobs = build_jobs(args.episodes, seeds, seasons, methods)
    print(f"jobs={len(jobs)} episodes={args.episodes} max_live={args.max_live}", flush=True)

    t, sftp = connect()
    try:
        sftp.stat(REMOTE)
    except OSError as exc:
        raise SystemExit(f"remote missing: {exc}") from exc

    if not args.launch_only:
        print("=== sync ===", flush=True)
        print(sync(sftp), flush=True)
        for rel in (
            "scripts/train_seasonal.py",
            "src/training/episode_starts.py",
            "src/training/ghtd3/train.py",
            "src/training/hybrid_td3/train.py",
            "src/training/hybrid_sac/train.py",
            "src/training/hybrid_common/eval_and_save.py",
            "src/config/ghtd3_config.yaml",
            "src/optimization/metrics.py",
            "src/optimization/pso_fmu.py",
        ):
            force_put(sftp, rel)

    if not args.no_clean:
        print("=== clean old seasonal + locks + cache tmp ===", flush=True)
        clean_bat = f"""@echo off
REM stop old queues / seasonal
wmic process where "CommandLine like '%%train_seasonal%%'" call terminate 2>nul
wmic process where "CommandLine like '%%fair_queue%%'" call terminate 2>nul
wmic process where "CommandLine like '%%wmic_queue%%'" call terminate 2>nul
timeout /t 2 /nobreak >nul
REM remove protocol-v0 seasonal runs (wrong eval week / single-week FORCE)
if exist "{REMOTE}\\runs\\seasonal" rmdir /s /q "{REMOTE}\\runs\\seasonal"
if exist "{CACHE}\\tmp" rmdir /s /q "{CACHE}\\tmp"
mkdir "{CACHE}\\tmp" 2>nul
mkdir "{CACHE}\\fmu_copies" 2>nul
mkdir "{REMOTE}\\runs\\seasonal_v1" 2>nul
mkdir "{REMOTE}\\logs" 2>nul
echo CLEAN_DONE
"""
        local_c = LOCAL / "logs" / "_remote_clean_fair.bat"
        local_c.parent.mkdir(parents=True, exist_ok=True)
        local_c.write_text(clean_bat, encoding="ascii")
        sftp.put(str(local_c), rpath("logs", "_remote_clean_fair.bat"), confirm=False)
        print(run(t, f'cmd /c "{rpath("logs", "_remote_clean_fair.bat")}"', timeout=180), flush=True)

    if args.sync_only:
        sftp.close()
        t.close()
        print("sync-only done", flush=True)
        return

    # smoke import
    smoke = (
        "import sys\n"
        f"sys.path.insert(0, r'{REMOTE}\\src')\n"
        "from training.episode_starts import eval_start_seconds\n"
        "from training.ghtd3.train import load_ghtd3_config\n"
        "c=load_ghtd3_config()['ghtd3']\n"
        "assert c.get('low_reward')=='ext'\n"
        "print('SMOKE_OK', c['goal_dim'], c['low_reward'])\n"
    )
    local_s = LOCAL / "logs" / "_fair_smoke.py"
    local_s.write_text(smoke, encoding="utf-8")
    sftp.put(str(local_s), rpath("logs", "_fair_smoke.py"), confirm=False)
    smoke_out = run(
        t,
        f'powershell -NoProfile -Command "Set-Location \'{REMOTE}\'; $env:OPTIMAL_DEMO_CACHE=\'{CACHE}\'; '
        f'$env:PYTHONPATH=\'{REMOTE}\\src\'; & \'{PY}\' \'{rpath("logs", "_fair_smoke.py")}\'"',
        timeout=120,
    )
    print("smoke:", smoke_out.strip()[:500], flush=True)
    if "SMOKE_OK" not in smoke_out:
        raise SystemExit("remote smoke failed")

    # upload bats + queue
    for job in jobs:
        local_b = LOCAL / "logs" / f"run_v1_{job['name']}.bat"
        local_b.write_text(make_bat(job), encoding="ascii")
        sftp.put(str(local_b), job["bat"], confirm=False)
    print(f"uploaded {len(jobs)} bats", flush=True)

    qj = [{"name": j["name"], "bat": j["bat"], "run_dir": j["run_dir"]} for j in jobs]
    local_q = LOCAL / "logs" / "fair_queue.json"
    local_q.write_text(json.dumps(qj, indent=2), encoding="utf-8")
    sftp.put(str(local_q), rpath("logs", "fair_queue.json"), confirm=False)

    qpy = QUEUE_PY.replace("__MAX__", str(int(args.max_live)))
    local_py = LOCAL / "logs" / "fair_queue.py"
    local_py.write_text(qpy, encoding="utf-8")
    sftp.put(str(local_py), rpath("logs", "fair_queue.py"), confirm=False)
    try:
        sftp.remove(rpath("logs", "fair_queue_state.json"))
    except OSError:
        pass

    start_bat = (
        f"@echo off\r\ncd /d {REMOTE}\r\n"
        f"set OPTIMAL_DEMO_CACHE={CACHE}\r\n"
        f"\"{PY}\" \"{REMOTE}\\logs\\fair_queue.py\"\r\n"
    )
    local_sb = LOCAL / "logs" / "start_fair_queue.bat"
    local_sb.write_text(start_bat, encoding="ascii")
    sftp.put(str(local_sb), rpath("logs", "start_fair_queue.bat"), confirm=False)

    out = run(t, f'wmic process call create "cmd.exe /c call \\"{rpath("logs", "start_fair_queue.bat")}\\""', timeout=40)
    print(out.strip()[:400], flush=True)

    time.sleep(20)
    print(
        run(
            t,
            f'powershell -NoProfile -Command "if(Test-Path \'{REMOTE}\\logs\\fair_queue.log\'){{ Get-Content \'{REMOTE}\\logs\\fair_queue.log\' -Tail 15 }} else {{ \'NOLOG\' }}"',
            timeout=40,
        ).strip(),
        flush=True,
    )
    man = LOCAL / "logs" / "remote_fair_suite_manifest.json"
    man.write_text(
        json.dumps(
            {
                "episodes": args.episodes,
                "max_live": args.max_live,
                "n_jobs": len(jobs),
                "jobs": [j["name"] for j in jobs],
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
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
