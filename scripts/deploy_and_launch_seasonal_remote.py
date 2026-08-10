#!/usr/bin/env python
"""Sync HMSD-min code to remote and launch formal seasonal 5000-ep jobs.

Default formal matrix (Cui protocol):
  seasons × methods × seeds = 3 × 2 × 3 = 18 jobs
  each: E_max=5000, T=168 → 840_000 steps

Usage:
  python scripts/deploy_and_launch_seasonal_remote.py
  python scripts/deploy_and_launch_seasonal_remote.py --seeds 0 --seasons winter --methods hmsd
  python scripts/deploy_and_launch_seasonal_remote.py --dry-run
  python scripts/deploy_and_launch_seasonal_remote.py --sync-only
  python scripts/deploy_and_launch_seasonal_remote.py --launch-only
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import paramiko

HOST = "172.16.1.80"
USER = "dell"
PASSWORD = "TR@SZ"
REMOTE_ROOT = r"D:\xuzh\demo_optimization"
LOCAL_ROOT = Path(__file__).resolve().parents[1]

# Lean sync set (training path only)
SYNC_GLOBS = [
    "src/actions/**/*.py",
    "src/config/**/*",
    "src/controllers/**/*.py",
    "src/envs/**/*.py",
    "src/fmu/**/*.py",
    "src/market/**/*.py",
    "src/optimization/**/*.py",
    "src/safety/**/*.py",
    "src/training/**/*.py",
    "src/utils/**/*.py",
    "scripts/train_seasonal.py",
    "scripts/train_ghtd3.py",
    "scripts/train_hybrid_td3.py",
    "docs/cui_seasonal_min_protocol.md",
]

SKIP_DIR_NAMES = {"__pycache__", ".git", ".pytest_cache", ".mypy_cache", "node_modules"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".pyd"}

SEASONS = ("winter", "transition", "summer")
METHODS = ("hmsd", "td3")


def run_ssh(t: paramiko.Transport, cmd: str, timeout: int = 120) -> str:
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


def remote_path(*parts: str) -> str:
    p = REMOTE_ROOT
    for part in parts:
        p = p.rstrip("\\/") + "\\" + part.replace("/", "\\")
    return p


def sftp_mkdirs(sftp: paramiko.SFTPClient, remote: str) -> None:
    remote = remote.replace("/", "\\")
    parts = [x for x in remote.split("\\") if x]
    if not parts:
        return
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


def collect_files() -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for pattern in SYNC_GLOBS:
        for fp in LOCAL_ROOT.glob(pattern):
            if not fp.is_file():
                continue
            if any(p in SKIP_DIR_NAMES for p in fp.parts):
                continue
            if fp.suffix.lower() in SKIP_SUFFIXES:
                continue
            rel = fp.relative_to(LOCAL_ROOT).as_posix()
            if rel in seen:
                continue
            seen.add(rel)
            out.append((fp, rel))
    out.sort(key=lambda x: x[1])
    return out


def sync_tree(sftp: paramiko.SFTPClient) -> dict[str, int]:
    files = collect_files()
    print(f"  candidates: {len(files)}", flush=True)
    n_ok = n_skip = n_fail = 0
    for i, (local, rel) in enumerate(files, 1):
        rpath = remote_path(*rel.split("/"))
        try:
            st_remote = None
            try:
                st_remote = sftp.stat(rpath)
            except OSError:
                pass
            local_size = local.stat().st_size
            if st_remote is not None and int(st_remote.st_size) == int(local_size):
                # size match → skip (fast path; enough for this deploy)
                n_skip += 1
                continue
            sftp_mkdirs(sftp, str(Path(rpath).parent))
            sftp.put(str(local), rpath)
            n_ok += 1
            if n_ok % 20 == 0 or i == len(files):
                print(f"  put {n_ok} (skip {n_skip}) @ {rel}", flush=True)
        except Exception as exc:
            n_fail += 1
            print(f"  FAIL {rel}: {exc}", flush=True)
    return {"uploaded": n_ok, "skipped_same_size": n_skip, "failed": n_fail, "total": len(files)}


def find_python(t: paramiko.Transport) -> str:
    candidates = [
        r"D:\xuzh\demo_optimization\.venv\Scripts\python.exe",
        r"D:\xuzh\demo_optimization\venv\Scripts\python.exe",
        r"C:\Users\dell\miniconda3\python.exe",
        r"C:\Users\dell\anaconda3\python.exe",
        r"C:\Users\dell\AppData\Local\Programs\Python\Python311\python.exe",
        r"C:\Python311\python.exe",
    ]
    for c in candidates:
        out = run_ssh(
            t,
            rf'cmd /c if exist "{c}" (echo EXISTS) else (echo MISS)',
            timeout=15,
        )
        if "EXISTS" in out:
            return c
    out = run_ssh(t, r"cmd /c where python", timeout=20)
    for ln in out.replace("\r", "").splitlines():
        ln = ln.strip()
        if ln.lower().endswith("python.exe"):
            return ln
    return "python"


def job_spec(season: str, method: str, seed: int, episodes: int) -> dict:
    run_rel = f"runs\\seasonal\\{season}\\{method}_s{seed}"
    log_rel = f"logs\\seasonal_{season}_{method}_s{seed}.log"
    return {
        "season": season,
        "method": method,
        "seed": seed,
        "episodes": episodes,
        "run_dir": run_rel,
        "log": log_rel,
        "name": f"{season}_{method}_s{seed}",
    }


def write_and_start(
    t: paramiko.Transport,
    sftp: paramiko.SFTPClient,
    py: str,
    jobs: list[dict],
    *,
    dry_run: bool,
    stagger_sec: float,
) -> list[dict]:
    root = REMOTE_ROOT
    sftp_mkdirs(sftp, remote_path("logs"))
    sftp_mkdirs(sftp, remote_path("runs", "seasonal"))
    results = []
    local_tmp_dir = LOCAL_ROOT / "logs"
    local_tmp_dir.mkdir(parents=True, exist_ok=True)

    for i, job in enumerate(jobs):
        run_abs = f"{root}\\{job['run_dir']}"
        log_abs = f"{root}\\{job['log']}"
        err_abs = log_abs + ".err"
        launcher = f"""$ErrorActionPreference = 'Continue'
$root = '{root}'
Set-Location $root
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONPATH = (Join-Path $root 'src')
New-Item -ItemType Directory -Force -Path (Join-Path $root 'logs') | Out-Null
New-Item -ItemType Directory -Force -Path '{run_abs}' | Out-Null
$py = '{py}'
$script = Join-Path $root 'scripts\\train_seasonal.py'
$arg = @(
  $script,
  '--method', '{job["method"]}',
  '--season', '{job["season"]}',
  '--episodes', '{job["episodes"]}',
  '--seed', '{job["seed"]}',
  '--run-dir', '{run_abs}'
)
$log = '{log_abs}'
$err = '{err_abs}'
if (Test-Path $log) {{ Remove-Item $log -Force }}
if (Test-Path $err) {{ Remove-Item $err -Force }}
$p = Start-Process -FilePath $py -ArgumentList $arg -WorkingDirectory $root `
  -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError $err -PassThru
Write-Output ("STARTED id=" + $p.Id + " name={job["name"]} run={run_abs}")
"""
        remote_ps1 = remote_path("logs", f"launch_{job['name']}.ps1")
        local_tmp = local_tmp_dir / f"_launch_{job['name']}.ps1"
        local_tmp.write_text(launcher, encoding="utf-8")
        sftp_mkdirs(sftp, str(Path(remote_ps1).parent))
        sftp.put(str(local_tmp), remote_ps1)

        if dry_run:
            print(f"[dry-run] {job['name']}", flush=True)
            results.append({**job, "status": "dry_run", "launcher": remote_ps1})
            continue

        out = run_ssh(
            t,
            rf'powershell -NoProfile -ExecutionPolicy Bypass -File "{remote_ps1}"',
            timeout=90,
        )
        print(out.strip(), flush=True)
        results.append({**job, "status": "started", "launch_out": out.strip(), "launcher": remote_ps1})
        if i + 1 < len(jobs) and stagger_sec > 0:
            time.sleep(stagger_sec)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Deploy HMSD-min + launch seasonal 5000ep remote")
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--seasons", type=str, default="winter,transition,summer")
    ap.add_argument("--methods", type=str, default="hmsd,td3")
    ap.add_argument("--sync-only", action="store_true")
    ap.add_argument("--launch-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stagger", type=float, default=5.0)
    ap.add_argument("--max-parallel", type=int, default=0, help="launch only first N jobs")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    seasons = [x.strip() for x in args.seasons.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    for s in seasons:
        if s not in SEASONS:
            raise SystemExit(f"bad season {s}")
    for m in methods:
        if m not in METHODS:
            raise SystemExit(f"bad method {m}")

    jobs = [
        job_spec(season, method, seed, args.episodes)
        for season in seasons
        for method in methods
        for seed in seeds
    ]
    if args.max_parallel and args.max_parallel > 0:
        jobs = jobs[: args.max_parallel]

    print(f"=== remote {HOST} jobs={len(jobs)} episodes={args.episodes} ===", flush=True)
    for j in jobs:
        print(f"  - {j['name']} -> {j['run_dir']}", flush=True)

    t = paramiko.Transport((HOST, 22))
    t.banner_timeout = 60
    t.auth_timeout = 60
    t.start_client(timeout=60)
    t.auth_password(USER, PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(t)
    assert sftp is not None

    try:
        sftp.stat(REMOTE_ROOT)
    except OSError as exc:
        raise SystemExit(f"remote root missing {REMOTE_ROOT}: {exc}") from exc

    if not args.launch_only:
        print("=== sync code (lean) ===", flush=True)
        stats = sync_tree(sftp)
        print(f"sync done: {stats}", flush=True)
        for key in (
            r"src\config\ghtd3_config.yaml",
            r"scripts\train_seasonal.py",
            r"src\training\ghtd3\agent.py",
            r"src\actions\caes_u.py",
            r"src\training\ghtd3\goals.py",
            r"src\training\ghtd3\networks.py",
        ):
            rp = remote_path(*key.split("\\"))
            try:
                st = sftp.stat(rp)
                print(f"  OK {key} size={st.st_size}", flush=True)
            except OSError:
                print(f"  MISS {key}", flush=True)

    if args.sync_only:
        sftp.close()
        t.close()
        print("=== sync-only done ===", flush=True)
        return

    py = find_python(t)
    print(f"=== python: {py} ===", flush=True)
    ver_ps = (
        f"& '{py}' -c \"import sys; print(sys.version)\""
    )
    print(
        run_ssh(
            t,
            rf'powershell -NoProfile -Command "{ver_ps}"',
            timeout=40,
        ).strip(),
        flush=True,
    )

    # remote smoke (write small script to avoid quote hell)
    sftp_mkdirs(sftp, remote_path("logs"))
    smoke_local = LOCAL_ROOT / "logs" / "_remote_smoke.py"
    smoke_local.parent.mkdir(parents=True, exist_ok=True)
    smoke_local.write_text(
        "import sys\n"
        f"sys.path.insert(0, r'{REMOTE_ROOT}\\src')\n"
        "from training.ghtd3.train import load_ghtd3_config\n"
        "from training.ghtd3.agent import GHTD3Agent\n"
        "c = load_ghtd3_config()['ghtd3']\n"
        "a = GHTD3Agent(16, c, device='cpu')\n"
        "print('goal_dim', a.goal_dim, 'her', c.get('goal_relabel_mode'), "
        "'prior', c.get('market_goal_prior'), 'fmle', c.get('f_mle_pretrain'))\n"
        "assert a.goal_dim == 2\n"
        "assert not hasattr(a, '_ms_her_weights')\n"
        "print('SMOKE_OK')\n",
        encoding="utf-8",
    )
    smoke_remote = remote_path("logs", "_remote_smoke.py")
    sftp.put(str(smoke_local), smoke_remote)
    smoke_ps = f"Set-Location '{REMOTE_ROOT}'; & '{py}' '{smoke_remote}'"
    smoke_out = run_ssh(
        t,
        rf'powershell -NoProfile -Command "{smoke_ps}"',
        timeout=180,
    )
    print("smoke:", smoke_out.strip()[:800], flush=True)
    if "SMOKE_OK" not in smoke_out:
        raise SystemExit("remote smoke failed; abort launch")

    # GPU info
    print("=== gpu ===", flush=True)
    print(run_ssh(t, r"cmd /c nvidia-smi -L", timeout=30).strip(), flush=True)

    print("=== launch ===", flush=True)
    results = write_and_start(
        t, sftp, py, jobs, dry_run=args.dry_run, stagger_sec=args.stagger
    )

    time.sleep(6.0)
    print("=== post-launch train_seasonal processes ===", flush=True)
    procs = run_ssh(
        t,
        r'cmd /c wmic process where "name=\"python.exe\"" get ProcessId,CommandLine /FORMAT:LIST',
        timeout=90,
    )
    n_live = 0
    for ln in procs.splitlines():
        if "train_seasonal" in ln.lower() or ln.strip().lower().startswith("processid"):
            print(ln[:320], flush=True)
            if "train_seasonal" in ln.lower():
                n_live += 1

    man = LOCAL_ROOT / "logs" / "remote_seasonal_launch_manifest.json"
    man.parent.mkdir(parents=True, exist_ok=True)
    man.write_text(
        json.dumps(
            {
                "host": HOST,
                "episodes": args.episodes,
                "jobs": results,
                "python": py,
                "live_train_seasonal_cmdlines": n_live,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"manifest -> {man}", flush=True)
    print(f"live train_seasonal cmdlines seen: {n_live}", flush=True)

    sftp.close()
    t.close()
    print("=== done ===", flush=True)


if __name__ == "__main__":
    main()
