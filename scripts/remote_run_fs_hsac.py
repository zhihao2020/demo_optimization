#!/usr/bin/env python
"""Sync FS-HSAC + monthly TOU to the training box, optionally launch seasonal jobs.

Default: sync only and print GPU / live python jobs.
  python scripts/remote_run_fs_hsac.py
  python scripts/remote_run_fs_hsac.py --launch --seasons winter,transition,summer --seeds 0
  python scripts/remote_run_fs_hsac.py --launch --support --episodes 5000

Run dirs: runs/seasonal_tou2026/<season>/fs_hsac[_support]_s<seed>
(isolated from old-price seasonal_v1).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import paramiko

HOST = "172.16.1.80"
USER = "dell"
PASSWORD = "TR@SZ"
REMOTE = r"D:\xuzh\demo_optimization"
LOCAL = Path(__file__).resolve().parents[1]
PY = REMOTE + r"\.venv\Scripts\python.exe"

SYNCS = [
    "scripts/train_seasonal.py",
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
    "src/training/hybrid_common/eval_and_save.py",
    "src/training/hybrid_common/policy_wrapper.py",
    "src/training/episode_starts.py",
    "src/config/env_config.yaml",
    "src/config/givesafe_config.yaml",
    "src/config/reward_config.yaml",
    "data/price_tou.csv",
    "data/price_tou_monthly_official.csv",
    "data/price_tou_meta.json",
    "data/price_tou_README.md",
    "tests/test_fs_hsac_support.py",
    "tests/test_fs_hsac_algorithm.py",
    "tests/test_fs_hsac_replay.py",
    "tests/test_fs_hsac_feasibility.py",
    "tests/test_market.py",
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--launch", action="store_true")
    p.add_argument("--support", action="store_true", help="FS-HSAC-support (no C_psi)")
    p.add_argument("--seasons", type=str, default="winter,transition,summer")
    p.add_argument("--seeds", type=str, default="0")
    p.add_argument("--episodes", type=int, default=5000)
    args = p.parse_args()
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    tag = "fs_hsac_support" if args.support else "fs_hsac"

    t = paramiko.Transport((HOST, 22))
    t.banner_timeout = 60
    t.auth_timeout = 60
    t.start_client(timeout=60)
    t.auth_password(USER, PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(t)

    print("=== GPU ===")
    print(run(t, "cmd /c nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader"))
    print("=== live python (train_seasonal) ===")
    live = run(
        t,
        r'cmd /c wmic process where "name=\"python.exe\"" get CommandLine /FORMAT:LIST',
        60,
    )
    for line in live.splitlines():
        if "train_seasonal" in line.lower() or "fs_hsac" in line.lower():
            print(line[:240])

    for rel in SYNCS:
        src = LOCAL / rel
        if not src.is_file():
            raise FileNotFoundError(src)
        rpath = REMOTE + "\\" + rel.replace("/", "\\")
        sftp_mkdirs(sftp, str(Path(rpath).parent))
        sftp.put(str(src), rpath)
        print("PUT", rel)

    if not args.launch:
        print("sync-only; pass --launch to start jobs")
        sftp.close()
        t.close()
        return

    for season in seasons:
        for seed in seeds:
            name = f"tou2026_{season}_{tag}_s{seed}"
            run_abs = rf"{REMOTE}\runs\seasonal_tou2026\{season}\{tag}_s{seed}"
            log = rf"{REMOTE}\logs\{name}.log"
            err = log + ".err"
            bat = rf"{REMOTE}\logs\run_{name}.bat"
            feas = "set FS_HSAC_NO_FEAS=1" if args.support else "set FS_HSAC_NO_FEAS="
            body = f"""@echo off
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
if not exist "%OPTIMAL_DEMO_TMP%" mkdir "%OPTIMAL_DEMO_TMP%"
if not exist "{run_abs}" mkdir "{run_abs}"
echo START %DATE% %TIME% > "{log}"
"{PY}" "{REMOTE}\\scripts\\train_seasonal.py" --method fs_hsac --season {season} --episodes {args.episodes} --seed {seed} --run-dir "{run_abs}" >> "{log}" 2>> "{err}"
echo EXIT %ERRORLEVEL% %DATE% %TIME% >> "{log}"
"""
            sftp_mkdirs(sftp, str(Path(bat).parent))
            sftp_mkdirs(sftp, run_abs)
            with sftp.file(bat, "wb") as f:
                f.write(body.replace("\n", "\r\n").encode("gbk", errors="replace"))
            ps = (
                "powershell -NoProfile -Command "
                f"\"Start-Process -FilePath 'cmd.exe' -ArgumentList '/c','{bat}' "
                f"-WorkingDirectory '{REMOTE}' -WindowStyle Hidden\""
            )
            print("START", name)
            print(run(t, ps, 30)[:400])

    time.sleep(4)
    print("=== GPU after start ===")
    print(run(t, "cmd /c nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader"))
    sftp.close()
    t.close()


if __name__ == "__main__":
    main()
