#!/usr/bin/env python
"""Kill remote jobs, delete remote src/training, full-sync local training stack, wipe runs, start fair.

Usage:
  python scripts/remote_resync_training_fair.py
  python scripts/remote_resync_training_fair.py --max-live 8
  python scripts/remote_resync_training_fair.py --dry-run
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import paramiko

HOST = "172.16.1.80"
USER = "dell"
PASSWORD = "TR@SZ"
REMOTE = r"D:\xuzh\demo_optimization"
LOCAL = Path(__file__).resolve().parents[1]

SYNC_DIRS = [
    "src/training",
    "src/actions",
    "src/envs",
    "src/safety",
    "src/config",
    "src/controllers",
    "src/fmu",
    "src/market",
    "src/optimization",
    "src/replay",
    "src/economics",
    "src/forecast",
]
SKIP_DIR_NAMES = {"__pycache__", ".git", ".pytest_cache", ".mypy_cache"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".pyd"}


def connect():
    t = paramiko.Transport((HOST, 22))
    t.banner_timeout = 120
    t.auth_timeout = 120
    t.start_client(timeout=120)
    t.auth_password(USER, PASSWORD)
    return t, paramiko.SFTPClient.from_transport(t)


def run(t: paramiko.Transport, cmd: str, timeout: int = 180) -> str:
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


def sftp_mkdirs(sftp: paramiko.SFTPClient, remote: str) -> None:
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


def collect_local_files() -> list[tuple[Path, str]]:
    """Return list of (local_path, remote_rel_posix)."""
    out: list[tuple[Path, str]] = []
    for d in SYNC_DIRS:
        root = LOCAL / d
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if any(part in SKIP_DIR_NAMES for part in p.parts):
                continue
            if p.suffix.lower() in SKIP_SUFFIXES:
                continue
            rel = p.relative_to(LOCAL).as_posix()
            out.append((p, rel))
    # train scripts
    for p in (LOCAL / "scripts").glob("train_*.py"):
        out.append((p, f"scripts/{p.name}"))
    # fair launcher pieces
    for name in ("fair_queue.py", "fair_queue.json", "start_fair_queue.bat"):
        p = LOCAL / "logs" / name
        if p.is_file():
            out.append((p, f"logs/{name}"))
    for p in (LOCAL / "logs").glob("run_v1_*.bat"):
        out.append((p, f"logs/{p.name}"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-live", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-start", action="store_true")
    ap.add_argument("--workers", type=int, default=8, help="parallel SFTP puts")
    args = ap.parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ensure local fair_queue MAX_LIVE
    fq = LOCAL / "logs" / "fair_queue.py"
    text = fq.read_text(encoding="utf-8")
    import re

    text2, n = re.subn(r"MAX_LIVE\s*=\s*\d+", f"MAX_LIVE = {int(args.max_live)}", text, count=1)
    if n:
        fq.write_text(text2, encoding="utf-8")
        print(f"local fair_queue MAX_LIVE -> {args.max_live}", flush=True)
    else:
        print("WARN: could not patch MAX_LIVE in fair_queue.py", flush=True)

    files = collect_local_files()
    print(f"=== resync @ {HOST} files={len(files)} max_live={args.max_live} dry={args.dry_run} ===", flush=True)

    t, sftp = connect()

    # 1) kill
    print("\n--- kill ---", flush=True)
    kill_ps = f"""
$ErrorActionPreference='Continue'
$keys=@('train_seasonal','fair_queue','wmic_queue','run_v1_','train_ghtd3','train_hybrid')
Get-CimInstance Win32_Process -Filter "name='python.exe' OR name='cmd.exe'" | ForEach-Object {{
  $cl=$_.CommandLine; if(-not $cl){{return}}
  foreach($k in $keys){{ if($cl -match [regex]::Escape($k)){{
    Write-Output ('KILL '+$_.ProcessId); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; break
  }}}}
}}
Start-Sleep -Seconds 3
$left=@(Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object {{
  $_.CommandLine -and ($_.CommandLine -match 'train_seasonal|fair_queue|wmic_queue')
}})
Write-Output ('LEFT='+$left.Count)
$left | ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }}
Write-Output 'KILL_DONE'
"""
    kill_local = LOCAL / "logs" / f"_resync_kill_{stamp}.ps1"
    kill_local.write_text(kill_ps, encoding="utf-8")
    if not args.dry_run:
        sftp.put(str(kill_local), REMOTE + rf"\logs\_resync_kill_{stamp}.ps1")
        print(run(t, f'powershell -NoProfile -ExecutionPolicy Bypass -File "{REMOTE}\\logs\\_resync_kill_{stamp}.ps1"', 120)[:2000], flush=True)
        time.sleep(2)
    else:
        print("dry-run kill", flush=True)

    # 2) delete remote training + wipe runs
    print("\n--- delete remote training + wipe runs ---", flush=True)
    wipe_ps = f"""
$ErrorActionPreference='Continue'
$root='{REMOTE}'
$dry=${str(args.dry_run).lower()}
function Kill-Path($p){{
  if(Test-Path $p){{ Write-Output ('REMOVE '+$p); if(-not $dry){{ Remove-Item -LiteralPath $p -Recurse -Force -ErrorAction Continue }} }}
  else {{ Write-Output ('MISS '+$p) }}
}}
Kill-Path (Join-Path $root 'src\\training')
Kill-Path (Join-Path $root 'runs\\seasonal_v1')
Kill-Path (Join-Path $root 'runs\\seasonal')
Get-ChildItem (Join-Path $root 'logs') -Filter 'seasonal_v1_*' -ErrorAction SilentlyContinue | ForEach-Object {{
  Write-Output ('REMOVE '+$_.FullName)
  if(-not $dry){{ Remove-Item -LiteralPath $_.FullName -Force -ErrorAction Continue }}
}}
Get-ChildItem (Join-Path $root 'logs') -Filter 'seasonal_*.log*' -ErrorAction SilentlyContinue | ForEach-Object {{
  Write-Output ('REMOVE '+$_.FullName)
  if(-not $dry){{ Remove-Item -LiteralPath $_.FullName -Force -ErrorAction Continue }}
}}
foreach($n in @('fair_queue_state.json','wmic_queue_state.json')){{ Kill-Path (Join-Path $root ('logs\\'+$n)) }}
$flog=Join-Path $root 'logs\\fair_queue.log'
if(Test-Path $flog){{
  if(-not $dry){{
    Copy-Item $flog (Join-Path $root ('logs\\fair_queue.log.bak_{stamp}')) -Force -ErrorAction SilentlyContinue
    Set-Content -LiteralPath $flog -Value '' -Encoding utf8
  }}
  Write-Output ('RESET '+$flog)
}}
# clear pyc under src
Get-ChildItem (Join-Path $root 'src') -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue | ForEach-Object {{
  Write-Output ('REMOVE '+$_.FullName)
  if(-not $dry){{ Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction Continue }}
}}
Write-Output 'WIPE_DONE'
"""
    wipe_local = LOCAL / "logs" / f"_resync_wipe_{stamp}.ps1"
    wipe_local.write_text(wipe_ps, encoding="utf-8")
    if not args.dry_run:
        sftp.put(str(wipe_local), REMOTE + rf"\logs\_resync_wipe_{stamp}.ps1")
        print(run(t, f'powershell -NoProfile -ExecutionPolicy Bypass -File "{REMOTE}\\logs\\_resync_wipe_{stamp}.ps1"', 600)[:3500], flush=True)
    else:
        print(wipe_ps[:500], flush=True)

    # 3) full sync
    print("\n--- sync files ---", flush=True)
    if args.dry_run:
        for _, rel in files[:30]:
            print("PUT", rel, flush=True)
        print(f"... total {len(files)}", flush=True)
    else:
        # serial mkdirs first for parents
        parents = sorted({str(Path(rel).parent).replace("/", "\\") for _, rel in files})
        for par in parents:
            sftp_mkdirs(sftp, REMOTE + "\\" + par)

        def put_one(item: tuple[Path, str]) -> str:
            local_p, rel = item
            rpath = REMOTE + "\\" + rel.replace("/", "\\")
            # each worker needs own sftp channel
            tt = paramiko.Transport((HOST, 22))
            tt.banner_timeout = 120
            tt.auth_timeout = 120
            tt.start_client(timeout=120)
            tt.auth_password(USER, PASSWORD)
            ss = paramiko.SFTPClient.from_transport(tt)
            try:
                ss.put(str(local_p), rpath)
                return rel
            finally:
                ss.close()
                tt.close()

        ok = 0
        fail = 0
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
            futs = {ex.submit(put_one, it): it[1] for it in files}
            for fut in as_completed(futs):
                rel = futs[fut]
                try:
                    fut.result()
                    ok += 1
                    if ok % 20 == 0 or ok == len(files):
                        print(f"uploaded {ok}/{len(files)}", flush=True)
                except Exception as exc:
                    fail += 1
                    print(f"FAIL {rel}: {exc}", flush=True)
        print(f"sync done ok={ok} fail={fail}", flush=True)
        if fail:
            raise SystemExit(f"sync failed for {fail} files")

    # 4) verify key files + start fair
    print("\n--- verify + start fair ---", flush=True)
    checks = [
        r"src\training\episode_starts.py",
        r"src\training\ghtd3\train.py",
        r"src\training\hybrid_td3\train.py",
        r"src\training\hybrid_sac\train.py",
        r"scripts\train_seasonal.py",
        r"logs\fair_queue.py",
    ]
    for c in checks:
        rp = REMOTE + "\\" + c
        if args.dry_run:
            print("check", c, flush=True)
            continue
        try:
            st = sftp.stat(rp)
            print(f"OK {c} size={st.st_size}", flush=True)
        except OSError:
            print(f"MISSING {c}", flush=True)
            raise SystemExit(f"missing {c}")

    if not args.dry_run:
        with sftp.file(REMOTE + r"\logs\fair_queue_state.json", "w") as f:
            f.write('{"started": []}\n')
        # confirm MAX_LIVE on remote
        cfg = run(t, f'powershell -NoProfile -Command "Select-String -Path \'{REMOTE}\\logs\\fair_queue.py\' -Pattern MAX_LIVE"', 30)
        print(cfg.strip(), flush=True)

    if args.skip_start or args.dry_run:
        print("skip start", flush=True)
    else:
        start_ps = f"""
$ErrorActionPreference='Continue'
$root='{REMOTE}'
$bat=Join-Path $root 'logs\\start_fair_queue.bat'
Set-Content -LiteralPath (Join-Path $root 'logs\\fair_queue_state.json') -Value '{{"started": []}}' -Encoding utf8
$cmd='cmd.exe /c call "'+$bat+'"'
$escaped=$cmd.Replace('"','\\"')
Write-Output ('START '+$cmd)
wmic process call create "$escaped"
Write-Output 'START_ISSUED'
"""
        sp = LOCAL / "logs" / f"_resync_start_{stamp}.ps1"
        sp.write_text(start_ps, encoding="utf-8")
        sftp.put(str(sp), REMOTE + rf"\logs\_resync_start_{stamp}.ps1")
        print(run(t, f'powershell -NoProfile -ExecutionPolicy Bypass -File "{REMOTE}\\logs\\_resync_start_{stamp}.ps1"', 90), flush=True)
        # wait for up to max_live starts
        time.sleep(15 + 12 * min(args.max_live, 8))

    print("\n--- status ---", flush=True)
    print(
        run(t, f'powershell -NoProfile -Command "Get-Content \'{REMOTE}\\logs\\fair_queue.log\' -Tail 25"', 40),
        flush=True,
    )
    live = run(
        t,
        r'cmd /c wmic process where "name=\"python.exe\"" get CommandLine /FORMAT:LIST',
        90,
    )
    lines = [ln for ln in live.splitlines() if "train_seasonal" in ln.lower() and "--run-dir" in ln.lower()]
    # unique run-dirs
    dirs = set()
    for ln in lines:
        if "--run-dir" in ln:
            part = ln.split("--run-dir", 1)[1].strip().strip('"')
            part = part.split(" --")[0].strip().strip('"')
            dirs.add(part.lower())
    print(f"live_cmd_lines={len(lines)} unique_run_dirs={len(dirs)}", flush=True)
    for d in sorted(dirs)[:12]:
        print(" ", d, flush=True)
    print("comfy", sum(1 for ln in lines if "comfyui" in ln.lower()), flush=True)
    print(
        run(t, "cmd /c nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader", 30),
        flush=True,
    )
    # list remote training top
    print(
        run(
            t,
            f'powershell -NoProfile -Command "Get-ChildItem \'{REMOTE}\\src\\training\' -Name"',
            30,
        ),
        flush=True,
    )

    sftp.close()
    t.close()
    print("\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
