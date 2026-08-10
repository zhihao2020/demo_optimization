#!/usr/bin/env python
"""Push fix scripts, reinstall CUDA torch on remote, start fair queue only."""
from __future__ import annotations

import time
from pathlib import Path

import paramiko

HOST = "172.16.1.80"
USER = "dell"
PASSWORD = "TR@SZ"
REMOTE = r"D:\xuzh\demo_optimization"
LOCAL = Path(__file__).resolve().parents[1]


def connect():
    t = paramiko.Transport((HOST, 22))
    t.banner_timeout = 120
    t.auth_timeout = 120
    t.start_client(timeout=120)
    t.auth_password(USER, PASSWORD)
    return t, paramiko.SFTPClient.from_transport(t)


def run(t: paramiko.Transport, cmd: str, timeout: int = 120) -> str:
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


def main() -> None:
    t, sftp = connect()
    for name in ("_fix_torch_cuda.ps1", "_start_fair_only.ps1"):
        local = LOCAL / "logs" / name
        remote = REMOTE + rf"\logs\{name}"
        sftp.put(str(local), remote)
        print("put", name, flush=True)

    print("--- fix torch ---", flush=True)
    out = run(
        t,
        f'powershell -NoProfile -ExecutionPolicy Bypass -File "{REMOTE}\\logs\\_fix_torch_cuda.ps1"',
        timeout=3600,
    )
    print(out[-3500:] if len(out) > 3500 else out, flush=True)
    if "TORCH_OK" not in out:
        raise SystemExit("torch fix failed")

    print("--- start fair ---", flush=True)
    print(
        run(
            t,
            f'powershell -NoProfile -ExecutionPolicy Bypass -File "{REMOTE}\\logs\\_start_fair_only.ps1"',
            timeout=120,
        ),
        flush=True,
    )
    time.sleep(25)

    print("--- verify ---", flush=True)
    print(run(t, f"cmd /c type {REMOTE}\\.venv\\pyvenv.cfg", timeout=30), flush=True)
    print(
        run(
            t,
            f'powershell -NoProfile -Command "Get-Content \'{REMOTE}\\logs\\fair_queue.log\' -Tail 20"',
            timeout=40,
        ),
        flush=True,
    )
    live = run(
        t,
        r'cmd /c wmic process where "name=\"python.exe\"" get CommandLine /FORMAT:LIST',
        timeout=90,
    )
    lines = [
        ln
        for ln in live.splitlines()
        if "train_seasonal" in ln.lower() or "fair_queue" in ln.lower()
    ]
    print("live_lines", len(lines), flush=True)
    for ln in lines[:20]:
        print(ln[:230], flush=True)
    print("comfy_hits", sum(1 for ln in lines if "comfyui" in ln.lower()), flush=True)
    print(
        run(
            t,
            "cmd /c nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader",
            timeout=30,
        ),
        flush=True,
    )
    sftp.close()
    t.close()
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
