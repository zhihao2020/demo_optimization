#!/usr/bin/env python
"""Snapshot fair seasonal_v1 status on remote."""
from __future__ import annotations
import re
import time
import paramiko

HOST = "172.16.1.80"
USER = "dell"
PASSWORD = "TR@SZ"
ROOT = r"D:\xuzh\demo_optimization"


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


def main():
    t = paramiko.Transport((HOST, 22))
    t.banner_timeout = 60
    t.auth_timeout = 60
    t.start_client(timeout=60)
    t.auth_password(USER, PASSWORD)

    print("=== queue tail ===")
    print(run(t, f"powershell -NoProfile -Command \"Get-Content '{ROOT}\\logs\\fair_queue.log' -Tail 8\"").strip())

    state = run(t, f"powershell -NoProfile -Command \"Get-Content '{ROOT}\\logs\\fair_queue_state.json' -Raw -ErrorAction SilentlyContinue\"").strip()
    print("=== state ===")
    print(state[:1500])

    # list logs with EXIT / last progress
    ps = rf"""
$root='{ROOT}\logs'
Get-ChildItem $root -Filter 'seasonal_v1_*.log' | ForEach-Object {{
  $n=$_.Name
  $tail = Get-Content $_.FullName -Tail 8 -ErrorAction SilentlyContinue
  $exit = ($tail | Where-Object {{ $_ -match 'EXIT' }} | Select-Object -Last 1)
  $prog = ($tail | Where-Object {{ $_ -match '\[progress\]|Hybrid-|Hybrid-SAC|PSO|status completed' }} | Select-Object -Last 1)
  $errp = $_.FullName + '.err'
  $errLast = ''
  if (Test-Path $errp) {{
    $et = Get-Content $errp -Tail 3 -ErrorAction SilentlyContinue
    $errLast = ($et -join ' ')
  }}
  [pscustomobject]@{{ name=$n; size=$_.Length; exit=$exit; prog=$prog; err=$errLast.Substring(0,[Math]::Min(120,$errLast.Length)) }}
}} | ConvertTo-Json -Compress
"""
    print("=== logs ===")
    print(run(t, f'powershell -NoProfile -Command "{ps}"', timeout=120).strip()[:8000])

    print("=== gpu ===")
    print(run(t, "cmd /c nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv").strip())
    t.close()


if __name__ == "__main__":
    main()
