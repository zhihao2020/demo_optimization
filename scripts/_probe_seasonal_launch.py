#!/usr/bin/env python
"""Probe seasonal launch status / errors on remote."""
from __future__ import annotations

import time

import paramiko

HOST = "172.16.1.80"
USER = "dell"
PASSWORD = "TR@SZ"
ROOT = r"D:\xuzh\demo_optimization"
PY = r"D:\xuzh\demo_optimization\.venv\Scripts\python.exe"


def run(t: paramiko.Transport, cmd: str, timeout: int = 90) -> str:
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
    t = paramiko.Transport((HOST, 22))
    t.banner_timeout = 60
    t.auth_timeout = 60
    t.start_client(timeout=60)
    t.auth_password(USER, PASSWORD)

    print("=== python processes ===")
    print(run(t, r'cmd /c tasklist /FI "IMAGENAME eq python.exe"'))

    print("=== wmic train lines ===")
    wmic = run(
        t,
        r'cmd /c wmic process where "name=\"python.exe\"" get ProcessId,CommandLine /FORMAT:LIST',
        timeout=90,
    )
    for ln in wmic.splitlines():
        if any(k in ln.lower() for k in ("train", "seasonal", "ghtd3", "processid=", "commandline=")):
            if "commandline=" in ln.lower() and "train" not in ln.lower() and "seasonal" not in ln.lower():
                continue
            print(ln[:400])

    print("=== logs dir ===")
    print(run(t, rf"cmd /c dir /B {ROOT}\logs"))

    names = [
        "seasonal_winter_hmsd_s0.log",
        "seasonal_winter_hmsd_s0.log.err",
        "seasonal_winter_td3_s0.log",
        "seasonal_winter_td3_s0.log.err",
        "seasonal_summer_td3_s2.log.err",
    ]
    for name in names:
        p = rf"{ROOT}\logs\{name}"
        ps = (
            f"if (Test-Path '{p}') {{ "
            f"$i=Get-Item '{p}'; Write-Output ('SIZE=' + $i.Length); "
            f"Get-Content '{p}' -TotalCount 40 "
            f"}} else {{ Write-Output 'MISS' }}"
        )
        print(f"--- {name} ---")
        print(run(t, f'powershell -NoProfile -Command "{ps}"', timeout=40).strip()[:2000])

    print("=== foreground help ===")
    help_ps = (
        f"Set-Location '{ROOT}'; "
        f"$env:PYTHONPATH='{ROOT}\\src'; "
        f"& '{PY}' '{ROOT}\\scripts\\train_seasonal.py' --help"
    )
    print(run(t, f'powershell -NoProfile -Command "{help_ps}"', timeout=60).strip()[:1500])

    print("=== foreground short import train path ===")
    # 1-episode would need FMU; just instantiate argparse path
    test_ps = (
        f"Set-Location '{ROOT}'; "
        f"$env:PYTHONPATH='{ROOT}\\src'; "
        f"& '{PY}' -c \"import scripts; print('ok')\""
    )
    # better: run python -c with sys.path
    test_ps = (
        f"Set-Location '{ROOT}'; "
        f"$env:PYTHONPATH='{ROOT}\\src'; "
        f"& '{PY}' -c \"import sys; sys.path.insert(0,r'{ROOT}\\src'); "
        f"from training.ghtd3.train import load_ghtd3_config; print(load_ghtd3_config()['ghtd3']['goal_dim'])\""
    )
    print(run(t, f'powershell -NoProfile -Command "{test_ps}"', timeout=60).strip()[:800])

    t.close()


if __name__ == "__main__":
    main()
