"""Launch three formal trainings via separate Start-Process calls."""
from __future__ import annotations

import sys
import time

import paramiko

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HOST = "172.16.1.80"
USER = "dell"
PASSWORD = "TR@SZ"
ROOT = r"D:\xuzh\demo_optimization"
PY = ROOT + r"\.venv\Scripts\python.exe"


def transport():
    t = paramiko.Transport((HOST, 22))
    t.banner_timeout = 30
    t.auth_timeout = 30
    t.start_client(timeout=30)
    t.auth_password(USER, PASSWORD)
    return t


def run(t, cmd, timeout=120):
    chan = t.open_session(timeout=30)
    chan.set_combine_stderr(True)
    chan.exec_command(cmd)
    chunks = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        if chan.recv_ready():
            chunks.append(chan.recv(65536))
        elif chan.exit_status_ready():
            while chan.recv_ready():
                chunks.append(chan.recv(65536))
            break
        else:
            time.sleep(0.05)
    code = chan.recv_exit_status() if chan.exit_status_ready() else -1
    text = b"".join(chunks).decode("utf-8", errors="replace")
    return code, text


t = transport()
_, out = run(t, "hostname")
print(out)

# Ensure runs dir and empty launch log
run(t, r'cmd /c "if not exist D:\xuzh\demo_optimization\runs mkdir D:\xuzh\demo_optimization\runs & echo RELAUNCH>%date% %time% > D:\xuzh\demo_optimization\runs\formal_launch.log"')

jobs = [
    (
        "td3",
        r"scripts\train_hybrid_td3.py --mode formal --annual-eval --run-dir runs\givesafe_td3_formal",
        r"runs\formal_td3.log",
    ),
    (
        "ppo",
        r"scripts\train_hybrid_ppo.py --mode formal --annual-eval --run-dir runs\givesafe_ppo_formal",
        r"runs\formal_ppo.log",
    ),
    (
        "sac",
        r"scripts\train_hybrid_sac.py --mode formal --annual-eval --run-dir runs\givesafe_sac_formal",
        r"runs\formal_sac.log",
    ),
]

for name, args, log in jobs:
    # Use cmd /c with set PYTHONPATH then python, redirect to log; wrap completion echo
    # Start-Process with Environment is hard; use a tiny per-job bat
    bat_body = f"""@echo off
cd /d D:\\xuzh\\demo_optimization
set PYTHONPATH=src
set PY=D:\\xuzh\\demo_optimization\\.venv\\Scripts\\python.exe
echo [%date% %time%] {name}_start>> runs\\formal_launch.log
%PY% {args} > {log} 2>&1
echo [%date% %time%] {name}_done=%ERRORLEVEL%>> runs\\formal_launch.log
"""
    sftp = paramiko.SFTPClient.from_transport(t)
    assert sftp is not None
    bat_path = f"/D:/xuzh/demo_optimization/run_formal_{name}.bat"
    with sftp.file(bat_path, "w") as f:
        f.write(bat_body.replace("\n", "\r\n"))
    sftp.close()

    ps = (
        "powershell -NoProfile -Command "
        f"\"$p=Start-Process -FilePath 'cmd.exe' "
        f"-ArgumentList '/c','D:\\xuzh\\demo_optimization\\run_formal_{name}.bat' "
        f"-WorkingDirectory 'D:\\xuzh\\demo_optimization' "
        f"-WindowStyle Hidden -PassThru; "
        f"Write-Output ('{name}_PID=' + $p.Id)\""
    )
    code, out = run(t, ps, timeout=60)
    print(out.strip())

time.sleep(8)
_, out = run(t, r'cmd /c "type D:\xuzh\demo_optimization\runs\formal_launch.log"')
print("LAUNCH_LOG:")
print(out)
_, out = run(t, r'cmd /c "tasklist /FI "IMAGENAME eq python.exe""')
print("PYTHON:")
print(out)
for name in ("td3", "ppo", "sac"):
    _, out = run(
        t,
        "powershell -NoProfile -Command "
        f"\"if (Test-Path 'D:\\xuzh\\demo_optimization\\runs\\formal_{name}.log') {{ "
        f"(Get-Item 'D:\\xuzh\\demo_optimization\\runs\\formal_{name}.log').Length; "
        f"Get-Content 'D:\\xuzh\\demo_optimization\\runs\\formal_{name}.log' -Tail 3 }} else {{ 'missing' }}\"",
    )
    print(f"LOG_{name}:", out.strip()[:500])

t.close()
print("LAUNCH2_OK")
