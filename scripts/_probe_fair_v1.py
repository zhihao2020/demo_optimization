#!/usr/bin/env python
from __future__ import annotations
import time
import paramiko

HOST="172.16.1.80"; USER="dell"; PW="TR@SZ"
ROOT=r"D:\xuzh\demo_optimization"

def run(t, cmd, timeout=60):
    ch=t.open_session(); ch.set_combine_stderr(True); ch.exec_command(cmd)
    buf=b""; end=time.time()+timeout
    while time.time()<end:
        if ch.recv_ready(): buf+=ch.recv(65536)
        elif ch.exit_status_ready():
            while ch.recv_ready(): buf+=ch.recv(65536)
            break
        else: time.sleep(0.05)
    return buf.decode("utf-8","replace")

t=paramiko.Transport((HOST,22)); t.banner_timeout=60; t.auth_timeout=60
t.start_client(timeout=60); t.auth_password(USER,PW)
print("=== queue ===")
print(run(t, f'powershell -NoProfile -Command "Get-Content \'{ROOT}\\logs\\fair_queue.log\' -Tail 25"').strip())
print("=== winter_hmsd_s0 err ===")
print(run(t, f'powershell -NoProfile -Command "Get-Content \'{ROOT}\\logs\\seasonal_v1_winter_hmsd_s0.log.err\' -Tail 40"').strip()[:2500])
print("=== winter_hmsd_s0 log ===")
print(run(t, f'powershell -NoProfile -Command "Get-Content \'{ROOT}\\logs\\seasonal_v1_winter_hmsd_s0.log\' -Tail 30"').strip()[:1500])
print("=== winter_hmsd_s1 err ===")
print(run(t, f'powershell -NoProfile -Command "Get-Content \'{ROOT}\\logs\\seasonal_v1_winter_hmsd_s1.log.err\' -Tail 40"').strip()[:2000])
print("=== live train_seasonal ===")
wmic=run(t, r'cmd /c wmic process where "name=\"python.exe\"" get ProcessId,CommandLine /FORMAT:LIST', 90)
for ln in wmic.splitlines():
    if "train_seasonal" in ln.lower():
        print(ln[:300])
t.close()
