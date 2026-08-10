#!/usr/bin/env python
from __future__ import annotations
import time
import paramiko

HOST="172.16.1.80"; USER="dell"; PW="TR@SZ"
ROOT=r"D:\xuzh\demo_optimization"

def run(t, cmd, timeout=90):
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

print("=== train_seasonal python ===")
wmic=run(t, r'cmd /c wmic process where "name=\"python.exe\"" get ProcessId,CommandLine /FORMAT:LIST')
n=0
for ln in wmic.splitlines():
    if "train_seasonal" in ln.lower():
        print(ln[:320]); n+=1
    if ln.strip().lower().startswith("processid=") and n:
        print(ln)
print("python_train_seasonal_lines", n)

print("=== queue logs ===")
for name in ["wmic_queue.log","wmic_queue_runner.log","wmic_queue_state.json"]:
    p=rf"{ROOT}\logs\{name}"
    ps=f"if(Test-Path '{p}'){{ $i=Get-Item '{p}'; 'SIZE='+$i.Length; Get-Content '{p}' -TotalCount 40 }} else {{ 'MISS' }}"
    print("---", name)
    print(run(t, f'powershell -NoProfile -Command "{ps}"').strip()[:1500])

print("=== job progress tails ===")
import re
names=["winter_hmsd_s0","winter_hmsd_s1","winter_hmsd_s2","winter_td3_s0","winter_td3_s1","transition_hmsd_s0"]
for name in names:
    p=rf"{ROOT}\logs\seasonal_{name}.log"
    ps=(
        f"if(Test-Path '{p}'){{ $i=Get-Item '{p}'; 'SIZE='+$i.Length; "
        f"Get-Content '{p}' -Tail 8 }} else {{ 'MISS' }}"
    )
    print("---", name)
    print(run(t, f'powershell -NoProfile -Command "{ps}"').strip()[:800])

print("=== gpu ===")
print(run(t, "cmd /c nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv").strip())
t.close()
