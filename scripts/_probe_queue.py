#!/usr/bin/env python
from __future__ import annotations
import time
import paramiko

HOST="172.16.1.80"; USER="dell"; PASSWORD="TR@SZ"
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
t.start_client(timeout=60); t.auth_password(USER,PASSWORD)
for name in ["queue_runner.log","queue_runner.log.err","remote_seasonal_queue_state.json","remote_seasonal_queue.json"]:
    p=rf"{ROOT}\logs\{name}"
    ps=f"if(Test-Path '{p}'){{ $i=Get-Item '{p}'; 'SIZE='+$i.Length; Get-Content '{p}' -TotalCount 80 }} else {{ 'MISS '+'{name}' }}"
    print("====", name)
    print(run(t, f'powershell -NoProfile -Command "{ps}"').strip()[:2500])
print("==== powershell procs")
print(run(t, r'cmd /c tasklist /FI "IMAGENAME eq powershell.exe"'))
print("==== python")
print(run(t, r'cmd /c tasklist /FI "IMAGENAME eq python.exe"'))
# run queue interactively once to see error
print("==== foreground queue 15s")
# just parse json
ps=f"powershell -NoProfile -Command \"try {{ $q=Get-Content '{ROOT}\\logs\\remote_seasonal_queue.json' -Raw | ConvertFrom-Json; 'count='+$q.Count; $q[0] | ConvertTo-Json }} catch {{ $_.Exception.Message }}\""
print(run(t, ps).strip()[:1500])
t.close()
