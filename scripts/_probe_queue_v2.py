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
for name in ["wmic_queue_v2.log","wmic_queue_state.json"]:
    p=rf"{ROOT}\logs\{name}"
    ps=f"if(Test-Path '{p}'){{ $i=Get-Item '{p}'; 'SIZE='+$i.Length; Get-Content '{p}' -Tail 25 }} else {{ 'MISS' }}"
    print("====", name)
    print(run(t, f'powershell -NoProfile -Command "{ps}"').strip()[:2000])
# td3 err tail
p=rf"{ROOT}\logs\seasonal_winter_td3_s0.log.err"
ps=f"if(Test-Path '{p}'){{ $i=Get-Item '{p}'; 'SIZE='+$i.Length; Get-Content '{p}' -Tail 5 }} else {{ 'MISS' }}"
print("==== td3 s0 err")
print(run(t, f'powershell -NoProfile -Command "{ps}"').strip()[-800:])
t.close()
