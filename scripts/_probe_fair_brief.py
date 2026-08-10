#!/usr/bin/env python
from __future__ import annotations
import time
import paramiko

HOST = "172.16.1.80"
USER = "dell"
PASSWORD = "TR@SZ"
ROOT = r"D:\xuzh\demo_optimization"


def run(t, cmd, timeout=60):
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

    jobs = [
        ("winter", "hmsd", 0),
        ("winter", "hmsd", 1),
        ("winter", "hmsd", 2),
        ("winter", "td3", 0),
        ("winter", "td3", 1),
        ("winter", "td3", 2),
        ("winter", "sac", 0),
    ]
    for season, method, seed in jobs:
        name = f"{season}_{method}_s{seed}"
        log = rf"{ROOT}\logs\seasonal_v1_{name}.log"
        err = log + ".err"
        run_dir = rf"{ROOT}\runs\seasonal_v1\{season}\{method}_s{seed}"
        res = run_dir + r"\train_result.json"
        ck = run_dir + r"\checkpoints"
        print(f"=== {name} ===")
        tail = run(t, f"powershell -NoProfile -Command \"Get-Content '{log}' -Tail 5\"")
        print("log:", " | ".join(x.strip() for x in tail.strip().splitlines()[-3:]))
        has = run(t, f"powershell -NoProfile -Command \"Test-Path '{res}'\"").strip()
        print("train_result:", has)
        print(
            "ckpt:",
            run(
                t,
                f"powershell -NoProfile -Command \"if(Test-Path '{ck}'){{ (Get-ChildItem '{ck}').Name -join ',' }} else {{ 'none' }}\"",
            ).strip(),
        )
        et = run(t, f"powershell -NoProfile -Command \"Get-Content '{err}' -Tail 2 -ErrorAction SilentlyContinue\"").strip()
        if et:
            print("err:", et[-200:])
    print("=== live ===")
    wmic = run(
        t,
        r'cmd /c wmic process where "name=\"python.exe\"" get CommandLine /FORMAT:LIST',
        timeout=90,
    )
    for ln in wmic.splitlines():
        if "train_seasonal" in ln.lower() and "--method" in ln.lower():
            print(ln.strip()[:220])
    t.close()


if __name__ == "__main__":
    main()
