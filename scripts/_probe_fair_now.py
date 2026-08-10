#!/usr/bin/env python
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

    print("=== fair_queue.log tail ===")
    print(
        run(
            t,
            f'powershell -NoProfile -Command "Get-Content \'{ROOT}\\logs\\fair_queue.log\' -Tail 20"',
        ).strip()
    )
    print("=== fair_queue_state ===")
    print(
        run(
            t,
            f'powershell -NoProfile -Command "Get-Content \'{ROOT}\\logs\\fair_queue_state.json\' -Raw"',
        ).strip()[:2000]
    )
    print("=== MAX_LIVE / pyvenv ===")
    print(
        run(
            t,
            f'powershell -NoProfile -Command "Select-String -Path \'{ROOT}\\logs\\fair_queue.py\' -Pattern MAX_LIVE"',
        ).strip()
    )
    print(run(t, f"cmd /c type {ROOT}\\.venv\\pyvenv.cfg").strip())
    print("=== gpu ===")
    print(
        run(
            t,
            "cmd /c nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader",
        ).strip()
    )

    live = run(
        t,
        r'cmd /c wmic process where "name=\"python.exe\"" get CommandLine /FORMAT:LIST',
        timeout=90,
    )
    seen = set()
    print("=== live train_seasonal (unique run-dir) ===")
    for ln in live.splitlines():
        low = ln.lower()
        if "train_seasonal" not in low or "--run-dir" not in low:
            continue
        part = ln.split("--run-dir", 1)[1].strip().strip('"')
        part = part.split(" --")[0].strip().strip('"')
        key = part.lower()
        if key in seen:
            continue
        seen.add(key)
        m = re.search(r"--method\s+(\w+)", ln, re.I)
        s = re.search(r"--season\s+(\w+)", ln, re.I)
        sd = re.search(r"--seed\s+(\d+)", ln, re.I)
        print(
            f"  LIVE {s.group(1) if s else '?'} {m.group(1) if m else '?'} s{sd.group(1) if sd else '?'}"
        )
    print("unique_run_dirs", len(seen))
    print("comfy_in_live", sum(1 for ln in live.splitlines() if "comfyui" in ln.lower()))

    listing = run(t, f"cmd /c dir /b {ROOT}\\logs\\seasonal_v1_*.log")
    names = sorted(
        n.strip()
        for n in listing.replace("\r", "").splitlines()
        if n.strip().startswith("seasonal_v1_")
    )
    print(f"=== job logs n={len(names)} ===")
    done = fail = runn = 0
    for n in names:
        p = f"{ROOT}\\logs\\{n}"
        tail = run(
            t,
            f'powershell -NoProfile -Command "if(Test-Path \'{p}\'){{Get-Content \'{p}\' -Tail 8}}"',
            timeout=40,
        ).strip()
        status = "run"
        if "EXIT 0" in tail:
            status = "done"
            done += 1
        elif re.search(r"EXIT\s+[1-9]", tail):
            status = "fail"
            fail += 1
        else:
            runn += 1
        pct = ""
        m = re.search(r"step=(\d+)/840000", tail)
        if m:
            pct = f"{100 * int(m.group(1)) / 840000:.1f}%"
        m2 = re.search(r"Hybrid-TD3:\s+(\d+)%", tail)
        if m2:
            pct = m2.group(1) + "%"
        m3 = re.search(r"Hybrid-SAC:\s+(\d+)%", tail)
        if m3:
            pct = m3.group(1) + "%"
        m4 = re.search(r"iter=(\d+)/(\d+)", tail)
        if m4 and "pso" in n:
            pct = f"pso {m4.group(1)}/{m4.group(2)}"
        short = n.replace("seasonal_v1_", "").replace(".log", "")
        last = " | ".join(x.strip() for x in tail.splitlines()[-2:] if x.strip())[:110]
        print(f"{status:5} {pct:10} {short:28} {last}")
    print(f"summary: done={done} fail={fail} run/other={runn} total_logs={len(names)}")
    print(
        "train_result count:",
        run(
            t,
            f'powershell -NoProfile -Command "(Get-ChildItem \'{ROOT}\\runs\\seasonal_v1\' -Recurse -Filter train_result.json -EA SilentlyContinue).Count"',
        ).strip(),
    )
    t.close()


if __name__ == "__main__":
    main()
