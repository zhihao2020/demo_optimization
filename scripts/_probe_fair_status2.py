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

    print("=== queue ===")
    print(run(t, f"powershell -NoProfile -Command \"Get-Content '{ROOT}\\logs\\fair_queue.log' -Tail 5\"").strip())
    print(run(t, f"powershell -NoProfile -Command \"Get-Content '{ROOT}\\logs\\fair_queue_state.json' -Raw\"").strip())

    listing = run(t, f"cmd /c dir /b {ROOT}\\logs\\seasonal_v1_*.log")
    names = [n.strip() for n in listing.replace("\r", "").splitlines() if n.strip().startswith("seasonal_v1_")]
    print(f"n_logs={len(names)}")
    rows = []
    for n in sorted(names):
        p = f"{ROOT}\\logs\\{n}"
        tail = run(
            t,
            f"powershell -NoProfile -Command \"Get-Content '{p}' -Tail 8\"",
            timeout=40,
        ).strip()
        last = " | ".join(x.strip() for x in tail.splitlines()[-4:] if x.strip())
        status = "run"
        if "EXIT 0" in tail:
            status = "done"
        elif re.search(r"EXIT\s+[1-9]", tail):
            status = "fail"
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
        if "PSO" in tail and "gbest" in tail:
            m4 = re.search(r"iter=(\d+)/(\d+)", tail)
            if m4:
                pct = f"pso {m4.group(1)}/{m4.group(2)}"
        if "status completed" in tail or "sum_delta" in tail:
            if status == "run":
                status = "done?"
        short = n.replace("seasonal_v1_", "").replace(".log", "")
        rows.append((status, pct, short, last[:120]))
        print(f"{status:5} {pct:8} {short:28} {last[:100]}")

    done = sum(1 for r in rows if r[0] in ("done", "done?"))
    fail = sum(1 for r in rows if r[0] == "fail")
    runn = sum(1 for r in rows if r[0] == "run")
    print(f"\nsummary: done={done} fail={fail} run/log={runn} total_logs={len(rows)}")
    print(run(t, "cmd /c nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader").strip())
    t.close()


if __name__ == "__main__":
    main()
