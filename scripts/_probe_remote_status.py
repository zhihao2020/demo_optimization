"""One-shot probe of remote training status (172.16.1.80)."""
from __future__ import annotations

import json
import time

import paramiko

HOST = "172.16.1.80"
USER = "dell"
PASSWORD = "TR@SZ"
ROOT = r"D:\xuzh\demo_optimization"


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
    t.banner_timeout = 30
    t.auth_timeout = 30
    t.start_client(timeout=30)
    t.auth_password(USER, PASSWORD)
    print("=== connected", HOST, "===")

    print("\n--- python.exe tasklist ---")
    print(run(t, r'cmd /c tasklist /FI "IMAGENAME eq python.exe"', timeout=40))

    print("\n--- relevant python CommandLines ---")
    wmic = run(
        t,
        r'cmd /c wmic process where "name=\"python.exe\"" get ProcessId,CommandLine /FORMAT:CSV',
        timeout=60,
    )
    for ln in wmic.splitlines():
        low = ln.lower()
        if any(
            k in low
            for k in (
                "train",
                "ghtd3",
                "td3",
                "sac",
                "hybrid",
                "demo_optimization",
                "optimal",
                "fmu",
                "ablation",
                "benchmark",
            )
        ):
            print(ln[:500])

    print("\n--- top run dirs (newest first, name only) ---")
    dirs_out = run(
        t,
        rf'cmd /c "cd /d {ROOT}\runs && dir /b /ad /o-d"',
        timeout=45,
    )
    dirs = [ln.strip() for ln in dirs_out.replace("\r", "").splitlines() if ln.strip()]
    # drop cmd noise
    dirs = [d for d in dirs if not d.lower().startswith("cmd") and "\\" not in d]
    print("\n".join(dirs[:40]))

    print("\n--- progress / summary for newest 25 runs ---")
    for d in dirs[:25]:
        base = rf"{ROOT}\runs\{d}"
        o = run(
            t,
            rf'cmd /c "if exist {base}\train\progress.json (type {base}\train\progress.json) '
            rf'else if exist {base}\summary.json (echo HAS_SUMMARY) '
            rf'else if exist {base}\vs_hybrid.json (echo HAS_VS) '
            rf'else (echo IDLE_OR_EMPTY)"',
            timeout=25,
        )
        o = o.replace("\r", "").strip()
        short = " ".join(o.split())
        if len(short) > 220:
            short = short[:220] + "..."
        print(f"{d}: {short}")

    print("\n--- instance.lock files ---")
    locks = run(
        t,
        rf'cmd /c "cd /d {ROOT}\runs && dir /s /b instance.lock 2>nul"',
        timeout=60,
    )
    print(locks.strip() or "(none)")

    print("\n--- recent train_ghtd3 / hybrid log files if any ---")
    logs = run(
        t,
        rf'cmd /c "cd /d {ROOT} && dir /b /o-d *.log 2>nul & dir /b /o-d logs\*.log 2>nul"',
        timeout=30,
    )
    print(logs.strip() or "(no top-level logs)")

    t.close()
    print("\n=== done ===")


if __name__ == "__main__":
    main()
