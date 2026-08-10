"""Tail SAC logs via PowerShell -LiteralPath."""
from __future__ import annotations

import time

import paramiko

HOST = "172.16.1.80"
USER = "dell"
PASSWORD = "TR@SZ"
ROOT = r"D:\xuzh\demo_optimization"


def run(t: paramiko.Transport, cmd: str, timeout: int = 50) -> str:
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

    for s in (0, 1, 2):
        log = f"{ROOT}\\runs\\sac_s{s}.log"
        ps = (
            f"$p='{log}'; "
            f"if(Test-Path -LiteralPath $p){{ "
            f"$i=Get-Item -LiteralPath $p; "
            f"Write-Output ('SIZE='+$i.Length); "
            f"Write-Output '===HEAD==='; "
            f"Get-Content -LiteralPath $p -TotalCount 8; "
            f"Write-Output '===TAIL==='; "
            f"Get-Content -LiteralPath $p -Tail 40 "
            f"}} else {{ Write-Output 'MISSING' }}"
        )
        # encode as single arg for powershell
        print(f"==== sac_s{s}.log ====")
        out = run(
            t,
            f'powershell -NoProfile -ExecutionPolicy Bypass -Command "{ps}"',
            timeout=70,
        )
        print(out[:4000])

    print("==== train_hybrid_sac processes ====")
    out = run(
        t,
        r'cmd /c wmic process where "name=\"python.exe\"" get ProcessId,CommandLine /FORMAT:CSV',
        timeout=50,
    )
    hits = [
        ln
        for ln in out.splitlines()
        if "train_hybrid_sac" in ln.lower() or "givesafe_sac" in ln.lower()
    ]
    print("\n".join(hits) if hits else "(none running)")
    t.close()


if __name__ == "__main__":
    main()
