"""Probe remote for SAC / hybrid-sac jobs and artifacts."""
from __future__ import annotations

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
    print("=== connected ===")

    print("\n--- all python CommandLines containing sac/hybrid/td3/ghtd3/train ---")
    wmic = run(
        t,
        r'cmd /c wmic process where "name=\"python.exe\"" get ProcessId,CommandLine /FORMAT:CSV',
        timeout=60,
    )
    for ln in wmic.splitlines():
        low = ln.lower()
        if any(k in low for k in ("sac", "hybrid", "td3", "ghtd3", "train_", "ppo")):
            print(ln[:600])

    print("\n--- dirs under runs matching *sac* *hybrid* *td3* *ghtd3* ---")
    print(
        run(
            t,
            rf'cmd /c "cd /d {ROOT}\runs && dir /b /ad 2>nul"',
            timeout=40,
        )
    )

    print("\n--- recursive names *sac* under runs (dirs) ---")
    print(
        run(
            t,
            rf'cmd /c "cd /d {ROOT}\runs && dir /s /b /ad *sac* 2>nul"',
            timeout=60,
        )
    )

    print("\n--- list contents of givesafe_sac_* ---")
    for s in (0, 1, 2):
        d = rf"{ROOT}\runs\givesafe_sac_s{s}"
        print(f"\n## {d}")
        print(
            run(
                t,
                rf'cmd /c "if exist {d} (dir /s /b {d} 2>nul) else (echo MISSING)"',
                timeout=45,
            )[:3000]
        )

    print("\n--- search summary.json / progress.json / checkpoints with sac in path ---")
    print(
        run(
            t,
            rf'cmd /c "cd /d {ROOT}\runs && dir /s /b *sac*\summary.json *sac*\progress.json *sac*\*.pt 2>nul"',
            timeout=60,
        )
    )

    print("\n--- scheduled tasks / bat files that might launch sac ---")
    print(
        run(
            t,
            rf'cmd /c "cd /d {ROOT} && dir /b /s *sac*.bat *sac*.ps1 *train*sac* 2>nul"',
            timeout=45,
        )[:2500]
    )

    print("\n--- recent files under runs modified (dir /o-d top) ---")
    print(
        run(
            t,
            rf'cmd /c "cd /d {ROOT}\runs && dir /ad /o-d /tw"',
            timeout=40,
        )
    )

    # any hybrid_sac elsewhere
    print("\n--- hybrid_sac / train_hybrid under ROOT ---")
    print(
        run(
            t,
            rf'cmd /c "cd /d {ROOT} && dir /s /b runs\*hybrid* runs\*sac* 2>nul | more +0"',
            timeout=90,
        )[:4000]
    )

    t.close()
    print("\n=== done ===")


if __name__ == "__main__":
    main()
