"""Short SAC-focused remote probe."""
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

    print("=== run dirs ===")
    print(run(t, f'cmd /c dir /b /ad "{ROOT}\\runs"'))

    for s in (0, 1, 2):
        d = f"{ROOT}\\runs\\givesafe_sac_s{s}"
        print(f"\n=== givesafe_sac_s{s} ===")
        cmd = (
            f'cmd /c "dir /a \"{d}\" & '
            f'if exist \"{d}\\summary.json\" (echo HAS_SUMMARY) else (echo NO_SUMMARY) & '
            f'if exist \"{d}\\train\\progress.json\" (type \"{d}\\train\\progress.json\") else (echo NO_PROGRESS) & '
            f'if exist \"{d}\\checkpoints\" (dir /b \"{d}\\checkpoints\") else (echo NO_CKPT) & '
            f'if exist \"{d}\\train\" (dir /b \"{d}\\train\") else (echo NO_TRAIN_DIR)"'
        )
        print(run(t, cmd))

    # alternate naming
    print("\n=== dirs with sac in name (recursive top) ===")
    print(run(t, f'cmd /c "cd /d \"{ROOT}\\runs\" && dir /b /s /ad *sac* 2>nul"'))

    print("\n=== process lines with sac/hybrid_sac/train_hybrid ===")
    out = run(
        t,
        r'cmd /c wmic process where "name=\"python.exe\"" get ProcessId,CommandLine /FORMAT:CSV',
        timeout=60,
    )
    for ln in out.splitlines():
        low = ln.lower()
        if "sac" in low or "hybrid" in low or "train_hybrid" in low:
            print(ln[:500])
    if not any("sac" in ln.lower() for ln in out.splitlines()):
        print("(no running process CommandLine contains 'sac')")

    # look for bat/ps1 launchers
    print("\n=== launch scripts mentioning sac ===")
    print(
        run(
            t,
            f'cmd /c "cd /d \"{ROOT}\" && findstr /s /i /m sac scripts\\*.bat scripts\\*.ps1 scripts\\*.cmd 2>nul & dir /b /s *sac*.bat 2>nul"',
            timeout=60,
        )[:2000]
    )

    t.close()


if __name__ == "__main__":
    main()
