from __future__ import annotations

import time

import paramiko

HOST = "172.16.1.80"
USER = "dell"
PASSWORD = "TR@SZ"
ROOT = r"D:\xuzh\demo_optimization"


def run(t: paramiko.Transport, cmd: str, timeout: int = 40) -> str:
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
        d = f"{ROOT}\\runs\\ghtd3_abs_s{s}"
        print(f"==== ghtd3_abs_s{s} ====")
        cmd = (
            f'cmd /c "if exist {d}\\summary.json (echo HAS_SUMMARY) else (echo NO_SUMMARY) '
            f"& if exist {d}\\checkpoints\\ghtd3.pt (echo HAS_CKPT) else (echo NO_CKPT) "
            f"& if exist {d}\\train\\instance.lock (echo HAS_LOCK) else (echo NO_LOCK) "
            f'& dir /b {d}\\checkpoints"'
        )
        print(run(t, cmd))

    print("==== python cmdline train/ghtd3/sac ====")
    out = run(
        t,
        r'cmd /c wmic process where "name=\"python.exe\"" get ProcessId,CommandLine /FORMAT:CSV',
        timeout=50,
    )
    hits = [
        ln
        for ln in out.splitlines()
        if any(k in ln.lower() for k in ("train_", "ghtd3", "sac", "hybrid"))
    ]
    print("\n".join(hits) if hits else "(none)")
    t.close()


if __name__ == "__main__":
    main()
