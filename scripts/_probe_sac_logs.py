"""Read SAC launch logs on remote."""
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
        log = f"{ROOT}\\runs\\sac_s{s}.log"
        print(f"==== sac_s{s}.log ====")
        print(
            run(
                t,
                f'cmd /c "if exist {log} (echo EXISTS & type {log}) else (echo MISSING)"',
            )
        )
        # also check config yaml copied
        cfg = f"{ROOT}\\runs\\givesafe_sac_s{s}\\config"
        print(
            run(
                t,
                f'cmd /c "if exist {cfg} (dir /b {cfg}) else echo no_config"',
            )
        )
        print(
            run(
                t,
                f'cmd /c "if exist {ROOT}\\runs\\givesafe_sac_s{s}\\train (dir /a {ROOT}\\runs\\givesafe_sac_s{s}\\train) else echo no_train"',
            )
        )

    t.close()


if __name__ == "__main__":
    main()
