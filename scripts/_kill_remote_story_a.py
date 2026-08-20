"""Kill remote story_a train jobs that block the fair queue."""
from __future__ import annotations

import time

import paramiko

HOST = "172.16.1.80"
USER = "dell"
PASSWORD = "TR@SZ"


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


def main() -> None:
    t = paramiko.Transport((HOST, 22))
    t.banner_timeout = 60
    t.start_client(timeout=60)
    t.auth_password(USER, PASSWORD)
    out = run(
        t,
        "wmic process where \"CommandLine like '%story_a%'\" call terminate",
    )
    print(out.encode("ascii", "replace").decode("ascii")[:1500])
    time.sleep(3)
    out2 = run(
        t,
        "wmic process where \"CommandLine like '%train_seasonal%'\" get ProcessId,CommandLine /FORMAT:LIST",
    )
    print(out2.encode("ascii", "replace").decode("ascii")[:2000])
    t.close()


if __name__ == "__main__":
    main()
