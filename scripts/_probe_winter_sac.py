"""Fetch winter SAC remote logs (stdout/err + run dir)."""
from __future__ import annotations

import time

import paramiko

HOST, USER, PASSWORD = "172.16.1.80", "dell", "TR@SZ"
REMOTE = r"D:\xuzh\demo_optimization"


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


def sp(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def main() -> None:
    t = paramiko.Transport((HOST, 22))
    t.banner_timeout = 60
    t.start_client(timeout=60)
    t.auth_password(USER, PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(t)
    assert sftp is not None

    paths = {
        "log": f"{REMOTE}/logs/seasonal_v1_winter_sac_s0.log".replace("\\", "/"),
        "err": f"{REMOTE}/logs/seasonal_v1_winter_sac_s0.log.err".replace("\\", "/"),
        "protocol": f"{REMOTE}/runs/seasonal_v1/winter/sac_s0/protocol.json".replace("\\", "/"),
        "progress": f"{REMOTE}/runs/seasonal_v1/winter/sac_s0/train/progress.json".replace("\\", "/"),
        "result": f"{REMOTE}/runs/seasonal_v1/winter/sac_s0/train_result.json".replace("\\", "/"),
    }
    for key, rp in paths.items():
        try:
            st = sftp.stat(rp)
            sp(f"{key}: size={st.st_size}")
        except OSError:
            sp(f"{key}: missing")

    sp("=== process list (sac/winter) ===")
    live = run(
        t,
        "wmic process where \"CommandLine like '%winter%' AND CommandLine like '%sac%'\" get ProcessId,CommandLine /FORMAT:LIST",
    )
    sp(live.strip()[:1500] or "(none)")

    for key in ("log", "err"):
        rp = paths[key]
        try:
            with sftp.open(rp, "r") as f:
                data = f.read().decode("utf-8", "replace")
        except OSError:
            continue
        sp(f"--- {key} head ---")
        sp(data[:1500])
        sp(f"--- {key} tail ---")
        sp(data[-2500:])

    sftp.close()
    t.close()


if __name__ == "__main__":
    main()
