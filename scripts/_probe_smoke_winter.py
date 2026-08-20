"""Probe fair_queue + winter smoke run status on remote."""
from __future__ import annotations

import json
import time

import paramiko

HOST = "172.16.1.80"
USER = "dell"
PASSWORD = "TR@SZ"
REMOTE = r"D:\xuzh\demo_optimization"


def run(t, cmd, timeout=60):
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

    sp("=== fair_queue.log (tail) ===")
    sp(
        run(
            t,
            f'powershell -NoProfile -Command "if(Test-Path \'{REMOTE}\\logs\\fair_queue.log\')'
            f'{{ Get-Content \'{REMOTE}\\logs\\fair_queue.log\' -Tail 12 }} else {{ \'NOLOG\' }}"',
        ).strip()
    )
    sp("=== live seasonal_v1 ===")
    live = run(
        t,
        "wmic process where \"CommandLine like '%seasonal_v1%'\" get ProcessId,CommandLine /FORMAT:LIST",
    )
    sp(live.strip()[:1500] or "(none)")

    paths = {
        "hmsd_train": f"{REMOTE}/runs/seasonal_v1/winter/hmsd_s0/train_result.json".replace("\\", "/"),
        "td3_train": f"{REMOTE}/runs/seasonal_v1/winter/td3_s0/train_result.json".replace("\\", "/"),
        "hmsd_pt": f"{REMOTE}/runs/seasonal_v1/winter/hmsd_s0/checkpoints/ghtd3.pt".replace("\\", "/"),
        "td3_pt": f"{REMOTE}/runs/seasonal_v1/winter/td3_s0/checkpoints/hybrid_givesafe_td3.pt".replace("\\", "/"),
        "hmsd_log": f"{REMOTE}/logs/seasonal_v1_winter_hmsd_s0.log".replace("\\", "/"),
        "td3_log": f"{REMOTE}/logs/seasonal_v1_winter_td3_s0.log".replace("\\", "/"),
        "hmsd_err": f"{REMOTE}/logs/seasonal_v1_winter_hmsd_s0.log.err".replace("\\", "/"),
        "td3_err": f"{REMOTE}/logs/seasonal_v1_winter_td3_s0.log.err".replace("\\", "/"),
    }
    done = True
    for key, rp in paths.items():
        try:
            st = sftp.stat(rp)
            sp(f"{key}: size={st.st_size}")
            if "train" in key:
                with sftp.open(rp, "r") as f:
                    obj = json.loads(f.read().decode("utf-8", "replace"))
                sp(
                    f"  status={obj.get('status')} observation_dim={obj.get('observation_dim')} "
                    f"keys={list(obj.keys())[:15]}"
                )
                if obj.get("observation_dim") not in (None, 166):
                    sp(f"  WARN unexpected obs dim {obj.get('observation_dim')}")
            if key.endswith("_log") or key.endswith("_err"):
                with sftp.open(rp, "r") as f:
                    data = f.read().decode("utf-8", "replace")
                tail = data[-800:].replace("\r", "")
                sp(f"--- {key} tail ---\n{tail}")
        except OSError:
            sp(f"{key}: missing")
            if key in ("hmsd_train", "hmsd_pt"):
                done = False

    sftp.close()
    t.close()
    if done:
        sp("SMOKE_COMPLETE")
    else:
        sp("SMOKE_PENDING")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
