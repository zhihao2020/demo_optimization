"""Probe formal fair suite queue (seasonal_v1, 5000 ep)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import paramiko

HOST = "172.16.1.80"
USER = "dell"
PASSWORD = "TR@SZ"
REMOTE = r"D:\xuzh\demo_optimization"
LOCAL = Path(__file__).resolve().parents[1]


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
    man = LOCAL / "logs" / "remote_fair_suite_manifest.json"
    if man.is_file():
        sp("local manifest: " + man.read_text(encoding="utf-8")[:800])

    t = paramiko.Transport((HOST, 22))
    t.banner_timeout = 60
    t.start_client(timeout=60)
    t.auth_password(USER, PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(t)
    assert sftp is not None

    sp("=== fair_queue.log tail ===")
    sp(
        run(
            t,
            f'powershell -NoProfile -Command "Get-Content \'{REMOTE}\\logs\\fair_queue.log\' -Tail 20"',
        ).strip()
    )
    sp("=== live seasonal_v1 ===")
    live = run(
        t,
        "wmic process where \"CommandLine like '%seasonal_v1%'\" get ProcessId,CommandLine /FORMAT:LIST",
    )
    sp(live.strip()[:2500] or "(none)")

    # count started run dirs / completed train_result
    try:
        state = sftp.open(f"{REMOTE}/logs/fair_queue_state.json".replace("\\", "/"), "r").read()
        sp("queue_state: " + state.decode("utf-8", "replace"))
    except OSError as exc:
        sp(f"queue_state missing: {exc}")

    completed = []
    for season in ("winter", "transition", "summer"):
        for method in ("hmsd", "td3", "sac", "pso", "linprog"):
            rp = f"{REMOTE}/runs/seasonal_v1/{season}/{method}_s0/train_result.json".replace("\\", "/")
            try:
                obj = json.loads(sftp.open(rp, "r").read().decode("utf-8", "replace"))
                completed.append(
                    f"{season}/{method}: status={obj.get('status')} obs={obj.get('observation_dim')}"
                )
            except OSError:
                pass
    sp("completed: " + ("; ".join(completed) if completed else "(none yet)"))
    sftp.close()
    t.close()


if __name__ == "__main__":
    main()
