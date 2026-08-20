"""Live queue + remaining job artifacts."""
from __future__ import annotations

import json
import time

import paramiko

HOST, USER, PASSWORD = "172.16.1.80", "dell", "TR@SZ"
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

    sp("=== live seasonal_v1 ===")
    live = run(
        t,
        "wmic process where \"CommandLine like '%seasonal_v1%'\" get ProcessId,CommandLine /FORMAT:LIST",
    )
    sp(live.strip()[:2500] or "(none)")

    sp("=== fair_queue.log tail ===")
    sp(
        run(
            t,
            f'powershell -NoProfile -Command "Get-Content \'{REMOTE}\\logs\\fair_queue.log\' -Tail 8"',
        ).strip()
    )

    jobs = []
    for season in ("winter", "transition", "summer"):
        for method in ("hmsd", "td3", "sac", "pso", "linprog"):
            jobs.append((season, method))

    sp("=== artifacts ===")
    for season, method in jobs:
        base = f"{REMOTE}/runs/seasonal_v1/{season}/{method}_s0".replace("\\", "/")
        tr = f"{base}/train_result.json"
        pr = f"{base}/train/progress.json"
        ck = f"{base}/checkpoints"
        status = "missing"
        extra = ""
        try:
            with sftp.open(tr, "r") as f:
                j = json.loads(f.read().decode("utf-8", "replace"))
            ev = j.get("eval") or {}
            status = j.get("status")
            extra = f"R={ev.get('episode_reward')} SOC={ev.get('terminal_soc_satisfied')} obs={j.get('observation_dim')}"
        except OSError:
            try:
                with sftp.open(pr, "r") as f:
                    p = json.loads(f.read().decode("utf-8", "replace"))
                frac = 100 * float(p.get("frac") or 0)
                status = "running"
                extra = f"step={p.get('valid_steps')}/{p.get('total_steps')} ({frac:.1f}%)"
            except OSError:
                extra = "no train_result/progress"
        sp(f"{season}/{method}: {status} {extra}")

    sftp.close()
    t.close()


if __name__ == "__main__":
    main()
