"""Quick remote seasonal_v1 progress + KPIs."""
from __future__ import annotations

import json

import paramiko

HOST, USER, PASSWORD = "172.16.1.80", "dell", "TR@SZ"
REMOTE = r"D:\xuzh\demo_optimization"


def main() -> None:
    t = paramiko.Transport((HOST, 22))
    t.banner_timeout = 60
    t.start_client(timeout=60)
    t.auth_password(USER, PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(t)
    assert sftp is not None

    def get(rel: str) -> str | None:
        path = f"{REMOTE}/{rel}".replace("\\", "/")
        try:
            with sftp.open(path, "r") as f:
                return f.read().decode("utf-8", "replace")
        except OSError:
            return None

    print("=== remaining / SAC ===")
    for season, method in (
        ("winter", "sac"),
        ("transition", "sac"),
        ("summer", "td3"),
        ("summer", "sac"),
    ):
        tr = get(f"runs/seasonal_v1/{season}/{method}_s0/train_result.json")
        pr = get(f"runs/seasonal_v1/{season}/{method}_s0/train/progress.json")
        if tr:
            j = json.loads(tr)
            ev = j.get("eval") or {}
            print(
                f"{season}/{method}: DONE status={j.get('status')} "
                f"obs={j.get('observation_dim')} R={ev.get('episode_reward')}"
            )
        elif pr:
            j = json.loads(pr)
            frac = 100 * float(j.get("frac") or 0)
            print(
                f"{season}/{method}: RUNNING "
                f"step={j.get('valid_steps')}/{j.get('total_steps')} ({frac:.1f}%) "
                f"ep={j.get('episode')} r_ext={j.get('r_ext')}"
            )
        else:
            print(f"{season}/{method}: no artifacts yet")

    print("=== completed HMSD / TD3 ===")
    for season in ("winter", "transition", "summer"):
        for method in ("hmsd", "td3"):
            tr = get(f"runs/seasonal_v1/{season}/{method}_s0/train_result.json")
            if not tr:
                print(f"{season}/{method}: missing")
                continue
            j = json.loads(tr)
            ev = j.get("eval") or {}
            terms = ev.get("cost_terms") or {}
            print(
                f"{season}/{method}: R={ev.get('episode_reward')} "
                f"Jgen={terms.get('generalized_cashflow_delta')} "
                f"SOC={ev.get('terminal_soc_satisfied')} "
                f"obs={j.get('observation_dim')}"
            )

    print("=== PSO / linprog ===")
    for season in ("winter", "transition", "summer"):
        for method in ("pso", "linprog"):
            tr = get(f"runs/seasonal_v1/{season}/{method}_s0/train_result.json")
            if not tr:
                print(f"{season}/{method}: missing")
                continue
            j = json.loads(tr)
            ev = j.get("eval") or {}
            kpi = j.get("kpi") or {}
            r = ev.get("episode_reward", kpi.get("episode_reward") or j.get("episode_reward"))
            print(f"{season}/{method}: status={j.get('status')} R={r}")

    sftp.close()
    t.close()


if __name__ == "__main__":
    main()
