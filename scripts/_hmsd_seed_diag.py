#!/usr/bin/env python
from __future__ import annotations
import json
import time
import paramiko

HOST = "172.16.1.80"
USER = "dell"
PASSWORD = "TR@SZ"
ROOT = r"D:\xuzh\demo_optimization"


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


def main():
    t = paramiko.Transport((HOST, 22))
    t.banner_timeout = 60
    t.auth_timeout = 60
    t.start_client(timeout=60)
    t.auth_password(USER, PASSWORD)
    for seed in (0, 1, 2):
        p = rf"{ROOT}\runs\seasonal_v1\winter\hmsd_s{seed}\train_result.json"
        raw = run(t, f"powershell -NoProfile -Command \"Get-Content '{p}' -Raw\"")
        d = json.loads(raw[raw.find("{") :])
        st = d.get("stats") or {}
        ev = d.get("eval") or {}
        terms = ev.get("cost_terms") or {}
        m = ev.get("metrics") or {}
        eps = max(int(d.get("episodes") or 1), 1)
        vs = int(d.get("valid_steps") or 0)
        print(f"=== hmsd s{seed} ===")
        print(
            f"  episodes={eps} mean_ep_len={vs/eps:.1f} "
            f"reject={st.get('givesafe_reject')} hi_goals={st.get('high_goal_count')}"
        )
        print(
            f"  reward={ev.get('episode_reward'):.2f} soc={ev.get('terminal_soc_satisfied')} "
            f"J_gen={float(terms.get('generalized_cashflow_delta') or 0):.3e}"
        )
        print(
            f"  thermal={m.get('thermal_generation_mwh'):.0f} bat={m.get('battery_throughput_mwh'):.0f} "
            f"caes={m.get('caes_throughput_mwh'):.0f} buy={float(terms.get('market_buy_cost') or 0):.0f} "
            f"sell={float(terms.get('market_sell_revenue') or 0):.0f}"
        )
    t.close()


if __name__ == "__main__":
    main()
