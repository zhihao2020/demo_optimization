#!/usr/bin/env python
"""Pull completed seasonal_v1 HMSD/TD3 train_result KPIs from remote."""
from __future__ import annotations

import json
import time
from pathlib import Path

import paramiko

HOST = "172.16.1.80"
USER = "dell"
PASSWORD = "TR@SZ"
ROOT = r"D:\xuzh\demo_optimization"

JOBS = [
    ("winter", "hmsd", 0),
    ("winter", "hmsd", 1),
    ("winter", "hmsd", 2),
    ("winter", "td3", 0),
    ("winter", "td3", 1),
    ("winter", "td3", 2),
]


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

    rows = []
    for season, method, seed in JOBS:
        path = rf"{ROOT}\runs\seasonal_v1\{season}\{method}_s{seed}\train_result.json"
        proto = rf"{ROOT}\runs\seasonal_v1\{season}\{method}_s{seed}\protocol.json"
        exists = run(t, f"powershell -NoProfile -Command \"Test-Path '{path}'\"").strip()
        if "True" not in exists:
            print(f"[skip] {season} {method} s{seed}: no train_result")
            continue
        raw = run(
            t,
            f"powershell -NoProfile -Command \"Get-Content '{path}' -Raw\"",
            timeout=60,
        )
        try:
            i = raw.find("{")
            data = json.loads(raw[i:])
        except Exception as exc:
            print(f"[parse fail] {method} s{seed}: {exc}")
            continue
        prot = {}
        pr = run(t, f"powershell -NoProfile -Command \"if(Test-Path '{proto}'){{Get-Content '{proto}' -Raw}}\"")
        if "{" in pr:
            try:
                prot = json.loads(pr[pr.find("{") :])
            except Exception:
                pass
        ev = data.get("eval") or {}
        terms = ev.get("cost_terms") or {}
        metrics = ev.get("metrics") or {}
        rule = data.get("rule") or {}
        row = {
            "season": season,
            "method": method,
            "seed": seed,
            "status": data.get("status"),
            "eval_start": data.get("eval_start_time_seconds") or prot.get("eval_start_seconds"),
            "train_weeks": prot.get("train_weeks"),
            "eval_week": prot.get("eval_week"),
            "episode_reward": ev.get("episode_reward"),
            "sum_delta_j_gen": terms.get("generalized_cashflow_delta"),
            "sum_delta_cf": terms.get("economic_cashflow_delta") or terms.get("cashflow_delta"),
            "weekly_raw_total_cost": ev.get("weekly_raw_total_cost"),
            "terminal_soc": ev.get("terminal_soc_satisfied"),
            "unserved": metrics.get("unserved_energy_mwh"),
            "curtail": metrics.get("curtailment_energy_mwh"),
            "bat_thr": metrics.get("battery_throughput_mwh"),
            "caes_thr": metrics.get("caes_throughput_mwh"),
            "thermal": metrics.get("thermal_generation_mwh"),
            "external_cost": terms.get("external_cost_cny"),
            "market_buy": terms.get("market_buy_cost"),
            "market_sell": terms.get("market_sell_revenue"),
            "rule_reward": rule.get("episode_reward"),
            "rule_j": (rule.get("cost_terms") or {}).get("generalized_cashflow_delta")
            or (rule.get("cost_terms") or {}).get("economic_cashflow_delta"),
        }
        rows.append(row)
        print(
            f"{method:4} s{seed}: reward={row['episode_reward']} "
            f"J_gen={row['sum_delta_j_gen']} CF={row['sum_delta_cf']} "
            f"soc={row['terminal_soc']} uns={row['unserved']} "
            f"eval_t={row['eval_start']} week={row['eval_week']}",
            flush=True,
        )

    out = Path("logs/remote_v1_done_compare.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("wrote", out)

    # summary
    hmsd = [r for r in rows if r["method"] == "hmsd" and r.get("episode_reward") is not None]
    td3 = [r for r in rows if r["method"] == "td3" and r.get("episode_reward") is not None]
    if hmsd:
        rw = [float(r["episode_reward"]) for r in hmsd]
        jg = [float(r["sum_delta_j_gen"] or 0) for r in hmsd]
        print(f"HMSD n={len(hmsd)} reward mean={sum(rw)/len(rw):.2f}  J_gen mean={sum(jg)/len(jg):.3e}")
    if td3:
        rw = [float(r["episode_reward"]) for r in td3]
        jg = [float(r["sum_delta_j_gen"] or 0) for r in td3]
        print(f"TD3  n={len(td3)} reward mean={sum(rw)/len(rw):.2f}  J_gen mean={sum(jg)/len(jg):.3e}")
    t.close()


if __name__ == "__main__":
    main()
