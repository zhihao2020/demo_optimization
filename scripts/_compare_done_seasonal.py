#!/usr/bin/env python
"""Compare completed remote seasonal HMSD vs TD3 runs."""
from __future__ import annotations

import json
import time
from pathlib import Path

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


def main() -> None:
    t = paramiko.Transport((HOST, 22))
    t.banner_timeout = 60
    t.auth_timeout = 60
    t.start_client(timeout=60)
    t.auth_password(USER, PASSWORD)

    # list seasonal runs
    print("=== seasonal tree ===")
    print(
        run(
            t,
            rf'cmd /c "dir /s /b {ROOT}\runs\seasonal\*\*\train_result.json 2>nul & dir /s /b {ROOT}\runs\seasonal\*\*\checkpoints\*.pt 2>nul"',
            timeout=60,
        )
    )

    candidates = []
    for season in ("winter", "transition", "summer"):
        for method in ("hmsd", "td3"):
            for seed in (0, 1, 2):
                run_dir = rf"{ROOT}\runs\seasonal\{season}\{method}_s{seed}"
                candidates.append((season, method, seed, run_dir))

    rows = []
    for season, method, seed, run_dir in candidates:
        # prefer train_result.json; also check log EXIT and summary files
        paths = [
            rf"{run_dir}\train_result.json",
            rf"{run_dir}\summary.json",
            rf"{run_dir}\train\summary.json",
            rf"{run_dir}\eval\summary.json",
        ]
        content = None
        used = None
        for p in paths:
            ps = (
                f"if (Test-Path '{p}') {{ Get-Content '{p}' -Raw }} else {{ '' }}"
            )
            out = run(t, f'powershell -NoProfile -Command "{ps}"', timeout=40).strip()
            if out and not out.startswith("if ") and len(out) > 20:
                content = out
                used = p
                break
        # log status
        log = rf"{ROOT}\logs\seasonal_{season}_{method}_s{seed}.log"
        log_tail = run(
            t,
            f'powershell -NoProfile -Command "if(Test-Path \'{log}\'){{ Get-Content \'{log}\' -Tail 15 }} else {{ \'NOLOG\' }}"',
            timeout=30,
        ).strip()
        done = "status completed" in log_tail or "EXIT 0" in log_tail
        ckpt = run(
            t,
            f'powershell -NoProfile -Command "if(Test-Path \'{run_dir}\\checkpoints\'){{ Get-ChildItem \'{run_dir}\\checkpoints\' | Select-Object -ExpandProperty Name }} else {{ \'NOCKPT\' }}"',
            timeout=30,
        ).strip()

        rec = {
            "season": season,
            "method": method,
            "seed": seed,
            "done_log": done,
            "result_path": used,
            "ckpt": ckpt[:200],
        }
        if content:
            try:
                # strip BOM / noise
                i = content.find("{")
                if i >= 0:
                    content = content[i:]
                data = json.loads(content)
                rec["status"] = data.get("status")
                rec["raw_keys"] = list(data.keys())[:30]
                # common metric locations
                for key in ("eval", "rule", "price_rule", "stats", "last_metrics"):
                    if key in data:
                        rec[key] = data[key]
                # flatten useful scalars
                eval_ = data.get("eval") or {}
                if isinstance(eval_, dict):
                    rec["episode_reward"] = eval_.get("episode_reward")
                    rec["weekly_raw_total_cost"] = eval_.get("weekly_raw_total_cost")
                    rec["economic_cashflow_total"] = eval_.get("economic_cashflow_total")
                    rec["terminal_soc_satisfied"] = eval_.get("terminal_soc_satisfied")
                    m = eval_.get("metrics") or {}
                    if isinstance(m, dict):
                        for mk in (
                            "total_cost",
                            "carbon_cost",
                            "unserved_energy",
                            "unserved_energy_mwh",
                            "buy_cost",
                            "thermal_cost",
                            "reward",
                            "soc_terminal_ok",
                            "terminal_soc_ok",
                        ):
                            if mk in m:
                                rec[f"m_{mk}"] = m[mk]
                        # keep small metrics dump
                        rec["metrics_keys"] = list(m.keys())[:40]
            except Exception as exc:
                rec["parse_error"] = str(exc)
                rec["content_head"] = content[:400]
        rows.append(rec)
        print(
            f"[{season} {method} s{seed}] done_log={done} result={used is not None} "
            f"reward={rec.get('episode_reward')} cost={rec.get('weekly_raw_total_cost')} "
            f"cash={rec.get('economic_cashflow_total')} soc={rec.get('terminal_soc_satisfied')}",
            flush=True,
        )
        if rec.get("metrics_keys"):
            print(f"  metrics_keys={rec['metrics_keys']}", flush=True)
        if rec.get("parse_error"):
            print(f"  parse_error={rec['parse_error']} head={rec.get('content_head')}", flush=True)

    # also dump full train_result for completed ones
    print("\n=== full train_result for done-ish ===")
    for rec in rows:
        if not (rec.get("done_log") or rec.get("status") == "completed" or rec.get("episode_reward") is not None):
            continue
        p = rec.get("result_path")
        if not p:
            continue
        out = run(
            t,
            f'powershell -NoProfile -Command "Get-Content \'{p}\' -Raw"',
            timeout=40,
        )
        print(f"\n----- {rec['season']} {rec['method']} s{rec['seed']} -----")
        print(out[:4000])

    out_path = Path("logs/remote_done_compare.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nwrote {out_path}", flush=True)
    t.close()


if __name__ == "__main__":
    main()
