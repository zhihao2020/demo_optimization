"""Poll remote STFR multi-seed runs; print DONE/FAILED for monitor.

Usage (local):
  python scripts/watch_remote_stfr.py --seeds 0,1,2,3,4,5 --steps 35000 --interval 60

Monitor-friendly: only prints DONE / FAILED / STATUS lines (no spam).
"""
from __future__ import annotations

import argparse
import sys
import time

import paramiko

HOST = "172.16.1.80"
USER = "dell"
PASSWORD = "TR@SZ"
ROOT = r"D:\xuzh\demo_optimization"


def run(t: paramiko.Transport, cmd: str, timeout: int = 60) -> str:
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


def snapshot(seeds: list[int], steps: int, run_prefix: str) -> dict:
    t = paramiko.Transport((HOST, 22))
    t.banner_timeout = 30
    t.auth_timeout = 30
    t.start_client(timeout=30)
    t.auth_password(USER, PASSWORD)
    try:
        done = []
        prog = []
        for s in seeds:
            rel = f"{run_prefix}_s{s}_{steps // 1000}k".replace("/", "\\")
            base = rf"{ROOT}\{rel}"
            out = run(
                t,
                rf'cmd /c "if exist {base}\vs_hybrid.json (echo VS) else (echo NOVS) '
                rf'& if exist {base}\summary.json (echo SUM) else (echo NOSUM) '
                rf'& if exist {base}\train\progress.json (type {base}\train\progress.json) else (echo NOPROG)"',
                timeout=45,
            )
            has_vs = "VS" in out and "NOVS" not in out.split("VS")[0][-5:] if "VS" in out else False
            # simpler flags
            has_vs = "\nVS" in ("\n" + out.replace("\r", "")) or out.strip().startswith("VS")
            lines = [ln.strip() for ln in out.replace("\r", "").splitlines() if ln.strip()]
            flags = set()
            for ln in lines:
                if ln in ("VS", "NOVS", "SUM", "NOSUM", "NOPROG"):
                    flags.add(ln)
            has_vs = "VS" in flags
            has_sum = "SUM" in flags
            frac = None
            if '"frac"' in out:
                try:
                    import json
                    # last json blob
                    i = out.rfind("{")
                    j = out.rfind("}")
                    if i >= 0 and j > i:
                        frac = float(json.loads(out[i : j + 1]).get("frac", 0))
                except Exception:
                    pass
            if has_vs and has_sum:
                done.append(s)
            prog.append((s, has_sum, has_vs, frac))
        # Prefer progress.json "frac" and vs/summary files; process list is unreliable over SSH/wmic.
        any_active = any(
            (f is not None and f < 0.999) or (not has_vs)
            for _s, _sum, has_vs, f in prog
        )
        return {
            "done": done,
            "prog": prog,
            "n_seeds": len(seeds),
            "any_active": any_active,
        }
    finally:
        t.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2,3,4,5")
    p.add_argument("--steps", type=int, default=35000)
    p.add_argument("--run-prefix", default="runs/stfr", help="e.g. runs/stfr or runs/ltar")
    p.add_argument("--interval", type=int, default=90)
    p.add_argument("--max-hours", type=float, default=8.0)
    args = p.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    deadline = time.time() + args.max_hours * 3600
    last_line = ""
    stagnant = 0
    last_sig = ""
    while time.time() < deadline:
        try:
            snap = snapshot(seeds, args.steps, args.run_prefix)
        except Exception as exc:
            print(f"STATUS poll_error={exc!r}", flush=True)
            time.sleep(args.interval)
            continue
        n_done = len(snap["done"])
        fracs = [f"{s}:{'' if f is None else f'{100 * f:.0f}%'}" for s, _, _, f in snap["prog"]]
        line = f"STATUS done={n_done}/{snap['n_seeds']} " + " ".join(fracs)
        if line != last_line:
            print(line, flush=True)
            last_line = line
        if n_done >= len(seeds):
            print("DONE all_seeds_finished", flush=True)
            return
        # stagnation: progress signature unchanged for many polls while incomplete
        sig = "|".join(f"{s}:{f}" for s, _, hv, f in snap["prog"])
        if sig == last_sig:
            stagnant += 1
        else:
            stagnant = 0
            last_sig = sig
        # ~30 min no progress change and none finished eval → soft fail
        if stagnant >= 20 and n_done == 0:
            print("FAILED stalled_no_progress", flush=True)
            sys.exit(2)
        time.sleep(args.interval)
    print("FAILED timeout", flush=True)
    sys.exit(3)


if __name__ == "__main__":
    main()
