#!/usr/bin/env python
"""Pull seasonal_v1 runs from remote fair suite into local runs/seasonal_v1.

Usage:
  python scripts/pull_seasonal_v1.py
  python scripts/pull_seasonal_v1.py --seasons winter --methods hmsd,td3,pso,linprog
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import paramiko

HOST = "172.16.1.80"
USER = "dell"
PASSWORD = "TR@SZ"
REMOTE_ROOT = r"D:/xuzh/demo_optimization"
LOCAL_ROOT = Path(__file__).resolve().parents[1]

INCLUDE_NAMES = {
    "summary.json",
    "progress.json",
    "train_result.json",
    "protocol.json",
    "bc_summary.json",
    "ghtd3_config.yaml",
    "env_config.yaml",
    "reward_config.yaml",
    "givesafe_config.yaml",
    "ghtd3.pt",
    "hybrid_givesafe_td3.pt",
    "hybrid_givesafe_sac.pt",
    "eval.csv",
}
INCLUDE_SUFFIXES = {".json", ".yaml", ".yml", ".csv", ".md", ".pt"}
SKIP_DIR_NAMES = {"__pycache__"}
SKIP_NAME_SUBSTR = ("step_log",)
MAX_FILE_BYTES = 800 * 1024 * 1024


def sftp_walk(sftp: paramiko.SFTPClient, remote: str):
    import stat as stmod

    try:
        entries = sftp.listdir_attr(remote)
    except OSError:
        return
    files, dirs = [], []
    for e in entries:
        path = remote.rstrip("/") + "/" + e.filename
        if stmod.S_ISDIR(e.st_mode):
            if e.filename not in SKIP_DIR_NAMES:
                dirs.append(path)
        else:
            files.append((path, e))
    yield remote, dirs, files
    for d in dirs:
        yield from sftp_walk(sftp, d)


def should_pull(name: str, size: int | None) -> bool:
    if any(s in name for s in SKIP_NAME_SUBSTR):
        return False
    if size is not None and size > MAX_FILE_BYTES:
        return False
    if name in INCLUDE_NAMES:
        return True
    ext = Path(name).suffix.lower()
    if ext in INCLUDE_SUFFIXES and (size is None or size < 50 * 1024 * 1024):
        if name.endswith(".log") and size is not None and size > 5 * 1024 * 1024:
            return False
        return True
    return False


def pull_tree(sftp: paramiko.SFTPClient, remote_rel: str, local_rel: str) -> dict:
    remote = f"{REMOTE_ROOT}/{remote_rel}".replace("\\", "/")
    local = LOCAL_ROOT / local_rel
    local.mkdir(parents=True, exist_ok=True)
    pulled, skipped = [], []
    try:
        sftp.stat(remote)
    except OSError:
        return {"remote": remote_rel, "error": "missing", "pulled": []}
    for _root, _dirs, files in sftp_walk(sftp, remote):
        for rpath, attr in files:
            name = Path(rpath).name
            size = getattr(attr, "st_size", None)
            if not should_pull(name, size):
                skipped.append(name)
                continue
            rel = rpath[len(remote) :].lstrip("/")
            lpath = local / rel
            lpath.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(rpath, str(lpath))
            pulled.append(rel)
    return {"remote": remote_rel, "local": str(local), "pulled": pulled, "skipped_n": len(skipped)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="winter,transition,summer")
    ap.add_argument("--methods", default="hmsd,td3,sac,pso,linprog")
    ap.add_argument("--seeds", default="0")
    args = ap.parse_args()
    seasons = [x.strip() for x in args.seasons.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    t = paramiko.Transport((HOST, 22))
    t.banner_timeout = 60
    t.start_client(timeout=60)
    t.auth_password(USER, PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(t)
    assert sftp is not None

    results = []
    for season in seasons:
        for method in methods:
            for seed in seeds:
                rel = f"runs/seasonal_v1/{season}/{method}_s{seed}"
                print(f"pulling {rel} ...", flush=True)
                info = pull_tree(sftp, rel, rel)
                results.append(info)
                n = len(info.get("pulled", []))
                err = info.get("error")
                print(f"  -> {n} files" + (f" ERR={err}" if err else ""), flush=True)

    # queue / protocol markers
    for extra in ("logs/fair_queue_state.json", "logs/remote_fair_suite_manifest.json"):
        try:
            rp = f"{REMOTE_ROOT}/{extra}".replace("\\", "/")
            sftp.stat(rp)
            lp = LOCAL_ROOT / extra
            lp.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(rp, str(lp))
            print("pulled", extra, flush=True)
        except OSError:
            pass

    man = LOCAL_ROOT / "logs" / "seasonal_v1_pull_manifest.json"
    man.parent.mkdir(parents=True, exist_ok=True)
    man.write_text(
        json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "results": results}, indent=2),
        encoding="utf-8",
    )
    print("wrote", man)
    sftp.close()
    t.close()


if __name__ == "__main__":
    main()
