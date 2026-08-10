"""Pull finished run artifacts from remote training server via SFTP."""
from __future__ import annotations

import json
import stat
from pathlib import Path

import paramiko

HOST = "172.16.1.80"
USER = "dell"
PASSWORD = "TR@SZ"
REMOTE_ROOT = r"D:/xuzh/demo_optimization"
LOCAL_ROOT = Path(__file__).resolve().parents[1]

# Prefer analysis-critical files; skip huge step logs / sac tqdm dumps unless needed.
INCLUDE_NAMES = {
    "summary.json",
    "progress.json",
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
SKIP_NAME_SUBSTR = ("step_log",)  # often huge
MAX_FILE_BYTES = 800 * 1024 * 1024  # 800MB cap per file


def sftp_walk(sftp: paramiko.SFTPClient, remote: str):
    try:
        entries = sftp.listdir_attr(remote)
    except IOError:
        return
    files = []
    dirs = []
    for e in entries:
        path = remote.rstrip("/") + "/" + e.filename
        if stat.S_ISDIR(e.st_mode):
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
    # small json/yaml/csv under config/train/trajectories
    ext = Path(name).suffix.lower()
    if ext in INCLUDE_SUFFIXES and (size is None or size < 50 * 1024 * 1024):
        # skip giant sac tqdm logs
        if name.startswith("sac_s") and name.endswith(".log"):
            return size is not None and size < 2 * 1024 * 1024
        if name.endswith(".log") and size is not None and size > 5 * 1024 * 1024:
            return False
        return True
    return False


def pull_run(sftp: paramiko.SFTPClient, run_name: str) -> dict:
    remote_run = f"{REMOTE_ROOT}/runs/{run_name}".replace("\\", "/")
    local_run = LOCAL_ROOT / "runs" / "remote_840k" / run_name
    local_run.mkdir(parents=True, exist_ok=True)
    pulled = []
    skipped = []
    try:
        sftp.stat(remote_run)
    except IOError:
        return {"run": run_name, "error": "missing_remote", "pulled": [], "skipped": []}

    for root, _dirs, files in sftp_walk(sftp, remote_run):
        for rpath, attr in files:
            name = Path(rpath).name
            size = getattr(attr, "st_size", None)
            if not should_pull(name, size):
                skipped.append(name)
                continue
            rel = rpath[len(remote_run) :].lstrip("/").replace("\\", "/")
            lpath = local_run / rel
            lpath.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(rpath, str(lpath))
            pulled.append(rel)
    # also pull compact sac logs heads if present at runs root
    return {"run": run_name, "local": str(local_run), "pulled": pulled, "skipped_n": len(skipped)}


def pull_extra_logs(sftp: paramiko.SFTPClient) -> list[str]:
    """Pull only last ~200KB of huge sac logs via remote powershell is hard; skip full logs.
    Pull short marker files if any.
    """
    out = []
    for s in (0, 1, 2):
        # try small companion if exists
        for rel in (f"runs/sac_s{s}_status.txt",):
            r = f"{REMOTE_ROOT}/{rel}".replace("\\", "/")
            try:
                sftp.stat(r)
            except IOError:
                continue
            l = LOCAL_ROOT / "runs" / "remote_840k" / Path(rel).name
            l.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(r, str(l))
            out.append(rel)
    return out


def main() -> None:
    runs = [
        "ghtd3_abs_s0",
        "ghtd3_abs_s1",
        "ghtd3_abs_s2",
        "td3_scratch_s0",
        "td3_scratch_s1",
        "td3_scratch_s2",
        "givesafe_sac_s0",
        "givesafe_sac_s1",
        "givesafe_sac_s2",
    ]
    t = paramiko.Transport((HOST, 22))
    t.banner_timeout = 60
    t.auth_timeout = 60
    t.start_client(timeout=60)
    t.auth_password(USER, PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(t)
    assert sftp is not None

    results = []
    for name in runs:
        print(f"pulling {name} ...", flush=True)
        info = pull_run(sftp, name)
        results.append(info)
        print(
            f"  -> {len(info.get('pulled', []))} files"
            + (f" ERR={info['error']}" if info.get("error") else ""),
            flush=True,
        )

    extra = pull_extra_logs(sftp)
    print("extra:", extra)

    # extract sac log tails via shell for analysis (no full 14MB)
    ch = t.open_session()
    ch.set_combine_stderr(True)
    ps = (
        "$root='D:\\xuzh\\demo_optimization\\runs'; "
        "foreach($s in 0,1,2){ "
        "$p=Join-Path $root (\"sac_s$s.log\"); "
        "Write-Output (\"==== sac_s$s ====\"); "
        "if(Test-Path -LiteralPath $p){ "
        "$i=Get-Item -LiteralPath $p; Write-Output ('SIZE='+$i.Length); "
        "Get-Content -LiteralPath $p -TotalCount 5; "
        "Write-Output '---TAIL---'; "
        "Get-Content -LiteralPath $p -Tail 25 "
        "} else { Write-Output 'MISSING' } }"
    )
    ch.exec_command(f'powershell -NoProfile -ExecutionPolicy Bypass -Command "{ps}"')
    buf = b""
    import time

    end = time.time() + 90
    while time.time() < end:
        if ch.recv_ready():
            buf += ch.recv(65536)
        elif ch.exit_status_ready():
            while ch.recv_ready():
                buf += ch.recv(65536)
            break
        else:
            time.sleep(0.05)
    sac_txt = buf.decode("utf-8", "replace")
    sac_path = LOCAL_ROOT / "runs" / "remote_840k" / "sac_logs_head_tail.txt"
    sac_path.parent.mkdir(parents=True, exist_ok=True)
    sac_path.write_text(sac_txt, encoding="utf-8")
    print("wrote", sac_path)

    manifest = LOCAL_ROOT / "runs" / "remote_840k" / "pull_manifest.json"
    manifest.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", manifest)

    sftp.close()
    t.close()
    print("DONE")


if __name__ == "__main__":
    main()
