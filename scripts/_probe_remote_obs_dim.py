"""One-shot remote obs/FMU probe."""
from __future__ import annotations

import time
from pathlib import Path

import paramiko

HOST = "172.16.1.80"
USER = "dell"
PASSWORD = "TR@SZ"
REMOTE = r"D:\xuzh\demo_optimization"
CACHE = r"D:\xuzh\demo_optimization_cache"
PY = r"D:\xuzh\demo_optimization\.venv\Scripts\python.exe"
LOCAL = Path(__file__).resolve().parents[1]


def main() -> None:
    py = (
        "import sys\n"
        f"sys.path.insert(0, r'{REMOTE}\\src')\n"
        "from pathlib import Path\n"
        "from envs.forecast_provider import DEFAULT_OBSERVATION_DIM\n"
        "assert int(DEFAULT_OBSERVATION_DIM) == 166\n"
        f"f = Path(r'{REMOTE}') / 'data' / 'TypicalScensrio_Example_TypicalScene_PowerSystem_8760h.fmu'\n"
        "assert f.is_file() and f.stat().st_size > 1e6\n"
        "print('PROBE_OK', DEFAULT_OBSERVATION_DIM, f.stat().st_size)\n"
    )
    local = LOCAL / "logs" / "_remote_obs_probe.py"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(py, encoding="utf-8")

    t = paramiko.Transport((HOST, 22))
    t.banner_timeout = 60
    t.start_client(timeout=60)
    t.auth_password(USER, PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(t)
    assert sftp is not None
    rp = rf"{REMOTE}\logs\_remote_obs_probe.py"
    sftp.put(str(local), rp, confirm=False)
    ch = t.open_session()
    ch.set_combine_stderr(True)
    cmd = (
        f'powershell -NoProfile -Command "Set-Location \'{REMOTE}\'; '
        f'$env:OPTIMAL_DEMO_CACHE=\'{CACHE}\'; $env:PYTHONPATH=\'{REMOTE}\\src\'; '
        f'& \'{PY}\' \'{rp}\'"'
    )
    ch.exec_command(cmd)
    buf = b""
    end = time.time() + 120
    while time.time() < end:
        if ch.recv_ready():
            buf += ch.recv(65536)
        elif ch.exit_status_ready():
            while ch.recv_ready():
                buf += ch.recv(65536)
            break
        else:
            time.sleep(0.05)
    out = buf.decode("utf-8", "replace")
    print(out)
    sftp.close()
    t.close()
    if "PROBE_OK" not in out:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
