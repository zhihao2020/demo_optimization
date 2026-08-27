"""Relaunch tou2026 FS-HSAC seed-0 on GPU (detached WMIC). Do not 3-way CPU."""
from __future__ import annotations

import time
from pathlib import Path

import paramiko

HOST, USER, PASSWORD = "172.16.1.80", "dell", "TR@SZ"
REMOTE = r"D:\xuzh\demo_optimization"
LOCAL = Path(__file__).resolve().parents[1]
PY = REMOTE + r"\.venv\Scripts\python.exe"
SEASONS = ("winter", "transition", "summer")


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


def main() -> None:
    t = paramiko.Transport((HOST, 22))
    t.banner_timeout = 60
    t.auth_timeout = 60
    t.start_client(timeout=60)
    t.auth_password(USER, PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(t)

    print("=== GPU before ===")
    print(run(t, "cmd /c nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader").strip())
    smoke = REMOTE + r"\logs\_smoke_torch_cuda.py"
    with sftp.file(smoke, "w") as f:
        f.write(
            "import torch\n"
            "print('torch', torch.__version__)\n"
            "print('cuda', torch.cuda.is_available())\n"
            "print('name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)\n"
        )
    smoke_out = run(t, PY + " " + smoke, 120)
    print("=== CUDA smoke ===")
    print(smoke_out.strip())
    if "cuda True" not in smoke_out:
        raise SystemExit("CUDA smoke failed; refusing to launch")

    live = run(
        t,
        r'cmd /c wmic process where "name=\"python.exe\"" get ProcessId,CommandLine /FORMAT:LIST',
        60,
    )
    for ln in live.splitlines():
        if "train_seasonal" in ln.lower():
            raise SystemExit("train_seasonal already live:\n" + ln[:300])

    for season in SEASONS:
        rel = f"logs/run_tou2026_{season}_fs_hsac_s0.bat"
        sftp.put(str(LOCAL / rel), REMOTE + "\\" + rel.replace("/", "\\"))
        print("PUT", rel)

    print("DEL logs", run(t, r"cmd /c del /q D:\xuzh\demo_optimization\logs\tou2026_*.log D:\xuzh\demo_optimization\logs\tou2026_*.log.err").strip())

    for season in SEASONS:
        bat = rf"{REMOTE}\logs\run_tou2026_{season}_fs_hsac_s0.bat"
        cmd = r'cmd /c wmic process call create "cmd.exe /c ' + bat + rf'","{REMOTE}"'
        print("WMIC", season)
        print(run(t, cmd, 30).strip()[:500])
        time.sleep(3)

    print("wait 40s for FMU init...")
    time.sleep(40)
    print("=== GPU after ===")
    print(run(t, "cmd /c nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader").strip())
    print("=== python ===")
    live = run(
        t,
        r'cmd /c wmic process where "name=\"python.exe\"" get ProcessId,CommandLine /FORMAT:LIST',
        60,
    )
    n = 0
    for ln in live.splitlines():
        if "train_seasonal" in ln.lower() or ln.strip().lower().startswith("processid="):
            print(ln.strip()[:300])
            if "train_seasonal" in ln.lower():
                n += 1
    print("live_train_seasonal_cmdlines", n)
    for season in SEASONS:
        name = f"tou2026_{season}_fs_hsac_s0"
        log = rf"{REMOTE}\logs\{name}.log"
        err = log + ".err"
        print("---", name)
        print("LOG", run(t, f'cmd /c if exist "{log}" (powershell -NoProfile -Command "Get-Content -LiteralPath \'{log}\' -Tail 8") else echo NOLOG').strip()[-800:])
        print("ERR", run(t, f'cmd /c if exist "{err}" (powershell -NoProfile -Command "Get-Content -LiteralPath \'{err}\' -Tail 12") else echo NOERR').strip()[-1200:])

    sftp.close()
    t.close()


if __name__ == "__main__":
    main()
