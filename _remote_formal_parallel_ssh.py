"""Run TD3/PPO/SAC formal in three parallel SSH sessions (foreground)."""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import paramiko

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HOST = "172.16.1.80"
USER = "dell"
PASSWORD = "TR@SZ"
ROOT = r"D:\xuzh\demo_optimization"


def run_once(cmd: str, timeout: int = 300) -> tuple[int, str]:
    t = paramiko.Transport((HOST, 22))
    t.banner_timeout = 30
    t.auth_timeout = 30
    t.start_client(timeout=30)
    t.auth_password(USER, PASSWORD)
    try:
        chan = t.open_session(timeout=30)
        chan.set_combine_stderr(True)
        chan.exec_command(cmd)
        chunks: list[bytes] = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            if chan.recv_ready():
                chunks.append(chan.recv(65536))
            elif chan.exit_status_ready():
                while chan.recv_ready():
                    chunks.append(chan.recv(65536))
                break
            else:
                time.sleep(0.05)
        else:
            return -1, "TIMEOUT " + cmd[:80]
        return chan.recv_exit_status(), b"".join(chunks).decode("utf-8", errors="replace")
    finally:
        t.close()


def prepare() -> None:
    print("=== kill old demo_optimization python ===", flush=True)
    code, out = run_once(
        "powershell -NoProfile -Command "
        "\"Get-CimInstance Win32_Process -Filter \\\"name='python.exe'\\\" | "
        "Where-Object { $_.CommandLine -and ($_.CommandLine -like '*demo_optimization*') } | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; "
        "Write-Output DONE\"",
        timeout=60,
    )
    print(out, flush=True)

    print("=== clear runs ===", flush=True)
    code, out = run_once(
        r'cmd /c "if exist D:\xuzh\demo_optimization\runs (rd /s /q D:\xuzh\demo_optimization\runs) & '
        r'mkdir D:\xuzh\demo_optimization\runs & echo CLEARED"',
        timeout=120,
    )
    print(out, flush=True)

    print("=== check nvidia / torch ===", flush=True)
    code, out = run_once(r"cmd /c nvidia-smi -L", timeout=60)
    print(out, flush=True)
    # write a tiny probe script via sftp to avoid quoting hell
    t = paramiko.Transport((HOST, 22))
    t.banner_timeout = 30
    t.auth_timeout = 30
    t.start_client(timeout=30)
    t.auth_password(USER, PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(t)
    assert sftp is not None
    with sftp.file("/D:/xuzh/demo_optimization/_probe_torch.py", "w") as f:
        f.write("import torch\nprint(torch.__version__, torch.cuda.is_available())\n")
    sftp.close()
    t.close()
    code, out = run_once(
        r"cmd /c cd /d D:\xuzh\demo_optimization && .venv\Scripts\python.exe _probe_torch.py",
        timeout=60,
    )
    print("torch:", out, flush=True)
    if "False" in out or "+cpu" in out:
        print("Installing torch CUDA 12.1 wheel (best effort)...", flush=True)
        code, out = run_once(
            r"cmd /c cd /d D:\xuzh\demo_optimization && .venv\Scripts\python.exe -m pip install --upgrade "
            r"torch --index-url https://download.pytorch.org/whl/cu121",
            timeout=1800,
        )
        print(out[-3000:] if len(out) > 3000 else out, flush=True)
        code, out = run_once(
            r"cmd /c cd /d D:\xuzh\demo_optimization && .venv\Scripts\python.exe _probe_torch.py",
            timeout=60,
        )
        print("torch after:", out, flush=True)

    code, out = run_once(
        r'cmd /c "echo PREPARE_OK> D:\xuzh\demo_optimization\runs\formal_launch.log"',
        timeout=30,
    )
    print(out, flush=True)


def worker(name: str, script_args: str, results: dict) -> None:
    log_remote = rf"D:\xuzh\demo_optimization\runs\formal_{name}.log"
    cmd = (
        rf'cmd /c "cd /d {ROOT} && set PYTHONPATH=src && '
        rf'echo [%date% %time%] {name}_start>> runs\formal_launch.log && '
        rf'.venv\Scripts\python.exe {script_args} > {log_remote} 2>&1 && '
        rf'echo [%date% %time%] {name}_done=0>> runs\formal_launch.log || '
        rf'echo [%date% %time%] {name}_done=%ERRORLEVEL%>> runs\formal_launch.log"'
    )
    print(f"[{name}] starting", flush=True)
    t = paramiko.Transport((HOST, 22))
    t.banner_timeout = 30
    t.auth_timeout = 30
    t.start_client(timeout=30)
    t.auth_password(USER, PASSWORD)
    try:
        chan = t.open_session(timeout=30)
        chan.set_combine_stderr(True)
        chan.exec_command(cmd)
        # Keep channel alive until training finishes (many hours)
        while True:
            if chan.recv_ready():
                _ = chan.recv(65536)
            if chan.exit_status_ready():
                while chan.recv_ready():
                    _ = chan.recv(65536)
                break
            time.sleep(1.0)
        code = chan.recv_exit_status()
        results[name] = code
        print(f"[{name}] finished exit={code}", flush=True)
    except Exception as exc:
        results[name] = -999
        print(f"[{name}] ERROR {exc}", flush=True)
    finally:
        t.close()


def verify() -> bool:
    t = paramiko.Transport((HOST, 22))
    t.banner_timeout = 30
    t.auth_timeout = 30
    t.start_client(timeout=30)
    t.auth_password(USER, PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(t)
    assert sftp is not None
    ok = True
    for algo in ("td3", "ppo", "sac"):
        base = f"/D:/xuzh/demo_optimization/runs/givesafe_{algo}_formal"
        try:
            with sftp.open(base + "/summary.json", "r") as f:
                summary = json.loads(f.read().decode("utf-8"))
            status = summary.get("status")
            sftp.stat(base + "/report/report.md")
            try:
                sftp.stat(base + "/trajectories/annual_eval")
                annual = True
            except FileNotFoundError:
                annual = False
            print(f"VERIFY {algo}: status={status} annual={annual}")
            if status != "completed" or not annual:
                ok = False
                print("  blockers", summary.get("formal_gate_blockers") or summary.get("blockers"))
            else:
                with sftp.open(base + "/report/report.md", "r") as rf:
                    print(rf.read().decode("utf-8", errors="replace").splitlines()[:12])
        except Exception as exc:
            print(f"VERIFY {algo}: FAIL {exc}")
            ok = False
    sftp.close()
    t.close()
    return ok


def poll_loop(stop_event: threading.Event) -> None:
    while not stop_event.wait(180):
        code, out = run_once(
            r'cmd /c "type D:\xuzh\demo_optimization\runs\formal_launch.log & '
            r'echo --- & '
            r'for %A in (td3 ppo sac) do @echo %A_log_bytes & '
            r'@for %F in (D:\xuzh\demo_optimization\runs\formal_%A.log) do @echo %~zF"',
            timeout=60,
        )
        print("=== poll ===", flush=True)
        print(out, flush=True)
        code, out = run_once(
            "powershell -NoProfile -Command "
            "\"(Get-CimInstance Win32_Process -Filter \\\"name='python.exe'\\\").Count\"",
            timeout=60,
        )
        print("python_count", out.strip(), flush=True)


def main() -> int:
    prepare()
    results: dict[str, int] = {}
    jobs = [
        ("td3", r"scripts\train_hybrid_td3.py --mode formal --annual-eval --run-dir runs\givesafe_td3_formal"),
        ("ppo", r"scripts\train_hybrid_ppo.py --mode formal --annual-eval --run-dir runs\givesafe_ppo_formal"),
        ("sac", r"scripts\train_hybrid_sac.py --mode formal --annual-eval --run-dir runs\givesafe_sac_formal"),
    ]
    stop = threading.Event()
    poller = threading.Thread(target=poll_loop, args=(stop,), daemon=True)
    poller.start()
    threads = [
        threading.Thread(target=worker, args=(name, args, results), daemon=False)
        for name, args in jobs
    ]
    for th in threads:
        th.start()
        time.sleep(2)
    for th in threads:
        th.join()
    stop.set()
    print("RESULTS", results, flush=True)
    ok = verify()
    return 0 if ok and all(v == 0 for v in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
