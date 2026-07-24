"""Remote formal parallel training launcher. Do not commit."""
from __future__ import annotations

import json
import sys
import time

import paramiko

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HOST = "172.16.1.80"
USER = "dell"
PASSWORD = "TR@SZ"
ROOT = r"D:\xuzh\demo_optimization"
PY = ROOT + r"\.venv\Scripts\python.exe"


def transport() -> paramiko.Transport:
    t = paramiko.Transport((HOST, 22))
    t.banner_timeout = 30
    t.auth_timeout = 30
    t.start_client(timeout=30)
    t.auth_password(USER, PASSWORD)
    return t


def run(t: paramiko.Transport, cmd: str, timeout: int = 120) -> tuple[int, str]:
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
        raise TimeoutError(cmd[:120])
    code = chan.recv_exit_status()
    return code, b"".join(chunks).decode("utf-8", errors="replace")


def kill_old(t: paramiko.Transport) -> None:
    # Kill python processes whose command line contains demo_optimization
    ps = (
        "powershell -NoProfile -Command "
        "\"Get-CimInstance Win32_Process -Filter \\\"name='python.exe'\\\" | "
        "Where-Object { $_.CommandLine -and ($_.CommandLine -like '*demo_optimization*') } | "
        "ForEach-Object { Write-Output ('KILL ' + $_.ProcessId + ' ' + $_.CommandLine); "
        "Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; "
        "Write-Output DONE\""
    )
    code, out = run(t, ps, timeout=60)
    print(out)
    print("kill_exit", code)


def clear_runs(t: paramiko.Transport) -> None:
    cmd = (
        r'cmd /c "if exist D:\xuzh\demo_optimization\runs '
        r"(rd /s /q D:\xuzh\demo_optimization\runs) & "
        r'mkdir D:\xuzh\demo_optimization\runs & echo CLEARED"'
    )
    code, out = run(t, cmd, timeout=120)
    print(out)
    if "CLEARED" not in out:
        raise RuntimeError("clear runs failed")


def launch_parallel(t: paramiko.Transport) -> None:
    # Write a launcher bat that starts 3 detached processes
    bat = r"""@echo off
setlocal
cd /d D:\xuzh\demo_optimization
set PYTHONPATH=src
set PY=D:\xuzh\demo_optimization\.venv\Scripts\python.exe
if not exist runs mkdir runs
echo [%date% %time%] LAUNCH > runs\formal_launch.log

start "formal_td3" /b cmd /c "%PY% scripts\train_hybrid_td3.py --mode formal --annual-eval --run-dir runs\givesafe_td3_formal > runs\formal_td3.log 2>&1 & echo [%date% %time%] td3_done=%ERRORLEVEL%>> runs\formal_launch.log"
start "formal_ppo" /b cmd /c "%PY% scripts\train_hybrid_ppo.py --mode formal --annual-eval --run-dir runs\givesafe_ppo_formal > runs\formal_ppo.log 2>&1 & echo [%date% %time%] ppo_done=%ERRORLEVEL%>> runs\formal_launch.log"
start "formal_sac" /b cmd /c "%PY% scripts\train_hybrid_sac.py --mode formal --annual-eval --run-dir runs\givesafe_sac_formal > runs\formal_sac.log 2>&1 & echo [%date% %time%] sac_done=%ERRORLEVEL%>> runs\formal_launch.log"

echo [%date% %time%] STARTED_3 >> runs\formal_launch.log
"""
    sftp = paramiko.SFTPClient.from_transport(t)
    assert sftp is not None
    with sftp.file("/D:/xuzh/demo_optimization/runs_formal_parallel.bat", "w") as f:
        f.write(bat.replace("\n", "\r\n"))
    sftp.close()

    # Use Start-Process so jobs survive SSH session
    ps = (
        "powershell -NoProfile -Command "
        "\"$p=Start-Process -FilePath 'cmd.exe' "
        "-ArgumentList '/c','D:\\xuzh\\demo_optimization\\runs_formal_parallel.bat' "
        "-WorkingDirectory 'D:\\xuzh\\demo_optimization' "
        "-WindowStyle Hidden -PassThru; "
        "Write-Output ('LAUNCHER_PID=' + $p.Id)\""
    )
    code, out = run(t, ps, timeout=60)
    print(out)
    time.sleep(5)
    code, out = run(
        t,
        r'cmd /c "type D:\xuzh\demo_optimization\runs\formal_launch.log & '
        r'echo --- & tasklist /FI "IMAGENAME eq python.exe""',
        timeout=60,
    )
    print(out)


def status(t: paramiko.Transport) -> dict:
    info: dict = {}
    _, launch = run(t, r'cmd /c "if exist D:\xuzh\demo_optimization\runs\formal_launch.log (type D:\xuzh\demo_optimization\runs\formal_launch.log) else (echo NO_LAUNCH)"')
    info["launch"] = launch
    _, py = run(t, r'cmd /c "tasklist /FI "IMAGENAME eq python.exe""')
    info["python"] = py
    for name in ("td3", "ppo", "sac"):
        log = rf"D:\xuzh\demo_optimization\runs\formal_{name}.log"
        _, tail = run(
            t,
            "powershell -NoProfile -Command "
            f"\"if (Test-Path '{log}') {{ Get-Content '{log}' -Tail 5 }} else {{ 'NO_LOG' }}\"",
        )
        info[f"tail_{name}"] = tail
        summary = rf"D:\xuzh\demo_optimization\runs\givesafe_{name}_formal\summary.json"
        _, exists = run(t, f'cmd /c "if exist {summary} (echo HAS_SUMMARY) else (echo NO_SUMMARY)"')
        info[f"summary_{name}"] = exists.strip()
    return info


def verify(t: paramiko.Transport) -> bool:
    ok = True
    sftp = paramiko.SFTPClient.from_transport(t)
    assert sftp is not None
    for algo in ("td3", "ppo", "sac"):
        base = f"/D:/xuzh/demo_optimization/runs/givesafe_{algo}_formal"
        try:
            with sftp.open(base + "/summary.json", "r") as f:
                summary = json.loads(f.read().decode("utf-8"))
            status_v = summary.get("status")
            report = base + "/report/report.md"
            sftp.stat(report)
            # annual eval dir
            try:
                sftp.stat(base + "/trajectories/annual_eval")
                annual = True
            except FileNotFoundError:
                annual = False
            print(f"{algo}: status={status_v} annual_eval={annual} report=OK")
            if status_v != "completed" or not annual:
                ok = False
                if summary.get("report_path"):
                    print("  report_path", summary.get("report_path"))
                if summary.get("blockers") or summary.get("formal_gate_blockers"):
                    print("  blockers", summary.get("blockers") or summary.get("formal_gate_blockers"))
            else:
                with sftp.open(report, "r") as rf:
                    head = rf.read().decode("utf-8", errors="replace").splitlines()[:14]
                print("\n".join("  " + x for x in head))
        except Exception as exc:
            print(f"{algo}: FAIL {exc}")
            ok = False
    sftp.close()
    return ok


def main() -> int:
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    t = transport()
    try:
        _, host = run(t, "echo HOST_OK && hostname")
        print(host)
        if stage in ("kill", "all"):
            print("=== kill old ===")
            kill_old(t)
        if stage in ("clear", "all"):
            print("=== clear runs ===")
            clear_runs(t)
        if stage in ("launch", "all"):
            print("=== launch parallel ===")
            launch_parallel(t)
        if stage == "status":
            info = status(t)
            for k, v in info.items():
                print(f"==== {k} ====")
                print(v)
        if stage == "verify":
            ok = verify(t)
            return 0 if ok else 1
        if stage == "wait":
            # poll until all three done markers or summaries exist
            while True:
                info = status(t)
                print(time.strftime("%H:%M:%S"), "---")
                print(info.get("launch", ""))
                for name in ("td3", "ppo", "sac"):
                    print(name, info.get(f"summary_{name}"), "|", (info.get(f"tail_{name}") or "")[-300:])
                launch = info.get("launch", "")
                if (
                    "td3_done=" in launch
                    and "ppo_done=" in launch
                    and "sac_done=" in launch
                ):
                    print("ALL_DONE_MARKERS")
                    break
                # also break if summaries all present and no python left with demo_optimization
                if all("HAS_SUMMARY" in info.get(f"summary_{n}", "") for n in ("td3", "ppo", "sac")):
                    # wait a bit more for done markers
                    if "python.exe" not in info.get("python", "").lower() or "没有" in info.get("python", ""):
                        print("SUMMARIES_READY")
                        break
                time.sleep(120)
            ok = verify(t)
            return 0 if ok else 1
        return 0
    finally:
        t.close()


if __name__ == "__main__":
    raise SystemExit(main())
