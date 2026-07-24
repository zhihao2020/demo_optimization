import sys
import time

import paramiko

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

t = paramiko.Transport(("172.16.1.80", 22))
t.banner_timeout = 30
t.auth_timeout = 30
t.start_client(timeout=30)
t.auth_password("dell", "TR@SZ")


def run(cmd, timeout=180):
    chan = t.open_session(timeout=30)
    chan.set_combine_stderr(True)
    chan.exec_command(cmd)
    chunks = []
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
    code = chan.recv_exit_status() if chan.exit_status_ready() else -1
    print("EXIT", code)
    print(b"".join(chunks).decode("utf-8", errors="replace")[-4000:])


print("=== processes ===")
run(
    "powershell -NoProfile -Command "
    "\"Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|cmd' -and $_.CommandLine -match 'demo_optimization|formal' } | "
    "Select-Object ProcessId,Name,CommandLine | Format-List\""
)

print("=== test venv python ===")
run(
    r'cmd /c "cd /d D:\xuzh\demo_optimization && set PYTHONPATH=src && '
    r'.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())""'
)

print("=== test import train ===")
run(
    r'cmd /c "cd /d D:\xuzh\demo_optimization && set PYTHONPATH=src && '
    r'.venv\Scripts\python.exe -c "from training.hybrid_td3.train import run_formal; print(\"ok\")""'
)

print("=== bat content ===")
run(r'cmd /c "type D:\xuzh\demo_optimization\run_formal_td3.bat"')

print("=== run bat sync briefly? check log after manual start ===")
# Start-Process again and wait longer
run(
    "powershell -NoProfile -Command "
    "\"Start-Process -FilePath 'D:\\xuzh\\demo_optimization\\.venv\\Scripts\\python.exe' "
    "-ArgumentList 'scripts\\train_hybrid_td3.py','--mode','formal','--annual-eval','--run-dir','runs\\givesafe_td3_formal' "
    "-WorkingDirectory 'D:\\xuzh\\demo_optimization' "
    "-RedirectStandardOutput 'D:\\xuzh\\demo_optimization\\runs\\formal_td3.log' "
    "-RedirectStandardError 'D:\\xuzh\\demo_optimization\\runs\\formal_td3.err' "
    "-WindowStyle Hidden; "
    "$env:PYTHONPATH='src'; "
    "Write-Output STARTED\""
)
# Note: Start-Process doesn't inherit env set after - need Environment
time.sleep(3)
run(
    "powershell -NoProfile -Command "
    "\"$psi=New-Object System.Diagnostics.ProcessStartInfo; "
    "$psi.FileName='D:\\xuzh\\demo_optimization\\.venv\\Scripts\\python.exe'; "
    "$psi.Arguments='scripts\\train_hybrid_ppo.py --mode formal --annual-eval --run-dir runs\\givesafe_ppo_formal'; "
    "$psi.WorkingDirectory='D:\\xuzh\\demo_optimization'; "
    "$psi.UseShellExecute=$false; "
    "$psi.RedirectStandardOutput=$true; "
    "$psi.RedirectStandardError=$true; "
    "$psi.CreateNoWindow=$true; "
    "$psi.EnvironmentVariables['PYTHONPATH']='src'; "
    "$p=[Diagnostics.Process]::Start($psi); "
    "Write-Output ('PPO_PID=' + $p.Id)\""
)

time.sleep(10)
run(
    "powershell -NoProfile -Command "
    "\"Get-CimInstance Win32_Process -Filter \\\"name='python.exe'\\\" | "
    "Select-Object ProcessId,CommandLine | Format-List\""
)
run(r'cmd /c "dir D:\xuzh\demo_optimization\runs\formal_*.log D:\xuzh\demo_optimization\runs\formal_*.err"')
run(
    "powershell -NoProfile -Command "
    "\"Get-Content D:\\xuzh\\demo_optimization\\runs\\formal_td3.err -ErrorAction SilentlyContinue -Tail 20; "
    "Get-Content D:\\xuzh\\demo_optimization\\runs\\formal_td3.log -ErrorAction SilentlyContinue -Tail 20\""
)

t.close()
