#!/usr/bin/env python
"""Stop queues, wipe seasonal runs/logs, rebuild .venv (non-Comfy), pip via CN mirror, start fair only.

Usage:
  python scripts/remote_rebuild_venv_fair.py --dry-run
  python scripts/remote_rebuild_venv_fair.py
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import paramiko

HOST = "172.16.1.80"
USER = "dell"
PASSWORD = "TR@SZ"
REMOTE = r"D:\xuzh\demo_optimization"
BASE_PY = r"D:\python3.11.8\python.exe"
PIP_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
PIP_TRUST = "pypi.tuna.tsinghua.edu.cn"
TORCH_INDEX = "https://download.pytorch.org/whl/cu124"
LOCAL = Path(__file__).resolve().parents[1]


def connect():
    t = paramiko.Transport((HOST, 22))
    t.banner_timeout = 120
    t.auth_timeout = 120
    t.start_client(timeout=120)
    t.auth_password(USER, PASSWORD)
    return t, paramiko.SFTPClient.from_transport(t)


def run(t: paramiko.Transport, cmd: str, timeout: int = 120) -> str:
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
    else:
        try:
            ch.close()
        except Exception:
            pass
    return buf.decode("utf-8", "replace")


def ps(t: paramiko.Transport, script: str, timeout: int = 300) -> str:
    # Escape for powershell -Command "..."
    one = script.replace("\r\n", "\n").replace("\n", "; ")
    # Prefer -File for long scripts: write temp on remote
    return run(t, f'powershell -NoProfile -ExecutionPolicy Bypass -Command "{one}"', timeout=timeout)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-start", action="store_true", help="rebuild only, do not start fair")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"=== remote rebuild @ {HOST} dry_run={args.dry_run} ===", flush=True)
    t, sftp = connect()

    def step(title: str) -> None:
        print(f"\n--- {title} ---", flush=True)

    # 0) sync requirements.txt
    step("sync requirements.txt")
    req_local = LOCAL / "requirements.txt"
    if req_local.is_file() and not args.dry_run:
        sftp.put(str(req_local), REMOTE + r"\requirements.txt")
        print("put requirements.txt", req_local.stat().st_size, flush=True)
    else:
        print("skip put or dry-run", flush=True)

    # 1) kill processes (PowerShell — wmic LIKE on CommandLine is fragile)
    step("kill train/queue processes")
    kill_ps = r"""
$ErrorActionPreference='Continue'
$keys=@('train_seasonal','fair_queue','wmic_queue','run_v1_','train_ghtd3','train_hybrid')
Get-CimInstance Win32_Process -Filter "name='python.exe' OR name='cmd.exe'" | ForEach-Object {
  $cl = $_.CommandLine
  if (-not $cl) { return }
  $hit = $false
  foreach ($k in $keys) { if ($cl -match [regex]::Escape($k)) { $hit = $true; break } }
  if (-not $hit) { return }
  Write-Output ("KILL pid=" + $_.ProcessId + " " + $cl.Substring(0, [Math]::Min(160, $cl.Length)))
  Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 3
$left = @(Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object {
  $_.CommandLine -and (
    $_.CommandLine -match 'train_seasonal' -or
    $_.CommandLine -match 'fair_queue' -or
    $_.CommandLine -match 'wmic_queue'
  )
})
Write-Output ("LEFT=" + $left.Count)
$left | ForEach-Object { Write-Output ("STILL " + $_.ProcessId + " " + $_.CommandLine.Substring(0,[Math]::Min(120,$_.CommandLine.Length))); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 2
Write-Output 'KILL_DONE'
"""
    remote_kill = REMOTE + rf"\logs\_kill_for_rebuild_{stamp}.ps1"
    local_kill = LOCAL / "logs" / f"_kill_for_rebuild_{stamp}.ps1"
    local_kill.parent.mkdir(parents=True, exist_ok=True)
    local_kill.write_text(kill_ps, encoding="utf-8")
    if not args.dry_run:
        sftp.put(str(local_kill), remote_kill)
        print(run(t, f'powershell -NoProfile -ExecutionPolicy Bypass -File "{remote_kill}"', timeout=120)[:2500], flush=True)
        time.sleep(3)
    else:
        print(kill_ps[:400], flush=True)

    # 2) wipe runs/logs state
    step("wipe seasonal runs + job logs + queue state")
    wipe_ps = rf"""
$ErrorActionPreference='Continue'
$root='{REMOTE}'
$paths=@(
  (Join-Path $root 'runs\seasonal_v1'),
  (Join-Path $root 'runs\seasonal')
)
foreach($p in $paths){{
  if(Test-Path $p){{ Write-Output ('REMOVE_DIR '+$p); if(-not ${{args.dry}}){{}} }}
}}
"""
    # Use explicit remote PowerShell without local interpolation bugs
    wipe_body = f"""
$ErrorActionPreference = 'Continue'
$root = '{REMOTE}'
$dry = ${str(args.dry_run).lower()}
function Kill-Path($p) {{
  if (Test-Path $p) {{
    Write-Output ("REMOVE " + $p)
    if (-not $dry) {{ Remove-Item -LiteralPath $p -Recurse -Force -ErrorAction Continue }}
  }} else {{
    Write-Output ("MISS " + $p)
  }}
}}
Kill-Path (Join-Path $root 'runs\\seasonal_v1')
Kill-Path (Join-Path $root 'runs\\seasonal')
Get-ChildItem (Join-Path $root 'logs') -Filter 'seasonal_v1_*' -ErrorAction SilentlyContinue | ForEach-Object {{
  Write-Output ("REMOVE " + $_.FullName)
  if (-not $dry) {{ Remove-Item -LiteralPath $_.FullName -Force -ErrorAction Continue }}
}}
Get-ChildItem (Join-Path $root 'logs') -Filter 'seasonal_*.log*' -ErrorAction SilentlyContinue | ForEach-Object {{
  if ($_.Name -like 'seasonal_v1_*') {{ return }}
  Write-Output ("REMOVE " + $_.FullName)
  if (-not $dry) {{ Remove-Item -LiteralPath $_.FullName -Force -ErrorAction Continue }}
}}
foreach ($name in @('fair_queue_state.json','wmic_queue_state.json','wmic_queue_v2.log','wmic_queue.log','wmic_queue_runner.log')) {{
  Kill-Path (Join-Path $root ('logs\\' + $name))
}}
$flog = Join-Path $root 'logs\\fair_queue.log'
if (Test-Path $flog) {{
  $bak = Join-Path $root ('logs\\fair_queue.log.bak_{stamp}')
  Write-Output ("BAK " + $flog + " -> " + $bak)
  if (-not $dry) {{
    Copy-Item -LiteralPath $flog -Destination $bak -Force -ErrorAction Continue
    Set-Content -LiteralPath $flog -Value '' -Encoding utf8
  }}
}}
Write-Output 'WIPE_DONE'
"""
    remote_wipe = REMOTE + rf"\logs\_wipe_for_rebuild_{stamp}.ps1"
    local_wipe = LOCAL / "logs" / f"_wipe_for_rebuild_{stamp}.ps1"
    local_wipe.parent.mkdir(parents=True, exist_ok=True)
    local_wipe.write_text(wipe_body, encoding="utf-8")
    if not args.dry_run:
        sftp.put(str(local_wipe), remote_wipe)
        print(run(t, f'powershell -NoProfile -ExecutionPolicy Bypass -File "{remote_wipe}"', timeout=600)[:3000], flush=True)
    else:
        print(wipe_body[:800], flush=True)
        print("(dry-run: wipe not executed)", flush=True)

    # 3) rebuild venv
    step("rebuild .venv from D:\\python3.11.8")
    rebuild_body = f"""
$ErrorActionPreference = 'Stop'
$root = '{REMOTE}'
$base = '{BASE_PY}'
$venv = Join-Path $root '.venv'
$bak = Join-Path $root '.venv_comfy_bak_{stamp}'
$dry = ${str(args.dry_run).lower()}
if (-not (Test-Path $base)) {{ throw "BASE_PY missing: $base" }}
if (Test-Path $venv) {{
  Write-Output ("RENAME " + $venv + " -> " + $bak)
  if (-not $dry) {{
    if (Test-Path $bak) {{ Remove-Item -LiteralPath $bak -Recurse -Force }}
    Rename-Item -LiteralPath $venv -NewName (Split-Path $bak -Leaf)
  }}
}}
Write-Output ("CREATE venv with " + $base)
if (-not $dry) {{
  & $base -m venv $venv
  if ($LASTEXITCODE -ne 0) {{ throw "venv create failed" }}
}}
$cfg = Join-Path $venv 'pyvenv.cfg'
if ((-not $dry) -and (Test-Path $cfg)) {{
  Write-Output '=== pyvenv.cfg ==='
  Get-Content $cfg
  $txt = Get-Content $cfg -Raw
  if ($txt -match 'ComfyUI') {{ throw 'pyvenv.cfg still points to ComfyUI' }}
  if ($txt -notmatch 'python3\\.11\\.8') {{ Write-Output 'WARN: home may not be python3.11.8' }}
}}
Write-Output 'VENV_OK'
"""
    remote_rb = REMOTE + rf"\logs\_rebuild_venv_{stamp}.ps1"
    local_rb = LOCAL / "logs" / f"_rebuild_venv_{stamp}.ps1"
    local_rb.write_text(rebuild_body, encoding="utf-8")
    if not args.dry_run:
        sftp.put(str(local_rb), remote_rb)
        out = run(t, f'powershell -NoProfile -ExecutionPolicy Bypass -File "{remote_rb}"', timeout=300)
        print(out[:2500], flush=True)
        if "VENV_OK" not in out and "throw" in out.lower():
            raise SystemExit("venv rebuild failed")
    else:
        print(rebuild_body[:600], flush=True)

    # 4) pip install via CN mirror + torch cu124
    step("pip install (tuna + torch cu124)")
    py = REMOTE + r"\.venv\Scripts\python.exe"
    pip_body = f"""
$ErrorActionPreference = 'Stop'
$py = '{py}'
$root = '{REMOTE}'
$req = Join-Path $root 'requirements.txt'
$index = '{PIP_INDEX}'
$trust = '{PIP_TRUST}'
$torchIndex = '{TORCH_INDEX}'
$dry = ${str(args.dry_run).lower()}
if ($dry) {{ Write-Output 'DRY pip'; exit 0 }}
if (-not (Test-Path $py)) {{ throw "venv python missing: $py" }}
function Invoke-Pip([string[]]$PipArgs) {{
  Write-Output ('PIP ' + ($PipArgs -join ' '))
  & $py -m pip @PipArgs
  if ($LASTEXITCODE -ne 0) {{ throw ('pip failed: ' + ($PipArgs -join ' ')) }}
}}
Invoke-Pip @('install','-U','pip','setuptools','wheel','-i',$index,'--trusted-host',$trust)
# Install non-torch deps from tuna (avoid tuna CPU torch overriding CUDA build)
$reqNoTorch = Join-Path $root 'logs\\_requirements_no_torch.txt'
Get-Content $req | Where-Object {{ $_ -notmatch '^\s*torch\b' }} | Set-Content -LiteralPath $reqNoTorch -Encoding utf8
Invoke-Pip @('install','-r',$reqNoTorch,'-i',$index,'--trusted-host',$trust)
# CUDA torch last (authoritative)
Invoke-Pip @('install','--force-reinstall','torch','--index-url',$torchIndex)
Write-Output '=== smoke imports ==='
$smoke = Join-Path $root 'logs\\_smoke_imports.py'
@'
import torch, numpy, yaml, gymnasium, fmpy, tqdm
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
assert torch.cuda.is_available(), "cuda false"
'@ | Set-Content -LiteralPath $smoke -Encoding utf8
& $py $smoke
if ($LASTEXITCODE -ne 0) {{ throw 'smoke import failed' }}
$cfg = Get-Content (Join-Path $root '.venv\\pyvenv.cfg') -Raw
if ($cfg -match 'ComfyUI') {{ throw 'ComfyUI still in pyvenv.cfg after install' }}
Write-Output 'PIP_OK'
"""
    remote_pip = REMOTE + rf"\logs\_pip_install_{stamp}.ps1"
    local_pip = LOCAL / "logs" / f"_pip_install_{stamp}.ps1"
    local_pip.write_text(pip_body, encoding="utf-8")
    if not args.dry_run:
        sftp.put(str(local_pip), remote_pip)
        # long install
        out = run(t, f'powershell -NoProfile -ExecutionPolicy Bypass -File "{remote_pip}"', timeout=3600)
        print(out[-4000:] if len(out) > 4000 else out, flush=True)
        if "PIP_OK" not in out:
            raise SystemExit("pip install failed (no PIP_OK)")
        # require cuda
        if "cuda True" not in out and "cuda', True" not in out:
            # print already has smoke line
            if "cuda False" in out or "cuda', False" in out:
                raise SystemExit("torch.cuda.is_available() is False")
    else:
        print(pip_body[:700], flush=True)

    # 5) empty fair state + start fair only
    step("start fair queue only")
    if not args.dry_run:
        state = '{"started": []}\n'
        sftp.open(REMOTE + r"\logs\fair_queue_state.json", "w").write(state)
        # ensure fair log exists
        try:
            sftp.open(REMOTE + r"\logs\fair_queue.log", "a").write(f"\n# rebuild {stamp}\n")
        except Exception:
            pass
    else:
        print("would write fair_queue_state.json started=[]", flush=True)

    if args.skip_start or args.dry_run:
        print("skip start fair", flush=True)
    else:
        start_cmd = (
            f'wmic process call create "cmd.exe /c call \\"{REMOTE}\\logs\\start_fair_queue.bat\\""'
        )
        print(start_cmd, flush=True)
        print(run(t, start_cmd, timeout=60)[:500], flush=True)
        time.sleep(20)

    # 6) verify
    step("verify")
    print(run(t, f'cmd /c type {REMOTE}\\.venv\\pyvenv.cfg', timeout=30), flush=True)
    live = run(
        t,
        r'cmd /c wmic process where "name=\"python.exe\"" get ProcessId,CommandLine /FORMAT:LIST',
        timeout=90,
    )
    train_lines = [ln for ln in live.splitlines() if "train_seasonal" in ln.lower() or "fair_queue" in ln.lower()]
    print("live train/fair lines:", len(train_lines), flush=True)
    for ln in train_lines[:20]:
        print(ln[:240], flush=True)
    comfy = [ln for ln in train_lines if "comfyui" in ln.lower()]
    print("comfy matched live lines:", len(comfy), flush=True)
    print(
        run(t, f'powershell -NoProfile -Command "Get-Content \'{REMOTE}\\logs\\fair_queue.log\' -Tail 15"', timeout=40),
        flush=True,
    )
    print(
        run(t, "cmd /c nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader", timeout=30),
        flush=True,
    )

    sftp.close()
    t.close()
    print("\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
