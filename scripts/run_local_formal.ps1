# 本机串行跑 TD3 / PPO / SAC formal（CPU），带 tqdm 进度条
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = "D:\Tool\Python311\python.exe"
$env:PYTHONPATH = "src"
$env:PYTHONUNBUFFERED = "1"

New-Item -ItemType Directory -Force -Path "runs\local_logs" | Out-Null

$jobs = @(
    @{ Name = "td3"; Script = "scripts\train_hybrid_td3.py"; RunDir = "runs\givesafe_td3_formal" },
    @{ Name = "ppo"; Script = "scripts\train_hybrid_ppo.py"; RunDir = "runs\givesafe_ppo_formal" },
    @{ Name = "sac"; Script = "scripts\train_hybrid_sac.py"; RunDir = "runs\givesafe_sac_formal" }
)

foreach ($job in $jobs) {
    $log = Join-Path $Root "runs\local_logs\$($job.Name)_formal.log"
    Write-Host "==== Starting $($job.Name) formal -> $($job.RunDir) ====" -ForegroundColor Cyan
    Write-Host "Log: $log"
    & $Python $job.Script --mode formal --annual-eval --run-dir $job.RunDir *> $log
    if ($LASTEXITCODE -ne 0) {
        Write-Host "---- last 40 log lines ----" -ForegroundColor Yellow
        Get-Content $log -Tail 40
        throw "$($job.Name) failed with exit $LASTEXITCODE"
    }
    Write-Host "==== Finished $($job.Name) ====" -ForegroundColor Green
    Get-Content $log -Tail 5
}

Write-Host "All formal runs completed." -ForegroundColor Green
