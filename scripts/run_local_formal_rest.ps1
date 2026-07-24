# TD3 formal 结束后继续跑 PPO / SAC（本机串行）
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = "D:\Tool\Python311\python.exe"
$env:PYTHONPATH = "src"
$env:PYTHONUNBUFFERED = "1"
New-Item -ItemType Directory -Force -Path "runs\local_logs" | Out-Null

$td3Summary = Join-Path $Root "runs\givesafe_td3_formal\summary.json"
Write-Host "Waiting for TD3 summary: $td3Summary"
while (-not (Test-Path $td3Summary)) {
    Start-Sleep -Seconds 30
}
Write-Host "TD3 done. Starting PPO / SAC..."

foreach ($job in @(
    @{ Name = "ppo"; Script = "scripts\train_hybrid_ppo.py"; RunDir = "runs\givesafe_ppo_formal" },
    @{ Name = "sac"; Script = "scripts\train_hybrid_sac.py"; RunDir = "runs\givesafe_sac_formal" }
)) {
    $log = "runs\local_logs\$($job.Name)_formal.log"
    Write-Host "==== $($job.Name) formal ====" -ForegroundColor Cyan
    & $Python $job.Script --mode formal --annual-eval --run-dir $job.RunDir 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { throw "$($job.Name) failed" }
}
Write-Host "PPO + SAC formal completed." -ForegroundColor Green
