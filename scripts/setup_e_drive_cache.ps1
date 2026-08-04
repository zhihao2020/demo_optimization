# Move optimal_demo compute cache to E: and junction D:\...\runs -> E:\optimal_demo_cache\runs
# Usage from repo root:
#   powershell -ExecutionPolicy Bypass -File scripts/setup_e_drive_cache.ps1

param(
    [string]$CacheRoot = "E:\optimal_demo_cache"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$LocalRuns = Join-Path $RepoRoot "runs"
$ERuns = Join-Path $CacheRoot "runs"
$ETmp = Join-Path $CacheRoot "tmp"
$EPy = Join-Path $CacheRoot "pycache"
$ETorch = Join-Path $CacheRoot "torch"
$EPip = Join-Path $CacheRoot "pip"
$EFmu = Join-Path $CacheRoot "fmu_work"
$ELogs = Join-Path $CacheRoot "logs"

foreach ($d in @($CacheRoot, $ERuns, $ETmp, $EPy, $ETorch, $EPip, $EFmu, $ELogs)) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}

Write-Host "CacheRoot=$CacheRoot"
Write-Host "RepoRoot=$RepoRoot"
Write-Host "LocalRuns=$LocalRuns"
Write-Host "ERuns=$ERuns"

function Test-IsJunction([string]$Path) {
    if (-not (Test-Path $Path)) { return $false }
    $item = Get-Item $Path -Force
    return [bool]($item.Attributes -band [IO.FileAttributes]::ReparsePoint)
}

if (Test-IsJunction $LocalRuns) {
    Write-Host "runs is already a junction"
    try { Write-Host "target=$($LocalRuns | Get-Item | Select-Object -ExpandProperty Target)" } catch {}
} elseif (Test-Path $LocalRuns) {
    Write-Host "Copying existing runs to E: (robocopy)..."
    & robocopy $LocalRuns $ERuns /E /COPY:DAT /R:2 /W:2 /NFL /NDL /NJH /NJS
    $rc = $LASTEXITCODE
    if ($rc -ge 8) { throw "robocopy failed code=$rc" }
    Write-Host "robocopy exit=$rc (0-7 ok)"

    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backupName = "runs_d_backup_$stamp"
    $backupPath = Join-Path $RepoRoot $backupName
    Write-Host "Rename local runs -> $backupName"
    Rename-Item -Path $LocalRuns -NewName $backupName

    Write-Host "Create junction..."
    $cmd = 'mklink /J "' + $LocalRuns + '" "' + $ERuns + '"'
    cmd /c $cmd
    if (-not (Test-IsJunction $LocalRuns)) { throw "junction create failed" }
    Write-Host "Junction OK: $LocalRuns -> $ERuns"
    Write-Host "Backup on D: $backupPath  (delete after verify to free D: space)"
} else {
    $cmd = 'mklink /J "' + $LocalRuns + '" "' + $ERuns + '"'
    cmd /c $cmd
    Write-Host "Created empty junction $LocalRuns -> $ERuns"
}

# User env (new shells)
[Environment]::SetEnvironmentVariable("OPTIMAL_DEMO_CACHE", $CacheRoot, "User")
[Environment]::SetEnvironmentVariable("OPTIMAL_DEMO_RUNS", $ERuns, "User")
[Environment]::SetEnvironmentVariable("OPTIMAL_DEMO_TMP", $ETmp, "User")
[Environment]::SetEnvironmentVariable("TEMP", $ETmp, "User")
[Environment]::SetEnvironmentVariable("TMP", $ETmp, "User")
[Environment]::SetEnvironmentVariable("PYTHONPYCACHEPREFIX", $EPy, "User")
[Environment]::SetEnvironmentVariable("TORCH_HOME", $ETorch, "User")
[Environment]::SetEnvironmentVariable("PIP_CACHE_DIR", $EPip, "User")

# Current session
$env:OPTIMAL_DEMO_CACHE = $CacheRoot
$env:OPTIMAL_DEMO_RUNS = $ERuns
$env:OPTIMAL_DEMO_TMP = $ETmp
$env:TEMP = $ETmp
$env:TMP = $ETmp
$env:TMPDIR = $ETmp
$env:PYTHONPYCACHEPREFIX = $EPy
$env:TORCH_HOME = $ETorch
$env:PIP_CACHE_DIR = $EPip

$snippet = @(
    "# optimal_demo E: cache",
    "`$env:OPTIMAL_DEMO_CACHE = '$CacheRoot'",
    "`$env:OPTIMAL_DEMO_RUNS = '$ERuns'",
    "`$env:OPTIMAL_DEMO_TMP = '$ETmp'",
    "`$env:TEMP = '$ETmp'",
    "`$env:TMP = '$ETmp'",
    "`$env:PYTHONPYCACHEPREFIX = '$EPy'",
    "`$env:TORCH_HOME = '$ETorch'",
    "`$env:PIP_CACHE_DIR = '$EPip'"
) -join "`n"
$snippetPath = Join-Path $CacheRoot "session_env.ps1"
Set-Content -Path $snippetPath -Value $snippet -Encoding ASCII
Write-Host "Wrote $snippetPath"
Write-Host "Done."
