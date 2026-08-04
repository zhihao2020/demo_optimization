# Source this before long training/benchmark jobs so TEMP and caches stay on E:
#   . .\scripts\with_e_cache.ps1
#   python scripts/run_full_benchmark.py ...

$CacheRoot = "E:\optimal_demo_cache"
if (Test-Path (Join-Path $CacheRoot "session_env.ps1")) {
    . (Join-Path $CacheRoot "session_env.ps1")
} else {
    $env:OPTIMAL_DEMO_CACHE = $CacheRoot
    $env:OPTIMAL_DEMO_RUNS = Join-Path $CacheRoot "runs"
    $env:OPTIMAL_DEMO_TMP = Join-Path $CacheRoot "tmp"
    $env:TEMP = $env:OPTIMAL_DEMO_TMP
    $env:TMP = $env:OPTIMAL_DEMO_TMP
    $env:PYTHONPYCACHEPREFIX = Join-Path $CacheRoot "pycache"
    $env:TORCH_HOME = Join-Path $CacheRoot "torch"
    $env:PIP_CACHE_DIR = Join-Path $CacheRoot "pip"
}
Write-Host "E-cache ready: TEMP=$env:TEMP RUNS=$env:OPTIMAL_DEMO_RUNS"
