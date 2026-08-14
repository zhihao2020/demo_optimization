<#
.SYNOPSIS
  Re-index optimal_demo qmd collections (and optionally refresh embeddings).

.EXAMPLE
  powershell -File scripts\qmd\reindex.ps1
  powershell -File scripts\qmd\reindex.ps1 -Embed
#>
param(
    [switch]$Embed,
    [switch]$ForceEmbed
)

$ErrorActionPreference = "Stop"

function Assert-Qmd {
    $cmd = Get-Command qmd -ErrorAction SilentlyContinue
    if (-not $cmd) {
        throw "qmd not found on PATH. Install: npm install -g @tobilu/qmd"
    }
}

Assert-Qmd

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$IndexYml = Join-Path $RepoRoot ".qmd\index.yml"

if (-not (Test-Path -LiteralPath $IndexYml)) {
    Write-Host "No .qmd/index.yml — running setup_collections.ps1 first"
    & powershell -NoProfile -File (Join-Path $PSScriptRoot "setup_collections.ps1") -SkipUpdate
}

Push-Location $RepoRoot
try {
    Write-Host "qmd update..."
    & qmd update
    if ($LASTEXITCODE -ne 0) { throw "qmd update failed" }

    if ($Embed -or $ForceEmbed) {
        Write-Host "qmd embed..."
        $embedArgs = @()
        if ($ForceEmbed) { $embedArgs += "-f" }
        & qmd embed @embedArgs
        if ($LASTEXITCODE -ne 0) { throw "qmd embed failed" }
    }

    & qmd status
}
finally {
    Pop-Location
}
