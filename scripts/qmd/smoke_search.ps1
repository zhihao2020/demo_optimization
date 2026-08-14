<#
.SYNOPSIS
  Smoke-test optimal_demo qmd collections against acceptance queries.

.EXAMPLE
  powershell -File scripts\qmd\smoke_search.ps1
#>
$ErrorActionPreference = "Stop"

if (-not (Get-Command qmd -ErrorAction SilentlyContinue)) {
    throw "qmd not found. Install: npm install -g @tobilu/qmd"
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
if (-not (Test-Path (Join-Path $RepoRoot ".qmd\index.yml"))) {
    throw "Missing .qmd index. Run: powershell -File scripts\qmd\setup_collections.ps1"
}

Push-Location $RepoRoot
try {
    # Use --format files to avoid console encoding corrupting JSON titles on Windows.
    $cases = @(
        @{
            Name   = "Q1-seasonal"
            Query  = "HMSD GHTD3 seasonal train"
            Coll   = @("docs-algo", "docs-protocol", "readme", "docs-all")
            Expect = @("cui_seasonal", "README", "GHTD3")
        },
        @{
            Name   = "Q2-CAES-GiveSafe"
            Query  = "CAES continuous GiveSafe"
            Coll   = @("docs-algo", "readme", "docs-all")
            Expect = @("GHTD3", "Safe_Market", "README", "principle")
        },
        @{
            Name   = "Q3-action"
            Query  = "u_caes"
            Coll   = @("readme", "docs-env", "docs-algo", "docs-all")
            Expect = @("README", "FMU", "GHTD3", "caes", "分层")
        },
        @{
            Name   = "Q4-goal"
            Query  = "her_mix goal_conditioned"
            Coll   = @("docs-algo", "docs-all", "readme")
            Expect = @("GHTD3", "Safe_Market", "principle", "cui_seasonal", "README")
        },
        @{
            Name   = "Q5-FMU-bounds"
            Query  = "FMU u_tp u_battery u_caes"
            Coll   = @("docs-env", "docs-all", "readme")
            Expect = @("FMU", "README", "data_dictionary", "economic")
        }
    )

    $failed = 0
    foreach ($case in $cases) {
        $qmdArgs = @("search", $case.Query, "--format", "files", "-n", "12")
        foreach ($c in $case.Coll) { $qmdArgs += @("-c", $c) }
        $hits = @(& qmd @qmdArgs 2>$null | Where-Object { $_ -and $_.Trim() -ne "" })
        $ok = $false
        foreach ($exp in $case.Expect) {
            if ($hits | Where-Object { $_ -match [regex]::Escape($exp) }) {
                $ok = $true
                break
            }
        }
        if ($ok) {
            Write-Host ("[PASS] {0}  query={1}" -f $case.Name, $case.Query)
        } else {
            Write-Host ("[FAIL] {0}  query={1}" -f $case.Name, $case.Query)
            $failed++
        }
        if ($hits.Count -gt 0) {
            Write-Host ("        hits: {0}" -f ($hits -join ", "))
        } else {
            Write-Host "        hits: (none)"
        }
    }

    if ($failed -gt 0) {
        throw ("smoke_search failed: {0} case(s)" -f $failed)
    }
    Write-Host "All smoke cases passed."
}
finally {
    Pop-Location
}
