$ErrorActionPreference = 'Continue'
$root = 'D:\xuzh\demo_optimization'
Set-Location $root
$bats = @(
  'D:\xuzh\demo_optimization\logs\run_tou2026_winter_fs_hsac_s0.bat',
  'D:\xuzh\demo_optimization\logs\run_tou2026_transition_fs_hsac_s0.bat',
  'D:\xuzh\demo_optimization\logs\run_tou2026_summer_fs_hsac_s0.bat'
)
foreach ($bat in $bats) {
  if (-not (Test-Path $bat)) { Write-Output ("MISSING " + $bat); continue }
  $p = Start-Process -FilePath 'cmd.exe' -ArgumentList @('/c', $bat) -WorkingDirectory $root -WindowStyle Hidden -PassThru
  Write-Output ("STARTED id=" + $p.Id + " bat=" + $bat)
}
