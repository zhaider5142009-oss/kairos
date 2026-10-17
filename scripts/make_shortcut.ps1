# Creates a KAIROS desktop shortcut pointing at launch_kairos.bat next to this repo.
$root = Split-Path -Parent $PSScriptRoot
$bat  = Join-Path $root 'launch_kairos.bat'
if (-not (Test-Path $bat)) { Write-Error "launch_kairos.bat not found in $root"; exit 1 }

$ws = New-Object -ComObject WScript.Shell
$desktop = [Environment]::GetFolderPath('Desktop')
$lnk = $ws.CreateShortcut((Join-Path $desktop 'KAIROS.lnk'))
$lnk.TargetPath = $bat
$lnk.WorkingDirectory = $root
$lnk.Description = 'KAIROS - market intelligence control tower'
$lnk.WindowStyle = 1
$lnk.Save()
Write-Output "shortcut created on Desktop -> $bat"
