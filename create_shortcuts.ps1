# =============================================================================
# Jarvis - desktop & start-menu shortcut installer
# =============================================================================
# Creates:
#   Desktop\Jarvis Reborn.lnk        -> visible console + auto-open dashboard
#                                      (wscript run_jarvis_visible.vbs)
#   Desktop\Jarvis Reborn UI.lnk     -> opens the web dashboard in your browser
#   Start Menu\Jarvis Reborn\...     -> same two shortcuts (optional)
#
# Run:
#   powershell -ExecutionPolicy Bypass -File .\create_shortcuts.ps1
# Re-running is safe (overwrites in place).
# =============================================================================

$ErrorActionPreference = "Stop"

$scriptDir = $PSScriptRoot
$vbsLauncher  = Join-Path $scriptDir "run_jarvis_visible.vbs"
$iconFile     = Join-Path $scriptDir "jarvis.ico"
$uiUrl        = "http://127.0.0.1:8765"

if (-not (Test-Path $vbsLauncher)) {
    Write-Host "[ERROR] Launcher not found: $vbsLauncher" -ForegroundColor Red
    exit 1
}

$ws = New-Object -ComObject WScript.Shell
$desktop = [Environment]::GetFolderPath("Desktop")

$icon = if (Test-Path $iconFile) { "`"$iconFile`"" } else { "" }

function New-JarvisShortcut {
    param(
        [string]$Path,
        [string]$Target,
        [string]$Arguments = "",
        [string]$Icon
    )
    $lnk = $ws.CreateShortcut($Path)
    $lnk.TargetPath = $Target
    if ($Arguments) { $lnk.Arguments = $Arguments }
    $lnk.WorkingDirectory = $scriptDir
    if ($Icon) { $lnk.IconLocation = $Icon }
    if (-not $Arguments) { $lnk.Description = "Jarvis Reborn - MT5 algo trader dashboard ($uiUrl)" }
    $lnk.Save()
}

# ---- Desktop -----------------------------------------------------------------
$startLnk = Join-Path $desktop "Jarvis Reborn.lnk"
$uiLnk    = Join-Path $desktop "Jarvis Reborn UI.lnk"

New-JarvisShortcut `
    -Path $startLnk `
    -Target "$env:SystemRoot\System32\wscript.exe" `
    -Arguments "`"$vbsLauncher`"" `
    -Icon $icon
Write-Host "[OK] $startLnk" -ForegroundColor Green

New-JarvisShortcut `
    -Path $uiLnk `
    -Target $uiUrl `
    -Icon $icon
Write-Host "[OK] $uiLnk" -ForegroundColor Green

# ---- Start menu (grouped) ----------------------------------------------------
$startMenuDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Jarvis Reborn"
New-Item -ItemType Directory -Force -Path $startMenuDir | Out-Null

New-JarvisShortcut `
    -Path (Join-Path $startMenuDir "Start Jarvis.lnk") `
    -Target "$env:SystemRoot\System32\wscript.exe" `
    -Arguments "`"$vbsLauncher`"" `
    -Icon $icon
New-JarvisShortcut `
    -Path (Join-Path $startMenuDir "Dashboard.lnk") `
    -Target $uiUrl `
    -Icon $icon
Write-Host "[OK] Start-menu shortcuts created." -ForegroundColor Green

Write-Host ""
Write-Host "Done. Double-click 'Jarvis Reborn' on your desktop to start Jarvis in a visible console."
Write-Host "The dashboard opens automatically once Jarvis is healthy. Open 'Jarvis Reborn UI' manually anytime."