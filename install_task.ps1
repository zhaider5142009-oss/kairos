# =============================================================================
# Jarvis — Windows Scheduled Task installer
# =============================================================================
# Registers a "never-dies" task that:
#   - Starts at user logon (no admin required).
#   - Runs hidden via run_jarvis.vbs (no console window).
#   - Restarts every 1 minute if it crashes, up to 3 times per failure window.
#   - Allows start on battery, never stops on idle end, runs indefinitely.
#   - Ignores duplicate starts (single-instance is enforced via the
#     http listener + signal handlers).
#
# Run from PowerShell:
#   powershell -ExecutionPolicy Bypass -File .\install_task.ps1
# Uninstall:
#   powershell -ExecutionPolicy Bypass -File .\uninstall_task.ps1
# =============================================================================

$ErrorActionPreference = "Stop"

$taskName   = "JarvisPredictor"
$scriptDir  = $PSScriptRoot
$vbsLauncher = Join-Path $scriptDir "run_jarvis.vbs"
$description = "Jarvis predictive trader - TimesFM + MT5, autostart at logon, self-healing."

if (-not (Test-Path $vbsLauncher)) {
    Write-Host "[ERROR] Launcher not found: $vbsLauncher" -ForegroundColor Red
    exit 1
}

# Remove any prior copy so re-running this script is safe.
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "[INFO] Removing existing task '$taskName' before recreating…" -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

# ---- Trigger: at logon ------------------------------------------------------
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

# ---- Action: hidden launch via wscript.exe -----------------------------------
$action = New-ScheduledTaskAction `
    -Execute "wscript.exe" `
    -Argument "`"$vbsLauncher`""

# ---- Settings: restart on failure, run forever, no battery restrictions -----
$restartInterval = New-TimeSpan -Minutes 1
$execLimit       = New-TimeSpan -Seconds 0  # 0 = unlimited
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -DontStopOnIdleEnd `
    -RestartCount 3 `
    -RestartInterval $restartInterval `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit $execLimit `
    -Priority 5

# ---- Register ---------------------------------------------------------------
try {
    Register-ScheduledTask `
        -TaskName $taskName `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Description $description `
        -Force | Out-Null
}
catch {
    Write-Host "[ERROR] Failed to register task: $_" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Task '$taskName' registered. Jarvis will start at your next logon." -ForegroundColor Green
Write-Host ""
Write-Host "Manage it:"
Write-Host "  Open Task Scheduler         : taskschd.msc"
Write-Host "  Start now (hidden)          : Start-ScheduledTask -TaskName '$taskName'"
Write-Host "  Stop                        : Stop-ScheduledTask  -TaskName '$taskName'"
Write-Host "  Remove                      : powershell -File .\\uninstall_task.ps1"
Write-Host "  View logs                   : Get-Content '$scriptDir\logs\jarvis.log' -Tail 50 -Wait"
