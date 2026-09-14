# =============================================================================
# Jarvis — Windows Scheduled Task uninstaller
# =============================================================================
# Removes the task created by install_task.ps1.
# Does NOT touch the project files, the venv, or the MT5 terminal.
# =============================================================================

$ErrorActionPreference = "Stop"
$taskName = "JarvisPredictor"

$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if (-not $existing) {
    Write-Host "[INFO] No task named '$taskName' was registered. Nothing to do." -ForegroundColor Yellow
    exit 0
}

try {
    Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}
catch {
    Write-Host "[ERROR] Failed to remove task: $_" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Task '$taskName' removed." -ForegroundColor Green
