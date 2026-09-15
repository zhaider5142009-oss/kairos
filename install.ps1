# =============================================================================
# Jarvis Reborn - one-shot bootstrap (Windows).
# Run:  powershell -ExecutionPolicy Bypass -File install.ps1
#
# Creates a .venv, installs runtime deps, writes config.yaml from the
# example (never overwrites an existing one), and optionally registers the
# permanent launchers (shortcut + keepalive + logon task).
# =============================================================================
param(
    [switch]$SkipLaunchers,
    [string]$Python = "py",
    [switch]$WithGPU
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host "[1/4] Creating virtual environment (.venv) ..."
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & $Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "failed to create venv (is Python installed?)" }
}
$Py = Join-Path $Root ".venv\Scripts\python.exe"

Write-Host "[2/4] Installing runtime dependencies ..."
& $Py -m pip install --upgrade pip
& $Py -m pip install -r requirements.txt
if ($WithGPU -and $IsWindows -eq $false) {
    Write-Host "  (DirectML is Windows-only; skipping GPU backends on non-Windows.)"
} elseif ($WithGPU) {
    & $Py -m pip install -r requirements-optional.txt
}
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

Write-Host "[3/4] Writing config.yaml from example (keeps existing file) ..."
if (-not (Test-Path "config.yaml")) {
    Copy-Item "config.example.yaml" "config.yaml"
    Write-Host "  config.yaml created - EDIT IT: set listener.http.auth_token and OpenRouter keys."
} else {
    Write-Host "  config.yaml already exists - left untouched."
}

if ($SkipLaunchers) { Write-Host "[4/4] Skipped launchers (-SkipLaunchers)."; return }

Write-Host "[4/4] Registering permanent launchers ..."
# Desktop + Start Menu shortcuts (visible console + auto-open dashboard).
if (Test-Path "create_shortcuts.ps1") {
    & powershell -ExecutionPolicy Bypass -File "create_shortcuts.ps1"
}
# Keepalive scheduled task: health-checks every 5 min, restarts if dead.
& powershell -ExecutionPolicy Bypass -Command "
    schtasks /Create /SC MINUTE /MO 5 /TN 'JarvisKeepalive' /TR 'powershell -NoProfile -ExecutionPolicy Bypass -File \"$Root\keepalive.ps1\"' /F | Out-Null
"
# Logon task: auto-start (hidden) at sign-in.
if (Test-Path "install_task.ps1") {
    & powershell -ExecutionPolicy Bypass -File "install_task.ps1"
}
Write-Host "Done. Double-click 'Jarvis Reborn' on your Desktop to start."