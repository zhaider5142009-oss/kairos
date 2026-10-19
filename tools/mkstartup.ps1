# Puts a hidden "KAIROS Daemon" auto-start entry into the Windows Startup folder.
$ErrorActionPreference = "Stop"
$startup  = [Environment]::GetFolderPath('Startup')
$target   = "C:\Users\Haider\Documents\Qoder\2026-10-16\f0bca35f\tools\kairos-daemon.vbs"
$icon     = "C:\Users\Haider\Documents\Qoder\2026-10-16\f0bca35f\tools\kairos.ico"
$lnkPath  = Join-Path $startup "KAIROS Daemon.lnk"
$sc = New-Object -ComObject WScript.Shell
$lnk = $sc.CreateShortcut($lnkPath)
$lnk.TargetPath = "C:\Windows\System32\wscript.exe"
$lnk.Arguments  = """$target"""
$lnk.IconLocation = $icon
$lnk.WorkingDirectory = "C:\Users\Haider\Documents\Qoder\2026-10-16\f0bca35f\tools"
$lnk.Description = "KAIROS always-on watchdog (control tower + autopilot auto-restart)"
$lnk.Save()
Write-Output "DAEMON_AUTOSTART_OK $lnkPath"
