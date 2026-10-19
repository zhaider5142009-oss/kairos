# Restart the KAIROS daemon cleanly: kill stray daemon loops + server, relaunch.
$ErrorActionPreference = "SilentlyContinue"
Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" |
  Where-Object { $_.CommandLine -match 'kairos-daemon' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force; Write-Output "KILLED_CMD $($_.ProcessId)" }
$pid8600 = (Get-NetTCPConnection -LocalPort 8600 -State Listen).OwningProcess | Select-Object -First 1
if ($pid8600) { Stop-Process -Id $pid8600 -Force; Write-Output "KILLED_SERVER $pid8600" }
Start-Sleep -Seconds 2
$sh = New-Object -ComObject WScript.Shell
$sh.Run("C:\Users\Haider\Documents\Qoder\2026-10-16\f0bca35f\tools\kairos-daemon.bat", 0, $false)
Write-Output "DAEMON_RELAUNCHED"
