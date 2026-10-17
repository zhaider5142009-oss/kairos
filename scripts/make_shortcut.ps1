$ws = New-Object -ComObject WScript.Shell
$desktop = [Environment]::GetFolderPath('Desktop')
$lnk = $ws.CreateShortcut("$desktop\KAIROS.lnk")
$lnk.TargetPath = 'C:\Users\Haider\Documents\Qoder\2026-10-16\f0bca35f\launch_kairos.bat'
$lnk.WorkingDirectory = 'C:\Users\Haider\Documents\Qoder\2026-10-16\f0bca35f'
$lnk.Description = 'KAIROS - market intelligence control tower'
$lnk.WindowStyle = 1
$lnk.Save()
Write-Output 'shortcut created'
