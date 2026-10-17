$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut('C:\Users\Haider\Desktop\KAIROS Desk.lnk')
$sc.TargetPath = 'C:\Windows\System32\wscript.exe'
$sc.Arguments = '"C:\Users\Haider\Documents\Qoder\2026-10-16\f0bca35f\tools\kairos-launch.vbs"'
$sc.IconLocation = 'C:\Users\Haider\Documents\Qoder\2026-10-16\f0bca35f\tools\kairos.ico'
$sc.Description = 'KAIROS quant desk — starts the server and opens the dashboard'
$sc.Save()
Write-Output SHORTCUT_OK
