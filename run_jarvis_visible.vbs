' =============================================================================
' Jarvis - visible launcher (console visible + auto-open dashboard)
' =============================================================================
' Runs run_jarvis.bat in a NEW visible console window (intWindowStyle=1), then
' spawns a small detached watchdog that polls /health and opens the dashboard
' in the default browser once Jarvis is ready (~20-40s).
'
' Use this from the desktop / start-menu shortcuts. For an invisible autostart
' at logon, keep run_jarvis.vbs (used by the scheduled task).
' =============================================================================

Option Explicit
Dim shell, fso, scriptDir, batPath, wdPath

Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
batPath = scriptDir & "\run_jarvis.bat"
wdPath  = scriptDir & "\watchdog_open_ui.ps1"

If Not fso.FileExists(batPath) Then
    WScript.Echo "Launcher not found: " & batPath
    WScript.Quit 1
End If

' intWindowStyle=1 -> normal visible console window. bWaitOnReturn=False.
shell.Run "cmd.exe /k " & """" & batPath & """", 1, False

' Detached watchdog: poll /health, open dashboard when ready.
If fso.FileExists(wdPath) Then
    shell.Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & wdPath & """", 0, False
End If

WScript.Quit 0