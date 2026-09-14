' =============================================================================
' Jarvis - hidden launcher (no console window)
' =============================================================================
' Runs run_jarvis.bat with intWindowStyle=0 (hidden) and bWaitOnReturn=False
' so this script returns immediately and the batch runs detached.
'
' Use this from the Windows Scheduled Task so Jarvis runs invisibly after
' logon. Output goes to %USERPROFILE%\jarvis\logs\jarvis.log via the
' RotatingFileHandler inside jarvis_predict.
' =============================================================================

Option Explicit
Dim shell, fso, scriptDir, batPath

Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
batPath = scriptDir & "\run_jarvis.bat"

If Not fso.FileExists(batPath) Then
    WScript.Echo "Jarvis launcher not found: " & batPath
    WScript.Quit 1
End If

' intWindowStyle=0 -> hide window. bWaitOnReturn=False -> return immediately.
shell.Run "cmd.exe /c " & """" & batPath & """", 0, False

WScript.Quit 0