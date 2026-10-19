' KAIROS daemon launcher — runs the watchdog loop with zero console windows.
Dim sh
Set sh = CreateObject("WScript.Shell")
sh.Run """C:\Users\Haider\Documents\Qoder\2026-10-16\f0bca35f\tools\kairos-daemon.bat""", 0, False
