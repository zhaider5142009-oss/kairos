"""Run smoke_ui.py capturing output to a UTF-8 file, then print key lines."""
import subprocess, sys, os
env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
env["PYTHONUTF8"] = "1"
env["PYTHONPATH"] = r"C:/Users/Haider/jarvis"
env["JARVIS_ROOT"] = r"C:/Users/Haider/jarvis"
out_path = r"C:/Users/Haider/jarvis/_smoke_output.txt"
with open(out_path, "wb") as f:
    r = subprocess.run(
        [sys.executable, r"C:/Users/Haider/jarvis/smoke_ui.py"],
        stdout=f, stderr=subprocess.STDOUT, env=env, timeout=90,
    )
print(f"Exit code: {r.returncode}")
print(f"Output saved to: {out_path}")
print(f"Size: {os.path.getsize(out_path)} bytes")
