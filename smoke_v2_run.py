"""Run smoke_v2.py capturing output to UTF-8 file."""
import subprocess, sys, os
env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
env["PYTHONUTF8"] = "1"
env["PYTHONPATH"] = r"C:/Users/Haider/jarvis"
env["JARVIS_ROOT"] = r"C:/Users/Haider/jarvis"
env["OPENROUTER_API_KEYS"] = "sk-test-fake"
out_path = r"C:/Users/Haider/jarvis/_smoke_v2.txt"
with open(out_path, "wb") as f:
    r = subprocess.run(
        [sys.executable, r"C:/Users/Haider/jarvis/smoke_v2.py"],
        stdout=f, stderr=subprocess.STDOUT, env=env, timeout=90,
    )
print(f"Exit: {r.returncode}, Size: {os.path.getsize(out_path)} bytes")
