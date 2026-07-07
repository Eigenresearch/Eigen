"""Environment fingerprint — captured for Appendix D."""
import sys
import platform
import os
import subprocess

lines = []
lines.append("=== ENVIRONMENT FINGERPRINT ===")
lines.append(f"OS: {platform.platform()}")
lines.append(f"Machine: {platform.machine()}")
lines.append(f"Processor: {platform.processor()}")
lines.append(f"Python: {sys.version}")
lines.append(f"Python executable: {sys.executable}")

# CPU info (Windows)
try:
    import psutil
    lines.append(f"CPU cores (physical): {psutil.cpu_count(logical=False)}")
    lines.append(f"CPU cores (logical): {psutil.cpu_count(logical=True)}")
    lines.append(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
except ImportError:
    lines.append(f"CPU cores (env): {os.environ.get('NUMBER_OF_PROCESSORS', 'unknown')}")

# Key libraries
try:
    import numpy as np
    lines.append(f"numpy: {np.__version__}")
except ImportError:
    lines.append("numpy: not installed")

try:
    import matplotlib
    lines.append(f"matplotlib: {matplotlib.__version__}")
except ImportError:
    lines.append("matplotlib: not installed")

try:
    import eigen_native
    lines.append("eigen_native: available")
except ImportError:
    lines.append("eigen_native: not available")

# Eigen version
try:
    with open("setup.py", "r") as f:
        for line in f:
            if "version" in line:
                lines.append(f"Eigen setup.py: {line.strip()}")
                break
except Exception:
    pass

output = "\n".join(lines)
print(output)
with open("appendix_logs/env_fingerprint.txt", "w") as f:
    f.write(output)
