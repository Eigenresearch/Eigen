import sys
import os
import json
import platform
import subprocess
from src.cli import register_command

@register_command("reproduce")
def reproduce_command(args, workspace_root):
    # Try to get git commit hash
    git_commit = "unknown"
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=workspace_root, capture_output=True, text=True)
        if res.returncode == 0:
            git_commit = res.stdout.strip()
    except Exception:
        pass
        
    audit_data = {
        "eigen_version": "2.3.0",
        "git_commit": git_commit,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "backend": "auto",
        "seed": 42
    }
    
    print(json.dumps(audit_data, indent=2))
