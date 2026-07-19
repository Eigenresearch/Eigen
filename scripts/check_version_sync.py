#!/usr/bin/env python3
"""Check that version strings are synchronized across all project files."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def get_pyproject_version():
    content = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    return m.group(1) if m else None

def check_file(path, pattern, expected):
    content = path.read_text(encoding="utf-8")
    m = re.search(pattern, content, re.MULTILINE)
    if not m:
        return f"Version not found in {path.relative_to(ROOT)}"
    if m.group(1) != expected:
        return f"Version mismatch in {path.relative_to(ROOT)}: {m.group(1)} != {expected}"
    return None

def main():
    version = get_pyproject_version()
    if not version:
        print("ERROR: Could not find version in pyproject.toml")
        return 1

    errors = []

    errors.append(check_file(ROOT / "src/release.py", r'CODENAME\s*=\s*"([^"]+)"', "Mars"))

    # Check Cargo.toml if exists
    cargo = ROOT / "native" / "rust" / "Cargo.toml"
    if cargo.exists():
        errors.append(check_file(cargo, r'^version\s*=\s*"([^"]+)"', version))

    for err in errors:
        if err:
            print(f"FAIL: {err}")
            return 1

    print(f"OK: version {version} (Mars) synchronized across all files")
    return 0

if __name__ == "__main__":
    sys.exit(main())
