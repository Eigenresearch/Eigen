import re
import sys
import os

def get_pyproject_version():
    with open("pyproject.toml", "r") as f:
        content = f.read()
    match = re.search(r'version\s*=\s*"([^"]+)"', content)
    return match.group(1) if match else None

def get_cargo_version():
    path = "native/rust/Cargo.toml"
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        content = f.read()
    match = re.search(r'version\s*=\s*"([^"]+)"', content)
    return match.group(1) if match else None

def main():
    py_version = get_pyproject_version()
    cargo_version = get_cargo_version()
    
    print(f"pyproject.toml version: {py_version}")
    
    errors = []
    if not py_version:
        errors.append("Could not find version in pyproject.toml")
        
    if cargo_version and cargo_version != py_version:
        errors.append(f"Version mismatch: pyproject.toml ({py_version}) != Cargo.toml ({cargo_version})")
    
    if errors:
        for err in errors:
            print(f"ERROR: {err}")
        sys.exit(1)
    
    print("Version consistency check passed.")

if __name__ == "__main__":
    main()
