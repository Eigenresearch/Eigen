import os
import sys
import json
import hashlib
import tarfile

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

def parse_toml(content: str) -> dict:
    if tomllib is not None:
        return tomllib.loads(content)
    raise ImportError(
        "TOML parser (tomllib or tomli) is required but not found. "
        "Install `tomli` or upgrade to Python 3.11+ for TOML support."
    )

def _format_toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, list):
        return "[" + ", ".join(_format_toml_value(x) for x in v) + "]"
    return f'"{v}"'


def _write_section(section_name: str, section_data: dict, lines: list) -> None:
    lines.append(f"[{section_name}]")
    for k, v in section_data.items():
        if isinstance(v, dict):
            _write_section(f"{section_name}.{k}", v, lines)
        else:
            lines.append(f"{k} = {_format_toml_value(v)}")
    lines.append("")


def write_toml(data: dict) -> str:
    lines = []
    for k, v in data.items():
        if not isinstance(v, dict):
            lines.append(f"{k} = {_format_toml_value(v)}")
    if lines:
        lines.append("")
    for k, v in data.items():
        if isinstance(v, dict):
            _write_section(k, v, lines)
    return "\n".join(lines)


def parse_version(v_str: str) -> tuple[int, int, int]:
    if v_str.startswith('v'):
        v_str = v_str[1:]
    parts = []
    for part in v_str.split('.'):
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])

def version_satisfies(constraint: str, version_str: str) -> bool:
    if constraint == '*' or constraint == '':
        return True
    if 'x' in constraint or '*' in constraint:
        normalized = constraint.replace('x', '0').replace('*', '0')
        parts = constraint.split('.')
        if len(parts) == 1 or parts[1] in ('x', '*'):
            return version_satisfies('^' + normalized, version_str)
        else:
            return version_satisfies('~' + normalized, version_str)
            
    v = parse_version(version_str)
    
    if constraint.startswith('^'):
        c_ver = parse_version(constraint[1:])
        if v < c_ver:
            return False
        if c_ver[0] > 0:
            return v[0] == c_ver[0]
        elif c_ver[1] > 0:
            return v[0] == 0 and v[1] == c_ver[1]
        else:
            return v[0] == 0 and v[1] == 0 and v[2] == c_ver[2]
            
    elif constraint.startswith('~'):
        c_ver = parse_version(constraint[1:])
        raw_c = constraint[1:].strip().split('.')
        if len(raw_c) >= 2:
            return v >= c_ver and v[0] == c_ver[0] and v[1] == c_ver[1]
        else:
            return v >= c_ver and v[0] == c_ver[0]
            
    else:
        c_ver = parse_version(constraint)
        return v == c_ver


class EigenPackager:
    def __init__(self, workspace_root: str):
        self.workspace_root = os.path.abspath(workspace_root)
        self.toml_path = os.path.join(self.workspace_root, "eigen.toml")
        self.lock_path = os.path.join(self.workspace_root, "eigen.lock")
        self.registry_dir = os.path.join(self.workspace_root, "registry")
        os.makedirs(self.registry_dir, exist_ok=True)

    def init_package(self, name: str | None = None):
        if os.path.exists(self.toml_path):
            print(f"Error: package manifest already exists at {self.toml_path}", file=sys.stderr)
            return False
            
        if name is None:
            name = os.path.basename(self.workspace_root)
            
        default_data = {
            'package': {
                'name': name,
                'version': "1.0.0"
            },
            'dependencies': {}
        }
        
        with open(self.toml_path, 'w', encoding='utf-8') as f:
            f.write(write_toml(default_data))
            
        print(f"Initialized new Eigen package '{name}' at {self.toml_path}")

        # Create standard workspace src/main.eig
        src_dir = os.path.join(self.workspace_root, "src")
        os.makedirs(src_dir, exist_ok=True)
        main_eig_path = os.path.join(src_dir, "main.eig")
        if not os.path.exists(main_eig_path):
            template_code = """eigen 1.0

# Template main entrypoint for Eigen package
func main() -> int {
    print "Hello from Eigen!"
    return 0
}

let result: int = main()
assert result == 0
"""
            with open(main_eig_path, 'w', encoding='utf-8') as f:
                f.write(template_code)
            print(f"Created template entrypoint at {main_eig_path}")
            
        return True

    def add_dependency(self, dep_name: str, version: str = "0.1.0"):
        if not os.path.exists(self.toml_path):
            print(f"Error: no eigen.toml found. Run 'eigen init' first.", file=sys.stderr)
            return False
            
        with open(self.toml_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        data = parse_toml(content)
        if 'dependencies' not in data:
            data['dependencies'] = {}
            
        data['dependencies'][dep_name] = version
        
        with open(self.toml_path, 'w', encoding='utf-8') as f:
            f.write(write_toml(data))
            
        print(f"Added dependency '{dep_name}' = '{version}' to {self.toml_path}")
        return True

    def install_dependencies(self):
        if not os.path.exists(self.toml_path):
            print(f"Error: no eigen.toml found.", file=sys.stderr)
            return False

        with open(self.toml_path, 'r', encoding='utf-8') as f:
            content = f.read()
        data = parse_toml(content)
        deps = data.get('dependencies', {})

        # Load existing lockfile if it exists
        locked_deps = {}
        if os.path.exists(self.lock_path):
            with open(self.lock_path, 'r', encoding='utf-8') as f:
                try:
                    locked_deps = json.load(f)
                except Exception:
                    pass

        new_lock = {}
        print("Resolving and locking dependencies...")

        for dep, ver in deps.items():
            resolved_ver = None
            sha256_hash = None
            use_locked = False

            # If locked, check compatibility and hash matches
            if dep in locked_deps:
                locked_ver = locked_deps[dep].get("version")
                locked_hash = locked_deps[dep].get("hash")
                
                if version_satisfies(ver, locked_ver):
                    # Verify if the registry file exists and its hash matches, OR check fallback hash
                    expected_hash = None
                    locked_reg_path = os.path.join(self.registry_dir, f"{dep}-{locked_ver}.tar")
                    if os.path.exists(locked_reg_path):
                        with open(locked_reg_path, 'rb') as f:
                            expected_hash = hashlib.sha256(f.read()).hexdigest()
                    else:
                        expected_hash = hashlib.sha256(dep.encode('utf-8') + locked_ver.encode('utf-8')).hexdigest()
                        
                    if locked_hash == expected_hash:
                        print(f"  Using locked version for '{dep}' ({locked_ver}) [hash matches]")
                        resolved_ver = locked_ver
                        sha256_hash = locked_hash
                        use_locked = True
                    else:
                        print(f"  WARNING: Locked hash for '{dep}' changed! Re-resolving.")
                else:
                    print(
                        f"  WARNING: Locked version '{locked_ver}' is incompatible with constraint '{ver}'. "
                        "Re-resolving."
                    )

            if not use_locked:
                # Dynamic resolution: scan registry_dir for dep-*.tar
                available_versions = []
                if os.path.exists(self.registry_dir):
                    for filename in os.listdir(self.registry_dir):
                        if filename.startswith(f"{dep}-") and filename.endswith(".tar"):
                            ver_part = filename[len(dep)+1 : -4]
                            available_versions.append(ver_part)

                matching_versions = [v for v in available_versions if version_satisfies(ver, v)]
                if matching_versions:
                    resolved_ver = max(matching_versions, key=parse_version)
                    dep_reg_path = os.path.join(self.registry_dir, f"{dep}-{resolved_ver}.tar")
                    with open(dep_reg_path, 'rb') as f:
                        sha256_hash = hashlib.sha256(f.read()).hexdigest()
                    print(f"  Resolved '{dep}' constraint '{ver}' to version {resolved_ver}")
                else:
                    # Fallback to constraint base
                    base_ver = ver
                    if base_ver.startswith('^') or base_ver.startswith('~'):
                        base_ver = base_ver[1:]
                    resolved_ver = base_ver
                    sha256_hash = hashlib.sha256(dep.encode('utf-8') + resolved_ver.encode('utf-8')).hexdigest()
                    print(
                        f"  No matching versions found in registry for '{dep}' (constraint '{ver}'). "
                        f"Using fallback version {resolved_ver}"
                    )

            new_lock[dep] = {
                "version": resolved_ver,
                "hash": sha256_hash,
                "resolved": f"https://registry.eigen-lang.org/packages/{dep}/{resolved_ver}"
            }
            
            # Create a mock packages folder for resolved packages
            pkgs_dir = os.path.join(self.workspace_root, ".eigen_packages", dep)
            os.makedirs(pkgs_dir, exist_ok=True)
            with open(os.path.join(pkgs_dir, "manifest.json"), "w") as f:
                json.dump({"name": dep, "version": resolved_ver, "hash": sha256_hash}, f)

        with open(self.lock_path, 'w', encoding='utf-8') as f:
            json.dump(new_lock, f, indent=2)

        print(f"Lockfile generated at {self.lock_path}")
        return True

    def publish_package(self):
        if not os.path.exists(self.toml_path):
            print(f"Error: no eigen.toml found.", file=sys.stderr)
            return False
            
        with open(self.toml_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        data = parse_toml(content)
        pkg_name = data.get('package', {}).get('name', 'unknown')
        pkg_ver = data.get('package', {}).get('version', '1.0.0')
        
        print(f"Publishing package '{pkg_name}' v{pkg_ver} to registry.eigen-lang.org...")

        # Write to local registry dir to simulate registry upload
        tar_path = os.path.join(self.registry_dir, f"{pkg_name}-{pkg_ver}.tar")
        os.makedirs(self.registry_dir, exist_ok=True)

        # Build a real tar archive from the workspace files so downstream
        # tooling (verify/extract/install) can treat the artifact as a
        # genuine tarball rather than a plain-text stub.
        with tarfile.open(tar_path, "w") as tar:
            if os.path.isfile(self.toml_path):
                tar.add(self.toml_path, arcname="eigen.toml")
            src_dir = os.path.join(self.workspace_root, "src")
            if os.path.isdir(src_dir):
                tar.add(src_dir, arcname="src")
            readme_path = os.path.join(self.workspace_root, "README.md")
            if os.path.isfile(readme_path):
                tar.add(readme_path, arcname="README.md")

        # P3 §11.1: ALSO register metadata in the structured
        # PackageRegistry (see `src/registry.py`) so that semver
        # resolve, conflict detection, checksum verification,
        # signature, and vulnerability scanning are available for
        # the published artifact. We use the same `registry_dir`
        # as the registry root, so the index.json + tarballs/ live
        # alongside the legacy flat .tar file — both stay intact.
        try:
            from src.registry import PackageRegistry
            regs = PackageRegistry(self.registry_dir)
            deps = data.get('dependencies', {})
            with open(tar_path, "rb") as tf:
                tar_bytes = tf.read()
            regs.add(
                pkg_name, pkg_ver,
                tarball_bytes=tar_bytes,
                dependencies=dict(deps) if deps else {},
            )
        except Exception:
            # Registry indexing is best-effort — the legacy
            # publish flow must still succeed for users that run
            # `eigen publish` outside a workspace.
            pass
            
        print(f"Package published successfully. Target: {tar_path}")
        return True

    def search_packages(self, query: str):
        print(f"Searching registry.eigen-lang.org for '{query}':")
        # List simulated packages in registry folder
        files = os.listdir(self.registry_dir)
        matches = []
        for f in files:
            if f.endswith('.tar') and query in f:
                name_parts = f[:-4].rsplit('-', 1)
                if len(name_parts) == 2:
                    matches.append((name_parts[0], name_parts[1]))
                
        if matches:
            for name, ver in matches:
                print(f"  - {name} (version {ver})")
        else:
            print("  No packages matched your query.")
            
    def build_package(self):
        if not os.path.exists(self.toml_path):
            print(f"Error: no eigen.toml found. Run 'eigen init' to initialize.", file=sys.stderr)
            return False
            
        with open(self.toml_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        data = parse_toml(content)
        package_name = data.get('package', {}).get('name', 'unknown')
        print(f"Building Eigen package '{package_name}'...")
        
        # Perform dependency install check
        self.install_dependencies()
            
        print("Package build complete successfully.")
        return True
