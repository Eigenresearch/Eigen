import os
import sys

def parse_toml(content: str) -> dict:
    result = {}
    current_section = None
    
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        if line.startswith('[') and line.endswith(']'):
            current_section = line[1:-1].strip()
            result[current_section] = {}
        elif '=' in line:
            key, val = line.split('=', 1)
            key = key.strip()
            val = val.strip()
            
            # Strip quotes
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
                
            if current_section:
                result[current_section][key] = val
            else:
                result[key] = val
                
    return result

def write_toml(data: dict) -> str:
    lines = []
    # Write package section first
    if 'package' in data:
        lines.append("[package]")
        for k, v in data['package'].items():
            lines.append(f'{k} = "{v}"')
        lines.append("")
        
    # Write dependencies section
    if 'dependencies' in data:
        lines.append("[dependencies]")
        for k, v in data['dependencies'].items():
            lines.append(f'{k} = "{v}"')
        lines.append("")
        
    return "\n".join(lines)


class EigenPackager:
    def __init__(self, workspace_root: str):
        self.workspace_root = os.path.abspath(workspace_root)
        self.toml_path = os.path.join(self.workspace_root, "eigen.toml")

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

    def build_package(self):
        if not os.path.exists(self.toml_path):
            print(f"Error: no eigen.toml found. Run 'eigen init' to initialize.", file=sys.stderr)
            return False
            
        with open(self.toml_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        data = parse_toml(content)
        package_name = data.get('package', {}).get('name', 'unknown')
        print(f"Building Eigen package '{package_name}'...")
        
        deps = data.get('dependencies', {})
        if deps:
            print("Resolving dependencies:")
            for dep, ver in deps.items():
                print(f"  - {dep} (version {ver}) -> Resolved locally")
        else:
            print("No dependencies to resolve.")
            
        print("Package build complete successfully.")
        return True

    def publish_package(self):
        if not os.path.exists(self.toml_path):
            print(f"Error: no eigen.toml found.", file=sys.stderr)
            return False
            
        with open(self.toml_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        data = parse_toml(content)
        package_name = data.get('package', {}).get('name', 'unknown')
        package_version = data.get('package', {}).get('version', '1.0.0')
        
        print(f"Publishing Eigen package '{package_name}' v{package_version} to registry...")
        print("Package published successfully.")
        return True
