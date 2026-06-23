import os
import sys
import hashlib
import json

from src.frontend.lexer import Lexer
from src.frontend.parser import Parser
from src.semantic.import_resolver import ImportResolver
from src.semantic.type_checker import TypeChecker, TypeErrorException
from src.ir.ir_converter import EQIRConverter
from src.ir.ir_graph import EQIRGraph
from src.backend.bytecode import Instruction

def get_workspace_root() -> str:
    return os.getcwd()

def get_project_hash(filepath: str, workspace_root: str) -> str:
    visited_files = set()
    files_to_process = [os.path.abspath(filepath)]
    stdlib_root = os.path.join(workspace_root, "stdlib")
    hasher = hashlib.sha256()
    processed_contents = []
    
    while files_to_process:
        current_path = files_to_process.pop(0)
        if current_path in visited_files:
            continue
        visited_files.add(current_path)
        
        if not os.path.isfile(current_path):
            continue
            
        with open(current_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        processed_contents.append((current_path, content))
        
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("import "):
                parts = line.split()
                if len(parts) >= 2:
                    module_name = parts[1]
                    relative_path = module_name.replace('.', '/') + ".eig"
                    local_path = os.path.join(workspace_root, relative_path)
                    if os.path.isfile(local_path):
                        files_to_process.append(os.path.abspath(local_path))
                    else:
                        stdlib_path = os.path.join(stdlib_root, relative_path)
                        if os.path.isfile(stdlib_path):
                            files_to_process.append(os.path.abspath(stdlib_path))
                            
    processed_contents.sort(key=lambda x: x[0])
    for path, content in processed_contents:
        hasher.update(path.encode('utf-8'))
        hasher.update(content.encode('utf-8'))
        
    return hasher.hexdigest()

def get_file_hash(filepath: str) -> str:
    hasher = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            hasher.update(f.read())
    except Exception:
        pass
    return hasher.hexdigest()

def load_from_cache(filepath: str, workspace_root: str, cache_type: str):
    import pickle
    try:
        file_hash = get_file_hash(filepath)
        cache_dir = os.path.join(workspace_root, ".eigen_cache")
        cache_path = os.path.join(cache_dir, f"{file_hash}.{cache_type}")
        if os.path.isfile(cache_path):
            if cache_type in ("ast", "ssa", "zx"):
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            else:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if cache_type == "ebc":
                    if isinstance(data, dict) and data.get("major") == 3:
                        return [Instruction.from_dict(d) for d in data["instructions"]]
                    elif isinstance(data, list):
                        return [Instruction.from_dict(d) for d in data]
                elif cache_type == "eqir":
                    return EQIRGraph.from_dict(data)
    except Exception:
        pass
    return None

def save_to_cache(filepath: str, workspace_root: str, cache_type: str, obj):
    import pickle
    try:
        file_hash = get_file_hash(filepath)
        cache_dir = os.path.join(workspace_root, ".eigen_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{file_hash}.{cache_type}")
        if cache_type in ("ast", "ssa", "zx"):
            with open(cache_path, 'wb') as f:
                pickle.dump(obj, f)
        else:
            if cache_type == "ebc":
                data = {
                    "major": 3,
                    "minor": 0,
                    "instructions": [inst.to_dict() for inst in obj]
                }
            elif cache_type == "eqir":
                data = obj.to_dict()
            else:
                return
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
    except Exception:
        pass

def compile_to_eqir(filepath: str, workspace_root: str) -> tuple:
    if not os.path.isfile(filepath):
        print(f"Error: File '{filepath}' not found.", file=sys.stderr)
        sys.exit(1)
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    lexer = Lexer(content)
    tokens = tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    resolver = ImportResolver(workspace_root)
    ast = resolver.resolve(ast)

    type_checker = TypeChecker()
    try:
        type_checker.check(ast)
    except TypeErrorException as e:
        print(f"Type Verification Failed:\n{e}", file=sys.stderr)
        sys.exit(1)

    from src.ir.mlir_dialect import ASTToMLIRConverter, MLIRToEQIRConverter
    mlir_converter = ASTToMLIRConverter()
    mlir_module = mlir_converter.convert(ast)
    
    eqir_converter = MLIRToEQIRConverter()
    graph = eqir_converter.convert(mlir_module)
    
    return graph, ast
