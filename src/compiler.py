import os
import sys
import hashlib
import hmac
import json

from src.frontend.lexer import Lexer
from src.frontend.parser import Parser
from src.semantic.import_resolver import ImportResolver
from src.semantic.type_checker import TypeChecker, TypeErrorException
from src.compiler_optimizations import IncrementalCache, LazyModuleLoader

def get_workspace_root() -> str:
    root = os.getcwd()
    for _ in range(10):
        if os.path.isfile(os.path.join(root, "eigen.toml")) or os.path.isfile(os.path.join(root, "pyproject.toml")):
            return root
        parent = os.path.dirname(root)
        if parent == root:
            break
        root = parent
    return os.getcwd()

def get_project_hash(filepath: str, workspace_root: str) -> str:
    resolver = ImportResolver(workspace_root)
    visited_files = {}
    
    def process_file(file_path: str):
        abs_path = os.path.abspath(file_path)
        if abs_path in visited_files:
            return
        
        try:
            with open(abs_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except (OSError, UnicodeDecodeError) as e:
            raise FileNotFoundError(f"Could not read source file '{abs_path}': {e}") from e
        
        visited_files[abs_path] = content
        
        lexer = Lexer(content)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        for imp in ast.imports:
            try:
                sub_file = resolver.resolve_module_file(imp.module_path)
                process_file(sub_file)
            except (FileNotFoundError, ImportError) as e:
                print(f"Warning: Failed to resolve import '{imp.module_path}': {e}", file=sys.stderr)

    process_file(filepath)
    
    if not visited_files:
        return get_file_hash(filepath)
        
    hasher = hashlib.sha256()
    for path in sorted(visited_files.keys()):
        hasher.update(path.encode('utf-8'))
        hasher.update(visited_files[path].encode('utf-8'))
        
    return hasher.hexdigest()

def get_file_hash(filepath: str) -> str:
    hasher = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            hasher.update(f.read())
    except OSError as e:
        print(f"Warning: Failed to hash file '{filepath}': {e}", file=sys.stderr)
    return hasher.hexdigest()

from src.compiler_db import QueryDb

_DBS = {}

# §1.2 — Module-level incremental compilation caches
_incremental_cache = IncrementalCache()
_lazy_loader = LazyModuleLoader()

def get_db(workspace_root: str) -> QueryDb:
    global _DBS
    workspace_root = os.path.abspath(workspace_root)
    if workspace_root not in _DBS:
        _DBS[workspace_root] = QueryDb(workspace_root)
    return _DBS[workspace_root]

def query_parse(filepath: str, workspace_root: str):
    db = get_db(workspace_root)
    db.add_input_file(filepath)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    lexer = Lexer(content)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    return parser.parse()

def parse(filepath: str, workspace_root: str):
    db = get_db(workspace_root)
    return db.execute_query("parse", filepath, query_parse, filepath, workspace_root)

def query_resolve_imports(filepath: str, workspace_root: str):
    db = get_db(workspace_root)
    ast = parse(filepath, workspace_root)
    resolver = ImportResolver(workspace_root)
    resolved_ast = resolver.resolve(ast)
    for imp in resolved_ast.imports:
        try:
            sub_file = resolver.resolve_module_file(imp.module_path)
            db.add_input_file(sub_file)
        except (FileNotFoundError, ImportError) as e:
            print(f"Warning: Failed to resolve import '{imp.module_path}': {e}", file=sys.stderr)
    return resolved_ast

def resolve_imports(filepath: str, workspace_root: str):
    db = get_db(workspace_root)
    return db.execute_query("resolve_imports", filepath, query_resolve_imports, filepath, workspace_root)

def query_type_check(filepath: str, workspace_root: str):
    ast = resolve_imports(filepath, workspace_root)
    type_checker = TypeChecker()
    type_checker.check(ast)
    
    from src.semantic.monomorphizer import Monomorphizer
    monomorphizer = Monomorphizer()
    ast = monomorphizer.monomorphize(ast)
    
    return ast

def type_check(filepath: str, workspace_root: str):
    db = get_db(workspace_root)
    return db.execute_query("type_check", filepath, query_type_check, filepath, workspace_root)

def query_to_eqir(filepath: str, workspace_root: str):
    ast = type_check(filepath, workspace_root)
    from src.ir.mlir_dialect import ASTToMLIRConverter, MLIRToEQIRConverter
    mlir_converter = ASTToMLIRConverter()
    mlir_module = mlir_converter.convert(ast)
    eqir_converter = MLIRToEQIRConverter()
    graph = eqir_converter.convert(mlir_module)
    return graph, ast

def to_eqir(filepath: str, workspace_root: str):
    db = get_db(workspace_root)
    return db.execute_query("to_eqir", filepath, query_to_eqir, filepath, workspace_root)

def load_from_cache(filepath: str, workspace_root: str, cache_type: str):
    db = get_db(workspace_root)
    # 1. Try legacy manual load
    query_key = f"{cache_type}:{filepath}"
    record = db.records.get(query_key)
    if record:
        valid, _ = db.verify_record(query_key)
        if valid:
            cache_path = os.path.join(workspace_root, record["cache_file"])
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "rb") as f:
                        raw = f.read()
                    obj = _deserialize_cache(raw, workspace_root)
                    if obj is not None:
                        return obj
                except (OSError, ValueError) as e:
                    print(f"Warning: Failed to load cache for '{query_key}': {e}", file=sys.stderr)
                    
    # 2. Try the QueryDb to_eqir check (for backwards compatibility with eqir query type)
    if cache_type == "eqir":
        eqir_key = f"to_eqir:{filepath}"
        record = db.records.get(eqir_key)
        if record:
            valid, _ = db.verify_record(eqir_key)
            if valid:
                cache_path = os.path.join(workspace_root, record["cache_file"])
                if os.path.exists(cache_path):
                    res = db.read_cache_file(cache_path, "to_eqir")
                    if res:
                        return res[0]
    return None

def save_to_cache(filepath: str, workspace_root: str, cache_type: str, obj):
    db = get_db(workspace_root)
    query_key = f"{cache_type}:{filepath}"
    
    db.current_query = query_key
    db.records[query_key] = {
        "cache_file": "",
        "result_hash": "",
        "input_files": {},
        "dependencies": []
    }
    db.add_input_file(filepath)
    
    payload_bytes = _serialize_cache(obj, workspace_root)
    hasher = hashlib.sha256()
    hasher.update(payload_bytes)
    result_hash = hasher.hexdigest()
    
    cache_file = os.path.join(db.cache_dir, f"manual_{result_hash}.{cache_type}")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "wb") as f:
        f.write(payload_bytes)
        
    try:
        rel_cache = os.path.relpath(cache_file, db.workspace_root)
    except ValueError:
        rel_cache = os.path.abspath(cache_file)
        
    db.records[query_key]["cache_file"] = rel_cache
    db.records[query_key]["result_hash"] = result_hash
    db.current_query = None
    db.save()

_CACHE_JSON_MAGIC = b"EIGENCJ1\n"
_CACHE_PKL_MAGIC = b"EIGENCP1\n"

def _cache_hmac_key(workspace_root: str) -> bytes:
    key_path = os.path.join(workspace_root, ".eigen_cache", ".hmac_key")
    if os.path.exists(key_path):
        try:
            with open(key_path, "rb") as f:
                key = f.read()
            if key:
                return key
        except OSError:
            pass
    key = os.urandom(32)
    try:
        os.makedirs(os.path.dirname(key_path), exist_ok=True)
        with open(key_path, "wb") as f:
            f.write(key)
    except OSError:
        pass
    return key

def _to_json_safe(obj):
    cls = type(obj)
    if hasattr(obj, "to_dict") and hasattr(cls, "from_dict"):
        return {
            "__obj__": True,
            "class": f"{cls.__module__}.{cls.__qualname__}",
            "data": obj.to_dict(),
        }
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    return obj

def _from_json_safe(data):
    if isinstance(data, dict):
        if data.get("__obj__") is True and "class" in data and "data" in data:
            cls_path = data["class"]
            module_path, _, qualname = cls_path.rpartition(".")
            if module_path:
                import importlib
                try:
                    mod = importlib.import_module(module_path)
                    cls = getattr(mod, qualname)
                    if hasattr(cls, "from_dict"):
                        return cls.from_dict(data["data"])
                except (ImportError, AttributeError):
                    pass
            return _from_json_safe(data["data"])
        return {k: _from_json_safe(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_from_json_safe(v) for v in data]
    return data

def _serialize_cache(obj, workspace_root: str) -> bytes:
    try:
        payload = {"__cache_format__": 1, "value": _to_json_safe(obj)}
        json_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        return _CACHE_JSON_MAGIC + json_bytes
    except (TypeError, ValueError):
        import pickle
        data = pickle.dumps(obj)
        key = _cache_hmac_key(workspace_root)
        sig = hmac.new(key, data, hashlib.sha256).hexdigest().encode("ascii")
        return _CACHE_PKL_MAGIC + sig + b"\n" + data

def _deserialize_cache(raw: bytes, workspace_root: str):
    if raw.startswith(_CACHE_JSON_MAGIC):
        try:
            payload = json.loads(raw[len(_CACHE_JSON_MAGIC):].decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None
        if isinstance(payload, dict) and payload.get("__cache_format__") == 1:
            return _from_json_safe(payload.get("value"))
        return _from_json_safe(payload)
    if raw.startswith(_CACHE_PKL_MAGIC):
        rest = raw[len(_CACHE_PKL_MAGIC):]
        nl = rest.find(b"\n")
        if nl < 0:
            return None
        sig = rest[:nl]
        data = rest[nl + 1:]
        key = _cache_hmac_key(workspace_root)
        expected = hmac.new(key, data, hashlib.sha256).hexdigest().encode("ascii")
        if not hmac.compare_digest(sig, expected):
            return None
        import pickle
        try:
            return pickle.loads(data)
        except (pickle.PickleError, EOFError, OSError, AttributeError):
            return None
    return None


def _has_classical_control_flow(node) -> bool:
    from src.frontend.ast import (
        WhileNode, ForNode, FuncDeclNode, TryCatchNode
    )
    if isinstance(node, (WhileNode, ForNode, FuncDeclNode, TryCatchNode)):
        return True
    if hasattr(node, "body") and node.body:
        if isinstance(node.body, list):
            for child in node.body:
                if _has_classical_control_flow(child):
                    return True
        elif _has_classical_control_flow(node.body):
            return True
    if hasattr(node, "else_body") and node.else_body:
        if isinstance(node.else_body, list):
            for child in node.else_body:
                if _has_classical_control_flow(child):
                    return True
        elif _has_classical_control_flow(node.else_body):
            return True
    if hasattr(node, "nodes") and node.nodes:
        for child in node.nodes:
            if _has_classical_control_flow(child):
                return True
    return False

def query_to_ebc(filepath: str, workspace_root: str, optimize: bool = False):
    graph, ast = to_eqir(filepath, workspace_root)
    from src.backend.ebc_compiler import EBCCompiler
    compiler = EBCCompiler(peephole=optimize)
    if optimize and not _has_classical_control_flow(ast):
        from src.ir.optimizer import EQIROptimizer
        optimizer = EQIROptimizer()
        graph = optimizer.optimize(graph)
        instructions = compiler.compile_eqir(graph)
    else:
        instructions = compiler.compile_ast(ast)
    if optimize:
        from src.ir.ssa.optimizer import optimize_ebc
        instructions = optimize_ebc(instructions)
    return instructions

def to_ebc(filepath: str, workspace_root: str, optimize: bool = False):
    db = get_db(workspace_root)
    query_name = "to_ebc_opt" if optimize else "to_ebc"
    return db.execute_query(query_name, filepath, query_to_ebc, filepath, workspace_root, optimize)

def compile_to_eqir(filepath: str, workspace_root: str) -> tuple:
    if not os.path.isfile(filepath):
        print(f"Error: File '{filepath}' not found.", file=sys.stderr)
        sys.exit(1)
    
    source = None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
    except OSError as e:
        print(f"Warning: Failed to read source '{filepath}': {e}", file=sys.stderr)
    
    # §1.2 — Check incremental AST/EQIR cache
    if source is not None:
        try:
            cached_ast, ast_hit = _incremental_cache.get_ast(source)
            if ast_hit and cached_ast is not None:
                # AST cache hit — skip parse + type_check, go straight to EQIR
                from src.ir.mlir_dialect import ASTToMLIRConverter, MLIRToEQIRConverter
                mlir_converter = ASTToMLIRConverter()
                mlir_module = mlir_converter.convert(cached_ast)
                eqir_converter = MLIRToEQIRConverter()
                graph = eqir_converter.convert(mlir_module)
                return graph, cached_ast
        except (OSError, EOFError, AttributeError, TypeError) as e:
            print(f"Warning: incremental cache lookup failed, falling through to normal pipeline: {e}", file=sys.stderr)
    
    try:
        result = to_eqir(filepath, workspace_root)
        # §1.2 — Cache the AST for incremental compilation
        if source is not None:
            try:
                _incremental_cache.put_ast(source, result[1])
            except (AttributeError, TypeError) as e:
                print(f"Warning: Failed to cache AST for '{filepath}': {e}", file=sys.stderr)
        return result
    except TypeErrorException as e:
        print(f"Type Verification Failed:\n{e}", file=sys.stderr)
        sys.exit(1)


def compile_multiple_parallel(filepaths: list[str],
                                workspace_root: str,
                                max_workers: int = 4) -> dict:
    """Compile multiple modules in parallel.

    §8.2: "Параллельная компиляция — несколько модулей одновременно"

    Uses the ParallelCompiler to compile independent modules
    simultaneously, respecting cross-module dependencies.
    Returns a dict of {filepath: (graph, ast) or error_string}.
    """
    from src.parallel_compiler import (
        CompilationTask, compile_in_parallel,
    )
    import os

    tasks = []
    for fp in filepaths:
        deps = []
        # Detect dependencies by parsing imports
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                content = f.read()
            from src.frontend.lexer import Lexer as _L
            from src.frontend.parser import Parser as _P
            ast = _P(_L(content).tokenize()).parse()
            base_dir = os.path.dirname(os.path.abspath(fp))
            for imp in ast.imports:
                # Map import path to a file in the same list
                dep_rel = imp.module_path.replace('.', '/') + '.eig'
                dep_abs = os.path.abspath(os.path.join(base_dir, dep_rel))
                for other in filepaths:
                    if os.path.abspath(other) == dep_abs:
                        deps.append(os.path.basename(other).replace('.eig', ''))
                        break
        except (OSError, SyntaxError) as e:
            print(f"Warning: Failed to detect dependencies for '{fp}': {e}", file=sys.stderr)
        tasks.append(CompilationTask(
            module_name=os.path.basename(fp).replace('.eig', ''),
            source_path=fp,
            dependencies=deps,
        ))

    def _compile_one(task: CompilationTask):
        return compile_to_eqir(task.source_path, workspace_root)

    result = compile_in_parallel(tasks, _compile_one, max_workers=max_workers)
    output = {}
    for t in result.tasks:
        if t.status == "done":
            output[t.source_path] = t.result
        else:
            output[t.source_path] = t.error or "unknown error"
    return output
