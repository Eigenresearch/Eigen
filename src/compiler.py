import os
import sys
import hashlib
import json

from src.frontend.lexer import Lexer
from src.frontend.parser import Parser
from src.semantic.import_resolver import ImportResolver
from src.semantic.type_checker import TypeChecker, TypeErrorException
from src.ir.ir_graph import EQIRGraph
from src.backend.bytecode import Instruction
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
        except Exception as e:
            raise FileNotFoundError(f"Could not read source file '{abs_path}': {e}")
        
        visited_files[abs_path] = content
        
        lexer = Lexer(content)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        for imp in ast.imports:
            try:
                sub_file = resolver.resolve_module_file(imp.module_path)
                process_file(sub_file)
            except Exception as e:
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
    except Exception:
        pass
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
        except Exception:
            pass
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
                        import pickle
                        return pickle.load(f)
                except Exception:
                    pass
                    
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
    # Support manual legacy saving using pickle
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
    
    # Compute result hash
    import hashlib
    import pickle
    hasher = hashlib.sha256()
    try:
        hasher.update(pickle.dumps(obj))
    except Exception:
        hasher.update(str(obj).encode('utf-8'))
    result_hash = hasher.hexdigest()
    
    cache_file = os.path.join(db.cache_dir, f"manual_{result_hash}.{cache_type}")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(obj, f)
        
    try:
        rel_cache = os.path.relpath(cache_file, db.workspace_root)
    except ValueError:
        rel_cache = os.path.abspath(cache_file)
        
    db.records[query_key]["cache_file"] = rel_cache
    db.records[query_key]["result_hash"] = result_hash
    db.current_query = None
    db.save()

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
    
    # §1.2 — Check incremental AST/EQIR cache
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        cached_ast, ast_hit = _incremental_cache.get_ast(source)
        if ast_hit and cached_ast is not None:
            # AST cache hit — skip parse + type_check, go straight to EQIR
            from src.ir.mlir_dialect import ASTToMLIRConverter, MLIRToEQIRConverter
            mlir_converter = ASTToMLIRConverter()
            mlir_module = mlir_converter.convert(cached_ast)
            eqir_converter = MLIRToEQIRConverter()
            graph = eqir_converter.convert(mlir_module)
            return graph, cached_ast
    except Exception:
        pass  # Fall through to normal pipeline on any cache error
    
    try:
        result = to_eqir(filepath, workspace_root)
        # §1.2 — Cache the AST for incremental compilation
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()
            _incremental_cache.put_ast(source, result[1])
        except Exception:
            pass
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
            for imp in ast.imports:
                # Map import path to a file in the same list
                dep_path = imp.module_path.replace('.', '/') + '.eig'
                for other in filepaths:
                    if other.endswith(dep_path):
                        deps.append(os.path.basename(other).replace('.eig', ''))
                        break
        except Exception:
            pass
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
