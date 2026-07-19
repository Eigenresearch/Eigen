import os
import concurrent.futures
import threading
from collections import OrderedDict
from src.frontend.lexer import Lexer
from src.frontend.parser import Parser
from src.frontend.ast import ProgramNode, QFuncDeclNode, FuncDeclNode, StructDeclNode, EnumDeclNode
from src.compiler_optimizations import ImportCache, LazyModuleLoader

# §5 (memory) — Parsed-AST cache is an LRU with a hard size bound so
# long-lived processes (LSP server) cannot leak unbounded ASTs, and a
# lock so concurrent compiles cannot race on eviction.
_PARSED_AST_CACHE_MAX = 256
_PARSED_AST_CACHE: "OrderedDict[str, tuple[str, ProgramNode]]" = OrderedDict()
_PARSED_AST_LOCK = threading.RLock()
# §1.2 — File-hash-based import cache for unchanged modules
_IMPORT_CACHE = ImportCache()
# §1.2 — Lazy module loader for on-demand parsing
_LAZY_LOADER = LazyModuleLoader()


def clear_import_caches() -> None:
    """Clear process-wide import artifacts for an isolated compiler session."""
    with _PARSED_AST_LOCK:
        _PARSED_AST_CACHE.clear()
    _IMPORT_CACHE.invalidate()
    _LAZY_LOADER.clear()


def _contained(path: str, root: str) -> bool:
    """True iff ``path`` stays inside ``root`` (no traversal escape)."""
    try:
        return os.path.commonpath([os.path.abspath(path), root]) == root
    except ValueError:
        # Raised on Windows when the paths are on different drives.
        return False


class ImportResolver:
    def __init__(self, workspace_root: str, stdlib_root: str | None = None):
        self.workspace_root = os.path.abspath(workspace_root)
        if stdlib_root:
            self.stdlib_root = os.path.abspath(stdlib_root)
        else:
            self.stdlib_root = os.path.join(self.workspace_root, "stdlib")
        self.resolved_modules = {}  # module_path (str) -> ProgramNode
        self._lock = threading.RLock()
        self._recursion_limit = 100

    def resolve_module_file(self, module_path: str) -> str:
        # §6 (security) — reject traversal segments before any path is
        # built: ``import foo....bar`` (or an absolute-path segment) must
        # never escape the workspace/stdlib roots.
        segments = module_path.split('.')
        if (not module_path or any(not s for s in segments)
                or any(s in ('..',) or '/' in s or '\\' in s or ':' in s
                       for s in segments)):
            raise ImportError(
                f"Illegal module path: {module_path!r} "
                f"(must be dotted identifiers without traversal)")

        # §1.2 — Check import cache first
        cached_path, fresh = _IMPORT_CACHE.get(module_path)
        if cached_path and fresh:
            return cached_path

        std_modules = {'math', 'std', 'collections', 'random', 'io', 'time', 'string', 'linalg', 'quantum'}
        relative_path = module_path.replace('.', '/') + ".eig"
        first_part = segments[0]

        # stdlib priority: search stdlib first for standard namespaces
        if first_part in std_modules:
            stdlib_path = os.path.abspath(os.path.join(self.stdlib_root, relative_path))
            if _contained(stdlib_path, self.stdlib_root) and os.path.isfile(stdlib_path):
                _IMPORT_CACHE.put(module_path, stdlib_path, stdlib_path)
                return stdlib_path

        local_path = os.path.abspath(os.path.join(self.workspace_root, relative_path))
        if _contained(local_path, self.workspace_root) and os.path.isfile(local_path):
            _IMPORT_CACHE.put(module_path, local_path, local_path)
            return local_path

        stdlib_path = os.path.abspath(os.path.join(self.stdlib_root, relative_path))
        if _contained(stdlib_path, self.stdlib_root) and os.path.isfile(stdlib_path):
            _IMPORT_CACHE.put(module_path, stdlib_path, stdlib_path)
            return stdlib_path

        raise FileNotFoundError(
            f"Could not resolve module '{module_path}'.\n"
            f"Looked in:\n"
            f"  Local: {local_path}\n"
            f"  Stdlib: {stdlib_path}"
        )

    def resolve(self, main_ast: ProgramNode) -> ProgramNode:
        """
        Recursively resolves all imports in the main_ast and returns a merged ProgramNode
        containing all declarations from imported files in topological order, detecting cycles.
        """
        imported_nodes = []
        visited = set()
        path_stack = []

        if main_ast.module_name:
            visited.add(main_ast.module_name)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        def parse_file(file_path: str) -> ProgramNode:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            import hashlib
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()

            with _PARSED_AST_LOCK:
                cached = _PARSED_AST_CACHE.get(file_path)
                if cached and cached[0] == content_hash:
                    _PARSED_AST_CACHE.move_to_end(file_path)
                    return cached[1]

            # §1.2 — LazyModuleLoader: register a lazy loader for
            # this file so it's only re-parsed on demand.
            module_name = os.path.basename(file_path).replace('.eig', '')
            if not _LAZY_LOADER.is_loaded(module_name):
                def _make_loader(path=file_path, ch=content_hash):
                    def _load():
                        with open(path, 'r', encoding='utf-8') as f2:
                            c = f2.read()
                        lexer = Lexer(c)
                        tokens = lexer.tokenize()
                        parser = Parser(tokens)
                        return parser.parse()
                    return _load
                _LAZY_LOADER.register(module_name, _make_loader())

            lexer = Lexer(content)
            tokens = lexer.tokenize()
            parser = Parser(tokens)
            ast = parser.parse()
            with _PARSED_AST_LOCK:
                _PARSED_AST_CACHE[file_path] = (content_hash, ast)
                _PARSED_AST_CACHE.move_to_end(file_path)
                while len(_PARSED_AST_CACHE) > _PARSED_AST_CACHE_MAX:
                    _PARSED_AST_CACHE.popitem(last=False)
            return ast

        def dfs_resolve(module_name: str, file_path: str, depth: int = 0):
            if depth > self._recursion_limit:
                raise ImportError(f"Import depth limit exceeded at {module_name}")
            if module_name in path_stack:
                cycle = " -> ".join(path_stack + [module_name])
                raise ImportError(f"Cyclic import detected: {cycle}")
            
            with self._lock:
                if module_name in visited:
                    return

            path_stack.append(module_name)

            module_ast = parse_file(file_path)
            with self._lock:
                self.resolved_modules[module_name] = module_ast

            sub_imports = []
            for sub_imp in module_ast.imports:
                with self._lock:
                    is_visited = sub_imp.module_path in visited
                if not is_visited:
                    sub_file = self.resolve_module_file(sub_imp.module_path)
                    sub_imports.append((sub_imp.module_path, sub_file))

            if sub_imports:
                futures = {executor.submit(parse_file, path): name for name, path in sub_imports}
                concurrent.futures.wait(futures)

            for sub_module, sub_file in sub_imports:
                dfs_resolve(sub_module, sub_file, depth + 1)

            for node in module_ast.body:
                if isinstance(node, (QFuncDeclNode, FuncDeclNode, StructDeclNode, EnumDeclNode)):
                    imported_nodes.append(node)

            with self._lock:
                visited.add(module_name)
            path_stack.pop()

        for imp in main_ast.imports:
            sub_module = imp.module_path
            sub_file = self.resolve_module_file(sub_module)
            dfs_resolve(sub_module, sub_file, 0)

        executor.shutdown()

        main_ast.body = imported_nodes + main_ast.body
        return main_ast
