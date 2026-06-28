import os
import concurrent.futures
from src.frontend.lexer import Lexer
from src.frontend.parser import Parser
from src.frontend.ast import ProgramNode, QFuncDeclNode, ImportNode, FuncDeclNode, StructDeclNode, EnumDeclNode

# Cache mapping file_path -> (mtime, ProgramNode)
_PARSED_AST_CACHE = {}

class ImportResolver:
    def __init__(self, workspace_root: str, stdlib_root: str | None = None):
        self.workspace_root = os.path.abspath(workspace_root)
        if stdlib_root:
            self.stdlib_root = os.path.abspath(stdlib_root)
        else:
            self.stdlib_root = os.path.join(self.workspace_root, "stdlib")
        self.resolved_modules = {}  # module_path (str) -> ProgramNode

    def resolve_module_file(self, module_path: str) -> str:
        std_modules = {'math', 'std', 'collections', 'random', 'io', 'time', 'string', 'linalg', 'quantum'}
        relative_path = module_path.replace('.', '/') + ".eig"
        first_part = module_path.split('.')[0]
        
        # stdlib priority: search stdlib first for standard namespaces
        if first_part in std_modules:
            stdlib_path = os.path.join(self.stdlib_root, relative_path)
            if os.path.isfile(stdlib_path):
                return stdlib_path

        local_path = os.path.join(self.workspace_root, relative_path)
        if os.path.isfile(local_path):
            return local_path

        stdlib_path = os.path.join(self.stdlib_root, relative_path)
        if os.path.isfile(stdlib_path):
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

            cached = _PARSED_AST_CACHE.get(file_path)
            if cached and cached[0] == content_hash:
                return cached[1]

            lexer = Lexer(content)
            tokens = lexer.tokenize()
            parser = Parser(tokens)
            ast = parser.parse()
            _PARSED_AST_CACHE[file_path] = (content_hash, ast)
            return ast

        def dfs_resolve(module_name: str, file_path: str):
            if module_name in path_stack:
                cycle = " -> ".join(path_stack + [module_name])
                raise ImportError(f"Cyclic import detected: {cycle}")
            if module_name in visited:
                return

            path_stack.append(module_name)

            module_ast = parse_file(file_path)
            self.resolved_modules[module_name] = module_ast

            sub_imports = []
            for sub_imp in module_ast.imports:
                if sub_imp.module_path not in visited:
                    sub_file = self.resolve_module_file(sub_imp.module_path)
                    sub_imports.append((sub_imp.module_path, sub_file))

            if sub_imports:
                futures = {executor.submit(parse_file, path): name for name, path in sub_imports}
                concurrent.futures.wait(futures)

            for sub_module, sub_file in sub_imports:
                dfs_resolve(sub_module, sub_file)

            for node in module_ast.body:
                if isinstance(node, (QFuncDeclNode, FuncDeclNode, StructDeclNode, EnumDeclNode)):
                    imported_nodes.append(node)

            visited.add(module_name)
            path_stack.pop()

        for imp in main_ast.imports:
            sub_module = imp.module_path
            sub_file = self.resolve_module_file(sub_module)
            dfs_resolve(sub_module, sub_file)

        executor.shutdown()

        main_ast.body = imported_nodes + main_ast.body
        return main_ast
