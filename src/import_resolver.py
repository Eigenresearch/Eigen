import os
from src.lexer import Lexer
from src.parser import Parser
from src.ast import ProgramNode, QFuncDeclNode, ImportNode

class ImportResolver:
    def __init__(self, workspace_root: str, stdlib_root: str | None = None):
        self.workspace_root = os.path.abspath(workspace_root)
        if stdlib_root:
            self.stdlib_root = os.path.abspath(stdlib_root)
        else:
            self.stdlib_root = os.path.join(self.workspace_root, "stdlib")
        self.resolved_modules = {}  # module_path (str) -> ProgramNode

    def resolve_module_file(self, module_path: str) -> str:
        # e.g., "quantum.bell" -> "quantum/bell.eig"
        relative_path = module_path.replace('.', '/') + ".eig"
        
        # Search in workspace first
        local_path = os.path.join(self.workspace_root, relative_path)
        if os.path.isfile(local_path):
            return local_path
            
        # Search in stdlib
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
        containing all declarations from imported files.
        """
        pending_imports = list(main_ast.imports)
        imported_qfuncs = []
        visited = set()

        if main_ast.module_name:
            visited.add(main_ast.module_name)

        while pending_imports:
            imp_node = pending_imports.pop(0)
            module_path = imp_node.module_path
            if module_path in visited:
                continue
            visited.add(module_path)

            file_path = self.resolve_module_file(module_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            lexer = Lexer(content)
            tokens = lexer.tokenize()
            parser = Parser(tokens)
            module_ast = parser.parse()

            # Add imports of this module to pending
            for sub_imp in module_ast.imports:
                if sub_imp.module_path not in visited:
                    pending_imports.append(sub_imp)

            # Store the module AST
            self.resolved_modules[module_path] = module_ast

            # Collect qfunc declarations
            for node in module_ast.body:
                if isinstance(node, QFuncDeclNode):
                    imported_qfuncs.append(node)

        # Merge imported qfuncs into the main AST's body
        # Put them at the beginning of the body so they are defined first
        main_ast.body = imported_qfuncs + main_ast.body
        return main_ast
