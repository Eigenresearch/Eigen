from src.frontend.ast import (
    ProgramNode, LetNode, VarDeclNode, FuncDeclNode, QFuncDeclNode, ASTNode
)

class TypeExtractor:
    def __init__(self):
        self.var_types = {}  # scope_name -> {var_name: type}
        self.funcs = {}      # func_name -> (param_types, return_type)

    def extract(self, ast: ProgramNode) -> tuple[dict, dict, dict]:
        # Walk the AST to collect types of variables and functions
        self.var_types = {"main": {}}
        self.funcs = {}
        self.func_params = {}

        # First pass: collect function signatures
        for node in ast.body:
            if isinstance(node, FuncDeclNode):
                param_types = [p[1] for p in node.params]
                self.funcs[node.name] = (param_types, node.return_type)
                self.func_params[node.name] = [p[0] for p in node.params]
            elif isinstance(node, QFuncDeclNode):
                param_types = [p[1] for p in node.params]
                self.funcs[node.name] = (param_types, "void")
                self.func_params[node.name] = [p[0] for p in node.params]

        # Second pass: walk all scopes and collect variable declarations
        current_scope = "main"

        def walk(node):
            nonlocal current_scope
            if isinstance(node, FuncDeclNode) or isinstance(node, QFuncDeclNode):
                old_scope = current_scope
                current_scope = node.name
                self.var_types[current_scope] = {}
                # Add parameters as local variables of this function
                for p_name, p_type in node.params:
                    self.var_types[current_scope][p_name] = p_type
                for stmt in node.body:
                    walk(stmt)
                current_scope = old_scope
            elif isinstance(node, LetNode):
                self.var_types[current_scope][node.name] = node.type_name
                walk(node.value)
            elif isinstance(node, VarDeclNode):
                self.var_types[current_scope][node.name] = node.type_name
            elif isinstance(node, ASTNode):
                # Walk all child fields
                for attr_name in dir(node):
                    if attr_name.startswith('_'):
                        continue
                    attr = getattr(node, attr_name)
                    if isinstance(attr, ASTNode):
                        walk(attr)
                    elif isinstance(attr, list):
                        for item in attr:
                            if isinstance(item, ASTNode):
                                walk(item)

        for node in ast.body:
            walk(node)

        return self.var_types, self.funcs, self.func_params
