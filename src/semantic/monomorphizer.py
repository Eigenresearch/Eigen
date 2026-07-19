from src.frontend.ast import (
    ProgramNode, FuncDeclNode, CallNode, VarRefNode, LiteralNode, LetNode, VarDeclNode,
    BinaryOpNode, UnaryOpNode, AssignmentNode, ReturnNode, IfNode, ForNode, WhileNode, PrintNode, AssertNode
)
import copy

class Monomorphizer:
    """Performs compile-time monomorphization of generic classic functions."""

    def __init__(self):
        self.generic_funcs = {}  # name -> FuncDeclNode
        self.specializations = {}  # (name, tuple_of_types) -> new_name
        self.new_decls = []
        self.var_types = {}  # local variables type mapping for inference

    def monomorphize(self, program: ProgramNode) -> ProgramNode:
        # 1. Collect generic functions and remove them from main body
        filtered_body = []
        for node in program.body:
            if isinstance(node, FuncDeclNode) and len(node.generic_params) > 0:
                self.generic_funcs[node.name] = node
            else:
                filtered_body.append(node)

        program.body = filtered_body

        # 2. Setup variable type inference from global variables and let statements
        for node in program.body:
            if isinstance(node, LetNode):
                self.var_types[node.name] = node.type_name
            elif isinstance(node, VarDeclNode):
                self.var_types[node.name] = node.type_name

        # 3. Recursively transform all calls in the main body
        transformed_body = []
        for node in program.body:
            transformed_body.append(self.visit(node))

        # 4. Append all generated specialized functions to the program body
        program.body = self.new_decls + transformed_body
        return program

    def visit(self, node):
        if node is None:
            return None

        if isinstance(node, LetNode):
            node.value = self.visit(node.value)
            self.var_types[node.name] = node.type_name
            return node

        elif isinstance(node, VarDeclNode):
            self.var_types[node.name] = node.type_name
            return node

        elif isinstance(node, AssignmentNode):
            node.value = self.visit(node.value)
            return node

        elif isinstance(node, ReturnNode):
            node.expr = self.visit(node.expr)
            return node

        elif isinstance(node, IfNode):
            node.condition_left = self.visit(node.condition_left)
            node.condition_right = self.visit(node.condition_right)
            node.body = [self.visit(stmt) for stmt in node.body]
            if hasattr(node, "else_body") and node.else_body:
                node.else_body = [self.visit(stmt) for stmt in node.else_body]
            return node

        elif isinstance(node, ForNode):
            node.iterable = self.visit(node.iterable)
            node.body = [self.visit(stmt) for stmt in node.body]
            return node

        elif isinstance(node, WhileNode):
            node.condition = self.visit(node.condition)
            node.body = [self.visit(stmt) for stmt in node.body]
            return node

        elif isinstance(node, PrintNode):
            node.expr = self.visit(node.expr)
            return node

        elif isinstance(node, AssertNode):
            node.condition_left = self.visit(node.condition_left)
            node.condition_right = self.visit(node.condition_right)
            return node

        elif isinstance(node, BinaryOpNode):
            node.left = self.visit(node.left)
            node.right = self.visit(node.right)
            return node

        elif isinstance(node, UnaryOpNode):
            node.operand = self.visit(node.operand)
            return node

        elif isinstance(node, CallNode):
            # Recursively visit arguments first
            node.args = [self.visit(arg) for arg in node.args]

            callee_name = None
            if isinstance(node.callee, VarRefNode):
                callee_name = node.callee.name
            elif isinstance(node.callee, str):
                callee_name = node.callee

            if callee_name and callee_name in self.generic_funcs:
                func = self.generic_funcs[callee_name]
                
                # Infer argument types
                arg_types = []
                for arg in node.args:
                    arg_types.append(self.infer_type(arg))
                
                # Solve generic parameters (bindings)
                bindings = {}
                for arg_type, (_p_name, p_type) in zip(arg_types, func.params, strict=False):
                    if p_type in func.generic_params:
                        bindings[p_type] = arg_type

                # Ensure all generic parameters are bound
                concrete_types = []
                for gp in func.generic_params:
                    concrete_types.append(bindings.get(gp, "any"))

                spec_key = (callee_name, tuple(concrete_types))
                if spec_key in self.specializations:
                    spec_name = self.specializations[spec_key]
                else:
                    # Create specialization name
                    types_suffix = "_".join(concrete_types)
                    spec_name = f"{callee_name}_{types_suffix}"
                    self.specializations[spec_key] = spec_name

                    # Deep copy and specialize the AST node
                    spec_func = copy.deepcopy(func)
                    spec_func.name = spec_name
                    spec_func.generic_params = []
                    
                    # Replace types in parameters
                    new_params = []
                    for p_name, p_type in spec_func.params:
                        resolved_type = bindings.get(p_type, p_type)
                        new_params.append((p_name, resolved_type))
                    spec_func.params = new_params
                    
                    # Replace return type
                    spec_func.return_type = bindings.get(spec_func.return_type, spec_func.return_type)

                    # Build local type mapping inside the function
                    old_var_types = self.var_types.copy()
                    self.var_types = {}
                    for p_name, p_type in spec_func.params:
                        self.var_types[p_name] = p_type

                    # Recursively specialize the body statements
                    spec_func.body = [self.visit(stmt) for stmt in spec_func.body]
                    self.var_types = old_var_types

                    # Register the new declaration
                    self.new_decls.append(spec_func)

                # Update CallNode callee to point to the specialized function
                if isinstance(node.callee, VarRefNode):
                    node.callee.name = spec_name
                else:
                    node.callee = spec_name

            return node

        return node

    def infer_type(self, node) -> str:
        if isinstance(node, LiteralNode):
            return node.type_name
        elif isinstance(node, VarRefNode):
            return self.var_types.get(node.name, "any")
        elif isinstance(node, BinaryOpNode):
            # Check left and right types
            left_type = self.infer_type(node.left)
            if left_type in ("float", "int"):
                return left_type
        elif isinstance(node, UnaryOpNode):
            return self.infer_type(node.operand)
        return "any"
