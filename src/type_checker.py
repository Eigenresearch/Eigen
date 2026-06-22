from src.ast import (
    ProgramNode, ImportNode, QFuncDeclNode, LetNode, VarDeclNode,
    BinaryOpNode, LiteralNode, VarRefNode, QFuncCallNode, GateNode,
    MeasureNode, IfNode, ReturnNode, TraceNode, PrintNode, AssertNode, ASTNode
)

class TypeErrorException(Exception):
    pass

class TypeChecker:
    def __init__(self):
        self.global_qfuncs = {}  # name -> QFuncDeclNode
        # Scope stack: list of dicts of name -> type_name
        self.scopes = [{}]

    def error(self, msg: str, node: ASTNode = None):
        if node:
            raise TypeErrorException(f"Type Error: {msg} at {node}")
        else:
            raise TypeErrorException(f"Type Error: {msg}")

    def enter_scope(self):
        self.scopes.append({})

    def exit_scope(self):
        if len(self.scopes) > 1:
            self.scopes.pop()

    def declare_var(self, name: str, type_name: str, node: ASTNode = None):
        current_scope = self.scopes[-1]
        if name in current_scope:
            self.error(f"Redeclaration of variable '{name}' in the same scope", node)
        current_scope[name] = type_name

    def lookup_var(self, name: str, node: ASTNode = None) -> str:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        self.error(f"Undeclared variable '{name}'", node)

    def check(self, program: ProgramNode):
        # Phase 1: Register all qfuncs in the global namespace
        for node in program.body:
            if isinstance(node, QFuncDeclNode):
                if node.name in self.global_qfuncs:
                    self.error(f"Duplicate declaration of qfunc '{node.name}'", node)
                self.global_qfuncs[node.name] = node

        # Phase 2: Type check all statements in the program body
        for node in program.body:
            self.check_node(node)

    def check_node(self, node: ASTNode) -> str | None:
        """
        Recursively type-checks a node and returns its type (if it is an expression),
        or None (if it is a statement).
        """
        if isinstance(node, QFuncDeclNode):
            # qfuncs are checked in isolation
            self.enter_scope()
            for p_name, p_type in node.params:
                self.declare_var(p_name, p_type, node)
            
            for stmt in node.body:
                self.check_node(stmt)
            self.exit_scope()
            return None

        elif isinstance(node, LetNode):
            # Check value expression type
            val_type = self.check_node(node.value)
            # Check compatibility
            if node.type_name != val_type:
                # Allow minor coercion: int to float
                if node.type_name == "float" and val_type == "int":
                    pass
                else:
                    self.error(f"Cannot assign expression of type '{val_type}' to variable '{node.name}' of type '{node.type_name}'", node)
            self.declare_var(node.name, node.type_name, node)
            return None

        elif isinstance(node, VarDeclNode):
            self.declare_var(node.name, node.type_name, node)
            return None

        elif isinstance(node, BinaryOpNode):
            left_type = self.check_node(node.left)
            right_type = self.check_node(node.right)
            
            # Binary operations are only valid for numeric types
            numeric_types = {"int", "float", "cbit"}
            if left_type not in numeric_types or right_type not in numeric_types:
                self.error(f"Binary operation '{node.op}' not supported between '{left_type}' and '{right_type}'", node)
            
            # If division or either operand is float, result is float. Else it is int (or cbit-coerced int)
            if node.op == "/" or left_type == "float" or right_type == "float":
                return "float"
            return "int"

        elif isinstance(node, LiteralNode):
            return node.type_name

        elif isinstance(node, VarRefNode):
            return self.lookup_var(node.name, node)

        elif isinstance(node, QFuncCallNode):
            if node.name not in self.global_qfuncs:
                self.error(f"Call to undefined qfunc '{node.name}'", node)
            qfunc = self.global_qfuncs[node.name]
            
            if len(node.args) != len(qfunc.params):
                self.error(f"Argument count mismatch for qfunc '{node.name}': expected {len(qfunc.params)}, got {len(node.args)}", node)
                
            for arg_name, (param_name, param_type) in zip(node.args, qfunc.params):
                arg_type = self.lookup_var(arg_name, node)
                if arg_type != param_type:
                    self.error(f"Type mismatch for argument '{arg_name}': expected '{param_type}', got '{arg_type}'", node)
            return None

        elif isinstance(node, GateNode):
            # Check targets are qubits
            for target in node.targets:
                t_type = self.lookup_var(target, node)
                if t_type != "qubit":
                    self.error(f"Gate '{node.gate_name}' target '{target}' must be of type 'qubit', got '{t_type}'", node)
            
            # Check rotation angle arguments evaluate to float/int
            for arg in node.args:
                arg_type = self.check_node(arg)
                if arg_type not in ("int", "float"):
                    self.error(f"Rotation gate '{node.gate_name}' angle must evaluate to a number, got '{arg_type}'", node)
            return None

        elif isinstance(node, MeasureNode):
            q_type = self.lookup_var(node.qubit_name, node)
            c_type = self.lookup_var(node.cbit_name, node)
            if q_type != "qubit":
                self.error(f"Measurement target '{node.qubit_name}' must be of type 'qubit', got '{q_type}'", node)
            if c_type != "cbit":
                self.error(f"Measurement destination '{node.cbit_name}' must be of type 'cbit', got '{c_type}'", node)
            return None

        elif isinstance(node, IfNode):
            left_type = self.check_node(node.condition_left)
            right_type = self.check_node(node.condition_right)
            
            # Comparison condition: check both are comparable (numeric or cbit)
            comparable = {"int", "float", "cbit"}
            if left_type not in comparable or right_type not in comparable:
                self.error(f"Condition comparison '{node.op}' not supported between '{left_type}' and '{right_type}'", node)
                
            self.enter_scope()
            for stmt in node.body:
                self.check_node(stmt)
            self.exit_scope()
            return None

        elif isinstance(node, ReturnNode):
            return None

        elif isinstance(node, TraceNode):
            return None

        elif isinstance(node, PrintNode):
            self.check_node(node.expr)
            return None

        elif isinstance(node, AssertNode):
            left_type = self.check_node(node.condition_left)
            right_type = self.check_node(node.condition_right)
            comparable = {"int", "float", "cbit"}
            if left_type not in comparable or right_type not in comparable:
                self.error(f"Assert comparison '{node.op}' not supported between '{left_type}' and '{right_type}'", node)
            return None

        elif isinstance(node, ProgramNode):
            self.check(node)
            return None

        else:
            self.error(f"Unknown AST Node type: {type(node).__name__}", node)
