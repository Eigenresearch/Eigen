from src.frontend.ast import (
    ProgramNode, QFuncDeclNode, LetNode, VarDeclNode,
    BinaryOpNode, LiteralNode, VarRefNode, QFuncCallNode, GateNode,
    MeasureNode, IfNode, ReturnNode, TraceNode, PrintNode, AssertNode, ASTNode
)
from src.ir.ir_graph import EQIRGraph, EQIRNode

class EQIRConverter:
    def __init__(self):
        self.graph = EQIRGraph()
        self.qfuncs = {}  # name -> QFuncDeclNode
        # Variable environment: maps name -> value (for static let variables)
        self.env = {}
        # Current active condition for conditional blocks (None or (cbit, op, val))
        self.current_condition = None

    def evaluate_expr(self, node: ASTNode) -> float | int | str:
        if isinstance(node, LiteralNode):
            return node.value
        elif isinstance(node, VarRefNode):
            if node.name in self.env:
                return self.env[node.name]
            # If it's not in self.env, it might be a qubit/cbit name, return as string
            return node.name
        elif isinstance(node, BinaryOpNode):
            try:
                left_val = self.evaluate_expr(node.left)
                right_val = self.evaluate_expr(node.right)
                if node.op == '+':
                    return left_val + right_val
                elif node.op == '-':
                    return left_val - right_val
                elif node.op == '*':
                    return left_val * right_val
                elif node.op == '/':
                    return left_val / right_val
                else:
                    return f"({left_val} {node.op} {right_val})"
            except Exception:
                return f"__expr_error__"
        else:
            return f"__unsupported_{type(node).__name__}__"

    def convert(self, program: ProgramNode) -> EQIRGraph:
        # Step 1: Register all qfuncs
        for node in program.body:
            if isinstance(node, QFuncDeclNode):
                self.qfuncs[node.name] = node

        # Step 2: Convert main body statements
        for node in program.body:
            # Skip qfunc declarations as they are processed via calls
            if isinstance(node, QFuncDeclNode):
                continue
            self.convert_node(node)

        return self.graph

    def convert_node(self, node: ASTNode, param_map: dict[str, str] = None):
        """
        Recursively converts AST nodes into EQIR operations.
        param_map is a dictionary mapping local qfunc parameter names to caller argument names.
        """
        if param_map is None:
            param_map = {}

        def resolve_qubit(name: str) -> str:
            # Resolve function parameter if applicable
            return param_map.get(name, name)

        if isinstance(node, VarDeclNode):
            if node.type_name == "qubit":
                self.graph.add_operation(
                    'ALLOC',
                    targets=[resolve_qubit(node.name)],
                    condition=self.current_condition
                )
            # Classical bits and other variables are handled dynamically, no compile-time DAG node needed for cbit alloc.
            # But let's add it if we want explicit classical bits tracking.
            return

        elif isinstance(node, LetNode):
            # Evaluate expression statically
            try:
                val = self.evaluate_expr(node.value)
                self.env[node.name] = val
            except Exception:
                self.env[node.name] = f"__unsupported_{type(node.value).__name__}__"
            return

        elif isinstance(node, GateNode):
            resolved_targets = [resolve_qubit(t) for t in node.targets]
            # Evaluate gate angles (if any)
            resolved_args = [self.evaluate_expr(arg) for arg in node.args]
            
            self.graph.add_operation(
                'GATE',
                gate_name=node.gate_name,
                targets=resolved_targets,
                args=resolved_args,
                condition=self.current_condition
            )
            return

        elif isinstance(node, MeasureNode):
            resolved_q = resolve_qubit(node.qubit_name)
            resolved_c = param_map.get(node.cbit_name, node.cbit_name)
            self.graph.add_operation(
                'MEASURE',
                targets=[resolved_q],
                cbit_name=resolved_c,
                condition=self.current_condition
            )
            return

        elif isinstance(node, QFuncCallNode):
            # Inline function call
            if node.name not in self.qfuncs:
                raise KeyError(f"Undefined qfunc '{node.name}'")
            qfunc = self.qfuncs[node.name]
            
            # Map parameters to arguments
            # node.args contains caller argument names, qfunc.params contains (name, type)
            local_map = {}
            for caller_arg, (param_name, param_type) in zip(node.args, qfunc.params):
                # If caller_arg is already mapped, resolve it
                resolved_arg = param_map.get(caller_arg, caller_arg)
                local_map[param_name] = resolved_arg

            # Compile qfunc body with the new parameter mapping
            for stmt in qfunc.body:
                self.convert_node(stmt, local_map)
            return

        elif isinstance(node, IfNode):
            # The condition is of the form: left == right
            # Left is usually a cbit variable, right is a literal value (like 0 or 1)
            left_val = self.evaluate_expr(node.condition_left)
            # If left_val resolves to a string, it means it's a variable reference (cbit name)
            if not isinstance(left_val, str):
                left_val = str(left_val)
            
            # Resolve cbit name if in a function call mapping
            cbit_name = param_map.get(left_val, left_val)
            right_val = self.evaluate_expr(node.condition_right)

            # Save existing condition to restore it later
            prev_cond = self.current_condition
            
            # For simplicity, we assume flat conditional block.
            # If there's an existing active condition, we combine it, but our language doesn't have nested ifs.
            # So just set it.
            self.current_condition = (cbit_name, node.op, right_val)

            # Compile body statements under this condition
            for stmt in node.body:
                self.convert_node(stmt, param_map)

            # Restore previous condition
            self.current_condition = prev_cond
            return

        elif isinstance(node, TraceNode):
            self.graph.add_operation('TRACE', condition=self.current_condition)
            return

        elif isinstance(node, PrintNode):
            # Print node can print a variable value
            resolved_expr = self.evaluate_expr(node.expr)
            # Resolve if it is mapped
            if isinstance(resolved_expr, str):
                resolved_expr = param_map.get(resolved_expr, resolved_expr)
            self.graph.add_operation('PRINT', print_expr=resolved_expr, condition=self.current_condition)
            return

        elif isinstance(node, AssertNode):
            left_val = self.evaluate_expr(node.condition_left)
            right_val = self.evaluate_expr(node.condition_right)
            if isinstance(left_val, str):
                left_val = param_map.get(left_val, left_val)
            self.graph.add_operation(
                'ASSERT',
                assert_cond=(left_val, node.op, right_val),
                condition=self.current_condition
            )
            return

        elif isinstance(node, ReturnNode):
            # Return inside qfunc does nothing during inlining, it just signals exit
            return

        else:
            # Other nodes (like ProgramNode, QFuncDeclNode) are not evaluated directly
            return
