from src.frontend.ast import (
    ProgramNode, QFuncDeclNode, LetNode, VarDeclNode,
    BinaryOpNode, UnaryOpNode, LiteralNode, VarRefNode, QFuncCallNode, GateNode,
    MeasureNode, IfNode, ReturnNode, TraceNode, PrintNode, AssertNode, ASTNode,
    FuncDeclNode, ForNode, WhileNode, StructDeclNode, AssignmentNode, CallNode,
    BreakNode, ContinueNode, ThrowNode, TryCatchNode,
    MatchNode, StringInterpolationNode
)
from src.ir.ir_graph import EQIRGraph

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
                elif node.op == '**':
                    return left_val ** right_val
                elif node.op == '%':
                    return left_val % right_val
                elif node.op == '==':
                    return left_val == right_val
                elif node.op == '!=':
                    return left_val != right_val
                elif node.op == '<':
                    return left_val < right_val
                elif node.op == '>':
                    return left_val > right_val
                elif node.op == '<=':
                    return left_val <= right_val
                elif node.op == '>=':
                    return left_val >= right_val
                elif node.op in ('and', 'or', 'not'):
                    if node.op == 'and':
                        return bool(left_val) and bool(right_val)
                    elif node.op == 'or':
                        return bool(left_val) or bool(right_val)
                    elif node.op == 'not':
                        return not bool(left_val)
                else:
                    return f"({left_val} {node.op} {right_val})"
            except Exception as e:
                import sys
                print(f"DiagnosticWarning: evaluate_expr failed for BinaryOpNode({node.op}): {e}", file=sys.stderr)
                return f"__expr_error__"
        elif isinstance(node, UnaryOpNode):
            try:
                val = self.evaluate_expr(node.operand)
                if node.op == 'not':
                    return not bool(val)
                elif node.op == '~':
                    return ~val
                elif node.op == '-':
                    return -val
                else:
                    return f"({node.op}{val})"
            except Exception as e:
                import sys
                print(f"DiagnosticWarning: evaluate_expr failed for UnaryOpNode({node.op}): {e}", file=sys.stderr)
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
            # Classical bits and other variables are handled dynamically,
            # no compile-time DAG node needed for cbit alloc.
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
            for caller_arg, (param_name, _param_type) in zip(node.args, qfunc.params, strict=False):
                # If caller_arg is already mapped, resolve it
                resolved_arg = param_map.get(caller_arg, caller_arg)
                local_map[param_name] = resolved_arg

            # Compile qfunc body with the new parameter mapping
            for stmt in qfunc.body:
                self.convert_node(stmt, local_map)
            return

        elif isinstance(node, IfNode):
            left_val = self.evaluate_expr(node.condition_left)
            if not isinstance(left_val, str):
                left_val = str(left_val)
            
            cbit_name = param_map.get(left_val, left_val)
            right_val = self.evaluate_expr(node.condition_right)

            prev_cond = self.current_condition
            
            # Combine condition if nested
            if prev_cond is not None:
                prev_cbit, prev_op, prev_val = prev_cond
                cond1 = (f"({prev_cbit} {prev_op} {repr(prev_val)})"
                         if ' ' in prev_cbit or prev_op != '=='
                         else f"{prev_cbit} {prev_op} {repr(prev_val)}")
                cond2 = f"{cbit_name} {node.op} {repr(right_val)}"
                combined_expr = f"({cond1}) and ({cond2})"
                self.current_condition = (combined_expr, '==', True)
            else:
                self.current_condition = (cbit_name, node.op, right_val)

            for stmt in node.body:
                self.convert_node(stmt, param_map)
                
            if hasattr(node, "else_body") and node.else_body:
                opp_map = {
                    "==": "!=",
                    "!=": "==",
                    "<": ">=",
                    ">": "<=",
                    "<=": ">",
                    ">=": "<",
                }
                opp_op = opp_map.get(node.op, "!=")
                if prev_cond is not None:
                    prev_cbit, prev_op, prev_val = prev_cond
                    cond1 = (f"({prev_cbit} {prev_op} {repr(prev_val)})"
                             if ' ' in prev_cbit or prev_op != '=='
                             else f"{prev_cbit} {prev_op} {repr(prev_val)}")
                    cond2 = f"{cbit_name} {opp_op} {repr(right_val)}"
                    combined_expr = f"({cond1}) and ({cond2})"
                    self.current_condition = (combined_expr, '==', True)
                else:
                    self.current_condition = (cbit_name, opp_op, right_val)
                for stmt in node.else_body:
                    self.convert_node(stmt, param_map)

            self.current_condition = prev_cond
            return

        elif isinstance(node, TraceNode):
            self.graph.add_operation('TRACE', condition=self.current_condition)
            return

        elif isinstance(node, PrintNode):
            resolved_expr = self.evaluate_expr(node.expr)
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
            return

        elif isinstance(node, FuncDeclNode):
            self.graph.add_operation('FUNC_DECL', name=node.name)
            for stmt in node.body:
                self.convert_node(stmt, param_map)
            return

        elif isinstance(node, ForNode):
            self.graph.add_operation('FOR', loop_var=node.variable)
            for stmt in node.body:
                self.convert_node(stmt, param_map)
            return

        elif isinstance(node, WhileNode):
            self.graph.add_operation('WHILE')
            for stmt in node.body:
                self.convert_node(stmt, param_map)
            return

        elif isinstance(node, TryCatchNode):
            self.graph.add_operation('TRY_CATCH')
            for stmt in node.try_body:
                self.convert_node(stmt, param_map)
            for stmt in node.catch_body:
                self.convert_node(stmt, param_map)
            if getattr(node, 'finally_body', None):
                self.graph.add_operation('FINALLY')
                for stmt in node.finally_body:
                    self.convert_node(stmt, param_map)
            return

        elif isinstance(node, StructDeclNode):
            self.graph.add_operation('STRUCT_DECL', name=node.name)
            return

        elif isinstance(node, AssignmentNode):
            self.graph.add_operation('ASSIGN', target=self.evaluate_expr(node.target))
            return

        elif isinstance(node, CallNode):
            self.graph.add_operation('CALL', callee=self.evaluate_expr(node.callee))
            return

        elif isinstance(node, BreakNode):
            self.graph.add_operation('BREAK')
            return

        elif isinstance(node, ContinueNode):
            self.graph.add_operation('CONTINUE')
            return

        elif isinstance(node, ThrowNode):
            self.graph.add_operation('THROW')
            return

        elif isinstance(node, (ProgramNode, QFuncDeclNode)):
            return

        elif isinstance(node, MatchNode):
            self.graph.add_operation('MATCH')
            for _pattern, body in node.cases:
                for stmt in body:
                    self.convert_node(stmt, param_map)
            if hasattr(node, 'default_body') and node.default_body:
                for stmt in node.default_body:
                    self.convert_node(stmt, param_map)
            return

        elif isinstance(node, StringInterpolationNode):
            self.graph.add_operation('STRING_INTERP')
            return

        else:
            import sys
            print(f"DiagnosticWarning: AST node type '{type(node).__name__}' "
                  f"is not natively supported by EQIR converter.", file=sys.stderr)
            self.graph.add_operation('UNSUPPORTED', node_class=type(node).__name__)
            return
