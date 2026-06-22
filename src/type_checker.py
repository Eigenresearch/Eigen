from src.ast import (
    ProgramNode, ImportNode, QFuncDeclNode, LetNode, VarDeclNode,
    BinaryOpNode, LiteralNode, VarRefNode, QFuncCallNode, GateNode,
    MeasureNode, IfNode, ReturnNode, TraceNode, PrintNode, AssertNode, ASTNode,
    FuncDeclNode, ForNode, WhileNode, BreakNode, ContinueNode, StructDeclNode,
    StructLiteralNode, DotAccessNode, ArrayLiteralNode, TupleLiteralNode,
    TryCatchNode, ThrowNode, EnumDeclNode, NoiseNode, AssignmentNode, CallNode,
    IndexAccessNode, MapAllocNode
)

class TypeErrorException(Exception):
    pass

class TypeChecker:
    def __init__(self):
        self.global_qfuncs = {}     # name -> QFuncDeclNode
        self.global_funcs = {}      # name -> FuncDeclNode
        self.global_structs = {}    # name -> StructDeclNode
        self.global_enums = {}      # name -> EnumDeclNode
        # Scope stack: list of dicts of name -> type_name
        self.scopes = [{}]
        self.current_function = None
        self.loop_depth = 0

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
        # Check if name is an enum variant
        for enum_name, enum_node in self.global_enums.items():
            if name in enum_node.variants:
                return enum_name
        self.error(f"Undeclared variable '{name}'", node)

    def types_compatible(self, expected: str, actual: str) -> bool:
        if expected == actual:
            return True
        if expected == "any" or actual == "any" or expected == "unknown" or actual == "unknown":
            return True
        if expected == "float" and actual == "int":
            return True
        if (expected == "cbit" and actual == "int") or (expected == "int" and actual == "cbit"):
            return True
        if expected == "null" or actual == "null":
            return True
        
        # Handle generic types like array<T>
        if expected.startswith("array<") and actual.startswith("array<"):
            t_exp = expected[6:-1]
            t_act = actual[6:-1]
            return self.types_compatible(t_exp, t_act)
            
        # Handle map<K, V>
        if expected.startswith("map<") and actual.startswith("map<"):
            parts_exp = expected[4:-1].split(",", 1)
            parts_act = actual[4:-1].split(",", 1)
            if len(parts_exp) == 2 and len(parts_act) == 2:
                k_exp, v_exp = parts_exp[0].strip(), parts_exp[1].strip()
                k_act, v_act = parts_act[0].strip(), parts_act[1].strip()
                return self.types_compatible(k_exp, k_act) and self.types_compatible(v_exp, v_act)
                
        return False

    def check(self, program: ProgramNode):
        self.global_qfuncs = {}
        self.global_funcs = {}
        self.global_structs = {}
        self.global_enums = {}
        self.scopes = [{}]
        self.current_function = None
        self.loop_depth = 0

        # Register all global declarations first (qfuncs, funcs, structs, enums)
        for node in program.body:
            if isinstance(node, QFuncDeclNode):
                if node.name in self.global_qfuncs or node.name in self.global_funcs:
                    self.error(f"Duplicate declaration of function '{node.name}'", node)
                self.global_qfuncs[node.name] = node
            elif isinstance(node, FuncDeclNode):
                if node.name in self.global_funcs or node.name in self.global_qfuncs:
                    self.error(f"Duplicate declaration of function '{node.name}'", node)
                self.global_funcs[node.name] = node
            elif isinstance(node, StructDeclNode):
                if node.name in self.global_structs:
                    self.error(f"Duplicate declaration of struct '{node.name}'", node)
                self.global_structs[node.name] = node
            elif isinstance(node, EnumDeclNode):
                if node.name in self.global_enums:
                    self.error(f"Duplicate declaration of enum '{node.name}'", node)
                self.global_enums[node.name] = node

        # Type check all global statements/functions
        for node in program.body:
            self.check_node(node)

    def check_node(self, node: ASTNode) -> str | None:
        if isinstance(node, QFuncDeclNode):
            self.enter_scope()
            for p_name, p_type in node.params:
                self.declare_var(p_name, p_type, node)
            for stmt in node.body:
                self.check_node(stmt)
            self.exit_scope()
            return None

        elif isinstance(node, FuncDeclNode):
            self.enter_scope()
            self.current_function = node
            for p_name, p_type in node.params:
                self.declare_var(p_name, p_type, node)
            for stmt in node.body:
                self.check_node(stmt)
            self.current_function = None
            self.exit_scope()
            return None

        elif isinstance(node, StructDeclNode):
            return None

        elif isinstance(node, EnumDeclNode):
            return None

        elif isinstance(node, LetNode):
            val_type = self.check_node(node.value)
            if not self.types_compatible(node.type_name, val_type):
                self.error(f"Cannot assign expression of type '{val_type}' to variable '{node.name}' of type '{node.type_name}'", node)
            self.declare_var(node.name, node.type_name, node)
            return None

        elif isinstance(node, VarDeclNode):
            self.declare_var(node.name, node.type_name, node)
            return None

        elif isinstance(node, AssignmentNode):
            target_type = self.check_node(node.target)
            val_type = self.check_node(node.value)
            if node.op in ("+=", "-=", "*=", "/="):
                if target_type not in ("int", "float") or val_type not in ("int", "float"):
                    self.error(f"Operator {node.op} not supported between types '{target_type}' and '{val_type}'", node)
            else:
                if not self.types_compatible(target_type, val_type):
                    self.error(f"Cannot assign type '{val_type}' to target of type '{target_type}'", node)
            return None

        elif isinstance(node, LiteralNode):
            return node.type_name

        elif isinstance(node, VarRefNode):
            return self.lookup_var(node.name, node)

        elif isinstance(node, BinaryOpNode):
            left_type = self.check_node(node.left)
            right_type = self.check_node(node.right)
            if node.op in ("==", "!=", "<", ">", "<=", ">="):
                if not self.types_compatible(left_type, right_type) and not self.types_compatible(right_type, left_type):
                    self.error(f"Comparison '{node.op}' not supported between '{left_type}' and '{right_type}'", node)
                return "bool"
            elif node.op in ("and", "or", "not"):
                if left_type != "bool" or (right_type and right_type != "bool"):
                    self.error(f"Logical operator '{node.op}' expects boolean arguments", node)
                return "bool"
            else:
                numeric_types = {"int", "float", "cbit"}
                if left_type not in numeric_types or right_type not in numeric_types:
                    self.error(f"Binary operation '{node.op}' not supported between '{left_type}' and '{right_type}'", node)
                if node.op == "/" or left_type == "float" or right_type == "float":
                    return "float"
                return "int"

        elif isinstance(node, ForNode):
            iter_type = self.check_node(node.iterable)
            if not iter_type.startswith("array<"):
                self.error(f"For loop expected array type, got '{iter_type}'", node)
            elem_type = iter_type[6:-1]
            self.enter_scope()
            self.declare_var(node.variable, elem_type, node)
            self.loop_depth += 1
            for stmt in node.body:
                self.check_node(stmt)
            self.loop_depth -= 1
            self.exit_scope()
            return None

        elif isinstance(node, WhileNode):
            cond_type = self.check_node(node.condition)
            if cond_type != "bool":
                self.error(f"While condition must be 'bool', got '{cond_type}'", node)
            self.enter_scope()
            self.loop_depth += 1
            for stmt in node.body:
                self.check_node(stmt)
            self.loop_depth -= 1
            self.exit_scope()
            return None

        elif isinstance(node, BreakNode) or isinstance(node, ContinueNode):
            if self.loop_depth <= 0:
                self.error(f"'{type(node).__name__}' statement outside of loop", node)
            return None

        elif isinstance(node, ReturnNode):
            if self.current_function:
                expected = self.current_function.return_type
                # ReturnNode has optional expr field (added in Phase 2)
                actual = "void"
                if hasattr(node, "expr") and node.expr is not None:
                    actual = self.check_node(node.expr)
                if not self.types_compatible(expected, actual):
                    self.error(f"Function return type mismatch: expected '{expected}', got '{actual}'", node)
            return None

        elif isinstance(node, StructLiteralNode):
            if node.struct_name not in self.global_structs:
                self.error(f"Undefined struct '{node.struct_name}'", node)
            s_decl = self.global_structs[node.struct_name]
            
            # If bindings is a list of tuples, convert to dict
            bindings = node.field_bindings
            if isinstance(bindings, list):
                bindings = dict(bindings)
                
            for f_name, f_type in s_decl.fields:
                if f_name not in bindings:
                    self.error(f"Missing field '{f_name}' in struct literal of '{node.struct_name}'", node)
                val_type = self.check_node(bindings[f_name])
                if not self.types_compatible(f_type, val_type):
                    self.error(f"Field '{f_name}' type mismatch: expected '{f_type}', got '{val_type}'", node)
            return node.struct_name

        elif isinstance(node, DotAccessNode):
            obj_type = self.check_node(node.obj)
            if obj_type not in self.global_structs:
                self.error(f"Cannot access member of non-struct object of type '{obj_type}'", node)
            s_decl = self.global_structs[obj_type]
            for f_name, f_type in s_decl.fields:
                if f_name == node.member:
                    return f_type
            self.error(f"Struct '{obj_type}' has no field '{node.member}'", node)

        elif isinstance(node, ArrayLiteralNode):
            if not node.elements:
                return "array<unknown>"
            elem_type = self.check_node(node.elements[0])
            for elem in node.elements[1:]:
                t = self.check_node(elem)
                if not self.types_compatible(elem_type, t):
                    elem_type = "any" # Coerce to any if mixed types
            return f"array<{elem_type}>"

        elif isinstance(node, MapAllocNode):
            if not node.keys:
                return "map<unknown,unknown>"
            k_type = self.check_node(node.keys[0])
            v_type = self.check_node(node.values[0])
            for k, v in zip(node.keys[1:], node.values[1:]):
                kt = self.check_node(k)
                vt = self.check_node(v)
                if not self.types_compatible(k_type, kt):
                    k_type = "any"
                if not self.types_compatible(v_type, vt):
                    v_type = "any"
            return f"map<{k_type},{v_type}>"

        elif isinstance(node, TupleLiteralNode):
            elem_types = [self.check_node(elem) for elem in node.elements]
            return f"tuple<{','.join(elem_types)}>"

        elif isinstance(node, TryCatchNode):
            for stmt in node.try_body:
                self.check_node(stmt)
            self.enter_scope()
            if node.catch_var:
                self.declare_var(node.catch_var, "string", node)
            for stmt in node.catch_body:
                self.check_node(stmt)
            self.exit_scope()
            return None

        elif isinstance(node, ThrowNode):
            self.check_node(node.expr)
            return None

        elif isinstance(node, NoiseNode):
            val_type = self.check_node(node.expr)
            if val_type not in ("int", "float"):
                self.error(f"Noise parameter must be numeric, got '{val_type}'", node)
            for target in node.targets:
                t_type = self.lookup_var(target, node)
                if t_type != "qubit":
                    self.error(f"Noise target '{target}' must be 'qubit', got '{t_type}'", node)
            return None

        elif isinstance(node, CallNode):
            # Callee can be a VarRefNode, or simple string identifier
            callee_name = None
            if isinstance(node.callee, VarRefNode):
                callee_name = node.callee.name
            elif isinstance(node.callee, str):
                callee_name = node.callee
                
            if callee_name and callee_name in self.global_funcs:
                func = self.global_funcs[callee_name]
                if len(node.args) != len(func.params):
                    self.error(f"Argument count mismatch for call to '{callee_name}': expected {len(func.params)}, got {len(node.args)}", node)
                
                # Perform simple generic binding if the function has generic parameters
                bindings = {}
                for arg, (p_name, p_type) in zip(node.args, func.params):
                    arg_type = self.check_node(arg)
                    
                    # If parameter is generic placeholder, bind it
                    if p_type in func.generic_params:
                        if p_type in bindings:
                            if not self.types_compatible(bindings[p_type], arg_type):
                                self.error(f"Generic parameter '{p_type}' bound to conflicting types '{bindings[p_type]}' and '{arg_type}'", node)
                        else:
                            bindings[p_type] = arg_type
                    else:
                        # Otherwise check type compatibility
                        if not self.types_compatible(p_type, arg_type):
                            self.error(f"Type mismatch for parameter '{p_name}': expected '{p_type}', got '{arg_type}'", node)
                
                # Resolve return type based on generic bindings
                ret_type = func.return_type
                if ret_type in bindings:
                    return bindings[ret_type]
                return ret_type
            else:
                self.error(f"Call to undefined classic function '{callee_name}'", node)

        elif isinstance(node, IndexAccessNode):
            obj_type = self.check_node(node.obj)
            idx_type = self.check_node(node.index)
            
            if obj_type.startswith("array<"):
                if idx_type != "int":
                    self.error(f"Array index must be 'int', got '{idx_type}'", node)
                return obj_type[6:-1]
                
            elif obj_type.startswith("map<"):
                parts = obj_type[4:-1].split(",", 1)
                if len(parts) == 2:
                    k_type, v_type = parts[0].strip(), parts[1].strip()
                    if not self.types_compatible(k_type, idx_type):
                        self.error(f"Map key type mismatch: expected '{k_type}', got '{idx_type}'", node)
                    return v_type
                    
            self.error(f"Index access not supported on type '{obj_type}'", node)

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
            for target in node.targets:
                t_type = self.lookup_var(target, node)
                if t_type != "qubit":
                    self.error(f"Gate '{node.gate_name}' target '{target}' must be of type 'qubit', got '{t_type}'", node)
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
            comparable = {"int", "float", "cbit", "bool"}
            if left_type not in comparable or right_type not in comparable:
                self.error(f"Condition comparison '{node.op}' not supported between '{left_type}' and '{right_type}'", node)
            self.enter_scope()
            for stmt in node.body:
                self.check_node(stmt)
            self.exit_scope()
            return None

        elif isinstance(node, TraceNode):
            return None

        elif isinstance(node, PrintNode):
            self.check_node(node.expr)
            return None

        elif isinstance(node, AssertNode):
            left_type = self.check_node(node.condition_left)
            right_type = self.check_node(node.condition_right)
            comparable = {"int", "float", "cbit", "bool"}
            if left_type not in comparable or right_type not in comparable:
                self.error(f"Assert comparison '{node.op}' not supported between '{left_type}' and '{right_type}'", node)
            return None

        elif isinstance(node, ProgramNode):
            self.check(node)
            return None

        else:
            self.error(f"Unknown AST Node type: {type(node).__name__}", node)
