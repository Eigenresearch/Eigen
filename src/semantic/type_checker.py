from src.frontend.ast import (
    ProgramNode, ImportNode, QFuncDeclNode, LetNode, VarDeclNode,
    BinaryOpNode, LiteralNode, VarRefNode, QFuncCallNode, GateNode,
    MeasureNode, IfNode, ReturnNode, TraceNode, PrintNode, AssertNode, ASTNode,
    FuncDeclNode, ForNode, WhileNode, BreakNode, ContinueNode, StructDeclNode,
    StructLiteralNode, DotAccessNode, ArrayLiteralNode, TupleLiteralNode,
    TryCatchNode, ThrowNode, EnumDeclNode, NoiseNode, AssignmentNode, CallNode,
    IndexAccessNode, MapAllocNode, ParallelBlockNode, TaskStatementNode,
    MatchNode, StringInterpolationNode,
    TraitDeclNode, TraitMethodSignatureNode, ImplBlockNode,
    TypeAliasDeclNode,
)

class TypeErrorException(Exception):
    pass

class TypeChecker:
    STDLIB_FUNCTIONS = {
        'sin', 'cos', 'tan', 'sqrt', 'log', 'exp', 'abs',
        'mean', 'variance', 'rand_float', 'rand_int',
        'append_int', 'remove_at', 'read_file', 'write_file',
        'print_format', 'now', 'sleep', 'concat', 'format_int',
        'len', 'range', 'print', 'push', 'pop',
    }

    def __init__(self):
        self.global_qfuncs = {}     # name -> QFuncDeclNode
        self.global_funcs = {}      # name -> FuncDeclNode
        self.global_structs = {}    # name -> StructDeclNode
        self.global_enums = {}      # name -> EnumDeclNode
        # §3.1 — partial trait/interface registry.
        self.global_traits = {}     # name -> TraitDeclNode
        self.global_impls = []      # list of ImplBlockNode (for trait
                                    # conformance auditing; not enforced
                                    # at call sites in this P2 cut).
        # Scope stack: list of dicts of name -> type_name
        self.scopes = [{}]
        self.current_function = None
        self.loop_depth = 0
        self._type_cache = {}

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
        # §3.3 — resolve type alias references lazily at declaration time
        # so the canonical type is what later lookups see. Aliases are
        # registered in pass #1 of check_program, so by the time we
        # reach _check_node_uncached for any user stmt the table is
        # fully populated.
        try:
            type_name = self._resolve_type_alias(type_name)
        except TypeErrorException:
            # Cyclic alias chain surfaced during declare_var; re-raise
            # with a clearer context rather than swallowing.
            raise
        current_scope[name] = type_name

    def _resolve_type_alias(self, type_name: str, seen: set | None = None) -> str:
        """Resolve a (possibly alias-chain) type name to its concrete
        target. ``type A = B`` followed by ``type B = int`` makes
        ``_resolve_type_alias("A")`` return ``"int"``. Cyclic chains
        are detected via ``seen`` and signalled as a TypeErrorException
        so users get a clear "Circular type alias" message rather than
        a hang.
        """
        if not isinstance(type_name, str):
            return type_name
        if seen is None:
            seen = set()
        if type_name in seen:
            self.error(f"Circular type alias detected involving '{type_name}'")
        seen.add(type_name)
        if type_name in self.global_type_aliases:
            return self._resolve_type_alias(
                self.global_type_aliases[type_name], seen)
        return type_name

    def lookup_var(self, name: str, node: ASTNode = None) -> str:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        # Check if name is an enum variant
        for enum_name, enum_node in self.global_enums.items():
            if name in enum_node.variants:
                return enum_name
        # §7.3 — surface a "Did you mean?" candidate when the name is a
        # close typo of something already in scope / in the global
        # function or struct vocabulary. We accumulate the vocabulary
        # on demand (not at __init__ time) so newly registered names
        # stay visible.
        from src.frontend.did_you_mean import format_suggestion
        vocab = set()
        for scope in self.scopes:
            vocab.update(scope.keys())
        vocab.update(self.global_qfuncs.keys())
        vocab.update(self.global_funcs.keys())
        vocab.update(self.global_structs.keys())
        vocab.update(self.global_enums.keys())
        vocab.update(self.global_traits.keys())
        vocab.update(self.global_type_aliases.keys())
        hint = format_suggestion(name, vocab)
        self.error(f"Undeclared variable '{name}'{hint}", node)

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
        from src.frontend.ast import NATIVE_AVAILABLE
        if NATIVE_AVAILABLE and hasattr(program, "source") and program.source is not None:
            import eigen_native
            try:
                from src.compiler import get_workspace_root
                workspace_root = get_workspace_root()
                eigen_native.type_check_source(program.source, workspace_root)
                return
            except TypeError as e:
                raise TypeErrorException(f"Type Error: {str(e)}")

        self.global_qfuncs = {}
        self.global_funcs = {}
        self.global_structs = {}
        self.global_enums = {}
        self.global_traits = {}
        self.global_impls = []
        # §3.3 — type alias table. `name -> target_type` (resolution done
        # lazily at lookup time, see `_resolve_type_alias`). Acyclic —
        # circular chains raise TypeErrorException at first reference.
        self.global_type_aliases = {}
        self.scopes = [{}]
        self.current_function = None
        self.loop_depth = 0
        self._type_cache = {}

        # Register all global declarations first (qfuncs, funcs, structs, enums, traits)
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
            elif isinstance(node, TraitDeclNode):
                if node.name in self.global_traits:
                    self.error(f"Duplicate declaration of trait '{node.name}'", node)
                self.global_traits[node.name] = node
            elif isinstance(node, TypeAliasDeclNode):
                if node.name in self.global_type_aliases:
                    self.error(
                        f"Duplicate declaration of type alias '{node.name}'",
                        node)
                self.global_type_aliases[node.name] = node.target_type

        # Type check all global statements/functions
        for node in program.body:
            self.check_node(node)

    def check_node(self, node: ASTNode) -> str | None:
        if node is None:
            return None
        node_id = id(node)
        if node_id in self._type_cache:
            return self._type_cache[node_id]
        res = self._check_node_uncached(node)
        if res is not None:
            self._type_cache[node_id] = res
        return res

    def _check_node_uncached(self, node: ASTNode) -> str | None:
        if isinstance(node, QFuncDeclNode):
            self.enter_scope()
            saved_func = self.current_function
            self.current_function = node
            if not hasattr(node, 'return_type'):
                node.return_type = "void"
            for p_name, p_type in node.params:
                self.declare_var(p_name, p_type, node)
            for stmt in node.body:
                self.check_node(stmt)
            self.current_function = saved_func
            self.exit_scope()
            return None

        elif isinstance(node, FuncDeclNode):
            self.enter_scope()
            saved_func = self.current_function
            self.current_function = node
            for p_name, p_type in node.params:
                self.declare_var(p_name, p_type, node)
            for stmt in node.body:
                self.check_node(stmt)
            self.current_function = saved_func
            self.exit_scope()
            return None

        elif isinstance(node, StructDeclNode):
            return None

        elif isinstance(node, EnumDeclNode):
            return None

        elif isinstance(node, TraitDeclNode):
            # §3.1 — Type-check each method signature inside the trait.
            for method in node.methods:
                self.check_node(method)
            return None

        elif isinstance(node, TraitMethodSignatureNode):
            # Trait method signatures don't have a body, so we just
            # walk the parameter list to register types. No scope is
            # entered — these don't end up in the user-visible namespace.
            for p_name, p_type in node.params:
                # We don't declare; we only validate the type strings.
                # Empty bodies mean there's nothing further to check here.
                pass
            return None

        elif isinstance(node, ImplBlockNode):
            # §3.1 — Trait conformance check (partial).
            #
            # If the impl declares a trait, verify that the trait
            # actually exists, and that EVERY method on the trait has
            # a corresponding FuncDeclNode in the impl's body with a
            # matching name. Type signature matching is intentionally
            # loose in this P2 cut — we accept any impl that has the
            # right method names so users can iterate on signatures
            # without the checker blocking them constantly.
            if node.trait_name is not None:
                trait = self.global_traits.get(node.trait_name)
                if trait is None:
                    self.error(
                        f"Impl references unknown trait '{node.trait_name}'",
                        node)
                else:
                    impl_methods = {m.name for m in node.methods}
                    missing = trait.method_names() - impl_methods
                    if missing:
                        self.error(
                            f"Impl of trait '{node.trait_name}' for "
                            f"type '{node.target_type}' is missing "
                            f"methods: {sorted(missing)}", node)
            # Type-check each method body via the standard FuncDeclNode
            # path; reuse the same scope-tracking as free functions.
            # We enter the method scope and pre-declare `self` of the
            # impl's target type so user code referencing `self.x` is
            # accepted (Rust-like convention). We don't currently
            # check that the method actually accesses only declared
            # fields — that's left for a future real type-checker.
            for m in node.methods:
                self.enter_scope()
                # Synthetic `self:` binding as the impl target type.
                self.scopes[-1]["self"] = node.target_type
                saved_func = self.current_function
                self.current_function = m
                for p_name, p_type in m.params:
                    self.declare_var(p_name, p_type, m)
                for stmt in m.body:
                    self.check_node(stmt)
                self.current_function = saved_func
                self.exit_scope()
            # Record the impl for downstream tooling.
            self.global_impls.append(node)
            return None

        elif isinstance(node, TypeAliasDeclNode):
            # §3.3 — aliases are registered during the pass-1 sweep over
            # `program.body`, so by the time we reach them in pass 2
            # there's nothing left to type-check (the target type is just
            # a free-form string). We validate that the alias doesn't
            # form a cycle by attempting resolution right away — this
            # surfaces circular chains at declaration time rather than
            # at first *use*.
            self._resolve_type_alias(node.name)
            return None

        elif isinstance(node, LetNode):
            val_type = self.check_node(node.value)
            expected = self._resolve_type_alias(node.type_name)
            if not self.types_compatible(expected, val_type):
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
            elif node.op in ("%", "&", "|", "^", "~", "<<", ">>"):
                integer_types = {"int", "cbit"}
                if left_type not in integer_types or (right_type and right_type not in integer_types):
                    self.error(f"Operator '{node.op}' is only supported on integer types, got '{left_type}' and '{right_type}'", node)
                return "int"
            else:
                numeric_types = {"int", "float", "cbit"}
                if left_type not in numeric_types or right_type not in numeric_types:
                    self.error(f"Binary operation '{node.op}' not supported between '{left_type}' and '{right_type}'", node)
                if node.op == "**" and right_type == "int":
                    return "int" if left_type == "int" else "float"
                if left_type == "float" or right_type == "float":
                    return "float"
                return "int"

        elif isinstance(node, ForNode):
            iter_type = self.check_node(node.iterable)
            if iter_type.startswith("array<"):
                elem_type = iter_type[6:-1]
            elif iter_type == "int":
                elem_type = "int"
            elif iter_type == "any":
                elem_type = "any"
            else:
                self.error(f"For loop expected array or range type, got '{iter_type}'", node)
                elem_type = "unknown"
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
                expected = getattr(self.current_function, "return_type", "void")
                # §3.3 — substitute any type aliases on the function's
                # declared return type so `func f() -> MyAlias { ... }`
                # is checked against the alias' concrete target.
                expected = self._resolve_type_alias(expected)
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
            elif callee_name and callee_name in self.STDLIB_FUNCTIONS:
                for arg in node.args:
                    self.check_node(arg)
                if callee_name in ('sin', 'cos', 'tan', 'sqrt', 'log', 'exp', 'abs'):
                    return "float"
                elif callee_name in ('len', 'rand_int', 'format_int', 'append_int', 'remove_at'):
                    return "int"
                elif callee_name in ('rand_float', 'mean', 'variance'):
                    return "float"
                elif callee_name in ('concat', 'read_file', 'print_format'):
                    return "string"
                elif callee_name in ('range',):
                    return "array<int>"
                return "any"
            elif isinstance(node.callee, DotAccessNode):
                obj_type = self.check_node(node.callee.obj)
                for arg in node.args:
                    self.check_node(arg)
                return "any"
            elif callee_name is None:
                for arg in node.args:
                    self.check_node(arg)
                return "any"
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
            comparable = {"int", "float", "cbit", "bool", "string"}
            if left_type not in comparable or right_type not in comparable:
                if not self.types_compatible(left_type, right_type):
                    self.error(f"Condition comparison '{node.op}' not supported between '{left_type}' and '{right_type}'", node)
            self.enter_scope()
            for stmt in node.body:
                self.check_node(stmt)
            self.exit_scope()
            if hasattr(node, "else_body") and node.else_body:
                self.enter_scope()
                for stmt in node.else_body:
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
            comparable = {"int", "float", "cbit", "bool", "string"}
            if left_type not in comparable or right_type not in comparable:
                if not self.types_compatible(left_type, right_type):
                    self.error(f"Assert comparison '{node.op}' not supported between '{left_type}' and '{right_type}'", node)
            return None

        elif isinstance(node, ProgramNode):
            self.check(node)
            return None

        elif isinstance(node, ParallelBlockNode):
            for task in node.tasks:
                self.check_node(task)
            return None

        elif isinstance(node, TaskStatementNode):
            self.check_node(node.call)
            return None

        elif isinstance(node, MatchNode):
            match_type = self.check_node(node.expr)
            for pattern, body in node.cases:
                pattern_type = self.check_node(pattern)
                self.enter_scope()
                for stmt in body:
                    self.check_node(stmt)
                self.exit_scope()
            if node.default_body:
                self.enter_scope()
                for stmt in node.default_body:
                    self.check_node(stmt)
                self.exit_scope()
            return None

        elif isinstance(node, StringInterpolationNode):
            for part in node.parts:
                if not isinstance(part, str):
                    self.check_node(part)
            return "string"

        else:
            self.error(f"Unknown AST Node type: {type(node).__name__}", node)
