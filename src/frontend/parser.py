from src.frontend.lexer import Token, TokenType
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

def node_to_str(node) -> str:
    from src.frontend.ast import VarRefNode, IndexAccessNode, LiteralNode
    if isinstance(node, VarRefNode):
        return node.name
    elif isinstance(node, IndexAccessNode):
        obj_str = node_to_str(node.obj)
        if isinstance(node.index, LiteralNode):
            index_str = str(node.index.value)
        elif isinstance(node.index, VarRefNode):
            index_str = node.index.name
        else:
            index_str = "0"
        return f"{obj_str}[{index_str}]"
    elif isinstance(node, LiteralNode):
        return str(node.value)
    else:
        return ""


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0
        self.errors = []
        self.recovery_tokens = {TokenType.SEMICOLON, TokenType.RBRACE, TokenType.IF,
                                  TokenType.LET, TokenType.FUNC, TokenType.QFUNC,
                                  TokenType.FOR, TokenType.WHILE, TokenType.RETURN,
                                  TokenType.BREAK, TokenType.CONTINUE, TokenType.EOF}
        self.declared_qfuncs = set()
        for i in range(len(tokens) - 1):
            if tokens[i].type == TokenType.QFUNC and tokens[i+1].type == TokenType.IDENTIFIER:
                self.declared_qfuncs.add(tokens[i+1].value)

    def error(self, msg: str):
        tok = self.current()
        err = SyntaxError(f"Parser Error at line {tok.line}, col {tok.column}: {msg} (found {tok})")
        self.errors.append(err)
        raise err

    def _recover(self):
        """Skip tokens until a recovery point is reached."""
        while self.current().type not in self.recovery_tokens:
            self.pos += 1
        if self.current().type == TokenType.SEMICOLON:
            self.pos += 1

    def current(self) -> Token:
        if self.pos >= len(self.tokens):
            return self.tokens[-1]
        return self.tokens[self.pos]

    def peek(self, offset: int = 1) -> Token:
        if self.pos + offset >= len(self.tokens):
            return self.tokens[-1]
        return self.tokens[self.pos + offset]

    def match(self, *expected_types: TokenType) -> Token | None:
        tok = self.current()
        if tok.type in expected_types:
            self.pos += 1
            return tok
        return None

    def consume(self, expected_type: TokenType, msg: str) -> Token:
        tok = self.match(expected_type)
        if tok is None:
            self.error(msg)
        return tok

    def parse(self) -> ProgramNode:
        from src.frontend.ast import NATIVE_AVAILABLE
        if NATIVE_AVAILABLE and hasattr(self.tokens, "source") and self.tokens.source is not None:
            import eigen_native
            try:
                return eigen_native.parse_native(self.tokens.source)
            except Exception:
                pass

        # File must start with version header: "eigen 1.0" or similar
        self.consume(TokenType.EIGEN, "Expected 'eigen' header directive at start of file")
        version_tok = self.match(TokenType.FLOAT_LIT, TokenType.INT_LIT)
        if not version_tok:
            self.error("Expected version number (e.g. 1.0) after 'eigen'")
        version = float(version_tok.value)

        # Optional module declaration: "module quantum.bell"
        module_name = None
        if self.match(TokenType.MODULE):
            module_name = self.parse_dotted_path()

        # Optional imports: "import quantum.bell"
        imports = []
        while self.match(TokenType.IMPORT):
            import_path = self.parse_dotted_path()
            imports.append(ImportNode(import_path))

        # Parse statements
        body = []
        while self.current().type != TokenType.EOF:
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)

        return ProgramNode(version, module_name, imports, body)

    def parse_dotted_path(self) -> str:
        parts = [self.consume(TokenType.IDENTIFIER, "Expected identifier").value]
        while self.match(TokenType.DOT):
            parts.append(self.consume(TokenType.IDENTIFIER, "Expected identifier after '.'").value)
        return ".".join(parts)

    def parse_type(self, allow_gates: bool = True) -> str:
        types_to_match = [
            TokenType.INT, TokenType.FLOAT, TokenType.STRING, TokenType.BOOL,
            TokenType.QUBIT, TokenType.CBIT, TokenType.ARRAY, TokenType.MAP,
            TokenType.IDENTIFIER
        ]
        if allow_gates:
            types_to_match.extend([
                TokenType.GATE_H, TokenType.GATE_X, TokenType.GATE_Y, TokenType.GATE_Z, TokenType.GATE_S, TokenType.GATE_T
            ])
        tok = self.match(*types_to_match)
        if not tok:
            # §7.3 — surface a "Did you mean?" suggestion when the caller's
            # token looks like a typo of a known primitive type or gate.
            from src.frontend.did_you_mean import format_suggestion
            cur = self.current()
            vocab = ["int", "float", "string", "bool", "qubit", "cbit",
                     "array", "map", "H", "X", "Y", "Z", "S", "T",
                     "CNOT", "CZ", "SWAP", "RX", "RY", "RZ", "CCX",
                     "CSWAP", "CP", "CRX", "CRY", "CRZ"]
            # Tight cap (1) — we only want close single-edit typos since
            # the type slot's vocab is short tokens where 2-edit matches
            # are too permissive ("42" → "H" is 2 edits but not a
            # meaningful suggestion).
            hint = format_suggestion(str(cur.value), vocab, max_distance=1)
            self.error(f"Expected type name{hint}")
        
        type_str = tok.value
        # Check if it has generic parameters, e.g. <T> or <K, V>
        if self.match(TokenType.LT):
            generic_types = [self.parse_type(allow_gates=True)]
            while self.match(TokenType.COMMA):
                generic_types.append(self.parse_type(allow_gates=True))
            self.consume(TokenType.GT, "Expected '>' after generic type parameters")
            type_str += "<" + ", ".join(generic_types) + ">"
        
        return type_str

    def parse_statement(self) -> ASTNode:
        tok = self.current()

        # qfunc declaration
        if tok.type == TokenType.QFUNC:
            return self.parse_qfunc_decl()

        # func declaration
        if tok.type == TokenType.FUNC:
            return self.parse_func_decl()

        # struct declaration
        if tok.type == TokenType.STRUCT:
            return self.parse_struct_decl()

        # enum declaration
        if tok.type == TokenType.ENUM:
            return self.parse_enum_decl()

        # §3.1 — Trait declaration: `trait Foo { fn bar(...) -> ...; ... }`
        if tok.type == TokenType.TRAIT:
            return self.parse_trait_decl()

        # §3.1 — Impl block: `impl Trait for Type { ... }` (or
        # `impl Type { ... }` inherent impl).
        if tok.type == TokenType.IMPL:
            return self.parse_impl_block()

        # §3.3 — Type alias: `type Name = Target;`
        if tok.type == TokenType.TYPE:
            return self.parse_type_alias_decl()

        # let assignment
        if tok.type == TokenType.LET:
            return self.parse_let()

        # for loop
        if tok.type == TokenType.FOR:
            return self.parse_for()

        # while loop
        if tok.type == TokenType.WHILE:
            return self.parse_while()

        # break / continue
        if tok.type == TokenType.BREAK:
            self.advance()
            return BreakNode()
        if tok.type == TokenType.CONTINUE:
            self.advance()
            return ContinueNode()

        # try-catch
        if tok.type == TokenType.TRY:
            return self.parse_try_catch()

        # throw
        if tok.type == TokenType.THROW:
            self.advance()
            return ThrowNode(self.parse_expr())

        # noise
        if tok.type == TokenType.NOISE:
            return self.parse_noise()

        # if statement
        if tok.type == TokenType.IF:
            return self.parse_if()

        # measure
        if tok.type == TokenType.MEASURE:
            self.advance()
            q_tok = self.consume(TokenType.IDENTIFIER, "Expected qubit identifier to measure")
            self.consume(TokenType.ARROW, "Expected '->' after qubit in measure statement")
            c_tok = self.consume(TokenType.IDENTIFIER, "Expected classical bit identifier to store measurement")
            return MeasureNode(q_tok.value, c_tok.value)

        if tok.type == TokenType.RETURN:
            self.advance()
            expr = None
            if self.current().type not in (TokenType.RBRACE, TokenType.EOF, TokenType.COMMA):
                # Also check if it's not a block ending
                expr = self.parse_expr()
            return ReturnNode(expr)

        # trace
        if tok.type == TokenType.TRACE:
            self.advance()
            return TraceNode()

        # print
        if tok.type == TokenType.PRINT:
            self.advance()
            expr = self.parse_expr()
            return PrintNode(expr)

        # assert
        if tok.type == TokenType.ASSERT:
            self.advance()
            left = self.parse_expr()
            op_tok = self.match(TokenType.EQ, TokenType.NE, TokenType.LT, TokenType.GT, TokenType.LE, TokenType.GE)
            if op_tok:
                right = self.parse_expr()
                return AssertNode(left, op_tok.value, right)
            else:
                return AssertNode(left, "==", LiteralNode(True, "bool"))

        # parallel block
        if tok.type == TokenType.PARALLEL:
            return self.parse_parallel_block()

        # match statement
        if tok.type == TokenType.MATCH:
            return self.parse_match()

        # Variable declarations: e.g. qubit q0, cbit c0, map<string, int> m, MyStruct s
        # We can backtrack to see if it is a type followed by an identifier.
        saved_pos = self.pos
        try:
            type_name = self.parse_type(allow_gates=False)
            if self.current().type == TokenType.IDENTIFIER:
                name_tok = self.consume(TokenType.IDENTIFIER, "Expected identifier")
                return VarDeclNode(name_tok.value, type_name)
        except Exception:
            pass
        self.pos = saved_pos

        # Custom qfunc/func call at statement level: e.g. bell(q0, q1)
        if tok.type == TokenType.IDENTIFIER and self.peek().type == TokenType.LPAREN:
            name_tok = self.consume(TokenType.IDENTIFIER, "Expected identifier")
            self.consume(TokenType.LPAREN, "Expected '('")
            args = []
            if self.current().type != TokenType.RPAREN:
                args.append(self.parse_expr())
                while self.match(TokenType.COMMA):
                    args.append(self.parse_expr())
            self.consume(TokenType.RPAREN, "Expected ')'")
            
            if name_tok.value in self.declared_qfuncs:
                return QFuncCallNode(name_tok.value, [node_to_str(arg) for arg in args])
            else:
                return CallNode(VarRefNode(name_tok.value), args)

        # Built-in gates
        gates_with_1_qubit = (
            TokenType.GATE_H, TokenType.GATE_X, TokenType.GATE_Y, TokenType.GATE_Z,
            TokenType.GATE_S, TokenType.GATE_T
        )
        if tok.type in gates_with_1_qubit:
            gate_tok = self.match(*gates_with_1_qubit)
            q_tok = self.consume(TokenType.IDENTIFIER, f"Expected qubit identifier for gate {gate_tok.value}")
            return GateNode(gate_tok.value, [q_tok.value], [])

        gates_with_2_qubit = (
            TokenType.GATE_CNOT, TokenType.GATE_CZ, TokenType.GATE_SWAP
        )
        if tok.type in gates_with_2_qubit:
            gate_tok = self.match(*gates_with_2_qubit)
            q1_tok = self.consume(TokenType.IDENTIFIER, f"Expected first qubit identifier for gate {gate_tok.value}")
            self.consume(TokenType.COMMA, f"Expected ',' between qubits for gate {gate_tok.value}")
            q2_tok = self.consume(TokenType.IDENTIFIER, f"Expected second qubit identifier for gate {gate_tok.value}")
            return GateNode(gate_tok.value, [q1_tok.value, q2_tok.value], [])

        rotation_gates = (
            TokenType.GATE_RX, TokenType.GATE_RY, TokenType.GATE_RZ
        )
        if tok.type in rotation_gates:
            gate_tok = self.match(*rotation_gates)
            q_tok = self.consume(TokenType.IDENTIFIER, f"Expected qubit identifier for rotation gate {gate_tok.value}")
            self.consume(TokenType.COMMA, f"Expected ',' before rotation angle for gate {gate_tok.value}")
            angle_expr = self.parse_expr()
            return GateNode(gate_tok.value, [q_tok.value], [angle_expr])

        gates_with_3_qubit = (
            TokenType.GATE_CCX, TokenType.GATE_CSWAP
        )
        if tok.type in gates_with_3_qubit:
            gate_tok = self.match(*gates_with_3_qubit)
            q1_tok = self.consume(TokenType.IDENTIFIER, f"Expected first qubit identifier for gate {gate_tok.value}")
            self.consume(TokenType.COMMA, f"Expected ',' between qubits for gate {gate_tok.value}")
            q2_tok = self.consume(TokenType.IDENTIFIER, f"Expected second qubit identifier for gate {gate_tok.value}")
            self.consume(TokenType.COMMA, f"Expected ',' between qubits for gate {gate_tok.value}")
            q3_tok = self.consume(TokenType.IDENTIFIER, f"Expected third qubit identifier for gate {gate_tok.value}")
            
            g_name = gate_tok.value.upper()
            if g_name in ("TOFFOLI", "CCX"):
                g_name = "CCX"
            elif g_name in ("FREDKIN", "CSWAP"):
                g_name = "CSWAP"
                
            return GateNode(g_name, [q1_tok.value, q2_tok.value, q3_tok.value], [])

        controlled_rotations = (
            TokenType.GATE_CP, TokenType.GATE_CRX, TokenType.GATE_CRY, TokenType.GATE_CRZ
        )
        if tok.type in controlled_rotations:
            gate_tok = self.match(*controlled_rotations)
            q1_tok = self.consume(TokenType.IDENTIFIER, f"Expected control qubit identifier for gate {gate_tok.value}")
            self.consume(TokenType.COMMA, f"Expected ',' between qubits for gate {gate_tok.value}")
            q2_tok = self.consume(TokenType.IDENTIFIER, f"Expected target qubit identifier for gate {gate_tok.value}")
            self.consume(TokenType.COMMA, f"Expected ',' before rotation angle for gate {gate_tok.value}")
            angle_expr = self.parse_expr()
            return GateNode(gate_tok.value.upper(), [q1_tok.value, q2_tok.value], [angle_expr])

        # Try parsing as an assignment or general expression statement
        expression_starters = (
            TokenType.IDENTIFIER, TokenType.INT_LIT, TokenType.FLOAT_LIT, TokenType.STRING_LIT,
            TokenType.TRUE, TokenType.FALSE, TokenType.NULL, TokenType.LPAREN, TokenType.LBRACK,
            TokenType.MINUS, TokenType.PLUS, TokenType.NOT, TokenType.PI, TokenType.TAU, TokenType.E
        )
        if tok.type in expression_starters:
            saved_pos = self.pos
            try:
                expr = self.parse_expr()
                assign_ops = (TokenType.EQUALS, TokenType.ADD_ASSIGN, TokenType.SUB_ASSIGN, TokenType.MUL_ASSIGN, TokenType.DIV_ASSIGN)
                if self.current().type in assign_ops:
                    op_tok = self.match(*assign_ops)
                    val_expr = self.parse_expr()
                    return AssignmentNode(expr, op_tok.value, val_expr)
                else:
                    return expr
            except Exception:
                self.pos = saved_pos

        self.error(f"Unexpected token in statement: {tok}")

    def advance(self):
        self.pos += 1

    def parse_qfunc_decl(self) -> QFuncDeclNode:
        self.consume(TokenType.QFUNC, "Expected 'qfunc'")
        name_tok = self.consume(TokenType.IDENTIFIER, "Expected identifier for qfunc name")
        self.consume(TokenType.LPAREN, "Expected '('")
        
        params = []
        if self.current().type != TokenType.RPAREN:
            p_type_tok = self.match(TokenType.QUBIT, TokenType.CBIT, TokenType.INT, TokenType.FLOAT)
            if not p_type_tok:
                self.error("Expected type for qfunc parameter")
            p_name_tok = self.consume(TokenType.IDENTIFIER, "Expected parameter name")
            params.append((p_name_tok.value, p_type_tok.value))
            
            while self.match(TokenType.COMMA):
                p_type_tok = self.match(TokenType.QUBIT, TokenType.CBIT, TokenType.INT, TokenType.FLOAT)
                if not p_type_tok:
                    self.error("Expected type for qfunc parameter")
                p_name_tok = self.consume(TokenType.IDENTIFIER, "Expected parameter name")
                params.append((p_name_tok.value, p_type_tok.value))
                
        self.consume(TokenType.RPAREN, "Expected ')'")
        self.consume(TokenType.LBRACE, "Expected '{'")
        
        body = []
        while self.current().type != TokenType.RBRACE and self.current().type != TokenType.EOF:
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
                
        self.consume(TokenType.RBRACE, "Expected '}'")
        return QFuncDeclNode(name_tok.value, params, body)

    def parse_generic_param_name(self) -> str:
        tok = self.current()
        if tok.type in (TokenType.IDENTIFIER, TokenType.GATE_H, TokenType.GATE_X, TokenType.GATE_Y, TokenType.GATE_Z, TokenType.GATE_S, TokenType.GATE_T):
            self.pos += 1
            return tok.value
        self.error("Expected generic parameter name")

    def parse_func_decl(self) -> FuncDeclNode:
        self.consume(TokenType.FUNC, "Expected 'func'")
        name_tok = self.consume(TokenType.IDENTIFIER, "Expected function name")
        
        generic_params = []
        if self.match(TokenType.LT):
            generic_params.append(self.parse_generic_param_name())
            while self.match(TokenType.COMMA):
                generic_params.append(self.parse_generic_param_name())
            self.consume(TokenType.GT, "Expected '>'")
            
        self.consume(TokenType.LPAREN, "Expected '('")
        params = []
        if self.current().type != TokenType.RPAREN:
            p_name_tok = self.consume(TokenType.IDENTIFIER, "Expected parameter name")
            self.consume(TokenType.COLON, "Expected ':'")
            p_type = self.parse_type()
            params.append((p_name_tok.value, p_type))
            
            while self.match(TokenType.COMMA):
                p_name_tok = self.consume(TokenType.IDENTIFIER, "Expected parameter name")
                self.consume(TokenType.COLON, "Expected ':'")
                p_type = self.parse_type()
                params.append((p_name_tok.value, p_type))
                
        self.consume(TokenType.RPAREN, "Expected ')'")
        return_type = "void"
        if self.match(TokenType.ARROW):
            return_type = self.parse_type()
        
        self.consume(TokenType.LBRACE, "Expected '{'")
        body = []
        while self.current().type != TokenType.RBRACE and self.current().type != TokenType.EOF:
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
        self.consume(TokenType.RBRACE, "Expected '}'")
        
        return FuncDeclNode(name_tok.value, generic_params, params, return_type, body)

    def parse_struct_decl(self) -> StructDeclNode:
        self.consume(TokenType.STRUCT, "Expected 'struct'")
        name_tok = self.consume(TokenType.IDENTIFIER, "Expected struct name")
        
        generic_params = []
        if self.match(TokenType.LT):
            generic_params.append(self.parse_generic_param_name())
            while self.match(TokenType.COMMA):
                generic_params.append(self.parse_generic_param_name())
            self.consume(TokenType.GT, "Expected '>'")
            
        self.consume(TokenType.LBRACE, "Expected '{'")
        fields = []
        while self.current().type != TokenType.RBRACE and self.current().type != TokenType.EOF:
            f_name_tok = self.consume(TokenType.IDENTIFIER, "Expected field name")
            self.consume(TokenType.COLON, "Expected ':'")
            f_type = self.parse_type()
            fields.append((f_name_tok.value, f_type))
            self.match(TokenType.COMMA)
            
        self.consume(TokenType.RBRACE, "Expected '}'")
        return StructDeclNode(name_tok.value, generic_params, fields)

    def parse_enum_decl(self) -> EnumDeclNode:
        self.consume(TokenType.ENUM, "Expected 'enum'")
        name_tok = self.consume(TokenType.IDENTIFIER, "Expected enum name")
        self.consume(TokenType.LBRACE, "Expected '{'")
        variants = []
        if self.current().type != TokenType.RBRACE:
            v_tok = self.consume(TokenType.IDENTIFIER, "Expected variant name")
            variants.append(v_tok.value)
            while self.match(TokenType.COMMA):
                if self.current().type == TokenType.RBRACE:
                    break
                v_tok = self.consume(TokenType.IDENTIFIER, "Expected variant name")
                variants.append(v_tok.value)
        self.consume(TokenType.RBRACE, "Expected '}'")
        return EnumDeclNode(name_tok.value, variants)

    # === §3.1 — Trait/Interface System (partial: AST/parser surface). =====

    def parse_trait_decl(self) -> TraitDeclNode:
        """Parse `trait Foo[<T>] { [fn name(...) -> ...;]* }`.

        Trait methods have signature-only bodies — they use
        `FuncDeclNode`'s parameter grammar but emit
        `TraitMethodSignatureNode` with an empty body.
        """
        self.consume(TokenType.TRAIT, "Expected 'trait'")
        name_tok = self.consume(TokenType.IDENTIFIER, "Expected trait name")

        generic_params = []
        if self.match(TokenType.LT):
            generic_params.append(self.parse_generic_param_name())
            while self.match(TokenType.COMMA):
                generic_params.append(self.parse_generic_param_name())
            self.consume(TokenType.GT, "Expected '>'")

        self.consume(TokenType.LBRACE, "Expected '{'")

        methods = []
        while self.current().type not in (TokenType.RBRACE, TokenType.EOF):
            # Require `fn` ... but Eigen uses `func` historically; accept
            # both so users migrating existing struct method syntax can
            # reuse `func`.
            method_kw = self.match(TokenType.FUNC, TokenType.QFUNC)
            if method_kw is None:
                self.error("Expected 'func' inside trait body")
            m_name = self.consume(TokenType.IDENTIFIER,
                                   "Expected method name").value
            # Per-method generics (rare): skip if present.
            m_generics = []
            if self.match(TokenType.LT):
                m_generics.append(self.parse_generic_param_name())
                while self.match(TokenType.COMMA):
                    m_generics.append(self.parse_generic_param_name())
                self.consume(TokenType.GT, "Expected '>'")
            self.consume(TokenType.LPAREN, "Expected '('")
            params = []
            if self.current().type != TokenType.RPAREN:
                p_name_tok = self.consume(TokenType.IDENTIFIER,
                                            "Expected parameter name")
                self.consume(TokenType.COLON, "Expected ':'")
                p_type = self.parse_type()
                params.append((p_name_tok.value, p_type))
                while self.match(TokenType.COMMA):
                    p_name_tok = self.consume(TokenType.IDENTIFIER,
                                                "Expected parameter name")
                    self.consume(TokenType.COLON, "Expected ':'")
                    p_type = self.parse_type()
                    params.append((p_name_tok.value, p_type))
            self.consume(TokenType.RPAREN, "Expected ')'")
            return_type = "void"
            if self.match(TokenType.ARROW):
                return_type = self.parse_type()
            # Trait method signatures end with ';' or '}' (we accept both
            # for compatibility with users who'd otherwise write the body).
            self.match(TokenType.SEMICOLON)
            methods.append(TraitMethodSignatureNode(
                m_name, m_generics, params, return_type,
            ))

        self.consume(TokenType.RBRACE, "Expected '}'")
        return TraitDeclNode(name_tok.value, generic_params, methods)

    # === §3.3 — Type Aliases (partial: AST/parser surface). =============

    def parse_type_alias_decl(self) -> TypeAliasDeclNode:
        """Parse `type Name = Target;`.

        The target type is parsed as the existing free-form type grammar
        via `parse_type` (allowing references to existing types and
        generic shapes like `Map<string, int>`). The body is single-
        statement and must end with either a `;` or the start of the
        next statement (`func`, `qfunc`, `let`, EOF, etc.) — we accept
        `;` optimally, otherwise we let the caller's statement-loop
        continue without consuming a lookahead token.

        Alias resolution is done lazily by the type checker at every
        type-reference site, so circular aliases (`type A = B; type B =
        A;`) will fail at first lookup with an explicit Undeclared-
        variable-style error rather than hanging.
        """
        self.consume(TokenType.TYPE, "Expected 'type'")
        name_tok = self.consume(TokenType.IDENTIFIER,
                                "Expected type alias name")
        self.consume(TokenType.EQUALS, "Expected '=' after type alias name")
        target = self.parse_type(allow_gates=False)
        # `;` is the canonical terminator; we accept no-semicolon for
        # newline-friendly source where the next token unambiguously
        # starts a new statement.
        self.match(TokenType.SEMICOLON)
        return TypeAliasDeclNode(name_tok.value, target)

    def parse_impl_block(self) -> ImplBlockNode:
        """Parse `impl Trait for Type { ...func... }` or `impl Type { ... }`.

        The first IDENT after `impl` may be either:
          * a trait name (if followed by `for Type`), or
          * the target type itself (an inherent impl; `trait_name = None`).
        """
        self.consume(TokenType.IMPL, "Expected 'impl'")
        first_name = self.consume(TokenType.IDENTIFIER, "Expected identifier after 'impl'").value

        # Optional generic parameters on the impl, e.g. `impl Vector<T> for Foo<T>`.
        if self.match(TokenType.LT):
            # Skip generics — we just record the name.
            self.parse_generic_param_name()
            while self.match(TokenType.COMMA):
                self.parse_generic_param_name()
            self.consume(TokenType.GT, "Expected '>'")

        # `for Type` form. We support the `for` keyword as a contextual
        # marker: if the next token is `FOR`, this is a trait impl; else
        # it's an inherent impl (`impl Type { ... }`).
        trait_name = None
        target_type = first_name
        if self.match(TokenType.FOR):
            target_type = self.consume(TokenType.IDENTIFIER,
                                        "Expected type name after 'for'").value
            trait_name = first_name

        # Strip any optional generic-args tail on the target type
        # (e.g. `impl Trait for Vector<T>` — we keep just `Vector`).
        if self.match(TokenType.LT):
            self.parse_generic_param_name()
            while self.match(TokenType.COMMA):
                self.parse_generic_param_name()
            self.consume(TokenType.GT, "Expected '>'")

        self.consume(TokenType.LBRACE, "Expected '{'")
        methods = []
        while self.current().type not in (TokenType.RBRACE, TokenType.EOF):
            # Use the existing func-decl parser to keep method signatures
            # and bodies consistent with free functions. The func-decl
            # parser expects `func` as the first token, so it composes
            # cleanly.
            stmt = self.parse_statement()
            if isinstance(stmt, FuncDeclNode):
                methods.append(stmt)
            else:
                self.error(f"Expected 'func' inside impl block, got {type(stmt).__name__}")

        self.consume(TokenType.RBRACE, "Expected '}'")
        return ImplBlockNode(trait_name, target_type, methods)

    def parse_let(self) -> LetNode:
        self.consume(TokenType.LET, "Expected 'let'")
        name_tok = self.consume(TokenType.IDENTIFIER, "Expected identifier for variable")
        self.consume(TokenType.COLON, "Expected ':' after variable name")
        type_name = self.parse_type()
        self.consume(TokenType.EQUALS, "Expected '=' in let statement")
        value_expr = self.parse_expr()
        return LetNode(name_tok.value, type_name, value_expr)

    def parse_if_tail(self) -> list:
        else_body = []
        if self.match(TokenType.ELSE):
            if self.match(TokenType.IF):
                left = self.parse_expr()
                op_tok = self.match(TokenType.EQ, TokenType.NE, TokenType.LT, TokenType.GT, TokenType.LE, TokenType.GE)
                if op_tok:
                    right = self.parse_expr()
                else:
                    op_tok = Token(TokenType.EQ, "==", self.current().line, self.current().column)
                    right = LiteralNode(True, "bool")
                self.consume(TokenType.LBRACE, "Expected '{'")
                body = []
                while self.current().type != TokenType.RBRACE and self.current().type != TokenType.EOF:
                    stmt = self.parse_statement()
                    if stmt:
                        body.append(stmt)
                self.consume(TokenType.RBRACE, "Expected '}'")
                
                nested_else = self.parse_if_tail()
                elif_node = IfNode(left, op_tok.value, right, body, nested_else)
                else_body.append(elif_node)
            else:
                self.consume(TokenType.LBRACE, "Expected '{' after 'else'")
                while self.current().type != TokenType.RBRACE and self.current().type != TokenType.EOF:
                    stmt = self.parse_statement()
                    if stmt:
                        else_body.append(stmt)
                self.consume(TokenType.RBRACE, "Expected '}'")
        elif self.match(TokenType.ELIF):
            left = self.parse_expr()
            op_tok = self.match(TokenType.EQ, TokenType.NE, TokenType.LT, TokenType.GT, TokenType.LE, TokenType.GE)
            if op_tok:
                right = self.parse_expr()
            else:
                op_tok = Token(TokenType.EQ, "==", self.current().line, self.current().column)
                right = LiteralNode(True, "bool")
            self.consume(TokenType.LBRACE, "Expected '{'")
            body = []
            while self.current().type != TokenType.RBRACE and self.current().type != TokenType.EOF:
                stmt = self.parse_statement()
                if stmt:
                    body.append(stmt)
            self.consume(TokenType.RBRACE, "Expected '}'")
            
            nested_else = self.parse_if_tail()
            elif_node = IfNode(left, op_tok.value, right, body, nested_else)
            else_body.append(elif_node)
        return else_body

    def parse_if(self) -> IfNode:
        self.consume(TokenType.IF, "Expected 'if'")
        left = self.parse_expr()
        op_tok = self.match(TokenType.EQ, TokenType.NE, TokenType.LT, TokenType.GT, TokenType.LE, TokenType.GE)
        if op_tok:
            right = self.parse_expr()
        else:
            op_tok = Token(TokenType.EQ, "==", self.current().line, self.current().column)
            right = LiteralNode(True, "bool")
        
        self.consume(TokenType.LBRACE, "Expected '{' to start if branch")
        body = []
        while self.current().type != TokenType.RBRACE and self.current().type != TokenType.EOF:
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
        self.consume(TokenType.RBRACE, "Expected '}' to end if branch")
        else_body = self.parse_if_tail()
        return IfNode(left, op_tok.value, right, body, else_body)

    def parse_for(self) -> ForNode:
        self.consume(TokenType.FOR, "Expected 'for'")
        var_tok = self.consume(TokenType.IDENTIFIER, "Expected loop variable")
        self.consume(TokenType.IN, "Expected 'in'")
        iterable = self.parse_expr()
        
        self.consume(TokenType.LBRACE, "Expected '{'")
        body = []
        while self.current().type != TokenType.RBRACE and self.current().type != TokenType.EOF:
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
        self.consume(TokenType.RBRACE, "Expected '}'")
        return ForNode(var_tok.value, iterable, body)

    def parse_while(self) -> WhileNode:
        self.consume(TokenType.WHILE, "Expected 'while'")
        condition = self.parse_expr()
        
        self.consume(TokenType.LBRACE, "Expected '{'")
        body = []
        while self.current().type != TokenType.RBRACE and self.current().type != TokenType.EOF:
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
        self.consume(TokenType.RBRACE, "Expected '}'")
        return WhileNode(condition, body)

    def parse_try_catch(self) -> TryCatchNode:
        self.consume(TokenType.TRY, "Expected 'try'")
        self.consume(TokenType.LBRACE, "Expected '{'")
        try_body = []
        while self.current().type != TokenType.RBRACE and self.current().type != TokenType.EOF:
            stmt = self.parse_statement()
            if stmt:
                try_body.append(stmt)
        self.consume(TokenType.RBRACE, "Expected '}'")
        
        self.consume(TokenType.CATCH, "Expected 'catch'")
        
        catch_var = None
        if self.match(TokenType.LPAREN):
            catch_var_tok = self.consume(TokenType.IDENTIFIER, "Expected catch variable name")
            catch_var = catch_var_tok.value
            self.consume(TokenType.RPAREN, "Expected ')'")
        elif self.current().type == TokenType.IDENTIFIER:
            catch_var_tok = self.consume(TokenType.IDENTIFIER, "Expected catch variable name")
            catch_var = catch_var_tok.value
            
        self.consume(TokenType.LBRACE, "Expected '{'")
        catch_body = []
        while self.current().type != TokenType.RBRACE and self.current().type != TokenType.EOF:
            stmt = self.parse_statement()
            if stmt:
                catch_body.append(stmt)
        self.consume(TokenType.RBRACE, "Expected '}'")
        
        return TryCatchNode(try_body, catch_var, catch_body)

    def parse_noise(self) -> NoiseNode:
        self.consume(TokenType.NOISE, "Expected 'noise'")
        noise_type_tok = self.match(TokenType.DEPOLARIZING, TokenType.BITFLIP)
        if not noise_type_tok:
            self.error("Expected depolarizing or bitflip after noise")
        self.consume(TokenType.LPAREN, "Expected '('")
        expr = self.parse_expr()
        self.consume(TokenType.RPAREN, "Expected ')'")
        
        targets = []
        if self.current().type == TokenType.IDENTIFIER:
            targets.append(self.consume(TokenType.IDENTIFIER, "Expected qubit identifier").value)
            while self.match(TokenType.COMMA):
                targets.append(self.consume(TokenType.IDENTIFIER, "Expected qubit identifier").value)
        return NoiseNode(noise_type_tok.value, expr, targets)

    def parse_parallel_block(self):
        self.consume(TokenType.PARALLEL, "Expected 'parallel'")
        self.consume(TokenType.LBRACE, "Expected '{' after 'parallel'")
        tasks = []
        while self.current().type != TokenType.RBRACE and self.current().type != TokenType.EOF:
            if self.match(TokenType.TASK):
                # Parse function call after 'task'
                call_expr = self.parse_expr()
                tasks.append(TaskStatementNode(call_expr))
            else:
                # Allow bare function calls inside parallel blocks too
                stmt = self.parse_statement()
                if stmt:
                    tasks.append(stmt)
        self.consume(TokenType.RBRACE, "Expected '}' to close parallel block")
        return ParallelBlockNode(tasks)

    def parse_match(self) -> MatchNode:
        self.consume(TokenType.MATCH, "Expected 'match'")
        match_expr = self.parse_expr()
        self.consume(TokenType.LBRACE, "Expected '{' after match expression")

        cases = []
        default_body = None

        while self.current().type != TokenType.RBRACE and self.current().type != TokenType.EOF:
            if self.match(TokenType.CASE):
                pattern = self.parse_expr()
                self.consume(TokenType.LBRACE, "Expected '{' after case pattern")
                body = []
                while self.current().type != TokenType.RBRACE and self.current().type != TokenType.EOF:
                    stmt = self.parse_statement()
                    if stmt:
                        body.append(stmt)
                self.consume(TokenType.RBRACE, "Expected '}' to close case body")
                cases.append((pattern, body))
            elif self.match(TokenType.DEFAULT):
                self.consume(TokenType.LBRACE, "Expected '{' after default")
                default_body = []
                while self.current().type != TokenType.RBRACE and self.current().type != TokenType.EOF:
                    stmt = self.parse_statement()
                    if stmt:
                        default_body.append(stmt)
                self.consume(TokenType.RBRACE, "Expected '}' to close default body")
            else:
                self.error("Expected 'case' or 'default' in match block")

        self.consume(TokenType.RBRACE, "Expected '}' to close match")
        return MatchNode(match_expr, cases, default_body)

    # Operator Precedence Parsing
    def parse_expr(self) -> ASTNode:
        return self.parse_logical_or()

    def parse_logical_or(self) -> ASTNode:
        node = self.parse_logical_and()
        while True:
            op_tok = self.match(TokenType.OR)
            if op_tok:
                right = self.parse_logical_and()
                node = BinaryOpNode(op_tok.value, node, right)
            else:
                break
        return node

    def parse_logical_and(self) -> ASTNode:
        node = self.parse_equality()
        while True:
            op_tok = self.match(TokenType.AND)
            if op_tok:
                right = self.parse_equality()
                node = BinaryOpNode(op_tok.value, node, right)
            else:
                break
        return node

    def parse_equality(self) -> ASTNode:
        node = self.parse_comparison()
        while True:
            op_tok = self.match(TokenType.EQ, TokenType.NE)
            if op_tok:
                right = self.parse_comparison()
                node = BinaryOpNode(op_tok.value, node, right)
            else:
                break
        return node

    def parse_comparison(self) -> ASTNode:
        node = self.parse_bitwise_or()
        while True:
            op_tok = self.match(TokenType.LT, TokenType.GT, TokenType.LE, TokenType.GE)
            if op_tok:
                right = self.parse_bitwise_or()
                node = BinaryOpNode(op_tok.value, node, right)
            else:
                break
        return node

    def parse_bitwise_or(self) -> ASTNode:
        node = self.parse_bitwise_xor()
        while True:
            op_tok = self.match(TokenType.PIPE)
            if op_tok:
                right = self.parse_bitwise_xor()
                node = BinaryOpNode(op_tok.value, node, right)
            else:
                break
        return node

    def parse_bitwise_xor(self) -> ASTNode:
        node = self.parse_bitwise_and()
        while True:
            op_tok = self.match(TokenType.CARET)
            if op_tok:
                right = self.parse_bitwise_and()
                node = BinaryOpNode(op_tok.value, node, right)
            else:
                break
        return node

    def parse_bitwise_and(self) -> ASTNode:
        node = self.parse_bitwise_shift()
        while True:
            op_tok = self.match(TokenType.AMP)
            if op_tok:
                right = self.parse_bitwise_shift()
                node = BinaryOpNode(op_tok.value, node, right)
            else:
                break
        return node

    def parse_bitwise_shift(self) -> ASTNode:
        node = self.parse_additive()
        while True:
            op_tok = self.match(TokenType.LSHIFT, TokenType.RSHIFT)
            if op_tok:
                right = self.parse_additive()
                node = BinaryOpNode(op_tok.value, node, right)
            else:
                break
        return node

    def parse_additive(self) -> ASTNode:
        node = self.parse_multiplicative()
        while True:
            op_tok = self.match(TokenType.PLUS, TokenType.MINUS)
            if op_tok:
                right = self.parse_multiplicative()
                node = BinaryOpNode(op_tok.value, node, right)
            else:
                break
        return node

    def parse_multiplicative(self) -> ASTNode:
        node = self.parse_power()
        while True:
            op_tok = self.match(TokenType.MUL, TokenType.DIV, TokenType.MOD)
            if op_tok:
                right = self.parse_power()
                node = BinaryOpNode(op_tok.value, node, right)
            else:
                break
        return node

    def parse_power(self) -> ASTNode:
        base = self.parse_unary()
        if self.match(TokenType.POW):
            exponent = self.parse_power()
            return BinaryOpNode("**", base, exponent)
        return base

    def parse_unary(self) -> ASTNode:
        if self.match(TokenType.NOT):
            right = self.parse_unary()
            return BinaryOpNode("not", right, LiteralNode(True, "bool"))
        if self.match(TokenType.TILDE):
            right = self.parse_unary()
            return BinaryOpNode("~", right, LiteralNode(0, "int"))
        if self.match(TokenType.MINUS):
            right = self.parse_unary()
            return BinaryOpNode("-", LiteralNode(0, "int"), right)
        if self.match(TokenType.PLUS):
            return self.parse_unary()
        return self.parse_postfix()

    def parse_postfix(self) -> ASTNode:
        node = self.parse_primary()
        while True:
            if self.match(TokenType.DOT):
                member_tok = self.consume(TokenType.IDENTIFIER, "Expected member name after '.'")
                node = DotAccessNode(node, member_tok.value)
            elif self.match(TokenType.LPAREN):
                args = []
                if self.current().type != TokenType.RPAREN:
                    args.append(self.parse_expr())
                    while self.match(TokenType.COMMA):
                        args.append(self.parse_expr())
                self.consume(TokenType.RPAREN, "Expected ')'")
                node = CallNode(node, args)
            elif self.match(TokenType.LBRACK):
                index_expr = self.parse_expr()
                self.consume(TokenType.RBRACK, "Expected ']'")
                node = IndexAccessNode(node, index_expr)
            else:
                break
        return node

    def parse_primary(self) -> ASTNode:
        # Map literal
        if self.match(TokenType.LBRACE):
            keys = []
            values = []
            if self.current().type != TokenType.RBRACE:
                key_expr = self.parse_expr()
                self.consume(TokenType.COLON, "Expected ':' after key in map literal")
                val_expr = self.parse_expr()
                keys.append(key_expr)
                values.append(val_expr)
                while self.match(TokenType.COMMA):
                    if self.current().type == TokenType.RBRACE:
                        break
                    key_expr = self.parse_expr()
                    self.consume(TokenType.COLON, "Expected ':' after key in map literal")
                    val_expr = self.parse_expr()
                    keys.append(key_expr)
                    values.append(val_expr)
            self.consume(TokenType.RBRACE, "Expected '}' to close map literal")
            return MapAllocNode(keys, values)

        # Parenthesized expression or Tuple literal
        if self.match(TokenType.LPAREN):
            if self.match(TokenType.RPAREN):
                return TupleLiteralNode([])
            exprs = [self.parse_expr()]
            is_tuple = False
            while self.match(TokenType.COMMA):
                is_tuple = True
                if self.current().type == TokenType.RPAREN:
                    break
                exprs.append(self.parse_expr())
            self.consume(TokenType.RPAREN, "Expected ')'")
            if is_tuple:
                return TupleLiteralNode(exprs)
            else:
                return exprs[0]

        # Array literal
        if self.match(TokenType.LBRACK):
            elements = []
            if self.current().type != TokenType.RBRACK:
                elements.append(self.parse_expr())
                while self.match(TokenType.COMMA):
                    if self.current().type == TokenType.RBRACK:
                        break
                    elements.append(self.parse_expr())
            self.consume(TokenType.RBRACK, "Expected ']'")
            return ArrayLiteralNode(elements)

        # Constants
        const_tok = self.match(TokenType.PI, TokenType.TAU, TokenType.E)
        if const_tok:
            val_map = {
                "PI": 3.141592653589793,
                "TAU": 6.283185307179586,
                "E": 2.718281828459045
            }
            return LiteralNode(val_map[const_tok.value], "float")

        # Literals
        lit_tok = self.match(TokenType.INT_LIT)
        if lit_tok:
            return LiteralNode(int(lit_tok.value), "int")
            
        lit_tok = self.match(TokenType.FLOAT_LIT)
        if lit_tok:
            return LiteralNode(float(lit_tok.value), "float")

        lit_tok = self.match(TokenType.STRING_LIT)
        if lit_tok:
            val = lit_tok.value
            # Check for string interpolation markers
            if '\x00' in val:
                parts = []
                idx = 0
                while idx < len(val):
                    marker_pos = val.find('\x00', idx)
                    if marker_pos == -1:
                        parts.append(val[idx:])
                        break
                    if marker_pos > idx:
                        parts.append(val[idx:marker_pos])
                    end_pos = val.find('\x00', marker_pos + 1)
                    if end_pos == -1:
                        parts.append(val[marker_pos:])
                        break
                    expr_str = val[marker_pos + 1:end_pos]
                    # Parse the expression
                    try:
                        expr_tokens = Lexer(expr_str).tokenize()
                        if expr_tokens and expr_tokens[-1].type.name == 'EOF':
                            expr_tokens.pop()
                        expr_node = Parser(expr_tokens).parse_expr()
                        parts.append(expr_node)
                    except Exception:
                        parts.append(expr_str)
                    idx = end_pos + 1
                return StringInterpolationNode(parts)
            return LiteralNode(val, "string")

        if self.match(TokenType.TRUE):
            return LiteralNode(True, "bool")

        if self.match(TokenType.FALSE):
            return LiteralNode(False, "bool")

        if self.match(TokenType.NULL):
            return LiteralNode(None, "null")

        # Struct literal or variable reference
        if self.current().type == TokenType.IDENTIFIER:
            if self.peek().type == TokenType.LBRACE:
                # Disambiguate: check if we have a struct literal (e.g. MyStruct { field: expr })
                # vs a variable reference followed by a block (e.g. for x in arr { ... })
                is_struct_literal = False
                if self.peek(2).type == TokenType.IDENTIFIER and self.peek(3).type == TokenType.COLON:
                    is_struct_literal = True
                elif self.peek(2).type == TokenType.RBRACE:
                    is_struct_literal = True
                
                if is_struct_literal:
                    struct_name_tok = self.consume(TokenType.IDENTIFIER, "Expected struct name")
                    self.consume(TokenType.LBRACE, "Expected '{'")
                    bindings = {}
                    if self.current().type != TokenType.RBRACE:
                        field_name = self.consume(TokenType.IDENTIFIER, "Expected field name").value
                        self.consume(TokenType.COLON, "Expected ':'")
                        field_val = self.parse_expr()
                        bindings[field_name] = field_val
                        while self.match(TokenType.COMMA):
                            if self.current().type == TokenType.RBRACE:
                                break
                            field_name = self.consume(TokenType.IDENTIFIER, "Expected field name").value
                            self.consume(TokenType.COLON, "Expected ':'")
                            field_val = self.parse_expr()
                            bindings[field_name] = field_val
                    self.consume(TokenType.RBRACE, "Expected '}'")
                    return StructLiteralNode(struct_name_tok.value, bindings)

            # If not a struct literal, parse as standard variable reference
            ref_tok = self.consume(TokenType.IDENTIFIER, "Expected identifier")
            return VarRefNode(ref_tok.value)

        self.error("Expected number, constant, literal, variable reference, or '('")
