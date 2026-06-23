from src.frontend.lexer import Token, TokenType
from src.frontend.ast import (
    ProgramNode, ImportNode, QFuncDeclNode, LetNode, VarDeclNode,
    BinaryOpNode, LiteralNode, VarRefNode, QFuncCallNode, GateNode,
    MeasureNode, IfNode, ReturnNode, TraceNode, PrintNode, AssertNode, ASTNode,
    FuncDeclNode, ForNode, WhileNode, BreakNode, ContinueNode, StructDeclNode,
    StructLiteralNode, DotAccessNode, ArrayLiteralNode, TupleLiteralNode,
    TryCatchNode, ThrowNode, EnumDeclNode, NoiseNode, AssignmentNode, CallNode,
    IndexAccessNode, MapAllocNode, ParallelBlockNode, TaskStatementNode
)

class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def error(self, msg: str):
        tok = self.current()
        raise SyntaxError(f"Parser Error at line {tok.line}, col {tok.column}: {msg} (found {tok})")

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

    def parse_type(self) -> str:
        tok = self.match(
            TokenType.INT, TokenType.FLOAT, TokenType.STRING, TokenType.BOOL,
            TokenType.QUBIT, TokenType.CBIT, TokenType.ARRAY, TokenType.MAP,
            TokenType.IDENTIFIER
        )
        if not tok:
            self.error("Expected type name")
        
        type_str = tok.value
        # Check if it has generic parameters, e.g. <T> or <K, V>
        if self.match(TokenType.LT):
            generic_types = [self.parse_type()]
            while self.match(TokenType.COMMA):
                generic_types.append(self.parse_type())
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

        # Variable declarations: e.g. qubit q0, cbit c0, map<string, int> m, MyStruct s
        # We can backtrack to see if it is a type followed by an identifier.
        saved_pos = self.pos
        try:
            type_name = self.parse_type()
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
            
            if all(isinstance(arg, VarRefNode) for arg in args):
                return QFuncCallNode(name_tok.value, [arg.name for arg in args])
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

    def parse_func_decl(self) -> FuncDeclNode:
        self.consume(TokenType.FUNC, "Expected 'func'")
        name_tok = self.consume(TokenType.IDENTIFIER, "Expected function name")
        
        generic_params = []
        if self.match(TokenType.LT):
            generic_params.append(self.consume(TokenType.IDENTIFIER, "Expected generic parameter").value)
            while self.match(TokenType.COMMA):
                generic_params.append(self.consume(TokenType.IDENTIFIER, "Expected generic parameter").value)
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
        self.consume(TokenType.ARROW, "Expected '->'")
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
            generic_params.append(self.consume(TokenType.IDENTIFIER, "Expected generic parameter").value)
            while self.match(TokenType.COMMA):
                generic_params.append(self.consume(TokenType.IDENTIFIER, "Expected generic parameter").value)
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

    def parse_let(self) -> LetNode:
        self.consume(TokenType.LET, "Expected 'let'")
        name_tok = self.consume(TokenType.IDENTIFIER, "Expected identifier for variable")
        self.consume(TokenType.COLON, "Expected ':' after variable name")
        type_name = self.parse_type()
        self.consume(TokenType.EQUALS, "Expected '=' in let statement")
        value_expr = self.parse_expr()
        return LetNode(name_tok.value, type_name, value_expr)

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
        return IfNode(left, op_tok.value, right, body)

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
        node = self.parse_additive()
        while True:
            op_tok = self.match(TokenType.LT, TokenType.GT, TokenType.LE, TokenType.GE)
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
        node = self.parse_unary()
        while True:
            op_tok = self.match(TokenType.MUL, TokenType.DIV)
            if op_tok:
                right = self.parse_unary()
                node = BinaryOpNode(op_tok.value, node, right)
            else:
                break
        return node

    def parse_unary(self) -> ASTNode:
        if self.match(TokenType.NOT):
            right = self.parse_unary()
            return BinaryOpNode("not", right, LiteralNode(True, "bool"))
        if self.match(TokenType.MINUS):
            right = self.parse_primary()
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
            return LiteralNode(lit_tok.value, "string")

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
