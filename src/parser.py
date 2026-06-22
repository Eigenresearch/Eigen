from src.lexer import Token, TokenType
from src.ast import (
    ProgramNode, ImportNode, QFuncDeclNode, LetNode, VarDeclNode,
    BinaryOpNode, LiteralNode, VarRefNode, QFuncCallNode, GateNode,
    MeasureNode, IfNode, ReturnNode, TraceNode, PrintNode, AssertNode, ASTNode
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
            module_name_tok = self.consume(TokenType.IDENTIFIER, "Expected module path after 'module'")
            module_name = module_name_tok.value

        # Optional imports: "import quantum.bell"
        imports = []
        while self.match(TokenType.IMPORT):
            import_path_tok = self.consume(TokenType.IDENTIFIER, "Expected module path after 'import'")
            imports.append(ImportNode(import_path_tok.value))

        # Parse statements
        body = []
        while self.current().type != TokenType.EOF:
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)

        return ProgramNode(version, module_name, imports, body)

    def parse_statement(self) -> ASTNode:
        tok = self.current()

        # qfunc declaration
        if tok.type == TokenType.QFUNC:
            return self.parse_qfunc_decl()

        # let assignment
        if tok.type == TokenType.LET:
            return self.parse_let()

        # qubit / cbit / int / float declarations
        if tok.type in (TokenType.QUBIT, TokenType.CBIT, TokenType.INT, TokenType.FLOAT):
            type_tok = self.match(TokenType.QUBIT, TokenType.CBIT, TokenType.INT, TokenType.FLOAT)
            name_tok = self.consume(TokenType.IDENTIFIER, f"Expected identifier after type '{type_tok.value}'")
            return VarDeclNode(name_tok.value, type_tok.value)

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

        # return
        if tok.type == TokenType.RETURN:
            self.advance()
            return ReturnNode()

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
            op_tok = self.consume(TokenType.EQ, "Expected '==' comparison in assert")
            right = self.parse_expr()
            return AssertNode(left, op_tok.value, right)

        # Gate operations or custom qfunc calls
        if tok.type == TokenType.IDENTIFIER:
            # Check if it's a qfunc call or gate application
            # If the next token is LPAREN, it's a qfunc call like 'bell(q0, q1)'
            if self.peek().type == TokenType.LPAREN:
                name_tok = self.consume(TokenType.IDENTIFIER, "Expected identifier")
                self.consume(TokenType.LPAREN, "Expected '('")
                args = []
                if self.current().type != TokenType.RPAREN:
                    arg_tok = self.consume(TokenType.IDENTIFIER, "Expected identifier argument")
                    args.append(arg_tok.value)
                    while self.match(TokenType.COMMA):
                        arg_tok = self.consume(TokenType.IDENTIFIER, "Expected identifier argument")
                        args.append(arg_tok.value)
                self.consume(TokenType.RPAREN, "Expected ')'")
                return QFuncCallNode(name_tok.value, args)
            else:
                self.error(f"Unexpected identifier '{tok.value}'. Did you mean to use a keyword or a gate?")

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

    def parse_let(self) -> LetNode:
        self.consume(TokenType.LET, "Expected 'let'")
        name_tok = self.consume(TokenType.IDENTIFIER, "Expected identifier for variable")
        self.consume(TokenType.COLON, "Expected ':' after variable name")
        type_tok = self.match(TokenType.INT, TokenType.FLOAT, TokenType.CBIT)
        if not type_tok:
            self.error("Expected type (int, float, cbit) after ':' in let statement")
        self.consume(TokenType.EQUALS, "Expected '=' in let statement")
        value_expr = self.parse_expr()
        return LetNode(name_tok.value, type_tok.value, value_expr)

    def parse_if(self) -> IfNode:
        self.consume(TokenType.IF, "Expected 'if'")
        left = self.parse_expr()
        op_tok = self.consume(TokenType.EQ, "Expected '==' in if condition")
        right = self.parse_expr()
        
        self.consume(TokenType.LBRACE, "Expected '{' to start if branch")
        body = []
        while self.current().type != TokenType.RBRACE and self.current().type != TokenType.EOF:
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
        self.consume(TokenType.RBRACE, "Expected '}' to end if branch")
        return IfNode(left, op_tok.value, right, body)

    # Expression parsing (Pratt Parser / Recursive Descent)
    def parse_expr(self) -> ASTNode:
        return self.parse_additive()

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
        node = self.parse_primary()
        while True:
            op_tok = self.match(TokenType.MUL, TokenType.DIV)
            if op_tok:
                right = self.parse_primary()
                node = BinaryOpNode(op_tok.value, node, right)
            else:
                break
        return node

    def parse_primary(self) -> ASTNode:
        # Prefix operators
        if self.match(TokenType.MINUS):
            # represent -x as 0 - x
            right = self.parse_primary()
            return BinaryOpNode("-", LiteralNode(0, "int"), right)
        if self.match(TokenType.PLUS):
            return self.parse_primary()

        # Parenthesized expression
        if self.match(TokenType.LPAREN):
            node = self.parse_expr()
            self.consume(TokenType.RPAREN, "Expected ')'")
            return node

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

        # Variable Reference
        ref_tok = self.consume(TokenType.IDENTIFIER, "Expected number, constant, variable reference, or '('")
        return VarRefNode(ref_tok.value)
