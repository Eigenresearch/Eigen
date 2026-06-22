import enum
import re

class TokenType(enum.Enum):
    # Directives
    EIGEN = "eigen"
    
    # Types
    QUBIT = "qubit"
    CBIT = "cbit"
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    BOOL = "bool"
    ARRAY = "array"
    MAP = "map"
    
    # Keywords
    MODULE = "module"
    IMPORT = "import"
    QFUNC = "qfunc"
    FUNC = "func"
    STRUCT = "struct"
    ENUM = "enum"
    TRY = "try"
    CATCH = "catch"
    THROW = "throw"
    NOISE = "noise"
    DEPOLARIZING = "depolarizing"
    BITFLIP = "bitflip"
    LET = "let"
    IF = "if"
    FOR = "for"
    IN = "in"
    WHILE = "while"
    BREAK = "break"
    CONTINUE = "continue"
    MEASURE = "measure"
    RETURN = "return"
    
    # Built-ins / Utilities
    TRACE = "trace"
    PRINT = "print"
    ASSERT = "assert"
    
    # Constants
    PI = "PI"
    TAU = "TAU"
    E = "E"
    
    # Gates
    GATE_H = "H"
    GATE_X = "X"
    GATE_Y = "Y"
    GATE_Z = "Z"
    GATE_S = "S"
    GATE_T = "T"
    GATE_CNOT = "CNOT"
    GATE_CZ = "CZ"
    GATE_SWAP = "SWAP"
    GATE_RX = "RX"
    GATE_RY = "RY"
    GATE_RZ = "RZ"
    
    # Literals
    STRING_LIT = "STRING_LIT"
    TRUE = "true"
    FALSE = "false"
    NULL = "null"
    
    # General
    IDENTIFIER = "IDENTIFIER"
    INT_LIT = "INT_LIT"
    FLOAT_LIT = "FLOAT_LIT"
    
    # Operators & Delimiters
    LPAREN = "("
    RPAREN = ")"
    LBRACE = "{"
    RBRACE = "}"
    LBRACK = "["
    RBRACK = "]"
    COMMA = ","
    COLON = ":"
    DOT = "."
    ARROW = "->"
    EQUALS = "="
    EQ = "=="
    NE = "!="
    LT = "<"
    GT = ">"
    LE = "<="
    GE = ">="
    
    PLUS = "+"
    MINUS = "-"
    MUL = "*"
    DIV = "/"
    
    ADD_ASSIGN = "+="
    SUB_ASSIGN = "-="
    MUL_ASSIGN = "*="
    DIV_ASSIGN = "/="
    
    AND = "and"
    OR = "or"
    NOT = "not"
    
    EOF = "EOF"


class Token:
    def __init__(self, type_: TokenType, value: str, line: int, column: int):
        self.type = type_
        self.value = value
        self.line = line
        self.column = column

    def __repr__(self):
        return f"Token({self.type.name}, {repr(self.value)}, line={self.line}, col={self.column})"


class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.length = len(source)
        self.pos = 0
        self.line = 1
        self.column = 1

    def error(self, msg: str):
        raise SyntaxError(f"Lexer Error at line {self.line}, col {self.column}: {msg}")

    def peek(self, offset: int = 0) -> str:
        if self.pos + offset >= self.length:
            return ""
        return self.source[self.pos + offset]

    def advance(self):
        if self.pos < self.length:
            char = self.source[self.pos]
            self.pos += 1
            if char == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1

    def tokenize(self) -> list[Token]:
        tokens = []
        
        KEYWORDS_MAP = {
            "eigen": TokenType.EIGEN,
            "qubit": TokenType.QUBIT,
            "cbit": TokenType.CBIT,
            "int": TokenType.INT,
            "float": TokenType.FLOAT,
            "string": TokenType.STRING,
            "bool": TokenType.BOOL,
            "array": TokenType.ARRAY,
            "map": TokenType.MAP,
            "module": TokenType.MODULE,
            "import": TokenType.IMPORT,
            "qfunc": TokenType.QFUNC,
            "func": TokenType.FUNC,
            "struct": TokenType.STRUCT,
            "enum": TokenType.ENUM,
            "try": TokenType.TRY,
            "catch": TokenType.CATCH,
            "throw": TokenType.THROW,
            "noise": TokenType.NOISE,
            "depolarizing": TokenType.DEPOLARIZING,
            "bitflip": TokenType.BITFLIP,
            "let": TokenType.LET,
            "if": TokenType.IF,
            "for": TokenType.FOR,
            "in": TokenType.IN,
            "while": TokenType.WHILE,
            "break": TokenType.BREAK,
            "continue": TokenType.CONTINUE,
            "measure": TokenType.MEASURE,
            "return": TokenType.RETURN,
            "trace": TokenType.TRACE,
            "print": TokenType.PRINT,
            "assert": TokenType.ASSERT,
            "PI": TokenType.PI,
            "TAU": TokenType.TAU,
            "E": TokenType.E,
            "H": TokenType.GATE_H,
            "X": TokenType.GATE_X,
            "Y": TokenType.GATE_Y,
            "Z": TokenType.GATE_Z,
            "S": TokenType.GATE_S,
            "T": TokenType.GATE_T,
            "CNOT": TokenType.GATE_CNOT,
            "CZ": TokenType.GATE_CZ,
            "SWAP": TokenType.GATE_SWAP,
            "RX": TokenType.GATE_RX,
            "RY": TokenType.GATE_RY,
            "RZ": TokenType.GATE_RZ,
            "true": TokenType.TRUE,
            "false": TokenType.FALSE,
            "null": TokenType.NULL,
            "and": TokenType.AND,
            "or": TokenType.OR,
            "not": TokenType.NOT,
        }

        while self.pos < self.length:
            char = self.peek()

            # Skip whitespace
            if char.isspace():
                self.advance()
                continue

            # Skip comments (both # and //)
            if char == '#' or (char == '/' and self.peek(1) == '/'):
                while self.pos < self.length and self.peek() != '\n':
                    self.advance()
                continue

            # Double-quoted string literals
            if char == '"':
                start_col = self.column
                self.advance()  # consume open quote
                string_val = ""
                while self.pos < self.length and self.peek() != '"':
                    if self.peek() == '\n':
                        self.error("Unterminated string literal")
                    # Handle escape sequences
                    if self.peek() == '\\':
                        self.advance()
                        if self.pos < self.length:
                            string_val += self.peek()
                            self.advance()
                    else:
                        string_val += self.peek()
                        self.advance()
                if self.peek() != '"':
                    self.error("Unterminated string literal")
                self.advance()  # consume close quote
                tokens.append(Token(TokenType.STRING_LIT, string_val, self.line, start_col))
                continue

            # Multi-character operators and symbols
            if char == '-' and self.peek(1) == '>':
                start_col = self.column
                self.advance()
                self.advance()
                tokens.append(Token(TokenType.ARROW, "->", self.line, start_col))
                continue

            if char == '=' and self.peek(1) == '=':
                start_col = self.column
                self.advance()
                self.advance()
                tokens.append(Token(TokenType.EQ, "==", self.line, start_col))
                continue

            if char == '!' and self.peek(1) == '=':
                start_col = self.column
                self.advance()
                self.advance()
                tokens.append(Token(TokenType.NE, "!=", self.line, start_col))
                continue

            if char == '<' and self.peek(1) == '=':
                start_col = self.column
                self.advance()
                self.advance()
                tokens.append(Token(TokenType.LE, "<=", self.line, start_col))
                continue

            if char == '>' and self.peek(1) == '=':
                start_col = self.column
                self.advance()
                self.advance()
                tokens.append(Token(TokenType.GE, ">=", self.line, start_col))
                continue

            if char == '+' and self.peek(1) == '=':
                start_col = self.column
                self.advance()
                self.advance()
                tokens.append(Token(TokenType.ADD_ASSIGN, "+=", self.line, start_col))
                continue

            if char == '-' and self.peek(1) == '=':
                start_col = self.column
                self.advance()
                self.advance()
                tokens.append(Token(TokenType.SUB_ASSIGN, "-=", self.line, start_col))
                continue

            if char == '*' and self.peek(1) == '=':
                start_col = self.column
                self.advance()
                self.advance()
                tokens.append(Token(TokenType.MUL_ASSIGN, "*=", self.line, start_col))
                continue

            if char == '/' and self.peek(1) == '=':
                start_col = self.column
                self.advance()
                self.advance()
                tokens.append(Token(TokenType.DIV_ASSIGN, "/=", self.line, start_col))
                continue

            # Single character operators & delimiters
            char_tokens = {
                '(': TokenType.LPAREN,
                ')': TokenType.RPAREN,
                '{': TokenType.LBRACE,
                '}': TokenType.RBRACE,
                '[': TokenType.LBRACK,
                ']': TokenType.RBRACK,
                ',': TokenType.COMMA,
                ':': TokenType.COLON,
                '.': TokenType.DOT,
                '=': TokenType.EQUALS,
                '+': TokenType.PLUS,
                '-': TokenType.MINUS,
                '*': TokenType.MUL,
                '/': TokenType.DIV,
                '<': TokenType.LT,
                '>': TokenType.GT,
            }
            if char in char_tokens:
                tokens.append(Token(char_tokens[char], char, self.line, self.column))
                self.advance()
                continue

            # Numbers (integers or floats)
            if char.isdigit():
                start_col = self.column
                num_str = ""
                while self.pos < self.length and self.peek().isdigit():
                    num_str += self.peek()
                    self.advance()
                
                # Check for decimal point
                if self.peek() == '.' and self.peek(1).isdigit():
                    num_str += '.'
                    self.advance()
                    while self.pos < self.length and self.peek().isdigit():
                        num_str += self.peek()
                        self.advance()
                    tokens.append(Token(TokenType.FLOAT_LIT, num_str, self.line, start_col))
                else:
                    tokens.append(Token(TokenType.INT_LIT, num_str, self.line, start_col))
                continue

            # Identifiers and keywords (no dots allowed anymore, dots are separate tokens)
            if char.isalpha() or char == '_':
                start_col = self.column
                ident_str = ""
                while self.pos < self.length and (self.peek().isalnum() or self.peek() == '_'):
                    ident_str += self.peek()
                    self.advance()

                if ident_str in KEYWORDS_MAP:
                    tokens.append(Token(KEYWORDS_MAP[ident_str], ident_str, self.line, start_col))
                else:
                    tokens.append(Token(TokenType.IDENTIFIER, ident_str, self.line, start_col))
                continue

            self.error(f"Unexpected character: {repr(char)}")

        tokens.append(Token(TokenType.EOF, "", self.line, self.column))
        return tokens
