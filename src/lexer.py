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
    
    # Keywords
    MODULE = "module"
    IMPORT = "import"
    QFUNC = "qfunc"
    LET = "let"
    IF = "if"
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
    
    # General
    IDENTIFIER = "IDENTIFIER"
    INT_LIT = "INT_LIT"
    FLOAT_LIT = "FLOAT_LIT"
    
    # Operators & Delimiters
    LPAREN = "("
    RPAREN = ")"
    LBRACE = "{"
    RBRACE = "}"
    COMMA = ","
    COLON = ":"
    ARROW = "->"
    EQUALS = "="
    EQ = "=="
    PLUS = "+"
    MINUS = "-"
    MUL = "*"
    DIV = "/"
    
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
            "module": TokenType.MODULE,
            "import": TokenType.IMPORT,
            "qfunc": TokenType.QFUNC,
            "let": TokenType.LET,
            "if": TokenType.IF,
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

            # Single character operators & delimiters
            char_tokens = {
                '(': TokenType.LPAREN,
                ')': TokenType.RPAREN,
                '{': TokenType.LBRACE,
                '}': TokenType.RBRACE,
                ',': TokenType.COMMA,
                ':': TokenType.COLON,
                '=': TokenType.EQUALS,
                '+': TokenType.PLUS,
                '-': TokenType.MINUS,
                '*': TokenType.MUL,
                '/': TokenType.DIV,
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

            # Identifiers and keywords (can contain dots for module paths in import)
            if char.isalpha() or char == '_':
                start_col = self.column
                ident_str = ""
                # Allow letters, digits, underscores, and dots (dots are for dotted module names in import/module statements)
                while self.pos < self.length and (self.peek().isalnum() or self.peek() == '_' or self.peek() == '.'):
                    ident_str += self.peek()
                    self.advance()

                # Clean up trailing dot if any, though syntactically invalid
                if ident_str.endswith('.'):
                    self.error("Identifier cannot end with a dot")

                if ident_str in KEYWORDS_MAP:
                    tokens.append(Token(KEYWORDS_MAP[ident_str], ident_str, self.line, start_col))
                else:
                    tokens.append(Token(TokenType.IDENTIFIER, ident_str, self.line, start_col))
                continue

            self.error(f"Unexpected character: {repr(char)}")

        tokens.append(Token(TokenType.EOF, "", self.line, self.column))
        return tokens
