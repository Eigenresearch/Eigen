import enum
import re

class TokensList(list):
    pass

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
    ELSE = "else"
    ELIF = "elif"
    FOR = "for"
    IN = "in"
    WHILE = "while"
    BREAK = "break"
    CONTINUE = "continue"
    MEASURE = "measure"
    RETURN = "return"
    PARALLEL = "parallel"
    TASK = "task"
    MATCH = "match"
    CASE = "case"
    DEFAULT = "default"
    
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
    GATE_CCX = "CCX"
    GATE_CSWAP = "CSWAP"
    GATE_CP = "CP"
    GATE_CRX = "CRX"
    GATE_CRY = "CRY"
    GATE_CRZ = "CRZ"
    
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
    MOD = "%"
    
    AMP = "&"
    PIPE = "|"
    CARET = "^"
    TILDE = "~"
    LSHIFT = "<<"
    RSHIFT = ">>"
    
    ADD_ASSIGN = "+="
    SUB_ASSIGN = "-="
    MUL_ASSIGN = "*="
    DIV_ASSIGN = "/="
    POW = "**"
    
    AND = "and"
    OR = "or"
    NOT = "not"
    SEMICOLON = ";"
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

    _KEYWORDS_MAP = {
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
        "else": TokenType.ELSE,
        "elif": TokenType.ELIF,
        "for": TokenType.FOR,
        "in": TokenType.IN,
        "while": TokenType.WHILE,
        "break": TokenType.BREAK,
        "continue": TokenType.CONTINUE,
        "measure": TokenType.MEASURE,
        "return": TokenType.RETURN,
        "parallel": TokenType.PARALLEL,
        "task": TokenType.TASK,
        "match": TokenType.MATCH,
        "case": TokenType.CASE,
        "default": TokenType.DEFAULT,
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
        "CCX": TokenType.GATE_CCX,
        "Toffoli": TokenType.GATE_CCX,
        "toffoli": TokenType.GATE_CCX,
        "CSWAP": TokenType.GATE_CSWAP,
        "Fredkin": TokenType.GATE_CSWAP,
        "fredkin": TokenType.GATE_CSWAP,
        "CP": TokenType.GATE_CP,
        "CRX": TokenType.GATE_CRX,
        "CRY": TokenType.GATE_CRY,
        "CRZ": TokenType.GATE_CRZ,
        "true": TokenType.TRUE,
        "false": TokenType.FALSE,
        "null": TokenType.NULL,
        "and": TokenType.AND,
        "or": TokenType.OR,
        "not": TokenType.NOT,
    }

    _CHAR_TOKENS = {
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
        '%': TokenType.MOD,
        '&': TokenType.AMP,
        '|': TokenType.PIPE,
        '^': TokenType.CARET,
        '~': TokenType.TILDE,
    }

    def tokenize(self) -> list[Token]:
        tokens = []
        
        KEYWORDS_MAP = self._KEYWORDS_MAP
        char_tokens = self._CHAR_TOKENS

        while self.pos < self.length:
            char = self.source[self.pos]

            # Skip whitespace
            if char.isspace():
                if char == '\n':
                    self.line += 1
                    self.column = 1
                    self.pos += 1
                else:
                    start_pos = self.pos
                    while self.pos < self.length and self.source[self.pos].isspace() and self.source[self.pos] != '\n':
                        self.pos += 1
                    self.column += (self.pos - start_pos)
                continue

            # Skip comments (both # and //)
            if char == '#' or (char == '/' and self.pos + 1 < self.length and self.source[self.pos + 1] == '/'):
                while self.pos < self.length and self.source[self.pos] != '\n':
                    self.pos += 1
                continue

            # Double-quoted string literals
            if char == '"':
                start_col = self.column
                self.pos += 1  # consume open quote
                start_pos = self.pos
                string_val = []
                while self.pos < self.length and self.source[self.pos] != '"':
                    if self.source[self.pos] == '\n':
                        self.error("Unterminated string literal")
                    if self.source[self.pos] == '\\':
                        self.pos += 1
                        if self.pos < self.length:
                            esc_char = self.source[self.pos]
                            escape_map = {
                                'n': '\n', 't': '\t', 'r': '\r',
                                '0': '\0', '\\': '\\', '"': '"',
                                "'": "'", 'a': '\a', 'b': '\b',
                                'f': '\f', 'v': '\v',
                            }
                            string_val.append(escape_map.get(esc_char, esc_char))
                            self.pos += 1
                    elif self.source[self.pos] == '$' and self.pos + 1 < self.length and self.source[self.pos + 1] == '{':
                        # String interpolation: ${expr}
                        self.pos += 2  # skip ${
                        expr_start = self.pos
                        brace_depth = 1
                        while self.pos < self.length and brace_depth > 0:
                            if self.source[self.pos] == '{':
                                brace_depth += 1
                            elif self.source[self.pos] == '}':
                                brace_depth -= 1
                                if brace_depth == 0:
                                    break
                            self.pos += 1
                        if self.pos >= self.length:
                            self.error("Unterminated string interpolation: expected '}'")
                        expr_str = self.source[expr_start:self.pos]
                        string_val.append(f"\x00{expr_str}\x00")  # Marker for interpolation
                        self.pos += 1  # skip }
                    else:
                        string_val.append(self.source[self.pos])
                        self.pos += 1
                if self.pos >= self.length or self.source[self.pos] != '"':
                    self.error("Unterminated string literal")
                self.pos += 1  # consume close quote
                self.column += (self.pos - start_pos + 1)
                tokens.append(Token(TokenType.STRING_LIT, "".join(string_val), self.line, start_col))
                continue

            # Multi-character operators and symbols
            if char == '-' and self.pos + 1 < self.length and self.source[self.pos + 1] == '>':
                start_col = self.column
                self.pos += 2
                self.column += 2
                tokens.append(Token(TokenType.ARROW, "->", self.line, start_col))
                continue

            if char == '=' and self.pos + 1 < self.length and self.source[self.pos + 1] == '=':
                start_col = self.column
                self.pos += 2
                self.column += 2
                tokens.append(Token(TokenType.EQ, "==", self.line, start_col))
                continue

            if char == '!' and self.pos + 1 < self.length and self.source[self.pos + 1] == '=':
                start_col = self.column
                self.pos += 2
                self.column += 2
                tokens.append(Token(TokenType.NE, "!=", self.line, start_col))
                continue

            if char == '<' and self.pos + 1 < self.length:
                if self.source[self.pos + 1] == '=':
                    start_col = self.column
                    self.pos += 2
                    self.column += 2
                    tokens.append(Token(TokenType.LE, "<=", self.line, start_col))
                    continue
                elif self.source[self.pos + 1] == '<':
                    start_col = self.column
                    self.pos += 2
                    self.column += 2
                    tokens.append(Token(TokenType.LSHIFT, "<<", self.line, start_col))
                    continue
 
            if char == '>' and self.pos + 1 < self.length:
                if self.source[self.pos + 1] == '=':
                    start_col = self.column
                    self.pos += 2
                    self.column += 2
                    tokens.append(Token(TokenType.GE, ">=", self.line, start_col))
                    continue
                elif self.source[self.pos + 1] == '>':
                    start_col = self.column
                    self.pos += 2
                    self.column += 2
                    tokens.append(Token(TokenType.RSHIFT, ">>", self.line, start_col))
                    continue

            if char == '+' and self.pos + 1 < self.length and self.source[self.pos + 1] == '=':
                start_col = self.column
                self.pos += 2
                self.column += 2
                tokens.append(Token(TokenType.ADD_ASSIGN, "+=", self.line, start_col))
                continue

            if char == '-' and self.pos + 1 < self.length and self.source[self.pos + 1] == '=':
                start_col = self.column
                self.pos += 2
                self.column += 2
                tokens.append(Token(TokenType.SUB_ASSIGN, "-=", self.line, start_col))
                continue

            if char == '*' and self.pos + 1 < self.length:
                if self.source[self.pos + 1] == '*':
                    start_col = self.column
                    self.pos += 2
                    self.column += 2
                    tokens.append(Token(TokenType.POW, "**", self.line, start_col))
                    continue
                elif self.source[self.pos + 1] == '=':
                    start_col = self.column
                    self.pos += 2
                    self.column += 2
                    tokens.append(Token(TokenType.MUL_ASSIGN, "*=", self.line, start_col))
                    continue

            if char == '/' and self.pos + 1 < self.length and self.source[self.pos + 1] == '=':
                start_col = self.column
                self.pos += 2
                self.column += 2
                tokens.append(Token(TokenType.DIV_ASSIGN, "/=", self.line, start_col))
                continue

            # Single character operators & delimiters
            if char in char_tokens:
                tokens.append(Token(char_tokens[char], char, self.line, self.column))
                self.pos += 1
                self.column += 1
                continue

            # Numbers (integers or floats) — including hex/binary/octal
            if char.isdigit():
                start_pos = self.pos
                start_col = self.column

                # Hex (0x), Binary (0b), Octal (0o)
                if char == '0' and self.pos + 1 < self.length:
                    next_char = self.source[self.pos + 1]
                    if next_char in ('x', 'X'):
                        self.pos += 2
                        hex_start = self.pos
                        while self.pos < self.length and self.source[self.pos] in '0123456789abcdefABCDEF':
                            self.pos += 1
                        if self.pos == hex_start:
                            self.error("Expected hex digits after '0x'")
                        num_str = str(int(self.source[start_pos:self.pos], 16))
                        self.column += (self.pos - start_pos)
                        tokens.append(Token(TokenType.INT_LIT, num_str, self.line, start_col))
                        continue
                    elif next_char in ('b', 'B'):
                        self.pos += 2
                        bin_start = self.pos
                        while self.pos < self.length and self.source[self.pos] in '01':
                            self.pos += 1
                        if self.pos == bin_start:
                            self.error("Expected binary digits after '0b'")
                        num_str = str(int(self.source[start_pos:self.pos], 2))
                        self.column += (self.pos - start_pos)
                        tokens.append(Token(TokenType.INT_LIT, num_str, self.line, start_col))
                        continue
                    elif next_char in ('o', 'O'):
                        self.pos += 2
                        oct_start = self.pos
                        while self.pos < self.length and self.source[self.pos] in '01234567':
                            self.pos += 1
                        if self.pos == oct_start:
                            self.error("Expected octal digits after '0o'")
                        num_str = str(int(self.source[start_pos:self.pos], 8))
                        self.column += (self.pos - start_pos)
                        tokens.append(Token(TokenType.INT_LIT, num_str, self.line, start_col))
                        continue

                # Scientific notation: 1.23e-5
                while self.pos < self.length and self.source[self.pos].isdigit():
                    self.pos += 1

                # Check for decimal point
                if self.pos < self.length and self.source[self.pos] == '.' and (self.pos + 1 < self.length and self.source[self.pos + 1].isdigit()):
                    self.pos += 2
                    while self.pos < self.length and self.source[self.pos].isdigit():
                        self.pos += 1
                    # Check for exponent
                    if self.pos < self.length and self.source[self.pos] in ('e', 'E'):
                        self.pos += 1
                        if self.pos < self.length and self.source[self.pos] in ('+', '-'):
                            self.pos += 1
                        while self.pos < self.length and self.source[self.pos].isdigit():
                            self.pos += 1
                    num_str = self.source[start_pos:self.pos]
                    self.column += (self.pos - start_pos)
                    tokens.append(Token(TokenType.FLOAT_LIT, num_str, self.line, start_col))
                elif self.pos < self.length and self.source[self.pos] in ('e', 'E') and self.pos + 1 < self.length and (self.source[self.pos+1].isdigit() or (self.source[self.pos+1] in ('+','-') and self.pos+2 < self.length and self.source[self.pos+2].isdigit())):
                    # Integer with exponent: 1e5, 2e-3
                    self.pos += 1
                    if self.pos < self.length and self.source[self.pos] in ('+', '-'):
                        self.pos += 1
                    while self.pos < self.length and self.source[self.pos].isdigit():
                        self.pos += 1
                    num_str = self.source[start_pos:self.pos]
                    self.column += (self.pos - start_pos)
                    tokens.append(Token(TokenType.FLOAT_LIT, num_str, self.line, start_col))
                else:
                    num_str = self.source[start_pos:self.pos]
                    self.column += (self.pos - start_pos)
                    tokens.append(Token(TokenType.INT_LIT, num_str, self.line, start_col))
                continue

            # Identifiers and keywords
            if char.isalpha() or char == '_':
                start_pos = self.pos
                start_col = self.column
                while self.pos < self.length and (self.source[self.pos].isalnum() or self.source[self.pos] == '_'):
                    self.pos += 1
                ident_str = self.source[start_pos:self.pos]
                self.column += (self.pos - start_pos)

                if ident_str in KEYWORDS_MAP:
                    tokens.append(Token(KEYWORDS_MAP[ident_str], ident_str, self.line, start_col))
                else:
                    tokens.append(Token(TokenType.IDENTIFIER, ident_str, self.line, start_col))
                continue

            self.error(f"Unexpected character: {repr(char)}")

        tokens.append(Token(TokenType.EOF, "", self.line, self.column))
        tokens_list = TokensList(tokens)
        tokens_list.source = self.source
        return tokens_list
