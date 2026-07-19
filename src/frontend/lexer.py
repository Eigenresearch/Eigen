import enum
import re
import bisect

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
    FINALLY = "finally"
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
    ASYNC = "async"
    AWAIT = "await"
    OPERATOR = "operator"
    MATCH = "match"
    CASE = "case"
    DEFAULT = "default"
    # §3.1 — Trait/Interface System (AST/parser surface). Used by the
    # parser to recognize `trait Foo { ... }` and `impl Foo for Bar { ... }`
    # blocks; the runtime still treats these as structural signatures,
    # not type-system-enforced bounds at this stage.
    TRAIT = "trait"
    IMPL = "impl"
    # §3.3 — Type aliases. `type Name = Target;` declares a substitution
    # that the type-checker resolves lazily on every type-reference site.
    TYPE = "type"
    
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
        "finally": TokenType.FINALLY,
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
        "async": TokenType.ASYNC,
        "await": TokenType.AWAIT,
        "operator": TokenType.OPERATOR,
        "match": TokenType.MATCH,
        "case": TokenType.CASE,
        "default": TokenType.DEFAULT,
        # §3.1 — Trait/Interface system keywords.
        "trait": TokenType.TRAIT,
        "impl": TokenType.IMPL,
        # §3.3 — Type alias declaration keyword.
        "type": TokenType.TYPE,
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
        ';': TokenType.SEMICOLON,
    }

    # === sol.md P0 §1.2 — regex-based fast lexer ============================
    # Master regex compiled once at class-load time. Each named alternative
    # maps to a Token type. ``finditer`` lets the re engine batch tokenization
    # in C, avoiding the per-character Python loop of ``_tokenize_slow``.
    #
    # Order matters: longer/more-specific prefixes must come first (e.g.
    # ``->`` before ``-``, ``<=`` before ``<``). The string alternative stops
    # at the first ``"`` so interpolation ``${...}`` is handled separately by
    # the per-token scan-string routine when the match value contains ``${``.
    _MASTER_PATTERN = re.compile(
        r"""
          (?P<ws>[ \t\r\f\v]+)
        | (?P<nl>\n)
        | (?P<hash_comment>\#[^\n]*)
        | (?P<block_comment>/\*[\s\S]*?\*/)
        | (?P<slash_comment>//[^\n]*)
        | (?P<float_e>\d+[eE][+-]?\d+)
        | (?P<float_dot>\d+\.\d*(?:[eE][+-]?\d+)?)
        | (?P<float_dot_lead>\.\d+(?:[eE][+-]?\d+)?)
        | (?P<hex>0[xX][0-9a-fA-F]+)
        | (?P<bin>0[bB][01]+)
        | (?P<oct>0[oO][0-7]+)
        | (?P<int_dec>\d+)
        | (?P<string_special>["'])
        | (?P<arrow>->)
        | (?P<eq>==)
        | (?P<ne>!=)
        | (?P<le><=)
        | (?P<ge>>=)
        | (?P<lshift><<)
        | (?P<rshift>>>)
        | (?P<add_assign>\+=)
        | (?P<sub_assign>-=)
        | (?P<pow>\*\*)
        | (?P<mul_assign>\*=)
        | (?P<div_assign>/=)
        | (?P<lparen>\()
        | (?P<rparen>\))
        | (?P<lbrace>\{)
        | (?P<rbrace>\})
        | (?P<lbrack>\[)
        | (?P<rbrack>\])
        | (?P<comma>,)
        | (?P<colon>:)
        | (?P<dot>\.)
        | (?P<equals>=)
        | (?P<lt><)
        | (?P<gt>>)
        | (?P<plus>\+)
        | (?P<minus>-)
        | (?P<mul>\*)
        | (?P<div>/)
        | (?P<mod>%)
        | (?P<amp>&)
        | (?P<pipe>\|)
        | (?P<caret>\^)
        | (?P<tilde>~)
        | (?P<semicolon>;)
        | (?P<identifier>[A-Za-z_][A-Za-z0-9_]*)
        """,
        re.VERBOSE,
    )

    # Maps master-regex group name to a callable that returns a
    # (TokenType, value_str) tuple, or None to skip the token.
    # Built lazily from keyword mapping and single-char table.

    # Maps master-regex group name to a callable that returns a
    # (TokenType, value_str) tuple, or None to skip the token.
    # Built lazily from keyword mapping and single-char table.

    def _make_escape_value(self, raw: str) -> str:
        """Backwards-compat helper retained for any external callers; new
        lexer path uses ``_scan_string`` instead which performs brace
        tracking directly on the source. Decodes escape sequences inside a
        string literal body (without surrounding quotes)."""
        out = []
        i = 0
        n = len(raw)
        escape_map = {
            'n': '\n', 't': '\t', 'r': '\r',
            '0': '\0', '\\': '\\', '"': '"',
            "'": "'", 'a': '\a', 'b': '\b',
            'f': '\f', 'v': '\v',
        }
        while i < n:
            ch = raw[i]
            if ch == '\\':
                i += 1
                if i >= n:
                    break
                nxt = raw[i]
                if nxt == 'u':
                    # \uXXXX — Unicode escape (4 hex digits)
                    hex_str = raw[i + 1:i + 5]
                    if len(hex_str) < 4 or not all(c in '0123456789abcdefABCDEF' for c in hex_str):
                        raise SyntaxError(
                            f"Lexer Error at line {self.line}, col {self.column}: "
                            "Invalid unicode escape: expected 4 hex digits after '\\u'"
                        )
                    out.append(chr(int(hex_str, 16)))
                    i += 5
                elif nxt == 'x':
                    # \xNN — Hex escape (2 hex digits)
                    hex_str = raw[i + 1:i + 3]
                    if len(hex_str) < 2 or not all(c in '0123456789abcdefABCDEF' for c in hex_str):
                        raise SyntaxError(
                            f"Lexer Error at line {self.line}, col {self.column}: "
                            "Invalid hex escape: expected 2 hex digits after '\\x'"
                        )
                    out.append(chr(int(hex_str, 16)))
                    i += 3
                else:
                    out.append(escape_map.get(nxt, nxt))
                    i += 1
            elif ch == '$' and i + 1 < n and raw[i + 1] == '{':
                i += 2
                expr_start = i
                depth = 1
                while i < n and depth > 0:
                    c = raw[i]
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            break
                    i += 1
                if depth != 0:
                    raise SyntaxError(
                        f"Lexer Error at line {self.line}, col {self.column}: "
                        "Unterminated string interpolation: expected '}'"
                    )
                expr = raw[expr_start:i]
                out.append(f"\x00{expr}\x00")
                i += 1
            else:
                out.append(ch)
                i += 1
        return "".join(out)

    # Direct group-name → TokenType map for cheap dispatch in the regex
    # hot path (avoids the long if/elif chain). Filled lazily below from
    # the inline alternatives of ``_MASTER_PATTERN``.
    _GROUP_TO_TOKENTYPE = {
        'arrow': TokenType.ARROW,
        'eq': TokenType.EQ, 'ne': TokenType.NE,
        'le': TokenType.LE, 'ge': TokenType.GE,
        'lshift': TokenType.LSHIFT, 'rshift': TokenType.RSHIFT,
        'add_assign': TokenType.ADD_ASSIGN, 'sub_assign': TokenType.SUB_ASSIGN,
        'pow': TokenType.POW, 'mul_assign': TokenType.MUL_ASSIGN,
        'div_assign': TokenType.DIV_ASSIGN,
        'lparen': TokenType.LPAREN, 'rparen': TokenType.RPAREN,
        'lbrace': TokenType.LBRACE, 'rbrace': TokenType.RBRACE,
        'lbrack': TokenType.LBRACK, 'rbrack': TokenType.RBRACK,
        'comma': TokenType.COMMA, 'colon': TokenType.COLON, 'dot': TokenType.DOT,
        'equals': TokenType.EQUALS, 'lt': TokenType.LT, 'gt': TokenType.GT,
        'plus': TokenType.PLUS, 'minus': TokenType.MINUS,
        'mul': TokenType.MUL, 'div': TokenType.DIV, 'mod': TokenType.MOD,
        'amp': TokenType.AMP, 'pipe': TokenType.PIPE, 'caret': TokenType.CARET,
        'tilde': TokenType.TILDE, 'semicolon': TokenType.SEMICOLON,
    }

    def _tokenize_regex(self) -> list[Token]:
        # Alternative regex-based tokenizer (sol.md §1.2). Public ``tokenize``
        # currently delegates to ``_tokenize_slow`` because the legacy path's
        # batched slicing beats Python's regex engine on real Eigen sources
        # in microbenchmarks. This method is kept for parity validation
        # (see ``tests/test_sol_p0_improvements.py``) and for future C/Rust
        # native lexers where a single regex sweep can take over.
        source = self.source
        length = self.length
        if length == 0:
            tokens_list = TokensList([Token(TokenType.EOF, "", 1, 1)])
            tokens_list.source = source
            return tokens_list

        KEYWORDS_MAP = self._KEYWORDS_MAP
        GROUP_TO_TT = self._GROUP_TO_TOKENTYPE
        match_pattern = self._MASTER_PATTERN
        tokens: list[Token] = []
        append_tok = tokens.append

        pos = 0
        cur_line = 1
        cur_col = 1
        while pos < length:
            m = match_pattern.match(source, pos)
            if m is None:
                raise SyntaxError(
                    f"Lexer Error at line {cur_line}, col {cur_col}: "
                    f"Unexpected character: {repr(source[pos])}"
                )
            kind = m.lastgroup
            end = m.end()
            tok_line = cur_line
            tok_col = cur_col
            # Whitespace / comments produce nothing — skip ``m.group()``.
            if (kind == 'ws' or kind == 'nl' or kind == 'hash_comment'
                    or kind == 'slash_comment' or kind == 'block_comment'):
                ch_count = end - pos
                # ``str.count`` is a C-level cursor, not a substring alloc.
                nl_count = source.count('\n', pos, end)
                if nl_count == 0:
                    cur_col += ch_count
                else:
                    cur_line += nl_count
                    last_nl_pos = source.rfind('\n', pos, end)
                    cur_col = end - last_nl_pos
                pos = end
                continue

            value = m.group()
            ch_count = len(value)
            nl_count = value.count('\n')
            if nl_count == 0:
                cur_col += ch_count
            else:
                cur_line += nl_count
                last_nl = value.rfind('\n')
                cur_col = ch_count - last_nl

            if kind == 'identifier':
                tt = KEYWORDS_MAP.get(value)
                if tt is None:
                    tt = TokenType.IDENTIFIER
                append_tok(Token(tt, value, tok_line, tok_col))
            elif kind == 'string_special':
                quote_char = source[pos]
                new_pos, decoded, cur_line, cur_col = self._scan_string(
                    source, pos, tok_line, tok_col, quote_char
                )
                append_tok(Token(TokenType.STRING_LIT, decoded, tok_line, tok_col))
                pos = new_pos
                continue
            elif kind == 'hex':
                append_tok(Token(TokenType.INT_LIT, str(int(value, 16)), tok_line, tok_col))
            elif kind == 'bin':
                append_tok(Token(TokenType.INT_LIT, str(int(value, 2)), tok_line, tok_col))
            elif kind == 'oct':
                append_tok(Token(TokenType.INT_LIT, str(int(value, 8)), tok_line, tok_col))
            elif kind == 'int_dec':
                append_tok(Token(TokenType.INT_LIT, value, tok_line, tok_col))
            elif kind == 'float_e' or kind == 'float_dot' or kind == 'float_dot_lead':
                append_tok(Token(TokenType.FLOAT_LIT, value, tok_line, tok_col))
            else:
                append_tok(Token(GROUP_TO_TT[kind], value, tok_line, tok_col))
            pos = end

        append_tok(Token(TokenType.EOF, "", cur_line, cur_col))
        tokens_list = TokensList(tokens)
        tokens_list.source = source
        return tokens_list

    def tokenize(self) -> list[Token]:
        # Public entry point. We currently route to the original
        # character-by-character path; the regex-based path
        # (``_tokenize_regex``) is available as an alternative implementation
        # for parity validation and as a building block for a future C-level
        # lexer. See ``tests/test_sol_p0_improvements.py`` for the parity
        # contract between the two paths.
        return self._tokenize_slow()

    @staticmethod
    def _offset_to_line_col(line_starts: list[int], pos: int) -> tuple[int, int]:
        line_idx = bisect.bisect_right(line_starts, pos)
        return line_idx, pos - line_starts[line_idx - 1] + 1

    def _scan_string(self, source: str, pos: int, start_line: int, start_col: int,
                     quote_char: str = '"') -> tuple[int, str, int, int]:
        """Scan a ``"..."`` (or ``'...'``) string literal starting at ``pos``
        (pointing at the opening quote). Returns ``(end_pos, decoded_value,
        end_line, end_col)`` where ``end_pos`` is one past the closing quote
        and ``end_line``/``end_col`` are the position immediately *after* the
        closing quote (i.e. where the next token begins).

        Faithfully mirrors the brace-tracking behaviour of ``_tokenize_slow``
        — in particular, a quote char inside a ``${...}`` interpolation is
        treated as a regular character (it does *not* close the string)."""
        n = source.__len__()
        i = pos + 1  # skip opening quote
        out = []
        escape_map = {
            'n': '\n', 't': '\t', 'r': '\r',
            '0': '\0', '\\': '\\', '"': '"',
            "'": "'", 'a': '\a', 'b': '\b',
            'f': '\f', 'v': '\v',
        }
        cur_line = start_line
        cur_col = start_col + 1  # one past the opening quote char
        while i < n:
            ch = source[i]
            if ch == quote_char:
                # closing quote found
                end_pos = i + 1
                # Return col one past the closing quote so the caller's
                # cursor sits at the next token's starting position.
                return end_pos, "".join(out), cur_line, cur_col + 1
            if ch == '\n':
                raise SyntaxError(
                    f"Lexer Error at line {start_line}, col {start_col}: "
                    "Unterminated string literal"
                )
            if ch == '\\':
                i += 1
                if i >= n:
                    break
                nxt = source[i]
                if nxt == 'u':
                    # \uXXXX — Unicode escape (4 hex digits)
                    hex_str = source[i + 1:i + 5]
                    if len(hex_str) < 4 or not all(c in '0123456789abcdefABCDEF' for c in hex_str):
                        raise SyntaxError(
                            f"Lexer Error at line {start_line}, col {start_col}: "
                            "Invalid unicode escape: expected 4 hex digits after '\\u'"
                        )
                    out.append(chr(int(hex_str, 16)))
                    i += 5
                    cur_col += 6
                elif nxt == 'x':
                    # \xNN — Hex escape (2 hex digits)
                    hex_str = source[i + 1:i + 3]
                    if len(hex_str) < 2 or not all(c in '0123456789abcdefABCDEF' for c in hex_str):
                        raise SyntaxError(
                            f"Lexer Error at line {start_line}, col {start_col}: "
                            "Invalid hex escape: expected 2 hex digits after '\\x'"
                        )
                    out.append(chr(int(hex_str, 16)))
                    i += 3
                    cur_col += 4
                else:
                    out.append(escape_map.get(nxt, nxt))
                    i += 1
                    cur_col += 2
            elif ch == '$' and i + 1 < n and source[i + 1] == '{':
                # String interpolation: ${expr}
                i += 2  # skip ${
                cur_col += 2
                expr_start = i
                depth = 1
                while i < n and depth > 0:
                    c = source[i]
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            break
                    i += 1
                if i >= n or depth != 0:
                    raise SyntaxError(
                        f"Lexer Error at line {start_line}, col {start_col}: "
                        "Unterminated string interpolation: expected '}'"
                    )
                expr_str = source[expr_start:i]
                out.append(f"\x00{expr_str}\x00")
                i += 1  # skip closing }
                cur_col += (len(expr_str) + 1) + 1
            else:
                out.append(ch)
                i += 1
                cur_col += 1
        raise SyntaxError(
            f"Lexer Error at line {start_line}, col {start_col}: "
            "Unterminated string literal"
        )

    def _tokenize_slow(self) -> list[Token]:
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

            # Skip block comments: /* ... */ (may span multiple lines)
            if char == '/' and self.pos + 1 < self.length and self.source[self.pos + 1] == '*':
                start_line = self.line
                start_col = self.column
                self.pos += 2
                self.column += 2
                closed = False
                while self.pos < self.length:
                    if self.source[self.pos] == '*' and self.pos + 1 < self.length and self.source[self.pos + 1] == '/':
                        self.pos += 2
                        self.column += 2
                        closed = True
                        break
                    if self.source[self.pos] == '\n':
                        self.line += 1
                        self.column = 1
                        self.pos += 1
                    else:
                        self.pos += 1
                        self.column += 1
                if not closed:
                    self.error(f"Unterminated block comment starting at line {start_line}, col {start_col}")
                continue

            # Skip comments (both # and //)
            if char == '#' or (char == '/' and self.pos + 1 < self.length and self.source[self.pos + 1] == '/'):
                while self.pos < self.length and self.source[self.pos] != '\n':
                    self.pos += 1
                continue

            # String literals (double- or single-quoted)
            if char == '"' or char == "'":
                quote_char = char
                start_col = self.column
                self.pos += 1  # consume open quote
                start_pos = self.pos
                string_val = []
                while self.pos < self.length and self.source[self.pos] != quote_char:
                    if self.source[self.pos] == '\n':
                        self.error("Unterminated string literal")
                    if self.source[self.pos] == '\\':
                        self.pos += 1
                        if self.pos < self.length:
                            esc_char = self.source[self.pos]
                            if esc_char == 'u':
                                # \uXXXX — Unicode escape (4 hex digits)
                                hex_str = self.source[self.pos + 1:self.pos + 5]
                                if len(hex_str) < 4 or not all(c in '0123456789abcdefABCDEF' for c in hex_str):
                                    self.error("Invalid unicode escape: expected 4 hex digits after '\\u'")
                                string_val.append(chr(int(hex_str, 16)))
                                self.pos += 5
                            elif esc_char == 'x':
                                # \xNN — Hex escape (2 hex digits)
                                hex_str = self.source[self.pos + 1:self.pos + 3]
                                if len(hex_str) < 2 or not all(c in '0123456789abcdefABCDEF' for c in hex_str):
                                    self.error("Invalid hex escape: expected 2 hex digits after '\\x'")
                                string_val.append(chr(int(hex_str, 16)))
                                self.pos += 3
                            else:
                                escape_map = {
                                    'n': '\n', 't': '\t', 'r': '\r',
                                    '0': '\0', '\\': '\\', '"': '"',
                                    "'": "'", 'a': '\a', 'b': '\b',
                                    'f': '\f', 'v': '\v',
                                }
                                string_val.append(escape_map.get(esc_char, esc_char))
                                self.pos += 1
                    elif (self.source[self.pos] == '$' and self.pos + 1 < self.length
                          and self.source[self.pos + 1] == '{'):
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
                if self.pos >= self.length or self.source[self.pos] != quote_char:
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
                if (self.pos < self.length and self.source[self.pos] == '.'
                        and (self.pos + 1 < self.length
                             and self.source[self.pos + 1].isdigit())):
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
                elif (self.pos < self.length and self.source[self.pos] in ('e', 'E')
                      and self.pos + 1 < self.length
                      and (self.source[self.pos+1].isdigit()
                           or (self.source[self.pos+1] in ('+','-')
                               and self.pos+2 < self.length
                               and self.source[self.pos+2].isdigit()))):
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
