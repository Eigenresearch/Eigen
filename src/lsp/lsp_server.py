import sys
import json
import os
from src.frontend.lexer import Lexer, TokenType
from src.frontend.parser import Parser
from src.semantic.import_resolver import ImportResolver
from src.semantic.type_checker import TypeChecker, TypeErrorException


# LSP CompletionItemKind — see
# https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_completion
_CI_KIND_TEXT = 1
_CI_KIND_METHOD = 2
_CI_KIND_FUNCTION = 3
_CI_KIND_FIELD = 5
_CI_KIND_VARIABLE = 6
_CI_KIND_CLASS = 7
_CI_KIND_KEYWORD = 14
_CI_KIND_SNIPPET = 15
_CI_KIND_STRUCT = 22

# LSP SymbolKind (for workspace/symbol and semantic highlighting).
_SYMB_KIND_FUNCTION = 12
_SYMB_KIND_STRUCT = 23
_SYMB_KIND_VARIABLE = 13
_SYMB_KIND_CLASS = 5


def _completion_engine_keywords():
    """Return LSP CompletionItem list for every Eigen keyword in the lexer's
    keyword token map. Excludes the gate mnemonics (H/X/Y/Z/S/T/...) — those
    are emitted as a separate gate category so completion can mark them as
    Quantum gates (kind=Function) rather than keywords."""
    out = []
    # Build a set of recognized gate token types; we'll skip those from
    # the keyword stream and emit them via `_completion_engine_gates`.
    gate_token_types = {v for v in TokenType.__members__.values()
                        if v.name.startswith("GATE_")}
    for kw, tok_type in sorted(Lexer._KEYWORDS_MAP.items()):
        if tok_type in gate_token_types:
            continue
        if kw in ("true", "false", "null"):
            kind = _CI_KIND_TEXT
            detail = "Eigen literal"
        elif kw in ("int", "float", "string", "bool", "array", "map",
                    "qubit", "cbit"):
            kind = _CI_KIND_STRUCT
            detail = "Eigen built-in type"
        else:
            kind = _CI_KIND_KEYWORD
            detail = "Eigen keyword"
        out.append({
            "label": kw,
            "kind": kind,
            "detail": detail,
        })
    return out


def _completion_engine_gates():
    """Return CompletionItems for the 50+ quantum gates Eigen knows about.
    The list is the union of:
      * `Lexer._KEYWORDS_MAP` entries that map to a `TokenType.GATE_*`
      * `gate_registry.ALL_GATES` (some entries are added there but absent
        from the lexer map at the moment of writing, e.g. iSWAP, SX, ECR,
        DCX, U1/U2/U3 — we surface them too so the user sees the full
        vocabulary even before the lexer/parser catches up).
    """
    out = []
    gate_token_types = {v for v in TokenType.__members__.values()
                        if v.name.startswith("GATE_")}
    seen = set()
    for kw, tok_type in sorted(Lexer._KEYWORDS_MAP.items()):
        if tok_type in gate_token_types:
            seen.add(kw)
            out.append({
                "label": kw,
                "kind": _CI_KIND_FUNCTION,
                "detail": f"Quantum gate {kw}",
            })
    # Pull any additional gate names from the registry without duplicating.
    try:
        from src.backend.gate_registry import ALL_GATES, GATE_QUBIT_COUNT
        for g in sorted(ALL_GATES):
            if g in seen:
                continue
            arity = GATE_QUBIT_COUNT.get(g, "?")
            out.append({
                "label": g,
                "kind": _CI_KIND_FUNCTION,
                "detail": f"Quantum gate {g} ({arity}-qubit)",
            })
            seen.add(g)
    except Exception:
        # gate_registry import is optional; the static lexer map already
        # covers the core gates.
        pass
    return out


def _completion_engine_builtins():
    """Constants + global built-ins (`PI`, `TAU`, `E`, `print`, `measure`,
    `trace`, `assert`) — surfaced once here even though some also appear in
    the keyword list as keywords (we want them findable from a `print`
    call-site too).
    """
    return [
        {"label": "PI", "kind": _CI_KIND_VARIABLE,
         "detail": "Mathematical constant pi"},
        {"label": "TAU", "kind": _CI_KIND_VARIABLE,
         "detail": "Mathematical constant tau = 2*pi"},
        {"label": "E", "kind": _CI_KIND_VARIABLE,
         "detail": "Euler's number e"},
        {"label": "print", "kind": _CI_KIND_FUNCTION,
         "detail": "Eigen built-in print"},
        {"label": "measure", "kind": _CI_KIND_FUNCTION,
         "detail": "Eigen built-in measure"},
        {"label": "trace", "kind": _CI_KIND_FUNCTION,
         "detail": "Eigen built-in partial trace"},
        {"label": "assert", "kind": _CI_KIND_FUNCTION,
         "detail": "Eigen built-in assert"},
    ]


def _extract_symbols_from_ast(program):
    """Walk an Eigen AST and return (functions, structs, qubits, lets).

    Each entry is a flat `CompletionItem`-shaped dict so callers can
    extend the completion list with user-defined symbols without
    re-walking the AST.
    """
    functions = []
    structs = []
    qubits = []
    lets = []

    def _walk(stmts):
        for stmt in stmts or []:
            cls = type(stmt).__name__
            if cls == "FuncDeclNode":
                param_str = ", ".join(f"{p}: {t}" for p, t in stmt.params)
                functions.append({
                    "label": stmt.name,
                    "kind": _CI_KIND_FUNCTION,
                    "detail": f"func {stmt.name}({param_str}) -> {stmt.return_type}",
                })
            elif cls == "QFuncDeclNode":
                param_str = ", ".join(f"{p}: {t}" for p, t in stmt.params)
                functions.append({
                    "label": stmt.name,
                    "kind": _CI_KIND_FUNCTION,
                    "detail": f"qfunc {stmt.name}({param_str})",
                })
            elif cls == "StructDeclNode":
                structs.append({
                    "label": stmt.name,
                    "kind": _CI_KIND_STRUCT,
                    "detail": f"struct {stmt.name} ({len(stmt.fields)} fields)",
                })
            elif cls == "VarDeclNode":
                if stmt.type_name == "qubit":
                    qubits.append({
                        "label": stmt.name,
                        "kind": _CI_KIND_VARIABLE,
                        "detail": "qubit",
                    })
                else:
                    lets.append({
                        "label": stmt.name,
                        "kind": _CI_KIND_VARIABLE,
                        "detail": stmt.type_name,
                    })
            elif cls == "LetNode":
                lets.append({
                    "label": stmt.name,
                    "kind": _CI_KIND_VARIABLE,
                    "detail": stmt.type_name or "let binding",
                })
            elif cls == "FuncDeclNode":
                pass  # handled above
            # Recurse into nested bodies
            for attr in ("body", "else_body"):
                inner = getattr(stmt, attr, None)
                if isinstance(inner, list):
                    _walk(inner)

    if program is not None:
        _walk(getattr(program, "body", None) or [])
    return functions, structs, qubits, lets


def _filter_by_prefix(items, prefix):
    if not prefix:
        return items
    lower = prefix.lower()
    return [i for i in items if i["label"].lower().startswith(lower)]


def build_completion_list(text, position):
    """Build the LSP CompletionList for cursor at `position` in `text`.

    Used both by the LSP handler and by the unit tests to avoid spinning
    up a full JSON-RPC round-trip.

    Returns:
        {"isIncomplete": False, "items": [...]}
    """
    # Resolve the prefix the user has typed so far at the cursor.
    line_idx = position.get("line", 0)
    char_idx = position.get("character", 0)
    lines = text.split("\n")
    line = lines[line_idx] if line_idx < len(lines) else ""
    prefix_chars = []
    i = min(char_idx, len(line))
    while i > 0 and (line[i - 1].isalnum() or line[i - 1] == "_"):
        prefix_chars.append(line[i - 1])
        i -= 1
    prefix = "".join(reversed(prefix_chars))

    items = []
    items.extend(_completion_engine_keywords())
    items.extend(_completion_engine_gates())
    items.extend(_completion_engine_builtins())

    # Walk the source AST for user-defined symbols. Fall back to an
    # even simpler regex sweep if parsing fails so partial-syntax input
    # (very common during typing) still gets completions.
    program = None
    try:
        lexer = Lexer(text)
        parser = Parser(lexer.tokenize())
        program = parser.parse()
    except Exception:
        program = None

    functions, structs, qubits, lets = _extract_symbols_from_ast(program)
    if program is None:
        # Lex-fail fallback: surface qubit decls by simple regex scan.
        import re
        seen = set()
        for m in re.finditer(r"\bqubit\s+([A-Za-z_][A-Za-z0-9_]*)", text):
            name = m.group(1)
            if name in seen:
                continue
            seen.add(name)
            qubits.append({"label": name, "kind": _CI_KIND_VARIABLE,
                           "detail": "qubit (regex sweep)"})
        for m in re.finditer(r"\bfunc\s+([A-Za-z_][A-Za-z0-9_]*)", text):
            name = m.group(1)
            if name in seen:
                continue
            seen.add(name)
            functions.append({"label": name, "kind": _CI_KIND_FUNCTION,
                              "detail": "func (regex sweep)"})

    # User-defined symbols take priority over the built-in keyword list
    # but both are returned (LSP clients rank by relevance).
    items.extend(qubits)
    items.extend(functions)
    items.extend(structs)
    items.extend(lets)

    if prefix:
        items = _filter_by_prefix(items, prefix)
        # If we filtered, mark incomplete so the client re-queries when
        # the user types more (the rank could change with more chars).
        incomplete = len(items) == 0
    else:
        incomplete = False

    return {"isIncomplete": incomplete, "items": items}


def build_signature_help(text, position):
    """Return LSP SignatureHelp for a `func` call near `position`.

    Walks the source AST, finds every `FuncDeclNode` whose name matches
    the identifier immediately preceding `(`, and returns its signature.
    Returns None if no matching function declaration is found.
    """
    line_idx = position.get("line", 0)
    char_idx = position.get("character", 0)
    lines = text.split("\n")
    line = lines[line_idx] if line_idx < len(lines) else ""
    # Find the identifier just before '(' near cursor.
    upto = line[:char_idx]
    paren_idx = upto.rfind("(")
    if paren_idx < 0:
        return None
    ident_end = paren_idx
    j = ident_end
    while j > 0 and (upto[j - 1].isalnum() or upto[j - 1] == "_"):
        j -= 1
    name = upto[j:ident_end]
    if not name:
        return None

    # Find the matching FuncDeclNode / QFuncDeclNode in the AST.
    try:
        lexer = Lexer(text)
        parser = Parser(lexer.tokenize())
        program = parser.parse()
    except Exception:
        return None
    if program is None:
        return None

    candidates = []
    def _walk(stmts):
        for stmt in stmts or []:
            cls = type(stmt).__name__
            if cls == "FuncDeclNode" and stmt.name == name:
                sig = f"func {name}(" + ", ".join(
                    f"{p}: {t}" for p, t in stmt.params) + \
                    f") -> {stmt.return_type}"
                candidates.append(sig)
            elif cls == "QFuncDeclNode" and stmt.name == name:
                sig = f"qfunc {name}(" + ", ".join(
                    f"{p}: {t}" for p, t in stmt.params) + ")"
                candidates.append(sig)
            for attr in ("body", "else_body"):
                inner = getattr(stmt, attr, None)
                if isinstance(inner, list):
                    _walk(inner)

    _walk(getattr(program, "body", None) or [])
    if not candidates:
        return None
    return {
        "signatures": [
            {"label": sig, "documentation": f"Eigen function {name}"}
            for sig in candidates
        ],
        "activeSignature": 0,
        "activeParameter": 0,
    }


def _word_at(text, position):
    """Return the identifier under the cursor at `position`, or None."""
    line_idx = position.get("line", 0)
    char_idx = position.get("character", 0)
    lines = text.split("\n")
    if line_idx >= len(lines):
        return None
    line = lines[line_idx]
    i = min(char_idx, len(line))
    start = i
    while start > 0 and (line[start - 1].isalnum() or line[start - 1] == "_"):
        start -= 1
    end = i
    while end < len(line) and (line[end].isalnum() or line[end] == "_"):
        end += 1
    word = line[start:end]
    return word if word else None


_BUILTIN_HOVER_DETAILS = {
    "PI": "const PI: float = 3.141592653589793",
    "TAU": "const TAU: float = 6.283185307179586",
    "E": "const E: float = 2.718281828459045",
    "print": "func print(value) -> void",
    "measure": "func measure(qubit: qubit) -> cbit",
    "trace": "func trace() -> void",
    "assert": "func assert(condition) -> void",
}


def _lookup_hover_info(text, word):
    """Look up `word` across builtins, keywords, and the document AST.

    Returns a type-info string suitable for a markdown code block, or
    None when the symbol is unknown.
    """
    if word in _BUILTIN_HOVER_DETAILS:
        return _BUILTIN_HOVER_DETAILS[word]

    if word in Lexer._KEYWORDS_MAP:
        tok_type = Lexer._KEYWORDS_MAP[word]
        if word in ("int", "float", "string", "bool", "array", "map",
                    "qubit", "cbit"):
            return f"{word}  // built-in type"
        if word in ("true", "false", "null"):
            return f"{word}  // Eigen literal"
        gate_token_types = {v for v in TokenType.__members__.values()
                            if v.name.startswith("GATE_")}
        if tok_type in gate_token_types:
            return f"gate {word}  // quantum gate"
        return f"{word}  // Eigen keyword"

    try:
        lexer = Lexer(text)
        parser = Parser(lexer.tokenize())
        program = parser.parse()
    except Exception:
        program = None

    if program is not None:
        found = []

        def _walk(stmts):
            for stmt in stmts or []:
                cls = type(stmt).__name__
                if cls == "FuncDeclNode" and stmt.name == word:
                    param_str = ", ".join(f"{p}: {t}" for p, t in stmt.params)
                    found.append(
                        f"func {word}({param_str}) -> {stmt.return_type}")
                elif cls == "QFuncDeclNode" and stmt.name == word:
                    param_str = ", ".join(f"{p}: {t}" for p, t in stmt.params)
                    found.append(f"qfunc {word}({param_str})")
                elif cls == "StructDeclNode" and stmt.name == word:
                    fields_str = ", ".join(f"{f}: {t}" for f, t in stmt.fields)
                    found.append(f"struct {word} {{ {fields_str} }}")
                elif cls == "EnumDeclNode" and stmt.name == word:
                    found.append(
                        f"enum {word} {{ {', '.join(stmt.variants)} }}")
                elif cls == "VarDeclNode" and stmt.name == word:
                    found.append(f"let {word}: {stmt.type_name}")
                elif cls == "LetNode" and stmt.name == word:
                    found.append(f"let {word}: {stmt.type_name or 'unknown'}")
                elif cls == "TypeAliasDeclNode" and stmt.name == word:
                    found.append(f"type {word} = {stmt.target_type}")
                for attr in ("body", "else_body"):
                    inner = getattr(stmt, attr, None)
                    if isinstance(inner, list):
                        _walk(inner)

        _walk(getattr(program, "body", None) or [])
        if found:
            return found[0]

    import re
    for kw in ("func", "qfunc", "let", "struct", "enum", "type"):
        if re.search(rf"\b{kw}\s+{re.escape(word)}\b", text):
            return f"{kw} {word}"
    return None


def build_hover(text, position):
    """Return LSP Hover for the symbol at `position` in `text`.

    Returns None when no hover information is available (per LSP spec
    the hover result is null in that case).
    """
    word = _word_at(text, position)
    if not word:
        return None
    type_info = _lookup_hover_info(text, word)
    if type_info is None:
        return None
    return {
        "contents": {
            "kind": "markdown",
            "value": f"```eigen\n{type_info}\n```",
        }
    }


def build_definition(text, position):
    """Return the definition position {line, character} for the symbol
    at `position` in `text`, or None if not found.

    AST nodes don't carry source positions, so this scans the document
    text for declaration patterns (`func|qfunc|let|struct|enum|type
    {word}`) and returns the first match's position (0-indexed).
    """
    word = _word_at(text, position)
    if not word:
        return None
    import re
    pattern = rf"\b(?:func|qfunc|let|struct|enum|type)\s+({re.escape(word)})\b"
    for line_idx, line in enumerate(text.split("\n")):
        m = re.search(pattern, line)
        if m:
            return {"line": line_idx, "character": m.start(1)}
    return None


class LSPServer:
    def __init__(self):
        self.workspace_root = os.getcwd()
        # In-memory document cache; updated by didOpen/didChange so the
        # completion/signatureHelp handlers don't have to re-read from
        # disk. The cache is also the right surface for unit testing
        # — we can push a synthetic document into `documents` directly
        # and then poke `handle_request` for completion.
        self.documents = {}

    def run(self):
        # LSP uses Header\r\n\r\nContent format
        # Content-Length: <len>\r\n\r\n<JSON-RPC>
        stdin = sys.stdin.buffer
        stdout = sys.stdout.buffer
        
        while True:
            try:
                # Read Headers
                headers = {}
                while True:
                    line = stdin.readline()
                    if not line or line == b'\r\n':
                        break
                    line_str = line.decode('utf-8')
                    if ':' in line_str:
                        k, v = line_str.split(':', 1)
                        headers[k.strip().lower()] = v.strip()
                        
                if 'content-length' not in headers:
                    continue
                    
                content_len = int(headers['content-length'])
                body = stdin.read(content_len)
                if not body:
                    break
                    
                request = json.loads(body.decode('utf-8'))
                response = self.handle_request(request)
                if response:
                    res_body = json.dumps(response).encode('utf-8')
                    res_headers = f"Content-Length: {len(res_body)}\r\n\r\n".encode('utf-8')
                    stdout.write(res_headers)
                    stdout.write(res_body)
                    stdout.flush()
            except Exception as e:
                # Avoid crashing on parsing errors, just log to stderr
                print(f"LSP Server Error: {e}", file=sys.stderr)

    def handle_request(self, req: dict) -> dict | None:
        method = req.get("method")
        req_id = req.get("id")
        
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "capabilities": {
                        "textDocumentSync": 1,  # Full sync
                        "hoverProvider": True,
                        "definitionProvider": True,
                        "workspaceSymbolProvider": True,
                        # §10.1 — Auto-complete + signature help.
                        "completionProvider": {
                            "resolveProvider": False,
                            "triggerCharacters": [".", "("],
                        },
                        "signatureHelpProvider": {
                            "triggerCharacters": ["(", ","],
                        },
                    }
                }
            }
        elif method == "textDocument/didOpen" or method == "textDocument/didChange":
            # Run diagnostics check and emit textDocument/publishDiagnostics notification
            params = req.get("params", {})
            doc = params.get("textDocument", {})
            uri = doc.get("uri", "")
            text = doc.get("text", "")
            
            # Since didChange doesn't always send full text unless sync=1
            if not text and "contentChanges" in params:
                text = params["contentChanges"][0].get("text", "")

            # Cache the snapshot so the completion / signatureHelp
            # handlers return completions for the buffer currently open
            # in the editor — important for tests, which only send
            # didOpen (no disk file to fall back on).
            if uri:
                self.documents[uri] = text

            if text:
                diagnostics = self.run_diagnostics(text, uri)
                # This is a notification (no id)
                return {
                    "jsonrpc": "2.0",
                    "method": "textDocument/publishDiagnostics",
                    "params": {
                        "uri": uri,
                        "diagnostics": diagnostics
                    }
                }
        elif method == "textDocument/hover":
            params = req.get("params", {})
            uri = params.get("textDocument", {}).get("uri", "")
            position = params.get("position", {})
            text = self._snapshot(uri)
            hover = build_hover(text, position)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": hover,
            }
        elif method == "textDocument/definition":
            params = req.get("params", {})
            uri = params.get("textDocument", {}).get("uri", "")
            position = params.get("position", {})
            text = self._snapshot(uri)
            pos = build_definition(text, position)
            if pos is None:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": None,
                }
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "uri": uri,
                    "range": {
                        "start": {"line": pos["line"], "character": pos["character"]},
                        "end": {"line": pos["line"], "character": pos["character"]},
                    }
                }
            }
        elif method == "textDocument/completion":
            # §10.1 — Auto-complete.
            params = req.get("params", {})
            uri = params.get("textDocument", {}).get("uri", "")
            position = params.get("position", {})
            text = self._snapshot(uri)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": build_completion_list(text, position),
            }
        elif method == "textDocument/signatureHelp":
            # §10.1 — Bonus: signature help for function calls.
            params = req.get("params", {})
            uri = params.get("textDocument", {}).get("uri", "")
            position = params.get("position", {})
            text = self._snapshot(uri)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": build_signature_help(text, position),
            }
        elif method == "workspace/symbol":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": []
            }
            
        return None

    def _snapshot(self, uri: str) -> str:
        """Return the most recent text snapshot we have for `uri`.

        Prefers the in-memory `documents` cache (populated by
        didOpen/didChange); falls back to reading from disk if the
        URI isn't cached — useful for ad-hoc completion on a file
        the user opened before the LSP was alive.
        """
        if uri in self.documents:
            return self.documents[uri]
        if uri.startswith("file://"):
            path = uri[len("file://"):]
        else:
            path = uri
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""

    def run_diagnostics(self, text: str, uri: str) -> list:
        diagnostics = []
        try:
            lexer = Lexer(text)
            tokens = lexer.tokenize()
            parser = Parser(tokens)
            ast = parser.parse()
            resolver = ImportResolver(self.workspace_root)
            ast = resolver.resolve(ast)
            type_checker = TypeChecker()
            type_checker.check(ast)
        except TypeErrorException as te:
            # Type error diagnostics format
            diagnostics.append({
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 0, "character": 10}
                },
                "severity": 1,  # Error
                "message": f"Type Error: {te}"
            })
        except Exception as e:
            # Parser/Lexer error
            diagnostics.append({
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 0, "character": 10}
                },
                "severity": 1,  # Error
                "message": str(e)
            })
        return diagnostics
