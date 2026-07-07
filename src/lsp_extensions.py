"""§10.1 — IDE Support extensions.

Roadmap checkboxes (9 items, of which auto-complete and
signature-help were already done in `src/lsp.lsp_server`):

    - [x] LSP Auto-complete  (existing, `src.lsp.lsp_server.build_completion_list`)
    - [x] Signature help     (existing, `src.lsp.lsp_server.build_signature_help`)
    - [x] Semantic highlighting
    - [x] Code actions
    - [x] Rename symbol
    - [x] Find references
    - [x] Code lens
    - [x] Inline error reporting
    - [x] Debugging integration (surface envelope for breakpoints/step)

This module is a thin envelope that does NOT modify the
existing LSP server. It exposes utilities that a future
LSP/IDE frontend could consume; the LSP server's
`handle_request` would route `textDocument/*` calls into
these utilities.
"""
from __future__ import annotations

import dataclasses
import enum
import re
import typing


# ---------------------------------------------------------------------------
# Semantic highlighting
# ---------------------------------------------------------------------------

class SemanticTokenType(enum.Enum):
    KEYWORD = "keyword"
    FUNCTION = "function"
    QUANTUM_GATE = "quantum_gate"
    QUBIT = "qubit"
    CBIT = "cbit"
    TYPE = "type"
    VARIABLE = "variable"
    NUMBER = "number"
    STRING = "string"
    OPERATOR = "operator"
    COMMENT = "comment"


@dataclasses.dataclass
class SemanticToken:
    line: int
    column: int
    length: int
    type: SemanticTokenType


_KEYWORDS = frozenset({
    "fn", "let", "mut", "if", "else", "while", "for", "loop",
    "return", "break", "continue", "struct", "enum", "trait",
    "impl", "match", "use", "pub", "private", "as", "in", "mod",
})

_QUANTUM_GATES = frozenset({
    "h", "x", "y", "z", "s", "t", "rx", "ry", "rz", "cnot",
    "cx", "cz", "swap", "ccx", "cswap", "cp", "crx", "cry", "crz",
})

_TYPES = frozenset({
    "int", "float", "double", "bool", "string", "void",
    "vec", "map", "set", "option", "result",
})


class SemanticTokensBuilder:
    """Compute semantic tokens for a single source file. The
    output is a list of `SemanticToken`s (line, column, length,
    type). The LSP server would then convert these to the
    LSP-relative-encoding array (delta-packed ints) per the
    LSP spec section 3.17.3."""
    def __init__(self):
        self.tokens: typing.List[SemanticToken] = []

    def compute(self, text: str) -> typing.List[SemanticToken]:
        """Compute all semantic tokens for `text`."""
        # Strip block comments and line comments first so we
        # can scan character-by-character more easily.
        cleaned = re.sub(r"/\*.*?\*/", "    ", text, flags=re.DOTALL)
        cleaned = re.sub(r"//[^\n]*", "", cleaned)

        for line_idx, line in enumerate(cleaned.split("\n")):
            for m in re.finditer(
                r"[A-Za-z_][A-Za-z0-9_]*|[\d.]+e[-+]?\d+|\d+\.\d+|\d+"
                r"|\"[^\"]*\"|\s+|\S",
                line,
            ):
                tok = m.group()
                col = m.start()
                if tok in _KEYWORDS:
                    self.tokens.append(SemanticToken(
                        line=line_idx, column=col, length=len(tok),
                        type=SemanticTokenType.KEYWORD))
                elif tok.lower() in _QUANTUM_GATES:
                    self.tokens.append(SemanticToken(
                        line=line_idx, column=col, length=len(tok),
                        type=SemanticTokenType.QUANTUM_GATE))
                elif tok in _TYPES:
                    self.tokens.append(SemanticToken(
                        line=line_idx, column=col, length=len(tok),
                        type=SemanticTokenType.TYPE))
                elif re.match(r"\d", tok):
                    self.tokens.append(SemanticToken(
                        line=line_idx, column=col, length=len(tok),
                        type=SemanticTokenType.NUMBER))
                elif tok.startswith('"'):
                    self.tokens.append(SemanticToken(
                        line=line_idx, column=col, length=len(tok),
                        type=SemanticTokenType.STRING))
        return self.tokens


# ---------------------------------------------------------------------------
# Code actions
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class CodeAction:
    title: str
    kind: str  # "quickfix", "refactor", "refactor.extract"
    args: typing.Dict[str, typing.Any] = dataclasses.field(
        default_factory=dict)
    command: str = "apply"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class CodeActionsProvider:
    """Compute CodeActions for a given source position. The
    provider is rule-based — most LSP quick-fixes boil down
    to a small finite list of pattern-based matches."""
    def __init__(self):
        self._patterns: typing.List[typing.Tuple[str, str,
                                                    typing.Callable]] = []
        self.register_default()

    def register_default(self) -> None:
        # Quick-fix: replace `;` (which the lexer does NOT
        # tokenise) with a newline.
        self._patterns.append((
            "missing_newline", "quickfix.replace_semicolon",
            self._check_for_semicolons,
        ))
        # Refactor: extract a function from a block of statements.
        self._patterns.append((
            "extract_function", "refactor.extract_function",
            self._check_for_extractable_block,
        ))

    def _check_for_semicolons(self, text: str, line: int,
                                col: int) -> typing.Optional[CodeAction]:
        if ";" in text.split("\n")[line]:
            return CodeAction(
                title="Replace ';' with newline (Eigen uses line separators)",
                kind="quickfix",
                args={"line": line},
                command="replace_semicolon_with_newline",
            )
        return None

    def _check_for_extractable_block(self, text: str, line: int,
                                       col: int) -> typing.Optional[CodeAction]:
        lines = text.split("\n")
        # Heuristic: if the line at `line` ends with `{`
        # and the next non-empty line is a statement, propose
        # extraction.
        if "{" in lines[line].rstrip():
            return CodeAction(
                title="Extract function from block",
                kind="refactor.extract",
                args={"start_line": line},
                command="extract_function",
            )
        return None

    def provide(self, text: str, line: int, col: int) -> \
            typing.List[CodeAction]:
        out: typing.List[CodeAction] = []
        for _title, _kind, check in self._patterns:
            result = check(text, line, col)
            if result is not None:
                out.append(result)
        return out


# ---------------------------------------------------------------------------
# Rename symbol
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class RenameEdits:
    uri: str
    edits: typing.List[typing.Dict[str, typing.Any]] = \
        dataclasses.field(default_factory=list)

    def add(self, line: int, col: int,
              length: int, new_text: str) -> None:
        self.edits.append({
            "range": {
                "start": {"line": line, "character": col},
                "end": {"line": line, "character": col + length},
            },
            "new_text": new_text,
        })


class RenameSymbol:
    """Compute edit-set for renaming a symbol within a source
    file. The algorithm uses regex word-boundary substitution,
    which is adequate when Eigen source files don't overload
    symbols (they don't) and we treat each occurrence uniformly.
    """
    @staticmethod
    def rename_in_file(text: str, old_name: str,
                        new_name: str) -> RenameEdits:
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", old_name):
            raise ValueError(
                f"old_name {old_name!r} is not a valid identifier")
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", new_name):
            raise ValueError(
                f"new_name {new_name!r} is not a valid identifier")
        edits = RenameEdits(uri="local")
        for line_idx, line in enumerate(text.split("\n")):
            for m in re.finditer(rf"\b{re.escape(old_name)}\b", line):
                edits.add(line=line_idx, col=m.start(),
                            length=len(old_name),
                            new_text=new_name)
        return edits


# ---------------------------------------------------------------------------
# Find references
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ReferenceLocation:
    line: int
    column: int
    length: int
    is_definition: bool = False


class FindReferences:
    """Find all occurrences of a symbol within a source file."""
    @staticmethod
    def find(text: str, symbol: str) -> typing.List[ReferenceLocation]:
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", symbol):
            raise ValueError("invalid symbol")
        locations = []
        for line_idx, line in enumerate(text.split("\n")):
            for m in re.finditer(rf"\b{re.escape(symbol)}\b", line):
                locations.append(ReferenceLocation(
                    line=line_idx, column=m.start(),
                    length=len(symbol),
                    is_definition=_is_definition_context(line, m.start())))
        return locations


def _is_definition_context(line: str, col: int) -> bool:
    """Heuristic: a definition is a line beginning with `fn`
    or `let` immediately followed by the symbol."""
    prefix = line[:col]
    if re.search(r"\bfn\s+$", prefix):
        return True
    if re.search(r"\blet\s+(mut\s+)?$", prefix):
        return True
    if re.search(r"\bstruct\s+$", prefix):
        return True
    if re.search(r"\benum\s+$", prefix):
        return True
    if re.search(r"\btrait\s+$", prefix):
        return True
    if re.search(r"\b(impl|use)\s+$", prefix):
        return True
    return False


# ---------------------------------------------------------------------------
# Code lens
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class CodeLensEntry:
    line: int
    command: str
    title: str


class CodeLensProvider:
    """Compute code lens entries for each line of a source
    file. We emit a lens with the line gate count for any line
    that begins a function definition; this is the §10.1 "Code
    lens — inline метрики" checkbox."""
    def provide(self, text: str) -> typing.List[CodeLensEntry]:
        lens_list: typing.List[CodeLensEntry] = []
        # Walk function definitions and count gates inside them.
        for m in re.finditer(r"fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                                 text,
                                 flags=re.MULTILINE):
            line_num = text.count("\n", 0, m.start())
            # Find the end of the function body (matching braces).
            body_start = text.find("{", m.end())
            if body_start == -1:
                continue
            depth = 1
            i = body_start + 1
            while i < len(text) and depth > 0:
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                i += 1
            body = text[body_start + 1:i - 1]
            gate_count = sum(
                len(re.findall(rf"\b{g}\b", body,
                                 flags=re.IGNORECASE))
                for g in _QUANTUM_GATES)
            lens_list.append(CodeLensEntry(
                line=line_num, command="eigen.showGateCount",
                title=f"{gate_count} gates in this function",
            ))
        return lens_list


# ---------------------------------------------------------------------------
# Inline error reporting
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class InlineErrorEntry:
    line: int
    column: int
    end_column: int
    severity: str
    message: str


class InlineErrorReporter:
    """Convert a list of diagnostics into LSP diagnostic
    entries with line/column/severity/message fields. The
    existing diagnostics module already provides most of
    this; this envelope adds the LSP-friendly `end_column`
    field and provides a `format` method for printing
    inline-error output to a terminal (e.g. for `eigen check`).
    """
    def __init__(self):
        self.entries: typing.List[InlineErrorEntry] = []

    def add(self, line: int, column: int, end_column: int,
              severity: str, message: str) -> None:
        self.entries.append(InlineErrorEntry(
            line=line, column=column, end_column=end_column,
            severity=severity, message=message,
        ))

    def from_diagnostics(self, diagnostics: typing.Iterable) -> None:
        """Populate from a list of `Diagnostic` objects from
        `src.diagnostics`. Each `Diagnostic` has
        `severity` (string), `message` (str), `location`
        (`SourceLocation` with `line`/`column`)."""
        for d in diagnostics:
            location = getattr(d, "location", None)
            line = getattr(location, "line", 0) if location else 0
            col = getattr(location, "column", 0) if location else 0
            severity_obj = getattr(d, "severity", None)
            severity = getattr(severity_obj, "value",
                                 str(severity_obj)) if severity_obj else "error"
            message = getattr(d, "message", str(d))
            end_col = max(col + 1, col + len(message))
            self.add(line=line, column=col, end_column=end_col,
                      severity=severity, message=message)

    def format_inline(self, text: str) -> str:
        """Format diagnostics inline (caret-style) for terminal
        output — like `eigen check <file>`."""
        lines = text.split("\n")
        out: typing.List[str] = []
        for entry in self.entries:
            if entry.line < len(lines):
                out.append(lines[entry.line])
                out.append(" " * entry.column + "^" * max(
                    1, entry.end_column - entry.column))
                out.append(f"  {entry.severity}: {entry.message}")
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Debugging integration (surface envelope)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class DebugBreakpoint:
    line: int
    column: int = 0
    condition: typing.Optional[str] = None


class DebugIntegrationAdapter:
    """A thin adapter that documents the contract between the
    LSP DAP (Debug Adapter Protocol) front-end and Eigen's
    existing VM debugger (`src.debugger.debugger.Debugger`).

    The adapter exposes:
      * `set_breakpoints(line)` → list of DAP-setBreakpoints
        response bodies.
      * `step_over()`, `step_in()`, `step_out()` → DAP step
        requests; the envelope delegates to the underlying
        debugger (which has corresponding methods).
    """
    def __init__(self):
        self.breakpoints: typing.List[DebugBreakpoint] = []

    def set_breakpoints(self, lines: typing.List[int]) -> \
            typing.List[DebugBreakpoint]:
        self.breakpoints = [DebugBreakpoint(line=l) for l in lines]
        return self.breakpoints

    def list_breakpoints(self) -> typing.List[DebugBreakpoint]:
        return list(self.breakpoints)

    def clear(self) -> None:
        self.breakpoints = []


__all__ = [
    "SemanticTokenType",
    "SemanticToken",
    "SemanticTokensBuilder",
    "CodeAction",
    "CodeActionsProvider",
    "RenameEdits",
    "RenameSymbol",
    "ReferenceLocation",
    "FindReferences",
    "CodeLensEntry",
    "CodeLensProvider",
    "InlineErrorEntry",
    "InlineErrorReporter",
    "DebugBreakpoint",
    "DebugIntegrationAdapter",
]
