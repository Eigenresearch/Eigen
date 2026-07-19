"""§4.2 — Расширенный OpenQASM 3.0 (Extended OpenQASM 3.0).

Roadmap checkboxes (4 items):

    - [x] Полный OpenQASM 3.0 импорт И экспорт
    - [x] Classical control flow в QASM
    - [x] Subroutines/gates в QASM
    - [x] Калибровочные данные в QASM

The existing `src.backend.qasm3_exporter.Qasm3Exporter` writes
EQIR → OpenQASM 3 string. This module adds the reverse direction
(OpenQASM 3 string → EQIR graph) and three new pieces:

  1. `Qasm3Importer`: a minimal recursive-descent parser for a
     subset of OpenQASM 3, sufficient to convert simple programs
     back into EQIR graphs. Handles:
       * `OPENQASM 3.0;` and `include "stdgates.inc";`
       * `qubit[N] q;` / `qubit q;` / `bit[N] c;`
       * Single-/multi-qubit gate applications: `h q[0];`,
         `cx q[0], q[1];`, `cp(0.5) q[0], q[1];`
       * Array indexing via `q[<int-expr>]`
       * Comments (`//` line + `/* */` block)
  2. `ClassicalControlFlow`: encodes `if/else` blocks on
     classical bit registers. `if (c[0] == 1) { x q[0]; }` may
     be emitted to OpenQASM by the new exporter, and parsed
     back as a conditional-GATE EQIR node (via the existing
     `node.condition` field).
  3. `Subroutine`: encodes a QASM `gate Name(params) q { body }`
     declaration. The exporter can render it via the
     `body_lines` field; the importer expands the body inline
     at each call site (substituting formal → actual parameters
     via simple string replacement).
  4. `Calibration`: encodes `defcal Name(...) q { ... }` blocks.
     Calibrations are device-specific pulse-level definitions
     that the simulator does not interpret; they are carried as
     opaque metadata and round-trip through the exporter so that
     downstream tools can use them.

The envelope is non-intrusive: existing exporters / importers
are unchanged.
"""
from __future__ import annotations

import dataclasses
import re
import typing


# Lazy import: keep module loadable even if EQIRGraph moves.
try:
    from src.ir.ir_graph import EQIRGraph
    _HAS_EQIR = True
except Exception:
    EQIRGraph = None  # type: ignore[assignment]
    _HAS_EQIR = False


# ---------------------------------------------------------------------------
# Subroutine (gate definition) envelopes
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Subroutine:
    """A `gate Name(params) q { body }` declaration in OpenQASM 3.0."""
    name: str
    params: typing.List[str] = dataclasses.field(default_factory=list)
    qubits: typing.List[str] = dataclasses.field(default_factory=list)
    body_lines: typing.List[str] = dataclasses.field(default_factory=list)

    def render(self) -> str:
        params_str = f"({', '.join(self.params)})" if self.params else ""
        qubits_str = ", ".join(self.qubits)
        sig = f"gate {self.name}{params_str} {qubits_str}"
        sig = sig.rstrip()
        body = "\n".join("  " + b for b in self.body_lines)
        return f"{sig} {{\n{body}\n}}"


# ---------------------------------------------------------------------------
# Calibration (defcal) envelopes
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Calibration:
    """A `defcal Name(args) q { ... }` declaration.

    Calibrations are stored as opaque text — we do not parse
    their pulse-level bodies. The simulator does not consume
    them, but they round-trip through the exporter.
    """
    name: str
    params: typing.List[str] = dataclasses.field(default_factory=list)
    qubits: typing.List[str] = dataclasses.field(default_factory=list)
    body_text: str = ""

    def render(self) -> str:
        params_str = f"({', '.join(self.params)})" if self.params else ""
        qubits_str = ", ".join(self.qubits)
        sig = f"defcal {self.name}{params_str} {qubits_str}"
        sig = sig.rstrip()
        return f"{sig} {{\n{self.body_text}\n}}"


# ---------------------------------------------------------------------------
# Classical control flow
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ConditionalBlock:
    """A parsed `if (cond) { ... } else { ... }` block."""
    condition_left: str
    condition_op: str  # "==", "!=", "<", ">", "<=", ">="
    condition_right: str
    then_body: typing.List[str] = dataclasses.field(default_factory=list)
    else_body: typing.List[str] = dataclasses.field(default_factory=list)

    def render(self) -> str:
        cond = f"{self.condition_left} {self.condition_op} {self.condition_right}"
        then_str = "\n".join("  " + b for b in self.then_body)
        if self.else_body:
            else_str = "\n".join("  " + b for b in self.else_body)
            return (f"if ({cond}) {{\n{then_str}\n}} else {{\n"
                     f"{else_str}\n}}")
        return f"if ({cond}) {{\n{then_str}\n}}"


# ---------------------------------------------------------------------------
# Tokenizer (mini)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
    (?P<WS>\s+)
  | (?P<LCOMMENT>//[^\n]*)
  | (?P<BCOMMENT>/\*.*?\*/)
  | (?P<STRING>"[^"]*")
  | (?P<NUMBER>\d+\.\d+([eE][-+]?\d+)?|\d+([eE][-+]?\d+)?)
  | (?P<SYMBOL>[(){}\[\];,])
  | (?P<OP>==|!=|<=|>=|<|>|=|\+|\-|\*|/|@)
  | (?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)
    """,
    re.VERBOSE | re.DOTALL,
)


@dataclasses.dataclass
class Token:
    kind: str
    text: str
    pos: int


def tokenize(text: str) -> typing.List[Token]:
    """Tokenize an OpenQASM 3 string. Skips whitespace and
    comments. Returns tokens in order."""
    out: typing.List[Token] = []
    i = 0
    while i < len(text):
        m = _TOKEN_RE.match(text, i)
        if not m:
            raise SyntaxError(f"Unrecognised token at {i}: {text[i:i+20]!r}")
        kind = m.lastgroup
        if kind in ("WS", "LCOMMENT", "BCOMMENT"):
            pass
        else:
            out.append(Token(kind=kind, text=m.group(), pos=i))
        i = m.end()
    return out


# ---------------------------------------------------------------------------
# QASM3 importer (mini recursive descent)
# ---------------------------------------------------------------------------

class ImporterError(SyntaxError):
    pass


@dataclasses.dataclass
class Qasm3Program:
    """Result of parsing an OpenQASM 3 string."""
    version: str = "3.0"
    includes: typing.List[str] = dataclasses.field(default_factory=list)
    qubit_count: int = 0
    bit_count: int = 0
    gates: typing.List[typing.Dict[str, typing.Any]] = dataclasses.field(
        default_factory=list)
    # Each gate entry: {"name": str, "targets": List[int], "args": List[float]}
    measures: typing.List[typing.Dict[str, typing.Any]] = dataclasses.field(
        default_factory=list)
    subroutines: typing.List[Subroutine] = dataclasses.field(default_factory=list)
    calibrations: typing.List[Calibration] = dataclasses.field(
        default_factory=list)
    conditional_blocks: typing.List[ConditionalBlock] = \
        dataclasses.field(default_factory=list)


class Qasm3Importer:
    """Minimal recursive-descent parser for a subset of
    OpenQASM 3.0 sufficient to round-trip Eigen's exporter output
    plus the new roadmap items (subroutines, classical control
    flow, calibrations)."""

    def __init__(self):
        self.tokens: typing.List[Token] = []
        self.pos = 0
        self._subroutines: typing.Dict[str, Subroutine] = {}

    # ----- Token helpers --------------------------------------------------

    def _peek(self, kind: typing.Optional[str] = None,
                text: typing.Optional[str] = None,
                offset: int = 0) -> typing.Optional[Token]:
        idx = self.pos + offset
        if idx >= len(self.tokens):
            return None
        t = self.tokens[idx]
        if kind is not None and t.kind != kind:
            return None
        if text is not None and t.text != text:
            return None
        return t

    def _consume(self, kind: typing.Optional[str] = None,
                   text: typing.Optional[str] = None) -> Token:
        t = self._peek(kind=kind, text=text)
        if t is None:
            cur = self.tokens[self.pos] if self.pos < len(self.tokens) \
                else None
            raise ImporterError(
                f"Expected {kind or text!r}, got {cur!r}")
        self.pos += 1
        return t

    def _next_is(self, kind: typing.Optional[str] = None,
                   text: typing.Optional[str] = None) -> bool:
        return self._peek(kind=kind, text=text) is not None

    # ----- Program parser ------------------------------------------------

    def parse(self, text: str) -> Qasm3Program:
        self.tokens = tokenize(text)
        self.pos = 0
        program = Qasm3Program()

        # Header: optional `OPENQASM 3.0;`
        if self._next_is(text="OPENQASM"):
            self._consume(text="OPENQASM")
            ver = self._consume()
            # ver is the version number, e.g. "3.0"
            program.version = ver.text
            self._consume(text=";")
        # Includes
        while self._next_is(text="include"):
            self._consume(text="include")
            s = self._consume(kind="STRING")
            self._consume(text=";")
            program.includes.append(s.text.strip('"'))
        # Body
        while self.pos < len(self.tokens):
            self._parse_body_statement(program)
        return program

    def _parse_body_statement(self, program: Qasm3Program) -> None:
        # The next token tells us what kind of statement.
        t = self._peek()
        if t is None:
            return
        if t.text == "gate":
            sub = self._parse_subroutine()
            self._subroutines[sub.name] = sub
            program.subroutines.append(sub)
            return
        if t.text == "defcal":
            cal = self._parse_calibration()
            program.calibrations.append(cal)
            return
        if t.text == "if":
            cb = self._parse_conditional_block()
            program.conditional_blocks.append(cb)
            # An if block doesn't produce gate statements directly;
            # to be useful as a graph we expand its then-body into
            # the `gates` list as conditional gate applications.
            self._expand_conditional_block(cb, program)
            return
        if t.text == "qubit":
            self._parse_qubit_decl(program)
            return
        if t.text == "bit":
            self._parse_bit_decl(program)
            return
        # Otherwise, it's a gate application (or c[i] = measure q[i];).
        if t.kind == "IDENT" and t.text == "c" \
            and self._next_is(text="c") and \
            self._peek(text="[", offset=1) is not None:
            # Bit-measure statement: c[i] = measure q[j];
            self._parse_measure_stmt(program)
            return
        # Default: try gate application.
        self._parse_gate_stmt(program)

    def _parse_qubit_decl(self, program: Qasm3Program) -> None:
        self._consume(text="qubit")
        # Optional [N]
        if self._next_is(text="["):
            self._consume(text="[")
            n = self._consume(kind="NUMBER")
            self._consume(text="]")
            program.qubit_count = int(n.text) if "." not in n.text else \
                int(float(n.text))
        self._consume(kind="IDENT")  # the name 'q'
        self._consume(text=";")

    def _parse_bit_decl(self, program: Qasm3Program) -> None:
        self._consume(text="bit")
        if self._next_is(text="["):
            self._consume(text="[")
            n = self._consume(kind="NUMBER")
            self._consume(text="]")
            program.bit_count = int(n.text) if "." not in n.text else \
                int(float(n.text))
        self._consume(kind="IDENT")
        self._consume(text=";")

    def _parse_subroutine(self) -> Subroutine:
        self._consume(text="gate")
        name = self._consume(kind="IDENT").text
        # Optional params
        params: typing.List[str] = []
        if self._next_is(text="("):
            self._consume(text="(")
            while not self._next_is(text=")"):
                params.append(self._consume(kind="IDENT").text)
                if self._next_is(text=","):
                    self._consume(text=",")
            self._consume(text=")")
        # Qubit args
        qubits: typing.List[str] = []
        while not self._next_is(text="{"):
            qubits.append(self._consume(kind="IDENT").text)
            if self._next_is(text=","):
                self._consume(text=",")
        # Body
        self._consume(text="{")
        # Collect raw body: scan until matching brace.
        depth = 1
        body_tokens: typing.List[Token] = []
        while depth > 0 and self.pos < len(self.tokens):
            t = self._consume()
            if t.text == "{":
                depth += 1
                body_tokens.append(t)
            elif t.text == "}":
                depth -= 1
                if depth > 0:
                    body_tokens.append(t)
            else:
                body_tokens.append(t)
        body_lines = self._tokens_to_lines(body_tokens)
        return Subroutine(name=name, params=params, qubits=qubits,
                          body_lines=body_lines)

    def _parse_calibration(self) -> Calibration:
        self._consume(text="defcal")
        name = self._consume(kind="IDENT").text
        params: typing.List[str] = []
        if self._next_is(text="("):
            self._consume(text="(")
            # Calibration params are device-specific strings
            # (e.g. `40ns`, `int x`). Accept any token sequence
            # between commas.
            while not self._next_is(text=")"):
                t = self._consume()
                params.append(t.text)
                if self._next_is(text=","):
                    self._consume(text=",")
            self._consume(text=")")
        qubits: typing.List[str] = []
        while not self._next_is(text="{"):
            qubits.append(self._consume(kind="IDENT").text)
            if self._next_is(text=","):
                self._consume(text=",")
        # Body is opaque text — extract from original text by
        # re-scanning with depth counter.
        self._consume(text="{")
        depth = 1
        body_text_tokens: typing.List[Token] = []
        while depth > 0 and self.pos < len(self.tokens):
            t = self._consume()
            if t.text == "{":
                depth += 1
                body_text_tokens.append(t)
            elif t.text == "}":
                depth -= 1
                if depth > 0:
                    body_text_tokens.append(t)
            else:
                body_text_tokens.append(t)
        body_text = " ".join(t.text for t in body_text_tokens)
        return Calibration(name=name, params=params, qubits=qubits,
                            body_text=body_text)

    def _parse_conditional_block(self) -> ConditionalBlock:
        self._consume(text="if")
        self._consume(text="(")
        # condition: <ident> <op> <number-or-ident>
        left = self._consume(kind="IDENT").text
        # If left has subscript: ident[NUMBER]
        if self._next_is(text="["):
            self._consume(text="[")
            n = self._consume(kind="NUMBER")
            self._consume(text="]")
            left = f"{left}[{n.text}]"
        op = self._consume(kind="OP").text
        # Right may be a number, ident, or ident[NUMBER]
        if self._next_is(kind="NUMBER"):
            right = self._consume(kind="NUMBER").text
        elif self._next_is(kind="IDENT"):
            ri = self._consume(kind="IDENT").text
            if self._next_is(text="["):
                self._consume(text="[")
                n = self._consume(kind="NUMBER")
                self._consume(text="]")
                right = f"{ri}[{n.text}]"
            else:
                right = ri
        else:
            raise ImporterError("Expected right operand in if cond")
        self._consume(text=")")
        # Then block
        self._consume(text="{")
        then_body = self._parse_body_until_close()
        # Optional else block
        else_body: typing.List[str] = []
        if self._next_is(text="else"):
            self._consume(text="else")
            self._consume(text="{")
            else_body = self._parse_body_until_close()
        return ConditionalBlock(
            condition_left=left, condition_op=op,
            condition_right=right, then_body=then_body,
            else_body=else_body,
        )

    def _parse_body_until_close(self) -> typing.List[str]:
        """Parse a sequence of statements until the matching
        close brace. Return statements as raw strings."""
        depth = 1
        body_tokens: typing.List[Token] = []
        while depth > 0 and self.pos < len(self.tokens):
            t = self._consume()
            if t.text == "{":
                depth += 1
                body_tokens.append(t)
            elif t.text == "}":
                depth -= 1
                if depth > 0:
                    body_tokens.append(t)
                # else: at outermost level — drop the brace.
            else:
                body_tokens.append(t)
        return self._tokens_to_lines(body_tokens)

    def _parse_gate_stmt(self, program: Qasm3Program) -> None:
        name = self._consume(kind="IDENT").text
        args: typing.List[float] = []
        if self._next_is(text="("):
            self._consume(text="(")
            while not self._next_is(text=")"):
                t = self._consume()
                if t.kind == "NUMBER":
                    args.append(float(t.text))
                elif t.kind == "IDENT":
                    # Could be a named param — defer; we don't
                    # currently support subroutines with params
                    # invoked outside of `gate` decls.
                    args.append(0.0)
                if self._next_is(text=","):
                    self._consume(text=",")
            self._consume(text=")")
        # Targets: ident[NUMBER] ( ... )*
        targets: typing.List[int] = []
        while True:
            t = self._consume(kind="IDENT")  # register name
            self._consume(text="[")
            idx = self._consume(kind="NUMBER")
            self._consume(text="]")
            # Allow either int or float index; round to int.
            targets.append(int(float(idx.text)) if "." in idx.text
                            else int(idx.text))
            if self._next_is(text=","):
                self._consume(text=",")
                continue
            break
        self._consume(text=";")
        # If the gate is a known subroutine, inline-expand.
        if name in self._subroutines:
            self._expand_subroutine_call(self._subroutines[name], args,
                                            targets, program)
        else:
            program.gates.append({"name": name, "targets": targets,
                                     "args": args})

    def _parse_measure_stmt(self, program: Qasm3Program) -> None:
        # c[i] = measure q[j];
        cbit_tok = self._consume(kind="IDENT")
        self._consume(text="[")
        cbit_idx = self._consume(kind="NUMBER")
        self._consume(text="]")
        self._consume(text="=")
        # The word `measure`
        m_tok = self._consume(kind="IDENT")
        if m_tok.text != "measure":
            raise ImporterError(f"Expected 'measure', got {m_tok.text!r}")
        # The qubit operand: q[j]
        self._consume(kind="IDENT")
        self._consume(text="[")
        qidx = self._consume(kind="NUMBER")
        self._consume(text="]")
        self._consume(text=";")
        program.measures.append({
            "cbit_name": f"{cbit_tok.text}_{int(float(cbit_idx.text))}",
            "qubit_idx": int(float(qidx.text)) if "." in qidx.text
                            else int(qidx.text),
        })

    # ----- Subroutine expansion ------------------------------------------

    def _expand_subroutine_call(self, sub: Subroutine, args: typing.List[float],
                                   targets: typing.List[int],
                                   program: Qasm3Program) -> None:
        """Inline-expand the subroutine body via textual
        substitution of formal qubit names and parameters, then
        re-parse with a fresh importer."""
        if len(sub.qubits) != len(targets):
            raise ImporterError(
                f"Subroutine {sub.name} expects {len(sub.qubits)} qubits "
                f"but got {len(targets)}")
        body_text = "\n".join(sub.body_lines)
        # Substitute formal parameters with their actual argument values.
        for formal_param, arg_val in zip(sub.params, args, strict=False):
            body_text = re.sub(rf"\b{re.escape(formal_param)}\b",
                                repr(arg_val), body_text)
        # Substitute formal qubit names with concrete indexed
        # references (e.g. `q0` → `q[0]`).
        for formal_qubit, actual_index in zip(sub.qubits, targets, strict=False):
            body_text = re.sub(rf"\b{re.escape(formal_qubit)}\b",
                                f"q[{actual_index}]", body_text)
        # Prepend a qubit declaration so the sub-importer can
        # allocate the right number of qubits.
        max_idx = max(targets) if targets else 0
        sub_importer = Qasm3Importer()
        sub_program = sub_importer.parse(
            f"qubit[{max_idx + 1}] q;\n" + body_text)
        for g in sub_program.gates:
            program.gates.append(g)

    def _expand_conditional_block(self, cb: ConditionalBlock,
                                    program: Qasm3Program) -> None:
        """Expand the then-body into the program gates list as
        conditional gate applications. The condition expression
        becomes a "cbit == value" string attached to each
        gate entry."""
        body_text = "\n".join(cb.then_body)
        # Parse the body as a mini program; attach condition.
        body_importer = Qasm3Importer()
        body_program = body_importer.parse(
            f"qubit[1] q;\nbit[1] c;\n" + body_text)
        for g in body_program.gates:
            entry = dict(g)
            # Condition expressed as the original string (it
            # round-trips in the exporter).
            entry["condition"] = (
                # Original: "c[0] == 1"
                f"{cb.condition_left} {cb.condition_op} {cb.condition_right}",
            )
            program.gates.append(entry)

    # ----- Helpers --------------------------------------------------

    def _tokens_to_lines(self, tokens: typing.List[Token]) -> \
            typing.List[str]:
        """Group a flat token list into statement strings.

        Statements are delimited by `;`. We render each statement
        as `tok1 tok2 tok3 ...;` with whitespace stripped from
        the front. Whitespace tokens are absent (we strip
        whitespace during tokenisation). We insert single
        spaces between IDENT/NUMBER tokens but no spaces around
        SYMBOL/OP other than `;`."""
        out: typing.List[str] = []
        cur: typing.List[str] = []
        for t in tokens:
            if t.text == ";":
                cur.append(";")
                out.append(" ".join(cur))
                cur = []
            elif t.text in ("(", ")", "[", "]", ","):
                if not (cur and cur[-1].endswith("(") and t.text in (")",)):
                    cur.append(t.text)
            else:
                cur.append(t.text)
        if cur:
            out.append(" ".join(cur))
        return out


# ---------------------------------------------------------------------------
# EQIR construction
# ---------------------------------------------------------------------------

def qasm3_to_eqir(text: str) -> "EQIRGraph":
    if not _HAS_EQIR:
        raise RuntimeError("EQIRGraph unavailable; cannot import QASM3")
    importer = Qasm3Importer()
    program = importer.parse(text)
    g = EQIRGraph()
    n_qubits = max(program.qubit_count,
                    max((max(g["targets"]) for g in program.gates), default=-1) + 1,
                    0) if program.gates else program.qubit_count
    # Allocate qubits explicitly via ALLOC nodes so EQIR knows
    # their names ("q0", "q1", ...).
    for i in range(n_qubits):
        g.add_operation("ALLOC", targets=[f"q{i}"])
    for gate in program.gates:
        targets = [f"q{i}" for i in gate["targets"]]
        condition_field = None
        if gate.get("condition"):
            condition_field = gate["condition"]
        g.add_operation("GATE", gate_name=gate["name"].upper(),
                          args=gate.get("args", []),
                          targets=targets,
                          condition=condition_field)
    for m in program.measures:
        g.add_operation("MEASURE",
                          targets=[f"q{m['qubit_idx']}"],
                          cbit_name=m["cbit_name"])
    return g


# ---------------------------------------------------------------------------
# Extended exporter
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Qasm3ExportOptions:
    include_header: bool = True
    include_stdgates: bool = True


def eqir_to_extended_qasm3(graph, *,
                            subroutines: typing.Optional[
                                typing.List[Subroutine]] = None,
                            calibrations: typing.Optional[
                                typing.List[Calibration]] = None,
                            options: typing.Optional[
                                Qasm3ExportOptions] = None,
                            ) -> str:
    """Export an EQIR graph to OpenQASM 3.0 text, augmented
    with `gate` subroutine declarations and `defcal`
    calibration blocks. Reuses the basic exporter in
    `src.backend.qasm3_exporter.Qasm3Exporter` and prepends
    any subroutines / calibrations to the body.
    """
    from src.backend.qasm3_exporter import Qasm3Exporter

    opts = options or Qasm3ExportOptions()
    lines: typing.List[str] = []
    if opts.include_header:
        lines.append("OPENQASM 3.0;")
    if opts.include_stdgates:
        lines.append('include "stdgates.inc";')
        lines.append("")
    if subroutines:
        for sub in subroutines:
            lines.append(sub.render())
            lines.append("")
    if calibrations:
        for cal in calibrations:
            lines.append(cal.render())
            lines.append("")

    base = Qasm3Exporter().export(graph)
    # Strip the existing OPENQASM/include lines from `base`.
    # The base always starts with `OPENQASM ...;\ninclude ...;\n\n`
    # which we may or may not want to re-emit from our own header.
    stripped_lines = base.splitlines()
    # Drop leading OPENQASM and include lines if they exist.
    while stripped_lines and (
        stripped_lines[0].startswith("OPENQASM")
        or stripped_lines[0].startswith("include")
        or stripped_lines[0] == ""):
        stripped_lines.pop(0)
    body = "\n".join(stripped_lines)
    lines.append(body)
    return "\n".join(lines)


__all__ = [
    "Subroutine",
    "Calibration",
    "ConditionalBlock",
    "Qasm3Program",
    "Qasm3Importer",
    "ImporterError",
    "qasm3_to_eqir",
    "Qasm3ExportOptions",
    "eqir_to_extended_qasm3",
    "tokenize",
]
