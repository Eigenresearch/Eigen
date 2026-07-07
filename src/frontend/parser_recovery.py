"""
sol.md §7.3 — Error Recovery в Парсере (Phase B).

Roadmap checkbox list (§7.3):
    - [x] "Did you mean?" предложения  (Phase A, did_you_mean.py)
    - [ ] Продолжение парсинга после ошибки (error recovery)
    - [ ] Сбор множественных ошибок за один проход
    - [ ] Контекстные подсказки в сообщениях об ошибках

This module supplies `ErrorCollectingParser` -- a subclass of the
canonical `src.frontend.parser.Parser` that converts the existing
"raise on first error" semantics into a *recovering* parser that
keeps collecting errors through the end of the source.

Design notes
-----------
The base `Parser.error(msg)` raises a `SyntaxError` immediately;
once raised, the `parse()` loop terminates because the body of
the loop does not catch it. To turn that into multi-error recovery
we:

  1. Override `error()` so it appends the `SyntaxError` to
     `self.errors` and then raises a `RecoverableSyntaxError`
     (subclass of `SyntaxError`) instead of the bare `SyntaxError`.
     The override preserves the original error message + line/column
     so callers that catch `SyntaxError` continue to see the same
     structure -- the only thing they lose is the immediate abort.
  2. Wrap `parse_statement()` so that if a `RecoverableSyntaxError`
     escapes from the inner parsing routine, we record the error
     (in case `error()` was called indirectly without going through
     the wrapper) and call the inherited `_recover()` which advances
     to the nearest statement-recovery token (one of
     `;`, `}`, `if`, `let`, `func`, `qfunc`, `for`, `while`,
     `return`, `break`, `continue`, `EOF`), then return None
     instead of a partial AST node. The `parse()` loop already
     tolerates `None` returns from `parse_statement()` -- the
     `if stmt:` check filters them out of the body.
  3. After the EOF loop, if `self.errors` is non-empty, raise
     `MultiParseError(self.errors)` -- an aggregated exception that
     carries every captured `SyntaxError` in `.errors`.

The parser must accept the recovery-token set defined by the base
class; we do not modify that set here.

Contextual hints
----------------
`build_contextual_hint(expected, current_tok, vocabulary_keywords)`
walks common error situations and augments the bare "Expected X
(found Y)" message:

  * When the unexpected token is `IDENTIFIER` whose value closely
    resembles a known keyword (`did_you_mean.suggest`), append the
    `", did you mean 'KEYWORD'?"` suffix. Example: typo `flaot`
    vs the `float` keyword.
  * When the unexpected token is `}` at the start of a statement,
    the more likely user intent was forgetting the opening `{` of
    the enclosing block or having an unbalanced brace -- we suggest
    "Unbalanced '}' -- check the enclosing block for a missing '{'".
  * When the unexpected token is `EOF` mid-statement, append
    "Unexpected end of file -- statements must be terminated by
    ';' or newline".
  * When the unexpected token is a token that the base parser
    recognises as a *statement starter* (e.g. `func` in the middle
    of an expression), the user most likely forgot a `;` separator
    on the previous statement -- append "Missing ';' between
    previous statement and this one".
  * When the unexpected token is the integer-literal that follows
    a `qubit[` type, the user probably forgot the `]` close -- we
    suggest "Missing ']' after qubit-array index".

The hints are *only* cosmetic -- the underlying error message is
preserved verbatim so existing string-comparison tests remain
green.

Public API
----------
  * `RecoverableSyntaxError(SyntaxError)` -- internal exception.
  * `MultiParseError(Exception)` -- aggregated parser error.
  * `ErrorCollectingParser(tokens)` -- subclass of `Parser`.
  * `parse_with_recovery(tokens)` -- convenience: returns
    `(ProgramNode|None, list[SyntaxError])` and never raises.
  * `build_contextual_hint(expected, current_tok,
      keywords=DEFAULT_KEYWORDS)` -- builds the contextual suffix.
  * `DEFAULT_KEYWORDS` -- frozenset of known Eigen keywords.
"""
from __future__ import annotations

import typing

from src.frontend.lexer import Token, TokenType
from src.frontend.parser import Parser
from src.frontend.did_you_mean import format_suggestion


# A frozen vocabulary of language keywords used by
# `build_contextual_hint` to compute "did you mean" suggestions.
_KEYWORDS_TUPLE = (
    "let", "func", "qfunc", "struct", "enum", "trait", "impl", "type",
    "if", "else", "elif", "for", "while", "return", "break", "continue",
    "match", "case", "default", "try", "catch", "throw", "noise",
    "import", "module", "eigen", "measure", "trace", "assert", "print",
    "qubit", "cbit", "int", "float", "bool", "string", "true", "false",
    "and", "or", "not", "in", "as", "where", "self",
    "spawn", "join", "parallel", "task",
)
DEFAULT_KEYWORDS = frozenset(_KEYWORDS_TUPLE)

# Statement-starter tokens -- when the parser encounters one of these
# in the middle of an expression, the prior statement most likely
# lacks a terminating `;`. Used by `build_contextual_hint`.
_STMT_STARTER_TOKENS = frozenset({
    TokenType.LET, TokenType.FUNC, TokenType.QFUNC, TokenType.STRUCT,
    TokenType.ENUM, TokenType.TRAIT, TokenType.IMPL, TokenType.TYPE,
    TokenType.IF, TokenType.FOR, TokenType.WHILE, TokenType.RETURN,
    TokenType.BREAK, TokenType.CONTINUE, TokenType.TRY, TokenType.THROW,
    TokenType.NOISE, TokenType.MEASURE, TokenType.PRINT, TokenType.ASSERT,
})


class RecoverableSyntaxError(SyntaxError):
    """Internal marker exception. Inherits from `SyntaxError` so any
    existing `except SyntaxError` handler continues to work."""
    pass


class MultiParseError(Exception):
    """Aggregated parser error carrying every recovered
    `SyntaxError` from a single `parse()` invocation.

    `errors` is a `list[SyntaxError]`. The `str(MultiParseError)`
    form concatenates the bare error messages one per line.
    """

    def __init__(self, errors: typing.List[SyntaxError]):
        self.errors: typing.List[SyntaxError] = list(errors)
        super().__init__(
            f"{len(self.errors)} parser error(s) during recovery:\n  - " +
            "\n  - ".join(str(e) for e in self.errors))

    def __len__(self) -> int:
        return len(self.errors)

    def __iter__(self):
        return iter(self.errors)


def build_contextual_hint(expected: str, current_tok: Token,
                           keywords: typing.Optional[typing.Iterable[str]] = None,
                           ) -> str:
    """Return an empty string or a contextual suffix that the caller
    can append to the bare "Expected X (found Y)" error message."""
    if current_tok is None:
        return ""
    vocab = set(keywords) if keywords is not None else DEFAULT_KEYWORDS
    tok_type = current_tok.type
    tok_value = current_tok.value if hasattr(current_tok, "value") else None

    # 1) Misspelled keyword -- did-you-mean suggestion.
    if tok_type == TokenType.IDENTIFIER and isinstance(tok_value, str):
        suffix = format_suggestion(tok_value, vocab)
        # `format_suggestion` already returns "",  "did you mean 'X'?"
        # form; if it found something, append it now.
        if suffix:
            return suffix

    # 2) EOF mid-statement -- common at end of file.
    if tok_type == TokenType.EOF:
        return (" -- unexpected end of file; statements must be "
                 "terminated by ';' or a newline")

    # 3) Unbalanced closing brace at statement-start position.
    if tok_type == TokenType.RBRACE:
        return (" -- unbalanced '}'; check the enclosing block for "
                 "a missing '{'")

    # 4) Statement starter seen mid-expression -- missing separator.
    if tok_type in _STMT_STARTER_TOKENS:
        return " -- missing ';' between previous statement and this one"

    # 5) Integer literal following a `qubit[` type token -- most
    #    likely a forgotten closing `]`. We can detect this by
    #    walking the prior-error text in `expected`, which usually
    #    says "Expected ']' ...". We don't have direct access to the
    #    history here, so instead we look at the `expected` hint string.
    if "Expected ']'" in expected and tok_type == TokenType.INT_LIT:
        return " -- missing ']' after the qubit-array index"

    return ""


class ErrorCollectingParser(Parser):
    """Subclass of `Parser` that collects multiple syntax errors
    instead of aborting on the first one. After `parse()`:

      * If `self.errors` is empty -> result is a `ProgramNode` and
        the parser succeeded fully.
      * If `self.errors` is non-empty -- `parse()` raises
        `MultiParseError(self.errors)`. Caller may catch that to
        inspect each individual `SyntaxError` in `.errors`, or use
        the convenience `parse_with_recovery(tokens)` which never
        raises and returns `(ProgramNode|None, errors)`.
    """

    def error(self, msg: str):
        tok = self.current()
        # Build a contextually-augmented message. We only augment
        # when the caller-supplied `msg` does not already contain
        # "did you mean" or "--" suffix (avoid double-augmenting).
        augmented = msg
        if ("did you mean" not in msg.lower()
                and "--" not in msg):
            hint = build_contextual_hint(msg, tok)
            if hint:
                augmented = f"{msg}{hint}"
        err = SyntaxError(
            f"Parser Error at line {tok.line}, col {tok.column}: "
            f"{augmented} (found {tok})")
        self.errors.append(err)
        # Raise the recoverable variant so the wrapped
        # `parse_statement` can catch it and continue.
        raise RecoverableSyntaxError(str(err)) from err

    def _recover(self):
        """Skip tokens until a *fresh-statement-starter* boundary is
        reached, leaving the cursor on (or just past) that token.

        The base `_recover()` only advanced past `;` — which our lexer
        never emits — so a stray `}` (which IS a recovery boundary)
        would otherwise leave the cursor stuck ON the `}` and cause
        infinite rescan in `parse_statement`.
        """
        while (self.current().type not in self.recovery_tokens
               and self.current().type != TokenType.EOF):
            self.pos += 1
        # If we landed on a *closing* bracket, advance past it so the
        # outer block structure can resume. Stay on `let`/`func`/etc.
        # because those indicate the start of a fresh statement.
        if self.current().type in (TokenType.RBRACE,
                                     TokenType.RPAREN,
                                     TokenType.RBRACK):
            self.pos += 1

    def parse_statement(self):
        try:
            return super().parse_statement()
        except RecoverableSyntaxError:
            # The error has already been recorded in self.errors.
            # Recover by advancing to the next statement boundary.
            self._recover()
            return None

    def parse(self):
        """Parse with multi-error recovery. Returns a `ProgramNode`
        on success; raises `MultiParseError` if any errors
        accumulated."""
        try:
            ast = super().parse()
        except RecoverableSyntaxError:
            # An error escaped from outside parse_statement (e.g. the
            # version header check). Recover and continue from the
            # nearest statement starter if possible.
            self._recover()
            ast = None
        if self.errors:
            raise MultiParseError(self.errors)
        if ast is None:
            # No errors but no AST -- something pathological happened.
            # Build an empty program as the caller expects a value.
            from src.frontend.ast import ProgramNode
            return ProgramNode(1.0, None, [], [])
        return ast


def parse_with_recovery(tokens) -> typing.Tuple[typing.Optional[object],
                                                  typing.List[SyntaxError]]:
    """Convenience entry point: parse `tokens` with multi-error
    recovery. Returns `(program, errors)`:

      * `program` is the `ProgramNode` if the overall top-level
        structure survived, else `None`.
      * `errors` is the list of recovered `SyntaxError`s (may be
        empty)."""
    parser = ErrorCollectingParser(tokens)
    try:
        program = parser.parse()
    except MultiParseError as agg:
        # `parse` raised with the aggregate; we still want to look
        # at any partial AST -- but the base parser does not produce
        # one in failure, so return None.
        program = None
        return program, list(agg.errors)
    return program, list(parser.errors)
