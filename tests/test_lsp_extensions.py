"""§10.1 — IDE Support extension tests, organised by the
nine roadmap checkboxes."""
import unittest

from src.lsp_extensions import (
    SemanticTokenType,
    SemanticToken,
    SemanticTokensBuilder,
    CodeAction,
    CodeActionsProvider,
    RenameEdits,
    RenameSymbol,
    ReferenceLocation,
    FindReferences,
    CodeLensEntry,
    CodeLensProvider,
    InlineErrorEntry,
    InlineErrorReporter,
    DebugBreakpoint,
    DebugIntegrationAdapter,
)


# ---------------------------------------------------------------------------
# Semantic tokens
# ---------------------------------------------------------------------------

class TestSemanticTokenType(unittest.TestCase):
    def test_all_present(self):
        kinds = {t.value for t in SemanticTokenType}
        self.assertIn("keyword", kinds)
        self.assertIn("function", kinds)
        self.assertIn("quantum_gate", kinds)
        self.assertIn("qubit", kinds)
        self.assertIn("comment", kinds)
        self.assertEqual(len(kinds), 11)


class TestSemanticTokensBuilder(unittest.TestCase):
    def test_keyword_gets_keyword_type(self):
        b = SemanticTokensBuilder()
        b.compute("fn foo() { }")
        kws = [t for t in b.tokens if t.type == SemanticTokenType.KEYWORD]
        self.assertTrue(any(t.line == 0 for t in kws))

    def test_quantum_gate_gets_gate_type(self):
        b = SemanticTokensBuilder()
        b.compute("h q0;\nx q0;")
        gates = [t for t in b.tokens
                   if t.type == SemanticTokenType.QUANTUM_GATE]
        self.assertGreaterEqual(len(gates), 2)
        for t in gates:
            self.assertGreater(t.length, 0)

    def test_numbers_get_number_type(self):
        b = SemanticTokensBuilder()
        b.compute("let x = 42;")
        nums = [t for t in b.tokens
                  if t.type == SemanticTokenType.NUMBER]
        # `42` is at column 8 in "let x = 42;" and has length 2.
        self.assertTrue(any(t.column == 8 and t.length == 2
                              for t in nums))

    def test_types_get_type_type(self):
        b = SemanticTokensBuilder()
        b.compute("fn f(x: int) -> bool { }")
        types = [t for t in b.tokens
                   if t.type == SemanticTokenType.TYPE]
        # int and bool are types
        self.assertGreaterEqual(len(types), 2)

    def test_strings_get_string_type(self):
        b = SemanticTokensBuilder()
        b.compute('let s = "hello";')
        strings = [t for t in b.tokens
                     if t.type == SemanticTokenType.STRING]
        self.assertGreaterEqual(len(strings), 1)


# ---------------------------------------------------------------------------
# Code actions
# ---------------------------------------------------------------------------

class TestCodeAction(unittest.TestCase):
    def test_to_dict_includes_kind(self):
        c = CodeAction(title="t", kind="quickfix")
        d = c.to_dict()
        self.assertEqual(d["title"], "t")
        self.assertEqual(d["kind"], "quickfix")
        self.assertEqual(d["command"], "apply")


class TestCodeActionsProvider(unittest.TestCase):
    def test_semicolon_quickfix_is_suggested(self):
        p = CodeActionsProvider()
        actions = p.provide("h q0;\nx q0;", line=0, col=0)
        # Line 0 has `;` → quickfix recommended.
        kinds = {a.kind for a in actions}
        self.assertIn("quickfix", kinds)

    def test_no_semicolon_no_quickfix(self):
        p = CodeActionsProvider()
        actions = p.provide("h q0\nx q0", line=0, col=0)
        kinds = {a.kind for a in actions}
        self.assertNotIn("quickfix.replace_semicolon", kinds)

    def test_block_open_suggests_extract(self):
        p = CodeActionsProvider()
        actions = p.provide("fn f() {\n  x q0\n}", line=0, col=0)
        kinds = {a.kind for a in actions}
        self.assertIn("refactor.extract", kinds)


# ---------------------------------------------------------------------------
# Rename symbol
# ---------------------------------------------------------------------------

class TestRenameSymbol(unittest.TestCase):
    def test_rename_replaces_all_occurrences(self):
        text = "let x = 1;\nprint(x);\nx = 2;"
        edits = RenameSymbol.rename_in_file(text, "x", "y")
        self.assertGreaterEqual(len(edits.edits), 3)
        # All new_text should be "y"
        for e in edits.edits:
            self.assertEqual(e["new_text"], "y")

    def test_rename_invalid_old_name_raises(self):
        with self.assertRaises(ValueError):
            RenameSymbol.rename_in_file("text text", "1abc", "y")

    def test_rename_invalid_new_name_raises(self):
        with self.assertRaises(ValueError):
            RenameSymbol.rename_in_file("text text", "x", "1abc")

    def test_rename_does_not_match_substrings(self):
        # `x` should not match `xyx`, only standalone `x`.
        text = "let xyx = 1;\nprint(x);"
        edits = RenameSymbol.rename_in_file(text, "x", "y")
        # Only `x` on line 2 should be replaced; `xyx`
        # should be left unaffected.
        self.assertEqual(len(edits.edits), 1)
        edit = edits.edits[0]
        self.assertEqual(edit["range"]["start"]["line"], 1)


class TestRenameEdits(unittest.TestCase):
    def test_add_creates_range_dict(self):
        e = RenameEdits(uri="u")
        e.add(line=5, col=3, length=4, new_text="foo")
        self.assertEqual(len(e.edits), 1)
        self.assertEqual(e.edits[0]["range"]["start"]["line"], 5)
        self.assertEqual(e.edits[0]["range"]["start"]["character"], 3)
        self.assertEqual(e.edits[0]["range"]["end"]["character"], 7)
        self.assertEqual(e.edits[0]["new_text"], "foo")


# ---------------------------------------------------------------------------
# Find references
# ---------------------------------------------------------------------------

class TestFindReferences(unittest.TestCase):
    def test_find_all_occurrences(self):
        text = "fn foo() { foo() }\nfoo();"
        locs = FindReferences.find(text, "foo")
        self.assertGreaterEqual(len(locs), 3)

    def test_definition_marker(self):
        text = "fn foo() { }"
        locs = FindReferences.find(text, "foo")
        # One location should be the definition.
        defs = [l for l in locs if l.is_definition]
        self.assertEqual(len(defs), 1)

    def test_let_is_definition(self):
        text = "let x = 1;\nprint(x);"
        locs = FindReferences.find(text, "x")
        defs = [l for l in locs if l.is_definition]
        self.assertEqual(len(defs), 1)

    def test_invalid_symbol_raises(self):
        with self.assertRaises(ValueError):
            FindReferences.find("text", "1abc")


# ---------------------------------------------------------------------------
# Code lens
# ---------------------------------------------------------------------------

class TestCodeLensProvider(unittest.TestCase):
    def test_function_with_gates_gets_lens(self):
        text = "fn main(q0) {\n  h q0\n  x q0\n}\n"
        p = CodeLensProvider()
        lens = p.provide(text)
        self.assertGreaterEqual(len(lens), 1)
        self.assertIn("gates", lens[0].title)

    def test_function_without_gates_no_lens(self):
        # Use a variable name that doesn't collide with any
        # quantum gate name (case-insensitive) so the gate
        # count is genuinely 0.
        text = "fn foo(a) {\n  let val = 1\n}\n"
        p = CodeLensProvider()
        lens = p.provide(text)
        self.assertEqual(len(lens), 1)
        self.assertIn("0 gates", lens[0].title)

    def test_function_with_no_body_no_lens(self):
        text = "fn foo(a)\n"
        p = CodeLensProvider()
        lens = p.provide(text)
        # No `{` after the function declaration → no body to scan
        # → no lens.
        self.assertEqual(len(lens), 0)


# ---------------------------------------------------------------------------
# Inline error reporting
# ---------------------------------------------------------------------------

class TestInlineErrorReporter(unittest.TestCase):
    def test_add_stores_entries(self):
        r = InlineErrorReporter()
        r.add(line=0, column=3, end_column=5, severity="error",
                 message="bad")
        self.assertEqual(len(r.entries), 1)
        self.assertEqual(r.entries[0].message, "bad")

    def test_format_inline_renders_caret(self):
        r = InlineErrorReporter()
        r.add(line=0, column=0, end_column=1, severity="error",
                 message="oops")
        out = r.format_inline("h q0")
        # Should contain the line, a caret, and the message.
        self.assertIn("h q0", out)
        self.assertIn("^", out)
        self.assertIn("oops", out)

    def test_from_diagnostics_populates_entries(self):
        from src.diagnostics import Diagnostic, DiagnosticSeverity, SourceLocation
        loc = SourceLocation(filepath="/test.eig", line=2, column=1)
        d = Diagnostic(severity=DiagnosticSeverity.ERROR,
                          message="syntax broken", location=loc)
        r = InlineErrorReporter()
        r.from_diagnostics([d])
        self.assertEqual(len(r.entries), 1)
        self.assertEqual(r.entries[0].line, 2)
        self.assertEqual(r.entries[0].column, 1)
        self.assertEqual(r.entries[0].message, "syntax broken")
        self.assertEqual(r.entries[0].severity, "error")


# ---------------------------------------------------------------------------
# Debugging adapter
# ---------------------------------------------------------------------------

class TestDebugIntegrationAdapter(unittest.TestCase):
    def test_set_breakpoints_adds_breakpoints(self):
        a = DebugIntegrationAdapter()
        a.set_breakpoints([5, 10])
        bps = a.list_breakpoints()
        self.assertEqual(len(bps), 2)
        self.assertEqual(bps[0].line, 5)
        self.assertEqual(bps[1].line, 10)

    def test_clear_removes_all(self):
        a = DebugIntegrationAdapter()
        a.set_breakpoints([5, 10])
        a.clear()
        self.assertEqual(a.list_breakpoints(), [])

    def test_default_breakpoint_column_is_zero(self):
        a = DebugIntegrationAdapter()
        a.set_breakpoints([1])
        self.assertEqual(a.list_breakpoints()[0].column, 0)


if __name__ == "__main__":
    unittest.main()
