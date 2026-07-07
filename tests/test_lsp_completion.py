"""§10.1 LSP Auto-complete tests.

These tests exercise the new `textDocument/completion` and bonus
`textDocument/signatureHelp` handlers added by the §10.1 work. They:

  * Drive `LSPServer.handle_request` end-to-end (initialize → didOpen →
    completion) so the wire shape is exercised.
  * Drive `build_completion_list` and `build_signature_help` directly
    for finer control over edge cases (empty positions, malformed
    input, parse-failure fallback).
"""

from __future__ import annotations

import unittest

from src.lsp.lsp_server import (
    LSPServer,
    build_completion_list,
    build_signature_help,
)


_BELL_SRC = """eigen 1.0
qubit q0
qubit q1
H q0
CNOT q0, q1
measure q0 -> c0
"""

_FUNC_SRC = """eigen 2.7
func add(a: int, b: int) -> int {
    return a + b
}
let result: int = add(1, 2)
"""


class TestCompletionListDirect(unittest.TestCase):
    def test_keyword_completion_includes_eigen_keywords(self):
        items = build_completion_list("", {"line": 0, "character": 0})["items"]
        labels = {i["label"] for i in items}
        for required in ("qubit", "func", "let", "if", "for", "while",
                         "struct", "import", "qfunc"):
            self.assertIn(required, labels)

    def test_keyword_completion_includes_builtin_types(self):
        items = build_completion_list("", {"line": 0, "character": 0})["items"]
        labels = {i["label"] for i in items}
        for t in ("int", "float", "bool", "string", "qubit", "cbit"):
            self.assertIn(t, labels)

    def test_gate_completion_includes_core_gates(self):
        items = build_completion_list("", {"line": 0, "character": 0})["items"]
        labels = {i["label"] for i in items}
        for g in ("H", "X", "Y", "Z", "S", "T", "CNOT", "CZ", "SWAP",
                  "RX", "RY", "RZ", "CCX", "CSWAP", "CP", "CRX", "CRY",
                  "CRZ"):
            self.assertIn(g, labels)

    def test_gate_completion_items_use_function_kind(self):
        items = build_completion_list("", {"line": 0, "character": 0})["items"]
        gates = [i for i in items if i.get("detail", "").startswith("Quantum gate ")]
        for g in gates:
            self.assertEqual(g["kind"], 3, f"{g['label']} should be Function (3)")

    def test_builtins_include_pi_tau_e(self):
        items = build_completion_list("", {"line": 0, "character": 0})["items"]
        labels = {i["label"] for i in items}
        for c in ("PI", "TAU", "E"):
            self.assertIn(c, labels)


class TestCompletionSourceSymbols(unittest.TestCase):
    def test_qubit_symbols_extracted(self):
        result = build_completion_list(_BELL_SRC, {"line": 0, "character": 0})
        labels = {i["label"] for i in result["items"]}
        self.assertIn("q0", labels)
        self.assertIn("q1", labels)

    def test_function_symbols_extracted(self):
        result = build_completion_list(_FUNC_SRC, {"line": 0, "character": 0})
        labels = {i["label"] for i in result["items"]}
        self.assertIn("add", labels)

    def test_let_binding_symbols_extracted(self):
        result = build_completion_list(_FUNC_SRC, {"line": 0, "character": 0})
        labels = {i["label"] for i in result["items"]}
        self.assertIn("result", labels)

    def test_parse_failure_falls_back_to_regex(self):
        # Malformed source — partial typing should still yield qubit
        # completions via the regex sweep fallback in build_completion_list.
        source = "eigen 1.0\nqubit q0\nqubit q1\nH q0\nCNOT q0  # missing target arg"
        result = build_completion_list(source, {"line": 0, "character": 0})
        # The regex sweep should pick up q0 and q1 even if the parser fails.
        labels = {i["label"] for i in result["items"]}
        self.assertIn("q0", labels)
        self.assertIn("q1", labels)


class TestCompletionPrefixFilter(unittest.TestCase):
    def test_prefix_filters_to_matching(self):
        # Type "q" in an empty file → should narrow to qubit/cbit and any
        # qubit-prefixed user symbols.
        result = build_completion_list("eigen 1.0\nq\n", {"line": 1, "character": 1})
        labels = {i["label"] for i in result["items"]}
        self.assertIn("qubit", labels)
        for lab in labels:
            self.assertTrue(lab.lower().startswith("q"),
                             f"label '{lab}' should start with 'q'")

    def test_prefix_empty_returns_all(self):
        result = build_completion_list("eigen 1.0\n", {"line": 1, "character": 0})
        self.assertFalse(result["isIncomplete"])
        self.assertGreater(len(result["items"]), 8)


class TestEndToEndCompletion(unittest.TestCase):
    """Drive LSPServer via JSON-RPC request shape."""

    def setUp(self):
        self.srv = LSPServer()

    def test_initialize_advertises_completion(self):
        res = self.srv.handle_request({
            "jsonrpc": "2.0", "id": 1, "method": "initialize"
        })
        self.assertIn("completionProvider", res["result"]["capabilities"])
        self.assertIn("(","(", res["result"]["capabilities"]["completionProvider"]["triggerCharacters"])
        self.assertIn("signatureHelpProvider", res["result"]["capabilities"])

    def test_completion_after_didOpen_returns_user_symbols(self):
        self.srv.handle_request({
            "jsonrpc": "2.0", "method": "textDocument/didOpen",
            "params": {"textDocument": {"uri": "file:///t.eig", "text": _BELL_SRC}},
        })
        res = self.srv.handle_request({
            "jsonrpc": "2.0", "id": 7, "method": "textDocument/completion",
            "params": {
                "textDocument": {"uri": "file:///t.eig"},
                "position": {"line": 0, "character": 0},
            },
        })
        self.assertEqual(res["id"], 7)
        labels = {i["label"] for i in res["result"]["items"]}
        self.assertIn("q0", labels)
        self.assertIn("H", labels)   # builtin gate
        self.assertIn("qubit", labels)


class TestSignatureHelpDirect(unittest.TestCase):
    def test_signature_help_returns_matching_func(self):
        result = build_signature_help(_FUNC_SRC, {"line": 4, "character": 22})
        # Cursor on `add(` — the signature should include "add(a: int, b: int) -> int"
        self.assertIsNotNone(result)
        labels = [sig["label"] for sig in result["signatures"]]
        joined = " ".join(labels)
        self.assertIn("add", joined)
        self.assertIn("a", joined)
        self.assertIn("b", joined)

    def test_signature_help_returns_none_when_no_paren(self):
        result = build_signature_help("eigen 1.0\nlet x: int = 5\n",
                                       {"line": 1, "character": 14})
        self.assertIsNone(result)

    def test_signature_help_returns_none_when_func_unknown(self):
        result = build_signature_help("eigen 1.0\nzzz(\n",
                                       {"line": 1, "character": 4})
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
