import unittest
from src.lsp.lsp_server import LSPServer

class TestLSPServer(unittest.TestCase):
    def setUp(self):
        self.server = LSPServer()

    def test_initialize(self):
        req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        }
        res = self.server.handle_request(req)
        self.assertIsNotNone(res)
        self.assertEqual(res["id"], 1)
        self.assertIn("capabilities", res["result"])
        self.assertTrue(res["result"]["capabilities"]["hoverProvider"])
        self.assertTrue(res["result"]["capabilities"]["definitionProvider"])

    def test_hover(self):
        req = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "textDocument/hover",
            "params": {
                "textDocument": {"uri": "file:///test.eig"},
                "position": {"line": 1, "character": 5}
            }
        }
        res = self.server.handle_request(req)
        self.assertIsNotNone(res)
        self.assertEqual(res["id"], 2)
        self.assertIn("contents", res["result"])

    def test_definition(self):
        req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "textDocument/definition",
            "params": {
                "textDocument": {"uri": "file:///test.eig"},
                "position": {"line": 1, "character": 5}
            }
        }
        res = self.server.handle_request(req)
        self.assertIsNotNone(res)
        self.assertEqual(res["id"], 3)
        self.assertEqual(res["result"]["uri"], "file:///test.eig")

    def test_diagnostics_ok(self):
        req = {
            "jsonrpc": "2.0",
            "method": "textDocument/didOpen",
            "params": {
                "textDocument": {
                    "uri": "file:///test.eig",
                    "text": "eigen 2.5\nlet x: int = 10\n"
                }
            }
        }
        res = self.server.handle_request(req)
        self.assertIsNotNone(res)
        self.assertEqual(res["method"], "textDocument/publishDiagnostics")
        self.assertEqual(len(res["params"]["diagnostics"]), 0)

    def test_diagnostics_error(self):
        req = {
            "jsonrpc": "2.0",
            "method": "textDocument/didOpen",
            "params": {
                "textDocument": {
                    "uri": "file:///test.eig",
                    "text": "eigen 2.5\nlet x: int = 10.5\n" # Type mismatch
                }
            }
        }
        res = self.server.handle_request(req)
        self.assertIsNotNone(res)
        self.assertEqual(res["method"], "textDocument/publishDiagnostics")
        self.assertGreater(len(res["params"]["diagnostics"]), 0)
        self.assertIn("Type Error", res["params"]["diagnostics"][0]["message"])

if __name__ == "__main__":
    unittest.main()
