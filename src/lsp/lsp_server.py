import sys
import json
import os
from src.frontend.lexer import Lexer
from src.frontend.parser import Parser
from src.semantic.import_resolver import ImportResolver
from src.semantic.type_checker import TypeChecker, TypeErrorException

class LSPServer:
    def __init__(self):
        self.workspace_root = os.getcwd()

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
                        "workspaceSymbolProvider": True
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
            position = params.get("position", {})
            # Sample hover details
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "contents": {
                        "kind": "markdown",
                        "value": "**Eigen Symbol**\nType: Quantum/Classical instruction keyword."
                    }
                }
            }
        elif method == "textDocument/definition":
            params = req.get("params", {})
            uri = params.get("textDocument", {}).get("uri", "")
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "uri": uri,
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 0}
                    }
                }
            }
        elif method == "workspace/symbol":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": []
            }
            
        return None

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
