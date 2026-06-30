import unittest
import subprocess
import json
import os
import sys

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.frontend.lexer import Lexer as PythonLexer
from src.frontend.parser import Parser as PythonParser

def python_ast_to_dict(node):
    if node is None:
        return None
    if isinstance(node, (int, float, str, bool)):
        return node
    if isinstance(node, list):
        return [python_ast_to_dict(x) for x in node]
    if isinstance(node, tuple):
        # Convert tuples to list for JSON compatibility
        return [python_ast_to_dict(x) for x in node]
    if isinstance(node, dict):
        return {k: python_ast_to_dict(v) for k, v in node.items()}
    
    # It's an ASTNode
    res = {"type": type(node).__name__}
    try:
        attrs = vars(node).items()
    except TypeError:
        print("FAILING NODE TYPE:", type(node), node)
        raise
    for k, v in attrs:
        # Skip internal properties if any
        if k.startswith("_"):
            continue
        res[k] = python_ast_to_dict(v)
    return res

def rust_arena_to_nested_dict(arena, node_id):
    if node_id is None:
        return None
    node = arena["nodes"][node_id]
    variant = list(node.keys())[0]
    fields = node[variant]
    
    res = {"type": variant + "Node"}
    for k, v in fields.items():
        if k in ("body", "imports", "targets", "args", "tasks", "params", "elements", "keys", "values", "try_body", "catch_body"):
            if isinstance(v, list):
                if k == "params" and len(v) > 0 and isinstance(v[0], list):
                    res[k] = v
                else:
                    res[k] = [rust_arena_to_nested_dict(arena, x) if isinstance(x, int) else x for x in v]
            else:
                res[k] = v
        elif k in ("left", "right", "condition_left", "condition_right", "expr", "iter", "cond", "obj", "index", "struct_expr", "value_expr", "array_expr", "index_expr", "key_expr", "map_expr", "call") or (k == "value" and variant != "Literal"):
            res[k] = rust_arena_to_nested_dict(arena, v) if isinstance(v, int) else v
        elif k == "value" and variant == "Literal":
            if isinstance(v, dict):
                inner_key = list(v.keys())[0]
                res[k] = v[inner_key]
            elif v == "Null":
                res[k] = None
            else:
                res[k] = v
        elif k == "callee":
            if "Node" in v:
                res[k] = rust_arena_to_nested_dict(arena, v["Node"])
            else:
                res[k] = v["Name"]
        elif k == "field_bindings":
            res[k] = {pair[0]: rust_arena_to_nested_dict(arena, pair[1]) for pair in v}
        else:
            res[k] = v
    return res

class TestFrontendEquivalence(unittest.TestCase):
    def setUp(self):
        self.workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.binary_path = os.path.join(self.workspace_root, "native", "rust", "target", "release", "eigen-frontend.exe")
        if not os.path.exists(self.binary_path):
            self.binary_path = os.path.join(self.workspace_root, "native", "rust", "target", "debug", "eigen-frontend.exe")
        if not os.path.exists(self.binary_path):
            self.binary_path = os.path.join(self.workspace_root, "native", "rust", "target", "release", "eigen_frontend")
        if not os.path.exists(self.binary_path):
            self.binary_path = os.path.join(self.workspace_root, "native", "rust", "target", "debug", "eigen_frontend")
        self.skip_rust = not os.path.exists(self.binary_path)

    def run_rust_binary(self, source: str):
        res = subprocess.run(
            [self.binary_path],
            input=source,
            capture_output=True,
            text=True,
            shell=True if os.name == 'nt' else False
        )
        return res

    def assert_equivalent(self, source: str):
        if self.skip_rust:
            self.skipTest("Rust frontend binary not available")
        # 1. Parse using Python (bypass native delegation)
        lexer = PythonLexer(source)
        tokens = lexer.tokenize()
        if hasattr(tokens, "source"):
            tokens.source = None # Bypass native delegation
        
        parser = PythonParser(tokens)
        try:
            py_ast = parser.parse()
            py_failed = False
            py_error = ""
        except Exception as e:
            py_failed = True
            py_error = str(e)

        # 2. Parse using Rust binary
        rust_res = self.run_rust_binary(source)
        if rust_res.returncode != 0:
            rust_failed = True
            rust_error = rust_res.stderr
        else:
            rust_failed = False
            rust_json_output = json.loads(rust_res.stdout)
            rust_root_id = rust_json_output["root_id"]
            rust_ast_arena = rust_json_output["ast"]

        self.assertEqual(py_failed, rust_failed, f"Parser failure mismatch on:\n{source}\nPython failed: {py_failed} ({py_error})\nRust failed: {rust_failed} ({rust_res.stderr})")
        
        if not py_failed:
            # Reconstruct Rust AST to nested dictionary starting at ProgramNode root_id
            rust_nested = rust_arena_to_nested_dict(rust_ast_arena, rust_root_id)
            py_nested = python_ast_to_dict(py_ast)
            self.assertEqual(py_nested, rust_nested, f"AST structure mismatch on:\n{source}\nPython AST: {py_nested}\nRust AST: {rust_nested}")

    def test_simple_program(self):
        self.assert_equivalent("eigen 1.0\nlet x: int = 5\n")

    def test_variables_and_arithmetic(self):
        self.assert_equivalent("eigen 1.0\nlet x: float = 3.14\nlet y: int = 2 + 3 * 4\n")

    def test_functions_and_calls(self):
        self.assert_equivalent("eigen 1.0\nfunc add(a: int, b: int) -> int {\n    return a + b\n}\nlet z: int = add(1, 2)\n")

    def test_loops_and_conditionals(self):
        self.assert_equivalent("eigen 1.0\nif 1 < 2 {\n    print(1)\n} else {\n    print(0)\n}\nfor i in 0..10 {\n    print(i)\n}\n")

    def test_quantum_constructs(self):
        self.assert_equivalent("eigen 1.0\nqubit q0\nqubit q1\nH q0\nCNOT q0 q1\nmeasure q0 -> c0\n")

    def test_structs(self):
        self.assert_equivalent("eigen 1.0\nstruct Point {\n    x: int,\n    y: int\n}\nlet p: Point = Point { x: 10, y: 20 }\nlet px: int = p.x\n")

    def test_arrays_and_maps(self):
        self.assert_equivalent("eigen 1.0\nlet a: array<int> = [1, 2, 3]\nlet a0: int = a[0]\nlet m: map<string, int> = {\"hello\": 1, \"world\": 2}\nlet mv: int = m[\"hello\"]\n")

if __name__ == "__main__":
    unittest.main()
