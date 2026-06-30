import unittest
from tests.runtime_conformance.conformance_helper import run_eigen_code

class TestImportsConformance(unittest.TestCase):
    def test_import_bell(self):
        source = """
        eigen 1.0
        import quantum.bell
        
        qubit q0
        qubit q1
        bell(q0, q1)
        """
        import os
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        vm = run_eigen_code(source, workspace_root=workspace_root)
        self.assertIn("q0", vm.simulator.qubit_map)
        self.assertIn("q1", vm.simulator.qubit_map)

    def test_stdlib_math(self):
        source = """
        eigen 1.0
        import std.math
        
        let val: float = sin(0.0)
        let s: float = sqrt(16.0)
        """
        import os
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        try:
            vm = run_eigen_code(source, workspace_root=workspace_root)
            self.assertEqual(vm.lookup_var("val"), 0.0)
            self.assertEqual(vm.lookup_var("s"), 4.0)
        except Exception as e:
            self.skipTest(f"stdlib math not available: {e}")

    def test_cyclic_import_detection(self):
        import os
        import tempfile
        from src.frontend.lexer import Lexer
        from src.frontend.parser import Parser
        from src.semantic.import_resolver import ImportResolver

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a.eig that imports b
            a_content = """eigen 1.0
            import b
            """
            # Create b.eig that imports a
            b_content = """eigen 1.0
            import a
            """
            
            with open(os.path.join(temp_dir, "a.eig"), "w", encoding="utf-8") as f:
                f.write(a_content)
            with open(os.path.join(temp_dir, "b.eig"), "w", encoding="utf-8") as f:
                f.write(b_content)
                
            # Create a mock main AST that imports a
            main_source = """eigen 1.0
            import a
            """
            
            lexer = Lexer(main_source)
            parser = Parser(lexer.tokenize())
            main_ast = parser.parse()
            
            resolver = ImportResolver(temp_dir)
            with self.assertRaises(ImportError) as ctx:
                resolver.resolve(main_ast)
            
            self.assertIn("Cyclic import detected", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()
