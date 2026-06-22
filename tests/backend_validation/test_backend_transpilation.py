import unittest
import ast as py_ast
from src.lexer import Lexer
from src.parser import Parser
from src.type_checker import TypeChecker
from src.ir_converter import EQIRConverter
from src.qiskit_backend import QiskitBackend
from src.import_resolver import ImportResolver

class TestBackendTranspilationValidation(unittest.TestCase):
    def transpile_code(self, source: str) -> tuple[str, object]:
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize())
        program_ast = parser.parse()
        
        resolver = ImportResolver(".")
        program_ast = resolver.resolve(program_ast)
        
        tc = TypeChecker()
        tc.check(program_ast)
        
        converter = EQIRConverter()
        graph = converter.convert(program_ast)
        
        backend = QiskitBackend()
        qiskit_script, report = backend.transpile(graph, program_ast)
        return qiskit_script, report

    def test_transpiled_valid_python_syntax(self):
        source = """
        eigen 1.0
        struct Person {
            age: int
        }
        func my_func(x: int) -> int {
            return x + 1
        }
        let p: Person = Person { age: 30 }
        let res: int = my_func(p.age)
        print res
        
        qubit q0
        H q0
        cbit c0
        measure q0 -> c0
        print c0
        """
        script, report = self.transpile_code(source)
        
        # 1. Verify output is syntactically valid Python
        try:
            py_ast.parse(script)
        except SyntaxError as e:
            self.fail(f"Transpiled script is not valid Python syntax:\n{script}\nError: {e}")
            
        # 2. Verify no placeholder text in output code
        self.assertNotIn("<CallNode>", script)
        self.assertNotIn("<DotAccessNode>", script)
        
        # 3. Verify report contains warning diagnostics and is structured
        self.assertGreater(report.unsupported_nodes, 0)
        self.assertIn("Qiskit", report.backend_name)
        self.assertTrue(any("not supported" in w for w in report.warnings))

    def test_transpiled_execution_smoke_test(self):
        source = """
        eigen 1.0
        qubit q0
        qubit q1
        H q0
        CNOT q0, q1
        cbit c0
        cbit c1
        measure q0 -> c0
        measure q1 -> c1
        print c0
        print c1
        assert c0 == c1
        """
        script, report = self.transpile_code(source)
        
        import sys
        from types import ModuleType
        
        # Create mock qiskit module
        mock_qiskit = ModuleType("qiskit")
        class MockQuantumCircuit:
            def __init__(self, qubits, cbits):
                self.qubits = qubits
                self.cbits = cbits
            def h(self, idx): pass
            def cx(self, c, t): pass
            def measure(self, q, c): pass
        mock_qiskit.QuantumCircuit = MockQuantumCircuit
        mock_qiskit.transpile = lambda qc, sim: "mock_compiled_circuit"
        
        # Create mock qiskit_aer module
        mock_aer = ModuleType("qiskit_aer")
        class MockAerSimulator:
            def run(self, *args, **kwargs):
                class MockJob:
                    def result(self):
                        class MockResult:
                            def get_counts(self, *args):
                                return {"00": 512, "11": 512}
                        return MockResult()
                return MockJob()
        mock_aer.AerSimulator = MockAerSimulator
        
        orig_qiskit = sys.modules.get("qiskit")
        orig_aer = sys.modules.get("qiskit_aer")
        
        sys.modules["qiskit"] = mock_qiskit
        sys.modules["qiskit_aer"] = mock_aer
        
        try:
            local_vars = {}
            exec(script, {"__builtins__": __builtins__, "np": __import__("numpy")}, local_vars)
            self.assertIn("counts", local_vars)
            self.assertEqual(local_vars["counts"], {"00": 512, "11": 512})
        except Exception as e:
            self.fail(f"Smoke test execution failed: {e}\nScript:\n{script}")
        finally:
            if orig_qiskit is not None:
                sys.modules["qiskit"] = orig_qiskit
            else:
                del sys.modules["qiskit"]
                
            if orig_aer is not None:
                sys.modules["qiskit_aer"] = orig_aer
            else:
                del sys.modules["qiskit_aer"]

if __name__ == "__main__":
    unittest.main()
