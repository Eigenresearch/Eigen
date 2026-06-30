import unittest
import os
import shutil
from src.ir.ir_graph import EQIRGraph
from src.compiler import to_eqir
from src.backend.qasm3_exporter import Qasm3Exporter
from src.backend.quil_exporter import QuilExporter

class TestExporters(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.workspace_root = os.path.dirname(self.test_dir)
        self.source_file = os.path.join(self.test_dir, "temp_export_test.eig")
        
        with open(self.source_file, "w", encoding="utf-8") as f:
            f.write("""eigen 2.5
            qubit q0
            qubit q1
            qubit q2
            cbit c0
            H q0
            CNOT q0, q1
            CCX q0, q1, q2
            CP q0, q1, 0.5
            measure q2 -> c0
            """)

    def tearDown(self):
        if os.path.exists(self.source_file):
            os.remove(self.source_file)
            
    def test_qasm3_export(self):
        graph, ast = to_eqir(self.source_file, self.workspace_root)
        exporter = Qasm3Exporter()
        qasm_code = exporter.export(graph)
        
        self.assertIn("OPENQASM 3.0;", qasm_code)
        self.assertIn('include "stdgates.inc";', qasm_code)
        self.assertIn("qubit[3] q;", qasm_code)
        self.assertIn("h q[", qasm_code)
        self.assertIn("cx q[", qasm_code)
        self.assertIn("ccx q[", qasm_code)
        self.assertIn("ctrl @ phase(0.5)", qasm_code)
        self.assertIn("measure q[", qasm_code)

    def test_quil_export(self):
        graph, ast = to_eqir(self.source_file, self.workspace_root)
        exporter = QuilExporter()
        quil_code = exporter.export(graph)
        
        self.assertIn("DECLARE ro BIT[1]", quil_code)
        self.assertIn("H ", quil_code)
        self.assertIn("CNOT ", quil_code)
        self.assertIn("CCNOT ", quil_code)
        self.assertIn("CPHASE(0.5)", quil_code)
        self.assertIn("MEASURE ", quil_code)

    def test_cli_qasm_export(self):
        import subprocess
        python_exe = os.path.join(self.workspace_root, ".venv", "Scripts", "python.exe")
        main_py = os.path.join(self.workspace_root, "src", "main.py")
        
        qasm_file = os.path.join(self.test_dir, "temp_export_test.qasm")
        if os.path.exists(qasm_file):
            os.remove(qasm_file)
            
        result = subprocess.run([
            python_exe, main_py, "build", self.source_file, "--qasm"
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0, f"CLI build failed: {result.stderr}")
        self.assertTrue(os.path.exists(qasm_file))
        
        with open(qasm_file, "r", encoding="utf-8") as f:
            content = f.read()
            
        self.assertIn("OPENQASM 3.0;", content)
        os.remove(qasm_file)

    def test_cli_quil_export(self):
        import subprocess
        python_exe = os.path.join(self.workspace_root, ".venv", "Scripts", "python.exe")
        main_py = os.path.join(self.workspace_root, "src", "main.py")
        
        quil_file = os.path.join(self.test_dir, "temp_export_test.quil")
        if os.path.exists(quil_file):
            os.remove(quil_file)
            
        result = subprocess.run([
            python_exe, main_py, "build", self.source_file, "--quil"
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0, f"CLI build failed: {result.stderr}")
        self.assertTrue(os.path.exists(quil_file))
        
        with open(quil_file, "r", encoding="utf-8") as f:
            content = f.read()
            
        self.assertIn("DECLARE ro BIT[1]", content)
        os.remove(quil_file)

if __name__ == "__main__":
    unittest.main()
