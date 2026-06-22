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
        vm = run_eigen_code(source, workspace_root="d:\\Nuras-7")
        self.assertIn("q0", vm.simulator.qubit_map)
        self.assertIn("q1", vm.simulator.qubit_map)

if __name__ == "__main__":
    unittest.main()
