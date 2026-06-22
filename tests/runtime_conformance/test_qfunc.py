import unittest
from tests.runtime_conformance.conformance_helper import run_eigen_code

class TestQFuncConformance(unittest.TestCase):
    def test_qfunc_call(self):
        source = """
        eigen 1.0
        qfunc prepare(qubit q) {
            H q
            return
        }
        qubit q0
        prepare(q0)
        """
        vm = run_eigen_code(source)
        self.assertIn("q0", vm.simulator.qubit_map)

if __name__ == "__main__":
    unittest.main()
