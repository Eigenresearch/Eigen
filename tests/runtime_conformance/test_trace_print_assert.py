import unittest
from tests.runtime_conformance.conformance_helper import run_eigen_code

class TestTracePrintAssertConformance(unittest.TestCase):
    def test_trace_and_print(self):
        source = """
        eigen 1.0
        qubit q0
        H q0
        trace
        print q0
        let x: int = 100
        print x
        assert x == 100
        """
        vm = run_eigen_code(source)
        # Verify assert passed and variable has value
        self.assertEqual(vm.lookup_var("x"), 100)

if __name__ == "__main__":
    unittest.main()
