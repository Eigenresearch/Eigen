import unittest
from tests.runtime_conformance.conformance_helper import run_eigen_code

class TestMeasurementConformance(unittest.TestCase):
    def test_measurement_and_condition(self):
        source = """
        eigen 1.0
        qubit q0
        qubit q1
        cbit c0
        
        H q0
        measure q0 -> c0
        
        if c0 == 1 {
            X q1
        }
        """
        vm = run_eigen_code(source)
        c0_val = vm.lookup_var("c0")
        self.assertIn(c0_val, (0, 1))

if __name__ == "__main__":
    unittest.main()
