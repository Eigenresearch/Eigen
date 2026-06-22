import unittest
from tests.runtime_conformance.conformance_helper import run_eigen_code

class TestHybridConformance(unittest.TestCase):
    def test_hybrid_quantum_loop(self):
        source = """
        eigen 1.0
        let targets: array<int> = [1, 2]
        let match_count: int = 0
        
        qubit q0
        cbit c0
        
        for t in targets {
            H q0
            measure q0 -> c0
            if c0 == 1 {
                match_count += 1
            }
        }
        """
        vm = run_eigen_code(source)
        match_count = vm.lookup_var("match_count")
        self.assertGreaterEqual(match_count, 0)
        self.assertLessEqual(match_count, 2)

if __name__ == "__main__":
    unittest.main()
