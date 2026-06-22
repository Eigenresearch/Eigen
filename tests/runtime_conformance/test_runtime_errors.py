import unittest
from tests.runtime_conformance.conformance_helper import run_eigen_code

class TestRuntimeErrorsConformance(unittest.TestCase):
    def test_uncaught_throw_panics(self):
        source = """
        eigen 1.0
        throw "uncaught_fatal_error"
        """
        with self.assertRaises(RuntimeError) as context:
            run_eigen_code(source)
        self.assertIn("Uncaught Exception: uncaught_fatal_error", str(context.exception))

    def test_failed_assertion_panics(self):
        source = """
        eigen 1.0
        let x: int = 42
        assert x == 99
        """
        with self.assertRaises(RuntimeError) as context:
            run_eigen_code(source)
        self.assertIn("Assertion Failed", str(context.exception))

if __name__ == "__main__":
    unittest.main()
