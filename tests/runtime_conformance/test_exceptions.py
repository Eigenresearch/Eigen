import unittest
from tests.runtime_conformance.conformance_helper import run_eigen_code

class TestExceptionsConformance(unittest.TestCase):
    def test_try_catch_success(self):
        source = """
        eigen 1.0
        let caught: string = "none"
        try {
            throw "my_error"
        } catch (e) {
            caught = e
        }
        """
        vm = run_eigen_code(source)
        self.assertEqual(vm.lookup_var("caught"), "my_error")

    def test_throw_propagates_up_call_stack(self):
        source = """
        eigen 1.0
        func helper() -> int {
            throw "deep_error"
            return 0
        }
        func middle() -> int {
            return helper()
        }
        let caught: string = "none"
        try {
            let x: int = middle()
        } catch (e) {
            caught = e
        }
        """
        vm = run_eigen_code(source)
        self.assertEqual(vm.lookup_var("caught"), "deep_error")

if __name__ == "__main__":
    unittest.main()
