import unittest
from tests.runtime_conformance.conformance_helper import run_eigen_code

class TestRecursionConformance(unittest.TestCase):
    def test_factorial(self):
        source = """
        eigen 1.0
        func fact(n: int) -> int {
            if n == 0 {
                return 1
            }
            return n * fact(n - 1)
        }
        let res: int = fact(5)
        """
        vm = run_eigen_code(source)
        self.assertEqual(vm.lookup_var("res"), 120)

    def test_fibonacci(self):
        source = """
        eigen 1.0
        func fib(n: int) -> int {
            if n <= 1 {
                return n
            }
            return fib(n - 1) + fib(n - 2)
        }
        let res: int = fib(6)
        """
        vm = run_eigen_code(source)
        self.assertEqual(vm.lookup_var("res"), 8)

if __name__ == "__main__":
    unittest.main()
