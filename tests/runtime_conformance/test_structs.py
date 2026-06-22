import unittest
from tests.runtime_conformance.conformance_helper import run_eigen_code

class TestStructsConformance(unittest.TestCase):
    def test_struct_operations(self):
        source = """
        eigen 1.0
        struct Person {
            age: int,
            score: float
        }
        let p: Person = Person { age: 25, score: 88.5 }
        let age1: int = p.age
        p.age = 26
        let age2: int = p.age
        """
        vm = run_eigen_code(source)
        self.assertEqual(vm.lookup_var("age1"), 25)
        self.assertEqual(vm.lookup_var("age2"), 26)

if __name__ == "__main__":
    unittest.main()
