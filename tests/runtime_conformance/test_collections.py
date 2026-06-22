import unittest
from tests.runtime_conformance.conformance_helper import run_eigen_code

class TestCollectionsConformance(unittest.TestCase):
    def test_array_operations(self):
        source = """
        eigen 1.0
        let arr: array<int> = [10, 20, 30]
        let item: int = arr[1]
        arr[1] = 99
        let mutated: int = arr[1]
        """
        vm = run_eigen_code(source)
        self.assertEqual(vm.lookup_var("item"), 20)
        self.assertEqual(vm.lookup_var("mutated"), 99)

    def test_map_operations(self):
        source = """
        eigen 1.0
        let m: map<string, int> = {"first": 1, "second": 2}
        let val: int = m["first"]
        m["first"] = 42
        let val_mut: int = m["first"]
        """
        vm = run_eigen_code(source)
        self.assertEqual(vm.lookup_var("val"), 1)
        self.assertEqual(vm.lookup_var("val_mut"), 42)

if __name__ == "__main__":
    unittest.main()
