import unittest
import os
import sys

# Add workspace root to path
sys.path.append(os.path.abspath('.'))

from src.backend.vm import EigenVM
from src.semantic.import_resolver import ImportResolver

class TestSecurityAudit(unittest.TestCase):
    def setUp(self):
        self.vm = EigenVM()

    def test_op_pow_resource_guard(self):
        # Test large exponent (already handled but let's check)
        self.vm.operand_stack = [2, 1000001]
        try:
            self.vm.op_pow(None)
        except RuntimeError as e:
            self.assertIn("OverflowError", str(e))
        
        # Test large base + moderate exponent that still explodes
        self.vm.operand_stack = [10**100, 100000] # (10^100)^100000 = 10^10,000,000
        # This currently is NOT guarded against base.
        print("Testing large base pow...")
        self.vm.op_pow(None)
        res = self.vm.operand_stack.pop()
        print(f"Result bit length: {res.bit_length()}")

    def test_op_shl_resource_guard(self):
        # Test large shift
        self.vm.operand_stack = [1, 100000000]
        print("Testing large shl...")
        try:
            self.vm.op_shl(None)
            res = self.vm.operand_stack.pop()
            print(f"Result bit length: {res.bit_length()}")
        except Exception as e:
            print(f"op_shl failed with: {e}")

    def test_finite_gate_angles(self):
        # Test NaN angle
        self.vm.operand_stack = [float('nan')]
        try:
            self.vm.op_q_gate(("RX", ["q0"]))
        except ValueError as e:
            self.assertIn("must be finite", str(e))

        # Test Inf angle
        self.vm.operand_stack = [float('inf')]
        try:
            self.vm.op_q_gate(("RX", ["q0"]))
        except ValueError as e:
            self.assertIn("must be finite", str(e))

    def test_import_resolver_traversal(self):
        resolver = ImportResolver(workspace_root=".", stdlib_root="./stdlib")
        # Test traversal segments
        with self.assertRaises(ImportError):
            resolver.resolve_module_file("foo...bar")
        
        # Test absolute path in segments (if not caught by '.')
        # Actually it's already caught by segments check.

    def test_import_resolver_recursion_limit(self):
        # Test deep recursion
        # This is harder to test without actual files, but we can mock it.
        pass

if __name__ == '__main__':
    unittest.main()
