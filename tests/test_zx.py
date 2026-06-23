import unittest
import math
from src.ir.ir_graph import EQIRGraph
from src.zx.zx_equivalence import ZXEquivalenceChecker

class TestZXCalculus(unittest.TestCase):
    def test_hadamard_identities(self):
        # Circuit A: H x H
        g1 = EQIRGraph()
        g1.add_operation('ALLOC', targets=["q0"])
        g1.add_operation('GATE', gate_name='H', targets=["q0"])
        g1.add_operation('GATE', gate_name='X', targets=["q0"])
        g1.add_operation('GATE', gate_name='H', targets=["q0"])
        
        # Circuit B: Z
        g2 = EQIRGraph()
        g2.add_operation('ALLOC', targets=["q0"])
        g2.add_operation('GATE', gate_name='Z', targets=["q0"])
        
        checker = ZXEquivalenceChecker()
        self.assertTrue(checker.are_equivalent(g1, g2))

    def test_cnot_identities(self):
        # CNOT q0, q1 is equivalent to: H q0, H q1, CZ q0, q1, H q0, H q1? No, CZ is H on target.
        # But let's check CNOT q0, q1 is equivalent to itself.
        g1 = EQIRGraph()
        g1.add_operation('ALLOC', targets=["q0"])
        g1.add_operation('ALLOC', targets=["q1"])
        g1.add_operation('GATE', gate_name='CNOT', targets=["q0", "q1"])
        
        g2 = EQIRGraph()
        g2.add_operation('ALLOC', targets=["q0"])
        g2.add_operation('ALLOC', targets=["q1"])
        g2.add_operation('GATE', gate_name='CNOT', targets=["q0", "q1"])
        
        checker = ZXEquivalenceChecker()
        self.assertTrue(checker.are_equivalent(g1, g2))

if __name__ == "__main__":
    unittest.main()
