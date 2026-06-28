import unittest
from src.frontend.lexer import Lexer
from src.frontend.parser import Parser
from src.ir.ir_converter import EQIRConverter
from src.ir.ir_graph import EQIRGraph
from src.equivalence import EquivalenceChecker

class TestEquivalence(unittest.TestCase):
    def test_equivalent_circuits(self):
        # Circuit 1: H; H on q0 (equivalent to identity / noop)
        source1 = """
        eigen 1.0
        qubit q0
        H q0
        H q0
        """
        # Circuit 2: Empty circuit (identity)
        source2 = """
        eigen 1.0
        qubit q0
        """
        
        c1 = EQIRConverter().convert(Parser(Lexer(source1).tokenize()).parse())
        c2 = EQIRConverter().convert(Parser(Lexer(source2).tokenize()).parse())
        
        checker = EquivalenceChecker()
        self.assertTrue(checker.are_equivalent(c1, c2))

    def test_non_equivalent_circuits(self):
        # Circuit 1: H on q0
        source1 = """
        eigen 1.0
        qubit q0
        H q0
        """
        # Circuit 2: X on q0
        source2 = """
        eigen 1.0
        qubit q0
        X q0
        """
        
        c1 = EQIRConverter().convert(Parser(Lexer(source1).tokenize()).parse())
        c2 = EQIRConverter().convert(Parser(Lexer(source2).tokenize()).parse())
        
        checker = EquivalenceChecker()
        self.assertFalse(checker.are_equivalent(c1, c2))

    def test_rotation_equivalence(self):
        # RX(1.0) followed by RX(2.0) is equivalent to RX(3.0)
        source1 = """
        eigen 1.0
        qubit q0
        RX q0, 1.0
        RX q0, 2.0
        """
        source2 = """
        eigen 1.0
        qubit q0
        RX q0, 3.0
        """
        
        c1 = EQIRConverter().convert(Parser(Lexer(source1).tokenize()).parse())
        c2 = EQIRConverter().convert(Parser(Lexer(source2).tokenize()).parse())
        
        checker = EquivalenceChecker()
        self.assertTrue(checker.are_equivalent(c1, c2))

    def test_direct_negative_cases(self):
        checker = EquivalenceChecker()
        
        # H vs X
        g1 = EQIRGraph()
        g1.add_operation('ALLOC', targets=["q0"])
        g1.add_operation('GATE', gate_name='H', targets=["q0"])
        g2 = EQIRGraph()
        g2.add_operation('ALLOC', targets=["q0"])
        g2.add_operation('GATE', gate_name='X', targets=["q0"])
        self.assertFalse(checker.are_equivalent(g1, g2))
        
        # CNOT vs SWAP
        g3 = EQIRGraph()
        g3.add_operation('ALLOC', targets=["q0"])
        g3.add_operation('ALLOC', targets=["q1"])
        g3.add_operation('GATE', gate_name='CNOT', targets=["q0", "q1"])
        g4 = EQIRGraph()
        g4.add_operation('ALLOC', targets=["q0"])
        g4.add_operation('ALLOC', targets=["q1"])
        g4.add_operation('GATE', gate_name='SWAP', targets=["q0", "q1"])
        self.assertFalse(checker.are_equivalent(g3, g4))

if __name__ == "__main__":
    unittest.main()
