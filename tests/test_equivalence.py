import unittest
from src.lexer import Lexer
from src.parser import Parser
from src.ir_converter import EQIRConverter
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

if __name__ == "__main__":
    unittest.main()
