import unittest
from src.lexer import Lexer
from src.parser import Parser
from src.ir_converter import EQIRConverter
from src.optimizer import EQIROptimizer
from src.equivalence import EquivalenceChecker

class TestDeadGateElimination(unittest.TestCase):
    def test_dead_gates(self):
        source = """
        eigen 1.0
        qubit q0
        X q0
        X q0
        H q0
        H q0
        """
        c = EQIRConverter().convert(Parser(Lexer(source).tokenize()).parse())
        
        optimizer = EQIROptimizer()
        c_opt = optimizer.optimize(c)
        
        # Verify optimized graph is empty of gates
        gates = [n for n in c_opt.nodes.values() if n.type == 'GATE']
        self.assertEqual(len(gates), 0)
        
        # Verify equivalence
        checker = EquivalenceChecker()
        self.assertTrue(checker.are_equivalent(c, c_opt))

if __name__ == "__main__":
    unittest.main()
