import unittest
from src.lexer import Lexer
from src.parser import Parser
from src.ir_converter import EQIRConverter
from src.optimizer import EQIROptimizer
from src.equivalence import EquivalenceChecker

class TestGateFusion(unittest.TestCase):
    def test_fusion(self):
        source = """
        eigen 1.0
        qubit q0
        RX q0, 0.5
        RX q0, 1.2
        RX q0, 0.3
        """
        c = EQIRConverter().convert(Parser(Lexer(source).tokenize()).parse())
        
        optimizer = EQIROptimizer()
        c_opt = optimizer.optimize(c)
        
        # Verify optimized has exactly 1 gate
        gates = [n for n in c_opt.nodes.values() if n.type == 'GATE']
        self.assertEqual(len(gates), 1)
        self.assertEqual(gates[0].gate_name, "RX")
        self.assertAlmostEqual(gates[0].args[0], 2.0)
        
        # Verify equivalence
        checker = EquivalenceChecker()
        self.assertTrue(checker.are_equivalent(c, c_opt))

if __name__ == "__main__":
    unittest.main()
