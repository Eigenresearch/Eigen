import unittest
from src.lexer import Lexer
from src.parser import Parser
from src.ir_converter import EQIRConverter
from src.optimizer import EQIROptimizer
from src.equivalence import EquivalenceChecker

class TestOptimizerEquivalence(unittest.TestCase):
    def test_complex_circuit_equivalence(self):
        source = """
        eigen 1.0
        qubit q0
        qubit q1
        H q0
        H q0
        CNOT q0, q1
        RY q0, 1.0
        RY q0, -1.0
        """
        c = EQIRConverter().convert(Parser(Lexer(source).tokenize()).parse())
        
        optimizer = EQIROptimizer()
        c_opt = optimizer.optimize(c)
        
        # Original: H, H, CNOT, RY, RY
        # Optimized: CNOT and RY(0.0)
        gates = [n for n in c_opt.nodes.values() if n.type == 'GATE']
        self.assertEqual(len(gates), 2)
        self.assertEqual(gates[0].gate_name, "CNOT")
        
        # Verify equivalence
        checker = EquivalenceChecker()
        self.assertTrue(checker.are_equivalent(c, c_opt))

if __name__ == "__main__":
    unittest.main()
