import unittest
from src.lexer import Lexer
from src.parser import Parser
from src.ir_converter import EQIRConverter
from src.optimizer import EQIROptimizer

class TestOptimizer(unittest.TestCase):
    def test_gate_cancellation(self):
        source = """
        eigen 1.0
        qubit q0
        H q0
        H q0
        X q0
        X q0
        """
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize())
        ast = parser.parse()
        
        converter = EQIRConverter()
        graph = converter.convert(ast)
        
        # Unoptimized graph has 4 gate nodes
        gates_unopt = [n for n in graph.nodes.values() if n.type == 'GATE']
        self.assertEqual(len(gates_unopt), 4)
        
        # Run optimizer
        optimizer = EQIROptimizer()
        opt_graph = optimizer.optimize(graph)
        
        # Optimized graph should have 0 gate nodes
        gates_opt = [n for n in opt_graph.nodes.values() if n.type == 'GATE']
        self.assertEqual(len(gates_opt), 0)

    def test_rotation_merging(self):
        source = """
        eigen 1.0
        qubit q0
        RX q0, 1.0
        RX q0, 2.0
        """
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize())
        ast = parser.parse()
        
        converter = EQIRConverter()
        graph = converter.convert(ast)
        
        optimizer = EQIROptimizer()
        opt_graph = optimizer.optimize(graph)
        
        # The two RX gates should merge into one RX gate with angle 3.0
        gates_opt = [n for n in opt_graph.nodes.values() if n.type == 'GATE']
        self.assertEqual(len(gates_opt), 1)
        self.assertEqual(gates_opt[0].gate_name, "RX")
        self.assertAlmostEqual(gates_opt[0].args[0], 3.0)

if __name__ == "__main__":
    unittest.main()
