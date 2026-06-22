import unittest
from src.lexer import Lexer
from src.parser import Parser
from src.ast import ProgramNode, LetNode, VarDeclNode, GateNode, MeasureNode, IfNode

class TestParser(unittest.TestCase):
    def test_parse_declarations(self):
        source = """
        eigen 1.0
        qubit q0
        cbit c1
        let angle: float = PI
        """
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        
        self.assertIsInstance(ast, ProgramNode)
        self.assertEqual(ast.version, 1.0)
        self.assertIsInstance(ast.body[0], VarDeclNode)
        self.assertEqual(ast.body[0].name, "q0")
        self.assertEqual(ast.body[0].type_name, "qubit")
        
        self.assertIsInstance(ast.body[1], VarDeclNode)
        self.assertEqual(ast.body[1].name, "c1")
        self.assertEqual(ast.body[1].type_name, "cbit")
        
        self.assertIsInstance(ast.body[2], LetNode)
        self.assertEqual(ast.body[2].name, "angle")
        self.assertEqual(ast.body[2].type_name, "float")

    def test_parse_gates_and_measures(self):
        source = """
        eigen 1.0
        qubit q0
        qubit q1
        H q0
        CNOT q0, q1
        RX q0, PI / 2
        measure q0 -> c0
        """
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        
        gates = [node for node in ast.body if isinstance(node, GateNode)]
        self.assertEqual(len(gates), 3)
        self.assertEqual(gates[0].gate_name, "H")
        self.assertEqual(gates[0].targets, ["q0"])
        
        self.assertEqual(gates[1].gate_name, "CNOT")
        self.assertEqual(gates[1].targets, ["q0", "q1"])
        
        self.assertEqual(gates[2].gate_name, "RX")
        self.assertEqual(gates[2].targets, ["q0"])
        
        measures = [node for node in ast.body if isinstance(node, MeasureNode)]
        self.assertEqual(len(measures), 1)
        self.assertEqual(measures[0].qubit_name, "q0")
        self.assertEqual(measures[0].cbit_name, "c0")

    def test_parse_if_stmt(self):
        source = """
        eigen 1.0
        if c0 == 1 {
            X q1
        }
        """
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        
        ifs = [node for node in ast.body if isinstance(node, IfNode)]
        self.assertEqual(len(ifs), 1)
        self.assertEqual(ifs[0].op, "==")
        self.assertIsInstance(ifs[0].body[0], GateNode)
        self.assertEqual(ifs[0].body[0].gate_name, "X")
        self.assertEqual(ifs[0].body[0].targets, ["q1"])

if __name__ == "__main__":
    unittest.main()
