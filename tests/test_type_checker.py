import unittest
from src.frontend.lexer import Lexer
from src.frontend.parser import Parser
from src.semantic.type_checker import TypeChecker, TypeErrorException

class TestTypeChecker(unittest.TestCase):
    def test_valid_types(self):
        source = """
        eigen 1.0
        qubit q0
        cbit c0
        let val: float = 3.14
        H q0
        measure q0 -> c0
        """
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize())
        ast = parser.parse()
        
        checker = TypeChecker()
        # Should not raise any exception
        checker.check(ast)

    def test_gate_on_non_qubit(self):
        source = """
        eigen 1.0
        cbit c0
        H c0
        """
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize())
        ast = parser.parse()
        
        checker = TypeChecker()
        with self.assertRaises(TypeErrorException):
            checker.check(ast)

    def test_measure_target_mismatch(self):
        source = """
        eigen 1.0
        qubit q0
        let val: float = 1.0
        measure q0 -> val
        """
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize())
        ast = parser.parse()
        
        checker = TypeChecker()
        with self.assertRaises(TypeErrorException):
            checker.check(ast)

    def test_parameter_count_mismatch(self):
        source = """
        eigen 1.0
        qfunc test(qubit q) {
            H q
        }
        qubit q0
        qubit q1
        test(q0, q1)
        """
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize())
        ast = parser.parse()
        
        checker = TypeChecker()
        with self.assertRaises(TypeErrorException):
            checker.check(ast)

if __name__ == "__main__":
    unittest.main()
