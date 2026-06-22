import unittest
from src.lexer import Lexer, TokenType

class TestLexer(unittest.TestCase):
    def test_simple_program(self):
        source = """
        eigen 1.0
        # This is a comment
        let x : float = PI / 2.0
        qubit q0
        H q0
        """
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        
        token_types = [t.type for t in tokens]
        expected_types = [
            TokenType.EIGEN, TokenType.FLOAT_LIT,
            TokenType.LET, TokenType.IDENTIFIER, TokenType.COLON, TokenType.FLOAT, TokenType.EQUALS,
            TokenType.PI, TokenType.DIV, TokenType.FLOAT_LIT,
            TokenType.QUBIT, TokenType.IDENTIFIER,
            TokenType.GATE_H, TokenType.IDENTIFIER,
            TokenType.EOF
        ]
        self.assertEqual(token_types, expected_types)

    def test_operators(self):
        source = "-> == = + - * /"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        token_types = [t.type for t in tokens]
        expected = [
            TokenType.ARROW, TokenType.EQ, TokenType.EQUALS,
            TokenType.PLUS, TokenType.MINUS, TokenType.MUL, TokenType.DIV,
            TokenType.EOF
        ]
        self.assertEqual(token_types, expected)

    def test_invalid_character(self):
        lexer = Lexer("qubit q0 @")
        with self.assertRaises(SyntaxError):
            lexer.tokenize()
            
if __name__ == "__main__":
    unittest.main()
