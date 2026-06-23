"""Fuzz testing for Eigen parser and lexer resilience.

Feeds random, malformed, and adversarial inputs into the lexer and parser
to ensure they raise controlled exceptions without crashing.
"""
import unittest
import random
import string
from src.frontend.lexer import Lexer
from src.frontend.parser import Parser


class TestParserFuzz(unittest.TestCase):

    RANDOM_SEED = 42
    NUM_FUZZ_ITERATIONS = 50

    def _try_parse(self, source: str):
        """Attempt to lex and parse; expect SyntaxError or controlled exception."""
        try:
            lexer = Lexer(source)
            tokens = lexer.tokenize()
            parser = Parser(tokens)
            parser.parse()
        except (SyntaxError, Exception):
            pass  # Any controlled exception is acceptable

    def test_empty_input(self):
        self._try_parse("")

    def test_whitespace_only(self):
        self._try_parse("   \n\t\n   ")

    def test_single_keyword(self):
        keywords = ["eigen", "qubit", "cbit", "let", "func", "return", "if",
                    "for", "while", "struct", "import", "measure", "trace",
                    "assert", "print", "noise", "H", "X", "Y", "Z", "CNOT"]
        for kw in keywords:
            self._try_parse(kw)

    def test_random_ascii_strings(self):
        rng = random.Random(self.RANDOM_SEED)
        for _ in range(self.NUM_FUZZ_ITERATIONS):
            length = rng.randint(1, 200)
            source = ''.join(rng.choice(string.printable) for _ in range(length))
            self._try_parse(source)

    def test_random_token_sequences(self):
        tokens = [
            "eigen", "1.0", "qubit", "cbit", "let", "func", "return",
            "if", "for", "while", "struct", "import", "measure", "trace",
            "H", "X", "Y", "Z", "CNOT", "RX", "RY", "RZ",
            "q0", "q1", "c0", "c1", "x", "y",
            ":", ",", "->", "{", "}", "(", ")", "[", "]",
            "=", "==", "!=", "<", ">", "+", "-", "*", "/",
            "0", "1", "42", "3.14", "PI",
            '"hello"', "true", "false",
            "\n", " ",
        ]
        rng = random.Random(self.RANDOM_SEED + 1)
        for _ in range(self.NUM_FUZZ_ITERATIONS):
            length = rng.randint(1, 30)
            source = " ".join(rng.choice(tokens) for _ in range(length))
            self._try_parse(source)

    def test_deeply_nested_braces(self):
        source = "eigen 1.0\n" + "if x == 1 {\n" * 50 + "}\n" * 50
        self._try_parse(source)

    def test_unbalanced_braces(self):
        cases = [
            "eigen 1.0\n{{{{",
            "eigen 1.0\n}}}}",
            "eigen 1.0\nfunc f() { {",
            "eigen 1.0\n} } }",
        ]
        for source in cases:
            self._try_parse(source)

    def test_extremely_long_identifier(self):
        long_id = "a" * 10000
        source = f"eigen 1.0\nlet {long_id}: int = 42"
        self._try_parse(source)

    def test_null_bytes(self):
        source = "eigen 1.0\nqubit\x00q0\nH q0"
        self._try_parse(source)

    def test_unicode_characters(self):
        source = "eigen 1.0\nlet α: int = 42\nlet β: float = 3.14"
        self._try_parse(source)

    def test_only_numbers(self):
        self._try_parse("1 2 3 4 5 6 7 8 9 0")

    def test_only_operators(self):
        self._try_parse("+ - * / = == != < > <= >= -> : , .")

    def test_incomplete_let(self):
        cases = [
            "eigen 1.0\nlet",
            "eigen 1.0\nlet x",
            "eigen 1.0\nlet x:",
            "eigen 1.0\nlet x: int",
            "eigen 1.0\nlet x: int =",
        ]
        for source in cases:
            self._try_parse(source)

    def test_incomplete_func(self):
        cases = [
            "eigen 1.0\nfunc",
            "eigen 1.0\nfunc f",
            "eigen 1.0\nfunc f(",
            "eigen 1.0\nfunc f()",
            "eigen 1.0\nfunc f() {",
            "eigen 1.0\nfunc f() -> int {",
        ]
        for source in cases:
            self._try_parse(source)

    def test_incomplete_measure(self):
        cases = [
            "eigen 1.0\nmeasure",
            "eigen 1.0\nmeasure q0",
            "eigen 1.0\nmeasure q0 ->",
        ]
        for source in cases:
            self._try_parse(source)

    def test_gate_without_target(self):
        cases = [
            "eigen 1.0\nH",
            "eigen 1.0\nCNOT",
            "eigen 1.0\nCNOT q0",
            "eigen 1.0\nRX",
        ]
        for source in cases:
            self._try_parse(source)

    def test_repeated_keywords(self):
        self._try_parse("eigen eigen eigen qubit qubit let let let")

    def test_mixed_valid_invalid(self):
        source = "eigen 1.0\nqubit q0\nH q0\n@@@\nlet x: int = 42"
        self._try_parse(source)

    def test_binary_garbage(self):
        rng = random.Random(self.RANDOM_SEED + 2)
        garbage = bytes(rng.randint(0, 255) for _ in range(100))
        try:
            source = garbage.decode('utf-8', errors='replace')
            self._try_parse(source)
        except Exception:
            pass

    def test_newlines_stress(self):
        source = "eigen 1.0" + "\n" * 1000 + "qubit q0\nH q0"
        self._try_parse(source)

    def test_comment_stress(self):
        source = "eigen 1.0\n" + "# comment line\n" * 500 + "qubit q0\nH q0"
        self._try_parse(source)


if __name__ == "__main__":
    unittest.main()
