"""Tests for language features added in this change:

1. Unicode/hex escape sequences in strings (``\\uXXXX``, ``\\xNN``)
2. Block comments (``/* ... */``)
3. Single-quoted strings (``'...'``)
4. ``finally`` block in try/catch
5. ``UnaryOpNode`` replacing the ``BinaryOpNode`` unary hack
"""
import unittest

from src.frontend.lexer import Lexer, TokenType
from src.frontend.parser import Parser
from src.frontend.ast import (
    TryCatchNode, UnaryOpNode, LiteralNode, LetNode,
)


def _parse(source: str):
    tokens = Lexer(source).tokenize()
    return Parser(tokens).parse()


# ---------------------------------------------------------------------------
# 1. Unicode / hex escape sequences
# ---------------------------------------------------------------------------
class TestUnicodeHexEscapes(unittest.TestCase):
    def test_unicode_escape_basic(self):
        toks = Lexer('"\\u00e9"').tokenize()
        self.assertEqual(toks[0].type, TokenType.STRING_LIT)
        self.assertEqual(toks[0].value, "\u00e9")

    def test_unicode_escape_uppercase(self):
        toks = Lexer('"\\u00C9"').tokenize()
        self.assertEqual(toks[0].value, "\u00c9")

    def test_unicode_escape_full_bmp(self):
        toks = Lexer('"\\u4e2d\\u6587"').tokenize()
        self.assertEqual(toks[0].value, "\u4e2d\u6587")

    def test_hex_escape_basic(self):
        toks = Lexer('"\\x41"').tokenize()
        self.assertEqual(toks[0].value, "A")

    def test_hex_escape_two_chars(self):
        toks = Lexer('"\\x1f"').tokenize()
        self.assertEqual(toks[0].value, "\x1f")

    def test_mixed_escapes(self):
        toks = Lexer('"\\x41\\u00e9\\n"').tokenize()
        self.assertEqual(toks[0].value, "A\u00e9\n")

    def test_invalid_unicode_escape_raises(self):
        with self.assertRaises(SyntaxError):
            Lexer('"\\uXYZ"').tokenize()

    def test_invalid_hex_escape_raises(self):
        with self.assertRaises(SyntaxError):
            Lexer('"\\xZZ"').tokenize()

    def test_unicode_escape_regex_path(self):
        toks = Lexer('"\\u00e9"')._tokenize_regex()
        self.assertEqual(toks[0].value, "\u00e9")

    def test_parity_escapes(self):
        src = '"\\u00e9 \\x42 \\n \\t"'
        slow = Lexer(src)._tokenize_slow()
        fast = Lexer(src)._tokenize_regex()
        self.assertEqual([t.value for t in slow], [t.value for t in fast])
        self.assertEqual([t.type for t in slow], [t.type for t in fast])


# ---------------------------------------------------------------------------
# 2. Block comments
# ---------------------------------------------------------------------------
class TestBlockComments(unittest.TestCase):
    def test_single_line_block_comment(self):
        toks = Lexer("x /* comment */ y").tokenize()
        self.assertEqual([t.type for t in toks],
                         [TokenType.IDENTIFIER, TokenType.IDENTIFIER, TokenType.EOF])

    def test_multi_line_block_comment(self):
        toks = Lexer("x /* line one\nline two */ y").tokenize()
        self.assertEqual([t.type for t in toks],
                         [TokenType.IDENTIFIER, TokenType.IDENTIFIER, TokenType.EOF])
        # The identifier ``y`` should be on line 2
        self.assertEqual(toks[1].line, 2)

    def test_block_comment_with_slash_slash_inside(self):
        toks = Lexer("/* // not a line comment */ x").tokenize()
        self.assertEqual([t.type for t in toks],
                         [TokenType.IDENTIFIER, TokenType.EOF])

    def test_block_comment_with_star_inside(self):
        toks = Lexer("/* a ** b */ x").tokenize()
        self.assertEqual([t.type for t in toks],
                         [TokenType.IDENTIFIER, TokenType.EOF])

    def test_unterminated_block_comment_raises(self):
        with self.assertRaises(SyntaxError):
            Lexer("x /* unterminated").tokenize()

    def test_line_comment_still_works(self):
        toks = Lexer("x // comment\ny").tokenize()
        self.assertEqual([t.type for t in toks],
                         [TokenType.IDENTIFIER, TokenType.IDENTIFIER, TokenType.EOF])

    def test_hash_comment_still_works(self):
        toks = Lexer("x # comment\ny").tokenize()
        self.assertEqual([t.type for t in toks],
                         [TokenType.IDENTIFIER, TokenType.IDENTIFIER, TokenType.EOF])

    def test_block_comment_regex_path(self):
        toks = Lexer("x /* c */ y")._tokenize_regex()
        self.assertEqual([t.type for t in toks],
                         [TokenType.IDENTIFIER, TokenType.IDENTIFIER, TokenType.EOF])

    def test_parity_block_comment(self):
        src = "x /* multi\nline */ y\nz"
        slow = Lexer(src)._tokenize_slow()
        fast = Lexer(src)._tokenize_regex()
        self.assertEqual([t.type for t in slow], [t.type for t in fast])
        self.assertEqual([t.value for t in slow], [t.value for t in fast])
        self.assertEqual([(t.line, t.column) for t in slow],
                         [(t.line, t.column) for t in fast])


# ---------------------------------------------------------------------------
# 3. Single-quoted strings
# ---------------------------------------------------------------------------
class TestSingleQuotedStrings(unittest.TestCase):
    def test_basic_single_quote(self):
        toks = Lexer("'hello'").tokenize()
        self.assertEqual(toks[0].type, TokenType.STRING_LIT)
        self.assertEqual(toks[0].value, "hello")

    def test_single_quote_with_escape(self):
        toks = Lexer("'line\\nbreak'").tokenize()
        self.assertEqual(toks[0].value, "line\nbreak")

    def test_single_quote_with_unicode_escape(self):
        toks = Lexer("'\\u00e9'").tokenize()
        self.assertEqual(toks[0].value, "\u00e9")

    def test_single_quote_with_hex_escape(self):
        toks = Lexer("'\\x41'").tokenize()
        self.assertEqual(toks[0].value, "A")

    def test_single_quote_interpolation(self):
        toks = Lexer("'val=${x}'").tokenize()
        self.assertIn("\x00x\x00", toks[0].value)

    def test_single_quote_can_contain_double_quote(self):
        toks = Lexer("'say \"hi\"'").tokenize()
        self.assertEqual(toks[0].value, 'say "hi"')

    def test_double_quote_can_contain_single_quote(self):
        toks = Lexer('"it\'s ok"').tokenize()
        self.assertEqual(toks[0].value, "it's ok")

    def test_single_quote_unterminated_raises(self):
        with self.assertRaises(SyntaxError):
            Lexer("'unterminated").tokenize()

    def test_parity_single_quote(self):
        src = "'hi' 'there'"
        slow = Lexer(src)._tokenize_slow()
        fast = Lexer(src)._tokenize_regex()
        self.assertEqual([t.value for t in slow], [t.value for t in fast])
        self.assertEqual([(t.line, t.column) for t in slow],
                         [(t.line, t.column) for t in fast])


# ---------------------------------------------------------------------------
# 4. Finally block in try/catch
# ---------------------------------------------------------------------------
class TestTryCatchFinally(unittest.TestCase):
    def test_finally_block_parsed(self):
        src = (
            "eigen 1.0\n"
            "func f() -> int {\n"
            "  try {\n"
            "    return 1\n"
            "  } catch (e) {\n"
            "    return 2\n"
            "  } finally {\n"
            "    print \"cleanup\"\n"
            "  }\n"
            "}\n"
        )
        ast = _parse(src)
        func = [n for n in ast.body if isinstance(n, LetNode) or hasattr(n, 'body')][0]
        # find the TryCatchNode in the function body
        tc = [s for s in func.body if isinstance(s, TryCatchNode)][0]
        self.assertEqual(len(tc.try_body), 1)
        self.assertEqual(tc.catch_var, "e")
        self.assertEqual(len(tc.catch_body), 1)
        self.assertEqual(len(tc.finally_body), 1)

    def test_finally_block_optional(self):
        src = (
            "eigen 1.0\n"
            "func f() -> int {\n"
            "  try {\n"
            "    return 1\n"
            "  } catch (e) {\n"
            "    return 2\n"
            "  }\n"
            "}\n"
        )
        ast = _parse(src)
        func = [n for n in ast.body if hasattr(n, 'body')][0]
        tc = [s for s in func.body if isinstance(s, TryCatchNode)][0]
        self.assertEqual(tc.finally_body, [])

    def test_catch_with_type_annotation(self):
        src = (
            "eigen 1.0\n"
            "func f() -> int {\n"
            "  try {\n"
            "    return 1\n"
            "  } catch (e: TypeError) {\n"
            "    return 2\n"
            "  }\n"
            "}\n"
        )
        ast = _parse(src)
        func = [n for n in ast.body if hasattr(n, 'body')][0]
        tc = [s for s in func.body if isinstance(s, TryCatchNode)][0]
        self.assertEqual(tc.catch_var, "e")
        self.assertEqual(tc.catch_type, "TypeError")

    def test_finally_without_catch_var(self):
        src = (
            "eigen 1.0\n"
            "func f() -> int {\n"
            "  try {\n"
            "    return 1\n"
            "  } catch {\n"
            "    return 2\n"
            "  } finally {\n"
            "    return 3\n"
            "  }\n"
            "}\n"
        )
        ast = _parse(src)
        func = [n for n in ast.body if hasattr(n, 'body')][0]
        tc = [s for s in func.body if isinstance(s, TryCatchNode)][0]
        self.assertIsNone(tc.catch_var)
        self.assertEqual(len(tc.finally_body), 1)

    def test_finally_keyword_is_reserved(self):
        toks = Lexer("finally").tokenize()
        self.assertEqual(toks[0].type, TokenType.FINALLY)


# ---------------------------------------------------------------------------
# 5. UnaryOpNode
# ---------------------------------------------------------------------------
class TestUnaryOpNode(unittest.TestCase):
    def test_not_produces_unary_op_node(self):
        src = "eigen 1.0\nlet x: bool = not true"
        ast = _parse(src)
        let = [n for n in ast.body if isinstance(n, LetNode)][0]
        self.assertIsInstance(let.value, UnaryOpNode)
        self.assertEqual(let.value.op, "not")
        self.assertIsInstance(let.value.operand, LiteralNode)

    def test_tilde_produces_unary_op_node(self):
        src = "eigen 1.0\nlet x: int = ~5"
        ast = _parse(src)
        let = [n for n in ast.body if isinstance(n, LetNode)][0]
        self.assertIsInstance(let.value, UnaryOpNode)
        self.assertEqual(let.value.op, "~")

    def test_unary_minus_produces_unary_op_node(self):
        src = "eigen 1.0\nlet x: int = -5"
        ast = _parse(src)
        let = [n for n in ast.body if isinstance(n, LetNode)][0]
        self.assertIsInstance(let.value, UnaryOpNode)
        self.assertEqual(let.value.op, "-")

    def test_unary_minus_on_variable(self):
        src = "eigen 1.0\nlet x: int = -y"
        ast = _parse(src)
        let = [n for n in ast.body if isinstance(n, LetNode)][0]
        self.assertIsInstance(let.value, UnaryOpNode)
        self.assertEqual(let.value.op, "-")

    def test_double_negation(self):
        src = "eigen 1.0\nlet x: bool = not not true"
        ast = _parse(src)
        let = [n for n in ast.body if isinstance(n, LetNode)][0]
        self.assertIsInstance(let.value, UnaryOpNode)
        self.assertEqual(let.value.op, "not")
        self.assertIsInstance(let.value.operand, UnaryOpNode)
        self.assertEqual(let.value.operand.op, "not")

    def test_unary_node_repr(self):
        node = UnaryOpNode("-", LiteralNode(5, "int"))
        self.assertIn("-", repr(node))

    def test_unary_node_to_source(self):
        node = UnaryOpNode("not", LiteralNode(True, "bool"))
        self.assertEqual(node.to_source(), "not True")
        node2 = UnaryOpNode("-", LiteralNode(5, "int"))
        self.assertEqual(node2.to_source(), "-5")


if __name__ == "__main__":
    unittest.main()
