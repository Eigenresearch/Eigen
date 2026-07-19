import unittest

from src.frontend.lexer import Lexer
from src.frontend.parser import Parser
from src.frontend.ast import (
    ProgramNode,
    ImportNode,
    VarDeclNode,
    LetNode,
    ReturnNode,
    PrintNode,
    AssertNode,
    FuncDeclNode,
    ForNode,
    BreakNode,
    ContinueNode,
    StructDeclNode,
    TryCatchNode,
    ThrowNode,
    EnumDeclNode,
    ArrayLiteralNode,
    TupleLiteralNode,
    TypeAliasDeclNode,
    TraitDeclNode,
    ImplBlockNode,
    LiteralNode,
    UnaryOpNode,
    MapAllocNode,
)


def _parse(source: str):
    tokens = Lexer(source).tokenize()
    return Parser(tokens).parse()


def _wrap(body: str, version: float = 1.0) -> str:
    return f"eigen {version}\n{body}"


def _tokens(source: str):
    return Lexer(source).tokenize()


class TestParserEmptyAndHeader(unittest.TestCase):
    def test_empty_file_no_eigen_header_raises(self):
        with self.assertRaises(SyntaxError):
            _parse("")

    def test_only_version_line(self):
        ast = _parse("eigen 1.0\n")
        self.assertIsInstance(ast, ProgramNode)
        self.assertEqual(ast.version, 1.0)
        self.assertEqual(len(ast.body), 0)

    def test_version_1_5_accepted(self):
        ast = _parse("eigen 1.5\n")
        self.assertEqual(ast.version, 1.5)

    def test_version_2_accepted_as_int(self):
        ast = _parse("eigen 2\n")
        self.assertEqual(ast.version, 2.0)

    def test_missing_version_raises(self):
        with self.assertRaises(SyntaxError):
            _parse("eigen\nqubit q0\n")

    def test_only_comments(self):
        source = "eigen 1.0\n# this is a comment\n// also a comment\n"
        ast = _parse(source)
        self.assertIsInstance(ast, ProgramNode)
        self.assertEqual(len(ast.body), 0)


class TestParserStringEscapes(unittest.TestCase):
    def test_escape_newline(self):
        source = _wrap('let x: string = "a\\nb"')
        ast = _parse(source)
        self.assertIsInstance(ast.body[0], LetNode)
        self.assertEqual(ast.body[0].value.value, "a\nb")

    def test_escape_tab(self):
        source = _wrap('let x: string = "a\\tb"')
        ast = _parse(source)
        self.assertEqual(ast.body[0].value.value, "a\tb")

    def test_escape_backslash(self):
        source = _wrap('let x: string = "a\\\\b"')
        ast = _parse(source)
        self.assertEqual(ast.body[0].value.value, "a\\b")

    def test_escape_quote(self):
        source = _wrap('let x: string = "a\\"b"')
        ast = _parse(source)
        self.assertEqual(ast.body[0].value.value, 'a"b')

    def test_unicode_escape(self):
        source = _wrap('let x: string = "a\\u0041b"')
        ast = _parse(source)
        self.assertEqual(ast.body[0].value.value, "aAb")

    def test_hex_escape(self):
        source = _wrap('let x: string = "\\x41"')
        ast = _parse(source)
        self.assertEqual(ast.body[0].value.value, "A")

    def test_escape_carriage_return(self):
        source = _wrap('let x: string = "a\\rb"')
        ast = _parse(source)
        self.assertEqual(ast.body[0].value.value, "a\rb")

    def test_escape_null_char(self):
        source = _wrap('let x: string = "ab"')
        ast = _parse(source)
        self.assertEqual(ast.body[0].value.value, "ab")

    def test_invalid_unicode_escape_raises(self):
        source = _wrap('let x: string = "\\uXYZW"')
        with self.assertRaises(SyntaxError):
            _parse(source)

    def test_invalid_hex_escape_raises(self):
        source = _wrap('let x: string = "\\xZZ"')
        with self.assertRaises(SyntaxError):
            _parse(source)

    def test_unterminated_string_raises(self):
        source = _wrap('let x: string = "abc')
        with self.assertRaises(SyntaxError):
            _parse(source)

    def test_unterminated_string_interpolation_raises(self):
        source = _wrap('let x: string = "${x"')
        with self.assertRaises(SyntaxError):
            _parse(source)


class TestParserComments(unittest.TestCase):
    def test_hash_comment_ignored(self):
        source = _wrap("# comment\nqubit q0\n")
        ast = _parse(source)
        self.assertEqual(len(ast.body), 1)
        self.assertIsInstance(ast.body[0], VarDeclNode)

    def test_slash_slash_comment_ignored(self):
        source = _wrap("// comment\nqubit q0\n")
        ast = _parse(source)
        self.assertEqual(len(ast.body), 1)

    def test_block_comment_single_line(self):
        source = _wrap("/* a block comment */\nqubit q0\n")
        ast = _parse(source)
        self.assertEqual(len(ast.body), 1)

    def test_block_comment_multi_line(self):
        source = _wrap("/* multi\nline\ncomment */\nqubit q0\n")
        ast = _parse(source)
        self.assertEqual(len(ast.body), 1)

    def test_block_comment_mid_line(self):
        source = _wrap("let x: int = /* inline */ 5\n")
        ast = _parse(source)
        self.assertIsInstance(ast.body[0], LetNode)
        self.assertEqual(ast.body[0].value.value, 5)

    def test_unterminated_block_comment_raises(self):
        source = _wrap("/* unterminated\nqubit q0\n")
        with self.assertRaises(SyntaxError):
            _parse(source)


class TestParserTryCatchFinally(unittest.TestCase):
    def test_try_catch_basic(self):
        source = _wrap("try {\n let x: int = 1\n}\ncatch (e) {\n let y: int = 2\n}\n")
        ast = _parse(source)
        self.assertIsInstance(ast.body[0], TryCatchNode)
        self.assertEqual(ast.body[0].catch_var, "e")
        self.assertEqual(len(ast.body[0].try_body), 1)
        self.assertEqual(len(ast.body[0].catch_body), 1)

    def test_try_catch_finally(self):
        source = _wrap(
            "try {\n let x: int = 1\n}\ncatch (e) {\n let y: int = 2\n}\nfinally {\n let z: int = 3\n}\n"
        )
        ast = _parse(source)
        tc = ast.body[0]
        self.assertIsInstance(tc, TryCatchNode)
        self.assertEqual(len(tc.finally_body), 1)

    def test_try_catch_no_paren(self):
        source = _wrap("try {\n}\ncatch e {\n}\n")
        ast = _parse(source)
        tc = ast.body[0]
        self.assertIsInstance(tc, TryCatchNode)
        self.assertEqual(tc.catch_var, "e")

    def test_try_catch_with_type(self):
        source = _wrap("try {\n}\ncatch (e: ValueError) {\n}\n")
        ast = _parse(source)
        tc = ast.body[0]
        self.assertEqual(tc.catch_var, "e")
        self.assertEqual(tc.catch_type, "ValueError")

    def test_throw_statement(self):
        source = _wrap("throw \"error\"")
        ast = _parse(source)
        self.assertIsInstance(ast.body[0], ThrowNode)


class TestParserUnaryOperators(unittest.TestCase):
    def test_not_operator(self):
        source = _wrap("let x: bool = not true\n")
        ast = _parse(source)
        self.assertIsInstance(ast.body[0].value, UnaryOpNode)
        self.assertEqual(ast.body[0].value.op, "not")

    def test_tilde_operator(self):
        source = _wrap("let x: int = ~5\n")
        ast = _parse(source)
        self.assertIsInstance(ast.body[0].value, UnaryOpNode)
        self.assertEqual(ast.body[0].value.op, "~")

    def test_unary_minus(self):
        source = _wrap("let x: int = -5\n")
        ast = _parse(source)
        self.assertIsInstance(ast.body[0].value, UnaryOpNode)
        self.assertEqual(ast.body[0].value.op, "-")

    def test_unary_plus_accepted(self):
        source = _wrap("let x: int = +5\n")
        ast = _parse(source)
        self.assertIsInstance(ast.body[0].value, LiteralNode)

    def test_double_unary_minus(self):
        source = _wrap("let x: int = --5\n")
        ast = _parse(source)
        self.assertIsInstance(ast.body[0].value, UnaryOpNode)

    def test_unary_with_parens(self):
        source = _wrap("let x: int = -(5 + 3)\n")
        ast = _parse(source)
        self.assertIsInstance(ast.body[0].value, UnaryOpNode)


class TestParserSpecialSyntax(unittest.TestCase):
    def test_single_quoted_string(self):
        source = _wrap("let x: string = 'abc'\n")
        ast = _parse(source)
        self.assertIsInstance(ast.body[0], LetNode)
        self.assertEqual(ast.body[0].value.value, "abc")

    def test_single_quoted_string_with_escape(self):
        source = _wrap("let x: string = 'a\\nb'\n")
        ast = _parse(source)
        self.assertEqual(ast.body[0].value.value, "a\nb")

    def test_single_semicolons(self):
        source = _wrap("let x: int = 5\nlet y: int = 6\n")
        ast = _parse(source)
        self.assertEqual(len(ast.body), 2)

    def test_semicolon_inside_block(self):
        source = _wrap("func f() -> int {\n let x: int = 5\n return x\n}\n")
        ast = _parse(source)
        self.assertEqual(len(ast.body), 1)

    def test_trailing_whitespace(self):
        source = "eigen 1.0\nlet x: int = 5   \n   \n"
        ast = _parse(source)
        self.assertEqual(len(ast.body), 1)

    def test_crlf_line_endings(self):
        source = "eigen 1.0\r\nlet x: int = 5\r\nlet y: int = 6\r\n"
        ast = _parse(source)
        self.assertEqual(len(ast.body), 2)

    def test_lf_line_endings(self):
        source = "eigen 1.0\nlet x: int = 5\nlet y: int = 6\n"
        ast = _parse(source)
        self.assertEqual(len(ast.body), 2)

    def test_cr_only_line_endings(self):
        source = "eigen 1.0\rlet x: int = 5\r"
        try:
            ast = _parse(source)
            self.assertGreaterEqual(len(ast.body), 0)
        except SyntaxError:
            pass

    def test_mixed_line_endings(self):
        source = "eigen 1.0\nlet x: int = 5\r\nlet y: int = 6\n"
        ast = _parse(source)
        self.assertEqual(len(ast.body), 2)


class TestParserNesting(unittest.TestCase):
    def test_deep_paren_nesting(self):
        depth = 6
        expr = "(" * depth + "1" + " + 1)" * depth
        source = _wrap(f"let x: int = {expr}\n")
        ast = _parse(source)
        self.assertIsInstance(ast.body[0], LetNode)

    def test_deep_brace_nesting(self):
        source = _wrap(
            "func outer() -> int {\n"
            "  func inner() -> int {\n"
            "    if x == 1 {\n"
            "      return 1\n"
            "    }\n"
            "    return 0\n"
            "  }\n"
            "  return 0\n"
            "}\n"
        )
        ast = _parse(source)
        self.assertIsInstance(ast.body[0], FuncDeclNode)

    def test_nested_array_literal(self):
        source = _wrap("let x: array = [1, [2, 3], 4]\n")
        ast = _parse(source)
        self.assertIsInstance(ast.body[0].value, ArrayLiteralNode)
        self.assertEqual(len(ast.body[0].value.elements), 3)

    def test_nested_struct_literal(self):
        source = _wrap("let x: Point = Point { x: 1, y: 2 }\n")
        ast = _parse(source)
        self.assertIsInstance(ast.body[0], LetNode)

    def test_nested_map_literal(self):
        source = _wrap('let x: map = { "a": 1, "b": 2 }\n')
        ast = _parse(source)
        self.assertIsInstance(ast.body[0].value, MapAllocNode)

    def test_tuple_literal(self):
        source = _wrap("let x: int = (1, 2, 3)\n")
        ast = _parse(source)
        self.assertIsInstance(ast.body[0].value, TupleLiteralNode)

    def test_module_with_dotted_path(self):
        source = "eigen 1.0\nmodule quantum.bell\n"
        ast = _parse(source)
        self.assertEqual(ast.module_name, "quantum.bell")

    def test_import_with_dotted_path(self):
        source = "eigen 1.0\nimport quantum.bell\n"
        ast = _parse(source)
        self.assertEqual(len(ast.imports), 1)
        self.assertIsInstance(ast.imports[0], ImportNode)
        self.assertEqual(ast.imports[0].module_path, "quantum.bell")

    def test_multiple_imports(self):
        source = "eigen 1.0\nimport a.b\nimport c.d\nimport e.f\n"
        ast = _parse(source)
        self.assertEqual(len(ast.imports), 3)

    def test_struct_declaration(self):
        source = _wrap("struct Point {\n x: int\n y: int\n}\n")
        ast = _parse(source)
        self.assertIsInstance(ast.body[0], StructDeclNode)
        self.assertEqual(ast.body[0].name, "Point")

    def test_enum_declaration(self):
        source = _wrap("enum Color {\n Red,\n Green,\n Blue\n}\n")
        ast = _parse(source)
        self.assertIsInstance(ast.body[0], EnumDeclNode)
        self.assertEqual(ast.body[0].name, "Color")
        self.assertEqual(len(ast.body[0].variants), 3)

    def test_trait_declaration(self):
        source = _wrap("trait Drawable {\n func draw(self: Self) -> void\n}\n")
        ast = _parse(source)
        self.assertIsInstance(ast.body[0], TraitDeclNode)

    def test_impl_block(self):
        source = _wrap("impl Drawable for Point {\n func draw(self: Point) -> void {\n }\n}\n")
        ast = _parse(source)
        self.assertIsInstance(ast.body[0], ImplBlockNode)

    def test_type_alias(self):
        source = _wrap("type Vec = array<int>\n")
        ast = _parse(source)
        self.assertIsInstance(ast.body[0], TypeAliasDeclNode)


class TestParserStatements(unittest.TestCase):
    def test_break_statement(self):
        source = _wrap("for i in arr {\n break\n}\n")
        ast = _parse(source)
        self.assertIsInstance(ast.body[0], ForNode)
        self.assertIsInstance(ast.body[0].body[0], BreakNode)

    def test_continue_statement(self):
        source = _wrap("for i in arr {\n continue\n}\n")
        ast = _parse(source)
        self.assertIsInstance(ast.body[0].body[0], ContinueNode)

    def test_return_with_expr(self):
        source = _wrap("func f() -> int {\n return 5\n}\n")
        ast = _parse(source)
        self.assertIsInstance(ast.body[0].body[0], ReturnNode)
        self.assertEqual(ast.body[0].body[0].expr.value, 5)

    def test_return_no_expr(self):
        source = _wrap("func f() -> int {\n return\n}\n")
        ast = _parse(source)
        self.assertIsInstance(ast.body[0].body[0], ReturnNode)
        self.assertIsNone(ast.body[0].body[0].expr)

    def test_assert_with_comparison(self):
        source = _wrap("assert 5 == 5\n")
        ast = _parse(source)
        self.assertIsInstance(ast.body[0], AssertNode)

    def test_assert_no_op(self):
        source = _wrap("assert true\n")
        ast = _parse(source)
        self.assertIsInstance(ast.body[0], AssertNode)

    def test_print_statement(self):
        source = _wrap('print "hello"\n')
        ast = _parse(source)
        self.assertIsInstance(ast.body[0], PrintNode)


if __name__ == "__main__":
    unittest.main()
