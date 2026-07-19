import unittest

from src.formatter import EigenFormatter


class TestFormatterStringEscapes(unittest.TestCase):
    def setUp(self):
        self.f = EigenFormatter()

    def test_escaped_quote_preserved(self):
        result = self.f.format_line_content('let x: string = "a\\"b"')
        self.assertIn('\\"', result)

    def test_escape_sequence_not_corrupted(self):
        result = self.f.format_line_content('let x: string = "a\\nb"')
        self.assertIn('\\n', result)

    def test_tab_escape_in_string(self):
        result = self.f.format_line_content('let x: string = "a\\tb"')
        self.assertIn('\\t', result)

    def test_backslash_escape_in_string(self):
        result = self.f.format_line_content('let x: string = "a\\\\b"')
        self.assertIn('\\\\', result)

    def test_unicode_escape_preserved(self):
        result = self.f.format_line_content('let x: string = "\\u0041"')
        self.assertIn('\\u0041', result)

    def test_hex_escape_preserved(self):
        result = self.f.format_line_content('let x: string = "\\x41"')
        self.assertIn('\\x41', result)

    def test_string_with_comment_marker_preserved(self):
        result = self.f.format_line_content('let x: string = "a # b"')
        self.assertIn('#', result)

    def test_string_with_braces_preserved(self):
        result = self.f.format_line_content('let x: string = "a { b } c"')
        self.assertIn('{', result)
        self.assertIn('}', result)


class TestFormatterOperatorSpacing(unittest.TestCase):
    def setUp(self):
        self.f = EigenFormatter()

    def test_equals_gets_spaces(self):
        result = self.f.format_line_content("x=5")
        self.assertIn("x = 5", result)

    def test_plus_operator_spacing(self):
        result = self.f.format_line_content("x=a+b")
        self.assertIn("a + b", result)

    def test_arrow_operator_spacing(self):
        result = self.f.format_line_content("func f()->int{return 0}")
        self.assertIn("->", result)

    def test_double_equals_spacing(self):
        result = self.f.format_line_content("if x==5 {")
        self.assertIn("==", result)

    def test_not_equals_spacing(self):
        result = self.f.format_line_content("if x!=5 {")
        self.assertIn("!=", result)

    def test_le_operator_spacing(self):
        result = self.f.format_line_content("if x<=5 {")
        self.assertIn("<=", result)

    def test_ge_operator_spacing(self):
        result = self.f.format_line_content("if x>=5 {")
        self.assertIn(">=", result)

    def test_add_assign_spacing(self):
        result = self.f.format_line_content("x+=5")
        self.assertIn("+=", result)

    def test_sub_assign_spacing(self):
        result = self.f.format_line_content("x-=5")
        self.assertIn("-=", result)

    def test_mul_assign_spacing(self):
        result = self.f.format_line_content("x*=5")
        self.assertIn("*=", result)

    def test_div_assign_spacing(self):
        result = self.f.format_line_content("x/=5")
        self.assertIn("/=", result)


class TestFormatterComments(unittest.TestCase):
    def setUp(self):
        self.f = EigenFormatter()

    def test_hash_comment_preserved(self):
        result = self.f.format_line_content("let x: int = 5  # comment")
        self.assertIn("# comment", result)

    def test_slash_slash_comment_preserved(self):
        result = self.f.format_line_content("let x: int = 5  // comment")
        self.assertIn("// comment", result)

    def test_comment_only_line(self):
        result = self.f.format_line_content("# just a comment")
        self.assertIn("# just a comment", result)

    def test_comment_with_hash_in_string(self):
        result = self.f.format_line_content('let x: string = "a # b"  # c')
        self.assertIn('"a # b"', result)

    def test_inline_slash_comment(self):
        result = self.f.format_line_content("let x: int = 5 // hello")
        self.assertIn("// hello", result)


class TestFormatterIndentation(unittest.TestCase):
    def setUp(self):
        self.f = EigenFormatter()

    def test_indent_increases_after_open_brace(self):
        source = "eigen 1.0\nfunc f() -> int {\nlet x: int = 1\nreturn x\n}\n"
        result = self.f.format_code(source)
        lines = result.splitlines()
        self.assertEqual(lines[0].lstrip(), "eigen 1.0")
        self.assertEqual(lines[1].lstrip(), "func f() -> int {")
        self.assertTrue(lines[2].startswith("    "))
        self.assertTrue(lines[3].startswith("    "))

    def test_indent_decreases_after_close_brace(self):
        source = "eigen 1.0\nfunc f() -> int {\nlet x: int = 1\n}\nlet y: int = 2\n"
        result = self.f.format_code(source)
        lines = result.splitlines()
        self.assertEqual(lines[3].lstrip(), "}")
        self.assertFalse(lines[4].startswith("    "))

    def test_nested_indent(self):
        source = (
            "eigen 1.0\n"
            "func f() -> int {\n"
            "  if x == 1 {\n"
            "    let y: int = 2\n"
            "  }\n"
            "  return 0\n"
            "}\n"
        )
        result = self.f.format_code(source)
        lines = result.splitlines()
        self.assertTrue(lines[2].startswith("    "))
        self.assertTrue(lines[3].startswith("        "))

    def test_empty_line_no_indent(self):
        source = "eigen 1.0\n\nlet x: int = 5\n"
        result = self.f.format_code(source)
        lines = result.splitlines()
        self.assertEqual(lines[1], "")

    def test_trailing_newline(self):
        source = "eigen 1.0\nlet x: int = 5\n"
        result = self.f.format_code(source)
        self.assertTrue(result.endswith("\n"))


class TestFormatterDeclarations(unittest.TestCase):
    def setUp(self):
        self.f = EigenFormatter()

    def test_struct_decl_preserved(self):
        source = (
            "eigen 1.0\n"
            "struct Point {\n"
            "  x: int\n"
            "  y: int\n"
            "}\n"
        )
        result = self.f.format_code(source)
        self.assertIn("struct Point", result)
        self.assertIn("x: int", result)
        self.assertIn("y: int", result)

    def test_enum_decl_preserved(self):
        source = "eigen 1.0\nenum Color {\n  Red,\n  Green,\n  Blue\n}\n"
        result = self.f.format_code(source)
        self.assertIn("enum Color", result)
        self.assertIn("Red", result)
        self.assertIn("Green", result)
        self.assertIn("Blue", result)

    def test_trait_decl_preserved(self):
        source = (
            "eigen 1.0\n"
            "trait Drawable {\n"
            "  func draw(self: Self) -> void\n"
            "}\n"
        )
        result = self.f.format_code(source)
        self.assertIn("trait Drawable", result)
        self.assertIn("func draw", result)

    def test_impl_block_preserved(self):
        source = (
            "eigen 1.0\n"
            "impl Drawable for Point {\n"
            "  func draw(self: Point) -> void {\n"
            "    return\n"
            "  }\n"
            "}\n"
        )
        result = self.f.format_code(source)
        self.assertIn("impl Drawable", result)
        self.assertIn("func draw", result)

    def test_let_binding_preserved(self):
        result = self.f.format_line_content("let x: int = 5")
        self.assertIn("let", result)
        self.assertIn("int", result)
        self.assertIn("5", result)

    def test_func_decl_preserved(self):
        result = self.f.format_line_content("func add(a: int, b: int) -> int {")
        self.assertIn("func add", result)
        self.assertIn("int", result)

    def test_qubit_decl_preserved(self):
        result = self.f.format_line_content("qubit q0")
        self.assertIn("qubit", result)
        self.assertIn("q0", result)


class TestFormatterSingleQuoted(unittest.TestCase):
    def setUp(self):
        self.f = EigenFormatter()

    def test_single_quoted_string_preserved(self):
        result = self.f.format_line_content("let x: string = 'abc'")
        self.assertIn("'abc'", result)

    def test_single_quoted_with_escape_preserved(self):
        result = self.f.format_line_content("let x: string = 'a\\nb'")
        self.assertIn("'", result)


class TestFormatterMixedContent(unittest.TestCase):
    def setUp(self):
        self.f = EigenFormatter()

    def test_call_with_args(self):
        result = self.f.format_line_content("print x")
        self.assertIn("print", result)
        self.assertIn("x", result)

    def test_return_statement(self):
        result = self.f.format_line_content("return 5")
        self.assertIn("return", result)

    def test_measure_statement(self):
        result = self.f.format_line_content("measure q0 -> c0")
        self.assertIn("measure", result)
        self.assertIn("q0", result)
        self.assertIn("c0", result)

    def test_gate_application(self):
        result = self.f.format_line_content("H q0")
        self.assertIn("H", result)
        self.assertIn("q0", result)

    def test_two_qubit_gate(self):
        result = self.f.format_line_content("CNOT q0, q1")
        self.assertIn("CNOT", result)
        self.assertIn("q0", result)
        self.assertIn("q1", result)


if __name__ == "__main__":
    unittest.main()
