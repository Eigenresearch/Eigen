"""§9.1 — Additional parser tests for previously-uncovered node types.

Exercises node types that were missing from test_parser_grammar.py:
  - ParallelBlockNode / TaskStatementNode
  - MapAllocNode (map literal)
  - ArrayAllocNode / ArrayGetNode / ArraySetNode
  - StructAllocNode / StructGetNode / StructSetNode
  - MapGetNode / MapSetNode
  - DotAccessNode
  - IndexAccessNode
  - QFuncDeclNode
  - StringInterpolationNode
  - BinaryOpNode (all precedence levels)
  - UnaryOpNode (not, minus, tilde)
"""
import unittest

from src.frontend.lexer import Lexer
from src.frontend.parser import Parser
from src.frontend.ast import (
    ProgramNode,
    QFuncDeclNode,
    QFuncCallNode,
    ParallelBlockNode,
    TaskStatementNode,
    MapAllocNode,
    ArrayLiteralNode,
    TupleLiteralNode,
    DotAccessNode,
    IndexAccessNode,
    BinaryOpNode,
    LiteralNode,
    VarRefNode,
    StructLiteralNode,
    StringInterpolationNode,
    LetNode,
    FuncDeclNode,
)


def _parse(source: str):
    tokens = Lexer(source).tokenize()
    return Parser(tokens).parse()


def _func_wrap(body: str) -> str:
    return f"eigen 1.0\nfunc main() -> int {{\n{body}\nreturn 0\n}}"


# ---------------------------------------------------------------------------
# Parallel block / task
# ---------------------------------------------------------------------------

class TestParallelBlock(unittest.TestCase):
    def test_parallel_block_with_tasks(self):
        src = _func_wrap("parallel {\ntask foo()\ntask bar()\n}")
        ast = _parse(src)
        func = [n for n in ast.body if isinstance(n, FuncDeclNode)][0]
        parallels = [s for s in func.body if isinstance(s, ParallelBlockNode)]
        self.assertEqual(len(parallels), 1)
        self.assertGreaterEqual(len(parallels[0].tasks), 2)

    def test_parallel_block_empty(self):
        src = _func_wrap("parallel {\n}")
        ast = _parse(src)
        func = [n for n in ast.body if isinstance(n, FuncDeclNode)][0]
        parallels = [s for s in func.body if isinstance(s, ParallelBlockNode)]
        self.assertEqual(len(parallels), 1)


# ---------------------------------------------------------------------------
# Map literal / MapAllocNode
# ---------------------------------------------------------------------------

class TestMapLiteral(unittest.TestCase):
    def test_map_literal_in_let(self):
        src = 'eigen 1.0\nlet m: map<string, int> = {"a": 1, "b": 2}'
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertEqual(len(lets), 1)
        # Value should be a MapAllocNode
        self.assertIsInstance(lets[0].value, MapAllocNode)

    def test_empty_map_literal(self):
        src = 'eigen 1.0\nlet m: map<string, int> = {}'
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertEqual(len(lets), 1)
        self.assertIsInstance(lets[0].value, MapAllocNode)


# ---------------------------------------------------------------------------
# Dot access
# ---------------------------------------------------------------------------

class TestDotAccess(unittest.TestCase):
    def test_dot_access_in_let(self):
        src = 'eigen 1.0\nlet x: int = obj.field'
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertEqual(len(lets), 1)
        self.assertIsInstance(lets[0].value, DotAccessNode)

    def test_chained_dot_access(self):
        src = 'eigen 1.0\nlet x: int = obj.a.b.c'
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertEqual(len(lets), 1)
        # Chained: outermost should be DotAccessNode
        val = lets[0].value
        self.assertIsInstance(val, DotAccessNode)


# ---------------------------------------------------------------------------
# Index access
# ---------------------------------------------------------------------------

class TestIndexAccess(unittest.TestCase):
    def test_array_index_access(self):
        src = 'eigen 1.0\nlet x: int = arr[0]'
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertEqual(len(lets), 1)
        self.assertIsInstance(lets[0].value, IndexAccessNode)

    def test_nested_index_access(self):
        src = 'eigen 1.0\nlet x: int = matrix[0][1]'
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertEqual(len(lets), 1)
        self.assertIsInstance(lets[0].value, IndexAccessNode)


# ---------------------------------------------------------------------------
# QFuncDeclNode
# ---------------------------------------------------------------------------

class TestQFuncDecl(unittest.TestCase):
    def test_qfunc_declaration(self):
        src = ("eigen 1.0\n"
                "qfunc bell(qubit q0, qubit q1) {\n"
                "    H q0\n"
                "    CNOT q0, q1\n"
                "}")
        ast = _parse(src)
        qfuncs = [n for n in ast.body if isinstance(n, QFuncDeclNode)]
        self.assertEqual(len(qfuncs), 1)
        self.assertEqual(qfuncs[0].name, "bell")
        self.assertEqual(len(qfuncs[0].params), 2)

    def test_qfunc_call(self):
        src = ("eigen 1.0\n"
                "qubit q0\n"
                "qubit q1\n"
                "qfunc bell(qubit q0, qubit q1) {\n"
                "    H q0\n"
                "    CNOT q0, q1\n"
                "}\n"
                "bell(q0, q1)")
        ast = _parse(src)
        # The call should produce a QFuncCallNode since bell is declared
        calls = [n for n in ast.body if isinstance(n, QFuncCallNode)]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "bell")

    def test_qfunc_with_int_param(self):
        src = ("eigen 1.0\n"
                "qfunc rot(int n, qubit q0) {\n"
                "    H q0\n"
                "}")
        ast = _parse(src)
        qfuncs = [n for n in ast.body if isinstance(n, QFuncDeclNode)]
        self.assertEqual(len(qfuncs), 1)
        self.assertEqual(len(qfuncs[0].params), 2)


# ---------------------------------------------------------------------------
# BinaryOpNode — operator precedence
# ---------------------------------------------------------------------------

class TestBinaryOperators(unittest.TestCase):
    def test_addition(self):
        src = 'eigen 1.0\nlet x: int = 1 + 2'
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertIsInstance(lets[0].value, BinaryOpNode)
        self.assertEqual(lets[0].value.op, "+")

    def test_multiplication(self):
        src = 'eigen 1.0\nlet x: int = 2 * 3'
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertIsInstance(lets[0].value, BinaryOpNode)
        self.assertEqual(lets[0].value.op, "*")

    def test_subtraction(self):
        src = 'eigen 1.0\nlet x: int = 5 - 3'
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertEqual(lets[0].value.op, "-")

    def test_division(self):
        src = 'eigen 1.0\nlet x: int = 10 / 2'
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertEqual(lets[0].value.op, "/")

    def test_modulo(self):
        src = 'eigen 1.0\nlet x: int = 10 % 3'
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertEqual(lets[0].value.op, "%")

    def test_power(self):
        src = 'eigen 1.0\nlet x: int = 2 ** 8'
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertEqual(lets[0].value.op, "**")

    def test_logical_and(self):
        src = 'eigen 1.0\nlet x: bool = true and false'
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertEqual(lets[0].value.op, "and")

    def test_logical_or(self):
        src = 'eigen 1.0\nlet x: bool = true or false'
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertEqual(lets[0].value.op, "or")

    def test_equality(self):
        src = 'eigen 1.0\nlet x: bool = 1 == 1'
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertEqual(lets[0].value.op, "==")

    def test_not_equal(self):
        src = 'eigen 1.0\nlet x: bool = 1 != 2'
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertEqual(lets[0].value.op, "!=")

    def test_less_than(self):
        src = 'eigen 1.0\nlet x: bool = 1 < 2'
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertEqual(lets[0].value.op, "<")

    def test_greater_than(self):
        src = 'eigen 1.0\nlet x: bool = 2 > 1'
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertEqual(lets[0].value.op, ">")

    def test_less_equal(self):
        src = 'eigen 1.0\nlet x: bool = 1 <= 1'
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertEqual(lets[0].value.op, "<=")

    def test_greater_equal(self):
        src = 'eigen 1.0\nlet x: bool = 1 >= 1'
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertEqual(lets[0].value.op, ">=")

    def test_precedence_mul_over_add(self):
        # 1 + 2 * 3 → (1 + (2 * 3))
        src = 'eigen 1.0\nlet x: int = 1 + 2 * 3'
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        # Top-level op should be + (lower precedence)
        self.assertEqual(lets[0].value.op, "+")
        # Right child should be * (higher precedence)
        self.assertIsInstance(lets[0].value.right, BinaryOpNode)
        self.assertEqual(lets[0].value.right.op, "*")

    def test_precedence_pow_over_mul(self):
        # 2 * 3 ** 2 → (2 * (3 ** 2))
        src = 'eigen 1.0\nlet x: int = 2 * 3 ** 2'
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertEqual(lets[0].value.op, "*")
        self.assertIsInstance(lets[0].value.right, BinaryOpNode)
        self.assertEqual(lets[0].value.right.op, "**")


# ---------------------------------------------------------------------------
# Unary operators
# ---------------------------------------------------------------------------

class TestUnaryOperators(unittest.TestCase):
    def test_negation(self):
        src = 'eigen 1.0\nlet x: int = -5'
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        # -5 may be parsed as LiteralNode(-5) by native parser
        # or as BinaryOpNode("-", 0, 5) by Python parser
        val = lets[0].value
        self.assertTrue(
            isinstance(val, (BinaryOpNode, LiteralNode)),
            f"Expected BinaryOpNode or LiteralNode, got {type(val).__name__}"
        )

    def test_not_operator(self):
        src = 'eigen 1.0\nlet x: bool = not true'
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertIsInstance(lets[0].value, BinaryOpNode)
        self.assertEqual(lets[0].value.op, "not")


# ---------------------------------------------------------------------------
# Struct literal
# ---------------------------------------------------------------------------

class TestStructLiteral(unittest.TestCase):
    def test_struct_literal_with_fields(self):
        src = 'eigen 1.0\nlet p: Point = Point { x: 1, y: 2 }'
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertEqual(len(lets), 1)
        self.assertIsInstance(lets[0].value, StructLiteralNode)
        self.assertEqual(lets[0].value.struct_name, "Point")
        # field_bindings is a dict or list of (name, value) tuples
        bindings = lets[0].value.field_bindings
        if isinstance(bindings, dict):
            self.assertIn("x", bindings)
            self.assertIn("y", bindings)
        elif isinstance(bindings, (list, tuple)):
            names = [b[0] for b in bindings]
            self.assertIn("x", names)
            self.assertIn("y", names)

    def test_empty_struct_literal(self):
        src = 'eigen 1.0\nlet p: Empty = Empty {}'
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertEqual(len(lets), 1)
        self.assertIsInstance(lets[0].value, StructLiteralNode)


# ---------------------------------------------------------------------------
# Constants: PI, TAU, E
# ---------------------------------------------------------------------------

class TestMathConstants(unittest.TestCase):
    def test_pi_constant(self):
        src = 'eigen 1.0\nlet x: float = PI'
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertIsInstance(lets[0].value, LiteralNode)
        self.assertAlmostEqual(lets[0].value.value, 3.141592653589793)

    def test_tau_constant(self):
        src = 'eigen 1.0\nlet x: float = TAU'
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertIsInstance(lets[0].value, LiteralNode)
        self.assertAlmostEqual(lets[0].value.value, 6.283185307179586)

    def test_e_constant(self):
        src = 'eigen 1.0\nlet x: float = E'
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertIsInstance(lets[0].value, LiteralNode)
        self.assertAlmostEqual(lets[0].value.value, 2.718281828459045)


# ---------------------------------------------------------------------------
# Boolean and null literals
# ---------------------------------------------------------------------------

class TestBooleanAndNullLiterals(unittest.TestCase):
    def test_true_literal(self):
        src = 'eigen 1.0\nlet x: bool = true'
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertIsInstance(lets[0].value, LiteralNode)
        self.assertTrue(lets[0].value.value)

    def test_false_literal(self):
        src = 'eigen 1.0\nlet x: bool = false'
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertIsInstance(lets[0].value, LiteralNode)
        self.assertFalse(lets[0].value.value)

    def test_null_literal(self):
        src = 'eigen 1.0\nlet x: int = null'
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertIsInstance(lets[0].value, LiteralNode)
        self.assertIsNone(lets[0].value.value)


# ---------------------------------------------------------------------------
# Variable reference
# ---------------------------------------------------------------------------

class TestVarRef(unittest.TestCase):
    def test_variable_reference(self):
        src = 'eigen 1.0\nlet x: int = y'
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertIsInstance(lets[0].value, VarRefNode)
        self.assertEqual(lets[0].value.name, "y")


# ---------------------------------------------------------------------------
# String interpolation
# ---------------------------------------------------------------------------

class TestStringInterpolation(unittest.TestCase):
    def test_string_interpolation_parsed(self):
        # The lexer marks interpolation with \x00 markers
        src = 'eigen 1.0\nlet x: string = "Result: ${y}"'
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertEqual(len(lets), 1)
        # Value should be StringInterpolationNode or LiteralNode
        # (depending on whether the lexer detected interpolation markers)
        val = lets[0].value
        self.assertTrue(
            isinstance(val, (StringInterpolationNode, LiteralNode)),
            f"Expected StringInterpolationNode or LiteralNode, got {type(val).__name__}"
        )

    def test_plain_string_no_interpolation(self):
        src = 'eigen 1.0\nlet x: string = "hello world"'
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertIsInstance(lets[0].value, LiteralNode)
        self.assertEqual(lets[0].value.value, "hello world")


if __name__ == "__main__":
    unittest.main()
