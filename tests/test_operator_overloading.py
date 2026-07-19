"""
Tests for src/language_extensions/operator_overloading.py — sol.md §3.1.
"""
import unittest

from src.language_extensions.operator_overloading import (
    Operator,
    OperatorOverloadTable,
    OperatorOverloadError,
)


class TestOperatorEnum(unittest.TestCase):
    def test_enum_values(self):
        self.assertEqual(Operator.ADD.value, "+")
        self.assertEqual(Operator.SUB.value, "-")
        self.assertEqual(Operator.EQ.value, "==")


class TestRegisterUnary(unittest.TestCase):
    def test_register_unary_neg(self):
        table = OperatorOverloadTable()
        table.register(Operator.NEG, "Vec", lambda v: Vec(-v.x, -v.y))
        self.assertTrue((Operator.NEG, "Vec") in table)
        v = Vec(1, 2)
        result = table.dispatch(Operator.NEG, v)
        self.assertEqual((result.x, result.y), (-1, -2))

    def test_register_unary_invert(self):
        class Bit:
            def __init__(self, v):
                self.v = v
        table = OperatorOverloadTable()
        table.register(Operator.INVERT, "Bit", lambda b: Bit(1 - b.v))
        result = table.dispatch(Operator.INVERT, Bit(1))
        self.assertEqual(result.v, 0)
        result2 = table.dispatch(Operator.INVERT, Bit(0))
        self.assertEqual(result2.v, 1)

    def test_register_unary_lookup_unknown_raises(self):
        table = OperatorOverloadTable()
        with self.assertRaises(OperatorOverloadError):
            table.lookup(Operator.NEG, "Nonexistent")

    def test_dispatch_unary_no_overload_raises(self):
        table = OperatorOverloadTable()
        with self.assertRaises(OperatorOverloadError):
            table.dispatch(Operator.NEG, 42)


class TestRegisterBinary(unittest.TestCase):
    def test_register_binary_add(self):
        class V:
            def __init__(self, x):
                self.x = x
        # Register binary op explicitly
        table = OperatorOverloadTable()
        table.register_binary(Operator.ADD, "V", "V", lambda a, b: V(a.x + b.x))
        result = table.dispatch(Operator.ADD, V(1), V(2))
        self.assertEqual(result.x, 3)

    def test_register_binary_commutative_via_reflected(self):
        class V:
            def __init__(self, x):
                self.x = x
        table = OperatorOverloadTable()
        # Register (V, V) only. With commutative EQ, the dispatch
        # should succeed for both (V, V) orderings.
        table.register_binary(Operator.EQ, "V", "V", lambda a, b: a.x == b.x)
        a = V(1)
        b = V(1)
        self.assertTrue(table.dispatch(Operator.EQ, a, b))
        self.assertTrue(table.dispatch(Operator.EQ, b, a))

    def test_register_binary_non_commutative(self):
        class V:
            def __init__(self, x):
                self.x = x
        table = OperatorOverloadTable()
        # SUB is non-commutative; (2, 1) is not the same as (1, 2).
        table.register_binary(Operator.SUB, "V", "V", lambda a, b: a.x - b.x)
        self.assertEqual(table.dispatch(Operator.SUB, V(5), V(3)), 2)

    def test_register_binary_raises_for_unary_op(self):
        table = OperatorOverloadTable()
        with self.assertRaises(OperatorOverloadError):
            table.register_binary(Operator.NEG, "V", "V", lambda a, b: None)


class TestDispatchLookup(unittest.TestCase):
    def test_dispatch_unknown_raises(self):
        table = OperatorOverloadTable()
        with self.assertRaises(OperatorOverloadError):
            table.dispatch(Operator.ADD, 1, 2)

    def test_dispatch_with_object_fallback(self):
        table = OperatorOverloadTable()
        # Register an "object" fallback that handles any pair.
        table.register_binary(Operator.ADD, "object", "object",
                                lambda a, b: 999)
        self.assertEqual(table.dispatch(Operator.ADD, "foo", "bar"), 999)

    def test_dispatch_prefer_specific_over_fallback(self):
        class V:
            def __init__(self, x):
                self.x = x
        table = OperatorOverloadTable()
        table.register_binary(Operator.ADD, "object", "object",
                                lambda a, b: "object-fallback")
        table.register_binary(Operator.ADD, "V", "V", lambda a, b: "v-specific")
        result = table.dispatch(Operator.ADD, V(1), V(2))
        self.assertEqual(result, "v-specific")

    def test_dispatch_commutable_fallback(self):
        """Even when no direct (left, right) match exists, commutative
        operators look up the reverse."""
        class A:
            def __init__(self, v):
                self.v = v
        class B:
            def __init__(self, v):
                self.v = v
        table = OperatorOverloadTable()
        # Register only (B, A) but call with (A, B) — commutative EQ
        # should detect the reverse.
        table.register_binary(Operator.EQ, "B", "A",
                                lambda a, b: a.v == b.v)
        # Direct (B, A) works.
        self.assertTrue(table.dispatch(Operator.EQ, B(1), A(1)))
        # Reverse (A, B) — commutative reflection finds the (B, A) entry.
        self.assertTrue(table.dispatch(Operator.EQ, A(1), B(1)))


class TestEntriesAndContains(unittest.TestCase):
    def test_entries_count(self):
        table = OperatorOverloadTable()
        self.assertEqual(table.entries(), 0)
        table.register(Operator.NEG, "V", lambda v: v)
        self.assertEqual(table.entries(), 1)

    def test_contains_check(self):
        table = OperatorOverloadTable()
        table.register(Operator.NEG, "V", lambda v: v)
        self.assertIn((Operator.NEG, "V"), table)
        self.assertNotIn((Operator.POS, "V"), table)


class TestRealisticCustomMatrixType(unittest.TestCase):
    """A realistic scenario: implement a 2D matrix type with
    add/mul/eq operator overloads. Demonstrates the full surface."""

    def setUp(self):
        class Matrix:
            def __init__(self, vals):
                self.vals = [list(row) for row in vals]
            def __eq__(self, other):
                return isinstance(other, Matrix) and self.vals == other.vals
            def __hash__(self):
                return hash(tuple(tuple(r) for r in self.vals))
            def __repr__(self):
                return f"Matrix({self.vals})"
            def __iter__(self):
                return iter(self.vals)
        self.Matrix = Matrix
        self.table = OperatorOverloadTable()
        # Matrix + Matrix → element-wise sum
        self.table.register_binary(
            Operator.ADD, "Matrix", "Matrix",
            lambda a, b: self.Matrix(
                [[a.vals[i][j] + b.vals[i][j]
                  for j in range(len(a.vals[0]))]
                 for i in range(len(a.vals))]))
        # Matrix * Matrix → classic matmul (2x2 only for simplicity)
        self.table.register_binary(
            Operator.MUL, "Matrix", "Matrix",
            lambda a, b: self.Matrix(
                [[sum(a.vals[i][k] * b.vals[k][j]
                       for k in range(len(b.vals)))
                   for j in range(len(b.vals[0]))]
                  for i in range(len(a.vals))]))
        # - Matrix → negated
        self.table.register(
            Operator.NEG, "Matrix",
            lambda m: self.Matrix([[-x for x in row] for row in m.vals]))
        # Matrix == Matrix
        self.table.register_binary(
            Operator.EQ, "Matrix", "Matrix",
            lambda a, b: a.vals == b.vals)

    def test_add(self):
        a = self.Matrix([[1, 2], [3, 4]])
        b = self.Matrix([[5, 6], [7, 8]])
        c = self.table.dispatch(Operator.ADD, a, b)
        self.assertEqual(c.vals, [[6, 8], [10, 12]])

    def test_neg(self):
        a = self.Matrix([[1, -2], [-3, 4]])
        b = self.table.dispatch(Operator.NEG, a)
        self.assertEqual(b.vals, [[-1, 2], [3, -4]])

    def test_mul(self):
        a = self.Matrix([[1, 2], [3, 4]])
        b = self.Matrix([[5, 6], [7, 8]])
        c = self.table.dispatch(Operator.MUL, a, b)
        self.assertEqual(c.vals, [[19, 22], [43, 50]])

    def test_eq(self):
        a = self.Matrix([[1, 2], [3, 4]])
        b = self.Matrix([[1, 2], [3, 4]])
        c = self.Matrix([[5, 6], [7, 8]])
        self.assertTrue(self.table.dispatch(Operator.EQ, a, b))
        self.assertFalse(self.table.dispatch(Operator.EQ, a, c))


# Helper class used in tests
class Vec:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Vec({self.x}, {self.y})"

    def __eq__(self, other):
        return isinstance(other, Vec) and (self.x, self.y) == (other.x, other.y)

    def __hash__(self):
        return hash((self.x, self.y))


class TestVecAddAndMulOverloads(unittest.TestCase):
    def test_add_returns_new_vec(self):
        table = OperatorOverloadTable()
        table.register_binary(Operator.ADD, "Vec", "Vec",
                                lambda a, b: Vec(a.x + b.x, a.y + b.y))
        result = table.dispatch(Operator.ADD, Vec(1, 2), Vec(3, 4))
        self.assertEqual(result, Vec(4, 6))

    def test_mul_not_registered_raises(self):
        table = OperatorOverloadTable()
        with self.assertRaises(OperatorOverloadError):
            table.dispatch(Operator.MUL, Vec(0, 0), Vec(1, 1))


if __name__ == "__main__":
    unittest.main()
