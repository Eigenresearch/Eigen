"""
Tests for src/language_extensions/type_system_extensions.py — sol.md §3.3.
"""
import unittest

from src.language_extensions.type_system_extensions import (
    TypeKind,
    TypeRef,
    Protocol,
    UnionType,
    OptionalType,
    ResultType,
    ConstGeneric,
    HigherKindedType,
)


class TestTypeRef(unittest.TestCase):
    def test_primitive_type_ref(self):
        t = TypeRef("int", kind=TypeKind.PRIMITIVE)
        self.assertEqual(t.name, "int")
        self.assertEqual(t.type_args, [])
        self.assertEqual(str(t), "int")

    def test_generic_with_type_args(self):
        t = TypeRef("Array", [TypeRef("int")], kind=TypeKind.GENERIC)
        self.assertEqual(str(t), "Array<int>")

    def test_equality_includes_kind(self):
        # Same name and args but different kind → not equal.
        a = TypeRef("X", [], kind=TypeKind.PRIMITIVE)
        b = TypeRef("X", [], kind=TypeKind.GENERIC)
        self.assertNotEqual(a, b)

    def test_hashable(self):
        a = TypeRef("int", [], kind=TypeKind.PRIMITIVE)
        b = TypeRef("int", [], kind=TypeKind.PRIMITIVE)
        self.assertEqual(hash(a), hash(b))
        self.assertIn(a, {a, b, "int"})


class TestProtocol(unittest.TestCase):
    def setUp(self):
        self.dessert = Protocol(
            name="Dessert",
            methods=[
                ("calories", [], "int"),
                ("is_vegan", [], "bool"),
            ])

    def test_check_object_with_all_methods_passes(self):
        class Cake:
            def calories(self): return 200
            def is_vegan(self): return False
        self.assertTrue(self.dessert.check(Cake()))

    def test_check_object_missing_one_method_fails(self):
        class Cake:
            def calories(self): return 200
            # missing is_vegan
        self.assertFalse(self.dessert.check(Cake()))

    def test_check_object_with_non_callable_attr_fails(self):
        class Cake:
            calories = 200   # not callable
            def is_vegan(self): return False
        self.assertFalse(self.dessert.check(Cake()))

    def test_check_with_registry_same_as_check(self):
        class Cake:
            def calories(self): return 200
            def is_vegan(self): return False
        # No registry needed in this envelope; check_with_registry
        # delegates to check.
        self.assertTrue(self.dessert.check_with_registry(Cake(), None))


class TestUnionType(unittest.TestCase):
    def test_str_representation(self):
        u = UnionType(members=[TypeRef("int"), TypeRef("str")])
        self.assertEqual(str(u), "int | str")

    def test_contains_matches_int(self):
        u = UnionType(members=[TypeRef("int"), TypeRef("str")])
        self.assertTrue(u.contains(5))
        self.assertTrue(u.contains("hello"))

    def test_contains_rejects_other_types(self):
        u = UnionType(members=[TypeRef("int"), TypeRef("str")])
        self.assertFalse(u.contains(5.0))
        self.assertFalse(u.contains([1, 2]))

    def test_contains_with_None_member(self):
        u = UnionType(members=[TypeRef("int"), TypeRef("None")])
        self.assertTrue(u.contains(None))

    def test_contains_without_None_member(self):
        u = UnionType(members=[TypeRef("int"), TypeRef("str")])
        self.assertFalse(u.contains(None))

    def test_any_matches_everything(self):
        u = UnionType(members=[TypeRef("Any")])
        for v in [5, "x", [], {}, None, object()]:
            with self.subTest(v=v):
                self.assertTrue(u.contains(v))

    def test_custom_type_resolver(self):
        class Foo:
            pass
        # Use a TypeRef with a name that doesn't match the actual class
        # __name__, then rely on the resolver to bridge the gap.
        class Bar:
            pass
        # TypeRef says "Bar" but the actual class is "Foo"
        # Without the resolver, contains() must return False.
        u = UnionType(members=[TypeRef("Bar")])
        self.assertFalse(u.contains(Foo()))
        # With a resolver that maps TypeRef name -> actual class:
        # Bar → Foo
        resolver = lambda name: {"Bar": Foo}.get(name, type(None))
        self.assertTrue(u.contains(Foo(), type_resolver=resolver))


class TestOptionalType(unittest.TestCase):
    def test_str_representation(self):
        t = OptionalType(inner=TypeRef("int"))
        self.assertEqual(str(t), "int?")

    def test_contains_None(self):
        t = OptionalType(inner=TypeRef("int"))
        self.assertTrue(t.contains(None))

    def test_contains_inner_value(self):
        t = OptionalType(inner=TypeRef("int"))
        self.assertTrue(t.contains(5))

    def test_rejects_other_types(self):
        t = OptionalType(inner=TypeRef("int"))
        self.assertFalse(t.contains("hello"))
        self.assertFalse(t.contains(5.0))


class TestResultType(unittest.TestCase):
    def test_str_representation(self):
        t = ResultType(ok=TypeRef("int"), err=TypeRef("str"))
        self.assertEqual(str(t), "Result<int, str>")

    def test_equality(self):
        a = ResultType(ok=TypeRef("int"), err=TypeRef("str"))
        b = ResultType(ok=TypeRef("int"), err=TypeRef("str"))
        c = ResultType(ok=TypeRef("int"), err=TypeRef("bool"))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


class TestConstGeneric(unittest.TestCase):
    def test_str_representation(self):
        t = ConstGeneric(name="Array",
                          type_arg=TypeRef("int"),
                          const_value=5)
        self.assertEqual(str(t), "Array<int, 5>")

    def test_equality(self):
        a = ConstGeneric("Array", TypeRef("int"), 5)
        b = ConstGeneric("Array", TypeRef("int"), 5)
        c = ConstGeneric("Array", TypeRef("int"), 10)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_const_value_can_be_anything(self):
        t = ConstGeneric("Map", TypeRef("str"), "literal_value")
        self.assertEqual(t.const_value, "literal_value")


class TestHigherKindedType(unittest.TestCase):
    def test_unary_arithmetic_kind_signature(self):
        hkt = HigherKindedType(name="Option", arity=1)
        self.assertEqual(hkt.kind_signature, "* -> *")
        self.assertEqual(str(hkt), "Option")

    def test_binary_arithmetic_kind_signature(self):
        hkt = HigherKindedType(name="Result", arity=2)
        self.assertEqual(hkt.kind_signature, "* -> * -> *")
        self.assertEqual(str(hkt), "Result")

    def test_apply_one_argument(self):
        hkt = HigherKindedType(name="Option", arity=1)
        applied = hkt.apply(TypeRef("int"))
        self.assertEqual(str(applied), "Option<int>")
        self.assertTrue(applied.is_fully_applied())

    def test_apply_two_arguments(self):
        hkt = HigherKindedType(name="Result", arity=2)
        applied = hkt.apply(TypeRef("int")).apply(TypeRef("str"))
        self.assertEqual(str(applied), "Result<int, str>")
        self.assertTrue(applied.is_fully_applied())

    def test_over_application_raises(self):
        hkt = HigherKindedType(name="Option", arity=1)
        fully = hkt.apply(TypeRef("int"))
        with self.assertRaises(TypeError):
            fully.apply(TypeRef("str"))

    def test_partially_applied(self):
        hkt = HigherKindedType(name="Result", arity=2)
        partial = hkt.apply(TypeRef("int"))
        self.assertEqual(str(partial), "Result<int>")
        self.assertFalse(partial.is_fully_applied())

    def test_hkt_with_zero_arity_treated_as_concrete(self):
        hkt = HigherKindedType(name="int", arity=0)
        self.assertEqual(hkt.kind_signature, "*")
        self.assertTrue(hkt.is_fully_applied())


class TestRealisticScenarios(unittest.TestCase):
    """End-to-end-style exercises combining the constructs."""

    def test_dessert_protocol_with_cake_class(self):
        # Define a Dessert protocol and an implementing class.
        Dessert = Protocol(
            name="Dessert",
            methods=[
                ("calories", [], "int"),
                ("is_vegan", [], "bool"),
            ],
        )

        class Cake:
            def calories(self): return 1000
            def is_vegan(self): return False
        class Salad:
            def calories(self): return 200
            def is_vegan(self): return True

        self.assertTrue(Dessert.check(Cake()))
        self.assertTrue(Dessert.check(Salad()))

    def test_int_or_string_union(self):
        StrOrInt = UnionType(members=[TypeRef("int"), TypeRef("str")])
        # Apply to a list filtering.
        data = [1, "hello", 2, "world", 3.14, None]
        valid = [v for v in data if StrOrInt.contains(v)]
        self.assertEqual(valid, [1, "hello", 2, "world"])

    def test_result_int_or_error(self):
        # `Result<int, str>` with str as an error-tag type.
        res = ResultType(ok=TypeRef("int"), err=TypeRef("str"))
        self.assertEqual(str(res), "Result<int, str>")

    def test_array_int_5_const_generic(self):
        cg = ConstGeneric("Array", TypeRef("int"), 5)
        self.assertEqual(cg.const_value, 5)
        self.assertEqual(cg.name, "Array")

    def test_nested_hkt_application(self):
        # Build `Result<Option<int>, str>` as a binary HKT applied
        # to HKT applied to concrete. We demonstrate this by first
        # fully applying Option<int>, then using that as an argument
        # to Result's first slot.
        Option = HigherKindedType("Option", arity=1)
        Result = HigherKindedType("Result", arity=2)
        option_int = Option.apply(TypeRef("int"))  # Option<int>, fully applied
        result_a = Result.apply(option_int)        # Result<Option<int>, _>
        result_a = result_a.apply(TypeRef("str"))  # Result<Option<int>, str>
        self.assertTrue(result_a.is_fully_applied())
        self.assertEqual(str(result_a), "Result<Option<int>, str>")


if __name__ == "__main__":
    unittest.main()
