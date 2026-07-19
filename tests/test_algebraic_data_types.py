"""
Tests for src/language_extensions/algebraic_data_types.py — sol.md §3.1.
"""
import unittest

from src.language_extensions.algebraic_data_types import (
    Variant,
    AlgebraicDataType,
    ADTValue,
    ADTValueError,
    ADTRegistry,
    StandardADTs,
)


class TestVariant(unittest.TestCase):
    def test_unit_variant_has_zero_arity(self):
        v = Variant("None")
        self.assertEqual(v.arity, 0)

    def test_variant_with_one_field_has_arity_1(self):
        v = Variant("Some", ["T"])
        self.assertEqual(v.arity, 1)

    def test_variant_with_many_fields(self):
        v = Variant("Cons", ["T", "List[T]"])
        self.assertEqual(v.arity, 2)

    def test_variant_is_value_equality(self):
        v1 = Variant("Some", ["int"])
        v2 = Variant("Some", ["int"])
        self.assertEqual(v1, v2)


class TestAlgebraicDataType(unittest.TestCase):
    def test_simple_definition(self):
        adt = AlgebraicDataType(
            name="Color",
            variants=[Variant("Red"), Variant("Green"), Variant("Blue")])
        self.assertEqual(adt.name, "Color")
        self.assertEqual(adt.variant_names, ["Red", "Green", "Blue"])
        self.assertEqual(adt.variant("Red").arity, 0)

    def test_parameterised_adt(self):
        adt = AlgebraicDataType(
            name="Option", type_params=["T"],
            variants=[Variant("Some", ["T"]), Variant("None", [])])
        self.assertEqual(adt.type_params, ["T"])
        self.assertEqual(len(adt.variants), 2)

    def test_variant_lookup_raises_on_unknown(self):
        adt = AlgebraicDataType(name="Color",
                                 variants=[Variant("Red")])
        with self.assertRaises(KeyError):
            adt.variant("Blue")


class TestADTValue(unittest.TestCase):
    def test_unit_constructor_repr(self):
        v = ADTValue("None", [])
        self.assertEqual(repr(v), "None")

    def test_single_field_repr(self):
        v = ADTValue("Some", [42])
        self.assertEqual(repr(v), "Some(42)")

    def test_multiple_field_repr(self):
        v = ADTValue("Cons", [1, ADTValue("Nil", [])])
        self.assertEqual(repr(v), "Cons(1, Nil)")

    def test_equality(self):
        v1 = ADTValue("Some", [5])
        v2 = ADTValue("Some", [5])
        v3 = ADTValue("Some", [6])
        v4 = ADTValue("None", [])
        self.assertEqual(v1, v2)
        self.assertNotEqual(v1, v3)
        self.assertNotEqual(v1, v4)

    def test_hashable(self):
        v1 = ADTValue("Some", [5])
        v2 = ADTValue("Some", [5])
        self.assertEqual(hash(v1), hash(v2))
        # As keys in a dict / set
        self.assertIn(v1, {v1, v2})

    def test_match_with_named_handlers(self):
        v = ADTValue("Some", [5])
        result = v.match({
            "Some": lambda x: f"got {x}",
            "None": lambda: "nothing",
        })
        self.assertEqual(result, "got 5")

    def test_match_with_underscore_fallback(self):
        v = ADTValue("Green", [])
        result = v.match({
            "Red": lambda: "red",
            "_": lambda: "other",
        })
        self.assertEqual(result, "other")

    def test_match_raises_when_non_exhaustive(self):
        v = ADTValue("Blue", [])
        with self.assertRaises(ValueError):
            v.match({"Red": lambda: "red"})


class TestADTRegistry(unittest.TestCase):
    def test_define_and_lookup(self):
        reg = ADTRegistry()
        adt = AlgebraicDataType(name="Color",
                                  variants=[Variant("Red"), Variant("Green")])
        reg.define(adt)
        self.assertIn("Color", reg)
        self.assertIs(reg.lookup("Color"), adt)

    def test_lookup_unknown_raises(self):
        reg = ADTRegistry()
        with self.assertRaises(ADTValueError):
            reg.lookup("Missing")

    def test_duplicate_definition_raises(self):
        reg = ADTRegistry()
        reg.define(AlgebraicDataType(name="X", variants=[Variant("A")]))
        with self.assertRaises(ADTValueError):
            reg.define(AlgebraicDataType(name="X",
                                            variants=[Variant("B")]))

    def test_duplicate_variant_name_across_adts_raises(self):
        reg = ADTRegistry()
        reg.define(AlgebraicDataType(name="A", variants=[Variant("Both")]))
        with self.assertRaises(ADTValueError):
            reg.define(AlgebraicDataType(name="B",
                                            variants=[Variant("Both")]))

    def test_constructor_with_explicit_adt(self):
        reg = ADTRegistry()
        adt = AlgebraicDataType(name="Option",
                                  variants=[Variant("Some", ["T"]),
                                             Variant("None", [])])
        reg.define(adt)
        v = reg.constructor("Some", 5, adt_name="Option")
        self.assertEqual(v.constructor_name, "Some")
        self.assertEqual(v.fields, [5])
        self.assertIs(v.adt, adt)

    def test_constructor_without_explicit_adt(self):
        reg = ADTRegistry()
        adt = AlgebraicDataType(name="Option",
                                  variants=[Variant("Some", ["T"]),
                                             Variant("None", [])])
        reg.define(adt)
        # Disambiguate by variant name alone — registry indexes
        # variant name → ADT name.
        v = reg.constructor("Some", 10)
        self.assertEqual(v.constructor_name, "Some")
        self.assertEqual(v.fields, [10])
        self.assertIs(v.adt, adt)

    def test_constructor_wrong_arity_raises(self):
        reg = ADTRegistry()
        reg.define(AlgebraicDataType(name="Option",
                                       variants=[Variant("Some", ["T"]),
                                                  Variant("None", [])]))
        with self.assertRaises(ADTValueError):
            reg.constructor("Some", 1, 2)
        with self.assertRaises(ADTValueError):
            reg.constructor("None", 1)

    def test_constructor_unknown_variant_raises(self):
        reg = ADTRegistry()
        reg.define(AlgebraicDataType(name="Color",
                                       variants=[Variant("Red")]))
        with self.assertRaises(ADTValueError):
            reg.constructor("Blue")

    def test_adt_for_variant(self):
        reg = ADTRegistry()
        adt = AlgebraicDataType(name="Option",
                                  variants=[Variant("Some", ["T"]),
                                             Variant("None", [])])
        reg.define(adt)
        self.assertIs(reg.adt_for_variant("Some"), adt)

    def test_iterating_registry(self):
        reg = ADTRegistry()
        reg.define(AlgebraicDataType(name="A", variants=[Variant("X")]))
        reg.define(AlgebraicDataType(name="B", variants=[Variant("Y")]))
        names = sorted(a.name for a in reg)
        self.assertEqual(names, ["A", "B"])

    def test_names_returns_sorted_set(self):
        reg = ADTRegistry()
        reg.define(AlgebraicDataType(name="Foo", variants=[Variant("A")]))
        reg.define(AlgebraicDataType(name="Bar", variants=[Variant("B")]))
        self.assertEqual(sorted(reg.names()), ["Bar", "Foo"])


class TestStandardADTs(unittest.TestCase):
    def setUp(self):
        self.std = StandardADTs()

    def test_option_adt_structure(self):
        adt = self.std.option
        self.assertEqual(adt.name, "Option")
        self.assertEqual(adt.type_params, ["T"])
        self.assertEqual(adt.variant_names, ["Some", "None"])
        self.assertEqual(adt.variant("Some").field_types, ["T"])
        self.assertEqual(adt.variant("None").arity, 0)

    def test_result_adt_structure(self):
        adt = self.std.result
        self.assertEqual(adt.name, "Result")
        self.assertEqual(adt.type_params, ["T", "E"])
        self.assertEqual(adt.variant_names, ["Ok", "Err"])
        self.assertEqual(adt.variant("Ok").field_types, ["T"])
        self.assertEqual(adt.variant("Err").field_types, ["E"])

    def test_either_adt_structure(self):
        adt = self.std.either
        self.assertEqual(adt.name, "Either")
        self.assertEqual(adt.variant_names, ["Left", "Right"])
        self.assertEqual(adt.variant("Left").field_types, ["L"])
        self.assertEqual(adt.variant("Right").field_types, ["R"])

    def test_list_adt_structure(self):
        adt = self.std.list
        self.assertEqual(adt.name, "List")
        self.assertEqual(adt.type_params, ["T"])
        self.assertEqual(adt.variant_names, ["Cons", "Nil"])
        self.assertEqual(adt.variant("Cons").field_types, ["T", "List"])
        self.assertEqual(adt.variant("Nil").arity, 0)

    def test_construct_some(self):
        v = self.std.registry.constructor("Some", 42)
        self.assertEqual(v.constructor_name, "Some")
        self.assertEqual(v.fields, [42])

    def test_construct_none(self):
        v = self.std.registry.constructor("None")
        self.assertEqual(v.constructor_name, "None")
        self.assertEqual(v.fields, [])

    def test_construct_ok(self):
        v = self.std.registry.constructor("Ok", "value")
        self.assertEqual(v.constructor_name, "Ok")
        self.assertEqual(v.fields, ["value"])

    def test_construct_err(self):
        v = self.std.registry.constructor("Err", ValueError("oops"))
        self.assertEqual(v.constructor_name, "Err")
        self.assertEqual(v.fields[0].args[0], "oops")

    def test_construct_cons_recursive(self):
        v1 = self.std.registry.constructor("Nil")
        v2 = self.std.registry.constructor("Cons", 1, v1)
        v3 = self.std.registry.constructor("Cons", 2, v2)
        # 3 → 2 → Nil
        self.assertEqual(v3.fields, [2, ADTValue("Cons",
                                                   [1, ADTValue("Nil", [])])])


class TestADTValueMatching(unittest.TestCase):
    def test_option_match_some(self):
        std = StandardADTs()
        v = std.registry.constructor("Some", 7)
        result = v.match({
            "Some": lambda x: f"got {x}",
            "None": lambda: "empty",
        })
        self.assertEqual(result, "got 7")

    def test_option_match_none(self):
        std = StandardADTs()
        v = std.registry.constructor("None")
        result = v.match({
            "Some": lambda x: f"got {x}",
            "None": lambda: "empty",
        })
        self.assertEqual(result, "empty")

    def test_list_match_cons_recursive(self):
        std = StandardADTs()
        v = std.registry.constructor("Cons", 1,
                                        std.registry.constructor("Nil"))
        result = v.match({
            "Cons": lambda head, tail: f"head={head}, tail={tail!r}",
            "Nil": lambda: "empty",
        })
        self.assertEqual(result, "head=1, tail=Nil")


class TestADTValueIntrospection(unittest.TestCase):
    def test_value_carries_adt_back_pointer(self):
        std = StandardADTs()
        v = std.registry.constructor("Some", 5)
        self.assertIsNotNone(v.adt)
        self.assertEqual(v.adt.name, "Option")

    def test_value_repr_no_duplicates_with_adt(self):
        std = StandardADTs()
        v = std.registry.constructor("Some", 5)
        self.assertEqual(repr(v), "Some(5)")  # adt name not in repr


if __name__ == "__main__":
    unittest.main()
