"""
Tests for src/language_extensions/default_named_variadic.py — sol.md §3.1
"""
import unittest

from src.language_extensions import (
    Parameter,
    ParameterKind,
    FunctionSignature,
    FunctionSignatureError,
    ArgumentCountError,
    UnknownKeywordArgumentError,
    DuplicateArgumentError,
    MissingArgumentError,
    bind_arguments,
    positional_list,
    validate_call,
    parameter,
    signature,
)


class TestParameterClass(unittest.TestCase):
    def test_basic_positional_or_named(self):
        p = Parameter("x", "int")
        self.assertEqual(p.name, "x")
        self.assertEqual(p.type_hint, "int")
        self.assertFalse(p.has_default)
        self.assertEqual(p.kind, ParameterKind.POSITIONAL_OR_NAMED)
        self.assertEqual(p.default, None)

    def test_with_default(self):
        p = Parameter("x", "int", default=42, has_default=True)
        self.assertTrue(p.has_default)
        self.assertEqual(p.default, 42)

    def test_variadic_kind(self):
        p = Parameter("args", "int", kind=ParameterKind.VARIADIC)
        self.assertEqual(p.kind, ParameterKind.VARIADIC)

    def test_variadic_cannot_have_default(self):
        with self.assertRaises(FunctionSignatureError):
            Parameter("args", "int", default=5, has_default=True,
                       kind=ParameterKind.VARIADIC)

    def test_is_frozen(self):
        import dataclasses
        p = Parameter("x", "int")
        with self.assertRaises((dataclasses.FrozenInstanceError,
                                  AttributeError)):
            p.name = "y"


class TestFunctionSignature(unittest.TestCase):
    def test_required_count_no_defaults(self):
        sig = FunctionSignature("foo", [
            Parameter("a"), Parameter("b"), Parameter("c"),
        ])
        self.assertEqual(sig.required_count, 3)
        self.assertFalse(sig.has_variadic)

    def test_required_count_with_defaults(self):
        sig = FunctionSignature("foo", [
            Parameter("a"),
            Parameter("b", default=2, has_default=True),
            Parameter("c", default=3, has_default=True),
        ])
        self.assertEqual(sig.required_count, 1)

    def test_required_count_with_variadic(self):
        sig = FunctionSignature("foo", [
            Parameter("a"),
            parameter("rest", variadic=True),
        ])
        self.assertEqual(sig.required_count, 1)
        self.assertTrue(sig.has_variadic)

    def test_total_params(self):
        sig = FunctionSignature("foo", [
            Parameter("a"), Parameter("b"),
        ])
        self.assertEqual(sig.total_params, 2)

    def test_param_by_name_found(self):
        sig = FunctionSignature("foo", [Parameter("a", "int")])
        p = sig.param_by_name("a")
        self.assertIsNotNone(p)
        self.assertEqual(p.type_hint, "int")

    def test_param_by_name_not_found(self):
        sig = FunctionSignature("foo", [Parameter("a")])
        self.assertIsNone(sig.param_by_name("b"))

    def test_param_index_returns_ordinal(self):
        sig = FunctionSignature("foo", [Parameter("a"), Parameter("b"),
                                          Parameter("c")])
        self.assertEqual(sig.param_index("a"), 0)
        self.assertEqual(sig.param_index("b"), 1)
        self.assertEqual(sig.param_index("c"), 2)

    def test_param_index_raises_on_unknown(self):
        sig = FunctionSignature("foo", [Parameter("a")])
        with self.assertRaises(UnknownKeywordArgumentError):
            sig.param_index("z")

    def test_signature_is_frozen(self):
        import dataclasses
        sig = FunctionSignature("foo", [Parameter("a")])
        with self.assertRaises((dataclasses.FrozenInstanceError,
                                  AttributeError)):
            sig.name = "bar"


class TestBindArguments(unittest.TestCase):
    def test_pure_positional_call(self):
        sig = FunctionSignature("foo", [Parameter("a"), Parameter("b")])
        bound = bind_arguments(sig, [1, 2], {})
        self.assertEqual(bound, {"a": 1, "b": 2})

    def test_pure_keyword_call(self):
        sig = FunctionSignature("foo", [Parameter("a"), Parameter("b")])
        bound = bind_arguments(sig, [], {"a": 1, "b": 2})
        self.assertEqual(bound, {"a": 1, "b": 2})

    def test_mixed_positional_and_keyword(self):
        sig = FunctionSignature("foo", [Parameter("a"), Parameter("b")])
        bound = bind_arguments(sig, [1], {"b": 2})
        self.assertEqual(bound, {"a": 1, "b": 2})

    def test_default_supplied_when_omitted(self):
        sig = FunctionSignature("foo", [
            Parameter("a"),
            Parameter("b", default=99, has_default=True),
        ])
        bound = bind_arguments(sig, [1], {})
        self.assertEqual(bound, {"a": 1, "b": 99})

    def test_default_overridden_positionally(self):
        sig = FunctionSignature("foo", [
            Parameter("a"),
            Parameter("b", default=99, has_default=True),
        ])
        bound = bind_arguments(sig, [1, 5], {})
        self.assertEqual(bound, {"a": 1, "b": 5})

    def test_default_overridden_by_keyword(self):
        sig = FunctionSignature("foo", [
            Parameter("a"),
            Parameter("b", default=99, has_default=True),
        ])
        bound = bind_arguments(sig, [1], {"b": 42})
        self.assertEqual(bound, {"a": 1, "b": 42})

    def test_variadic_collects_surplus_positional(self):
        sig = FunctionSignature("foo", [
            Parameter("a"),
            parameter("rest", variadic=True),
        ])
        bound = bind_arguments(sig, [1, 2, 3, 4], {})
        self.assertEqual(bound["a"], 1)
        self.assertEqual(bound["rest"], [2, 3, 4])

    def test_variadic_empty_when_no_surplus(self):
        sig = FunctionSignature("foo", [
            Parameter("a"),
            parameter("rest", variadic=True),
        ])
        bound = bind_arguments(sig, [1], {})
        self.assertEqual(bound["a"], 1)
        self.assertEqual(bound["rest"], [])

    def test_variadic_only_signature(self):
        sig = FunctionSignature("foo", [parameter("rest", variadic=True)])
        bound = bind_arguments(sig, [1, 2, 3], {})
        self.assertEqual(bound, {"rest": [1, 2, 3]})
        bound = bind_arguments(sig, [], {})
        self.assertEqual(bound, {"rest": []})

    def test_missing_required_raises(self):
        sig = FunctionSignature("foo", [Parameter("a"), Parameter("b")])
        with self.assertRaises(MissingArgumentError):
            bind_arguments(sig, [1], {})

    def test_unknown_keyword_raises(self):
        sig = FunctionSignature("foo", [Parameter("a")])
        with self.assertRaises(UnknownKeywordArgumentError):
            bind_arguments(sig, [1], {"unknown": 99})

    def test_duplicate_positional_and_keyword_raises(self):
        sig = FunctionSignature("foo", [Parameter("a")])
        with self.assertRaises(DuplicateArgumentError):
            bind_arguments(sig, [1], {"a": 2})

    def test_too_many_positional_with_no_variadic(self):
        sig = FunctionSignature("foo", [Parameter("a"), Parameter("b")])
        with self.assertRaises(ArgumentCountError):
            bind_arguments(sig, [1, 2, 3], {})

    def test_too_many_positional_with_variadic_ok(self):
        sig = FunctionSignature("foo", [
            Parameter("a"),
            parameter("rest", variadic=True),
        ])
        # Should NOT raise
        bound = bind_arguments(sig, [1, 2, 3, 4, 5], {})
        self.assertEqual(bound["a"], 1)
        self.assertEqual(bound["rest"], [2, 3, 4, 5])

    def test_variadic_after_named_param(self):
        # `func foo(int a, int... rest, int b=10)`
        sig = FunctionSignature("foo", [
            Parameter("a"),
            parameter("rest", variadic=True),
            Parameter("b", default=10, has_default=True),
        ])
        bound = bind_arguments(sig, [1, 2, 3, 4], {"b": 99})
        self.assertEqual(bound["a"], 1)
        self.assertEqual(bound["rest"], [2, 3, 4])
        self.assertEqual(bound["b"], 99)


class TestPositionalList(unittest.TestCase):
    def test_round_trip_simple(self):
        sig = FunctionSignature("foo", [Parameter("a"), Parameter("b")])
        bound = bind_arguments(sig, [1, 2], {})
        pos = positional_list(sig, bound)
        self.assertEqual(pos, [1, 2])

    def test_with_default_filled(self):
        sig = FunctionSignature("foo", [
            Parameter("a"),
            Parameter("b", default=99, has_default=True),
        ])
        bound = bind_arguments(sig, [1], {})
        pos = positional_list(sig, bound)
        self.assertEqual(pos, [1, 99])

    def test_variadic_in_pos_list(self):
        sig = FunctionSignature("foo", [
            Parameter("a"),
            parameter("rest", variadic=True),
        ])
        bound = bind_arguments(sig, [1, 2, 3], {})
        pos = positional_list(sig, bound)
        self.assertEqual(pos, [1, [2, 3]])

    def test_variadic_empty_in_pos_list(self):
        sig = FunctionSignature("foo", [parameter("rest", variadic=True)])
        bound = bind_arguments(sig, [], {})
        pos = positional_list(sig, bound)
        self.assertEqual(pos, [[]])


class TestValidateCall(unittest.TestCase):
    def test_returns_positional_list_on_success(self):
        sig = FunctionSignature("foo", [
            Parameter("a"),
            Parameter("b", default=99, has_default=True),
        ])
        pos = validate_call(sig, [1], {})
        self.assertEqual(pos, [1, 99])

    def test_translates_keyword_to_positional(self):
        sig = FunctionSignature("foo", [Parameter("a"), Parameter("b")])
        pos = validate_call(sig, [1], {"b": 42})
        self.assertEqual(pos, [1, 42])

    def test_variadic_round_trip(self):
        sig = FunctionSignature("foo", [
            Parameter("a"),
            parameter("rest", variadic=True),
        ])
        pos = validate_call(sig, [1, 2, 3, 4], {})
        self.assertEqual(pos[0], 1)
        self.assertEqual(pos[1], [2, 3, 4])

    def test_raises_on_validation_failure(self):
        sig = FunctionSignature("foo", [Parameter("a")])
        with self.assertRaises(MissingArgumentError):
            validate_call(sig, [], {})


class TestConvenienceConstructors(unittest.TestCase):
    def test_parameter_with_default(self):
        p = parameter("x", "int", default=10, has_default=True)
        self.assertTrue(p.has_default)
        self.assertEqual(p.default, 10)
        self.assertEqual(p.type_hint, "int")

    def test_parameter_variadic(self):
        p = parameter("args", "int", variadic=True)
        self.assertEqual(p.kind, ParameterKind.VARIADIC)

    def test_parameter_no_type_hint(self):
        p = parameter("x")
        self.assertIsNone(p.type_hint)

    def test_signature_constructor(self):
        sig = signature("foo", [parameter("x")], return_type="int")
        self.assertEqual(sig.name, "foo")
        self.assertEqual(sig.return_type, "int")
        self.assertEqual(sig.total_params, 1)


class TestVariadicWithNoDefaultSemantics(unittest.TestCase):
    def test_only_variadic_no_required_no_defaults(self):
        sig = FunctionSignature("printf", [parameter("args", variadic=True)])
        # Should accept any number of positional args
        for n in [0, 1, 5, 100]:
            with self.subTest(n=n):
                bound = bind_arguments(sig, list(range(n)), {})
                self.assertEqual(len(bound["args"]), n)

    def test_required_plus_variadic_with_keywords(self):
        # `func foo(int a, int... rest, str b="hi")` — `b` is keyword-only
        # because it comes AFTER the variadic. Caller:
        #   foo(1, 2, 3, b="custom")  → a=1, rest=[2,3], b="custom"
        sig = FunctionSignature("foo", [
            Parameter("a"),
            parameter("rest", variadic=True),
            Parameter("b", "str", default="hi", has_default=True),
        ])
        # Call: foo(1, 2, 3, b="custom")
        bound = bind_arguments(sig, [1, 2, 3], {"b": "custom"})
        self.assertEqual(bound["a"], 1)
        self.assertEqual(bound["b"], "custom")
        self.assertEqual(bound["rest"], [2, 3])

    def test_keyword_only_param_missing_raises(self):
        sig = FunctionSignature("foo", [
            Parameter("a"),
            parameter("rest", variadic=True),
            Parameter("b"),  # keyword-only, no default
        ])
        # Caller forgot the required keyword-only `b`
        with self.assertRaises(MissingArgumentError):
            bind_arguments(sig, [1, 2, 3], {})


class TestErrorMessages(unittest.TestCase):
    def test_unknown_keyword_message_includes_name(self):
        sig = FunctionSignature("foo", [Parameter("a")])
        try:
            bind_arguments(sig, [1], {"unknown": 99})
            self.fail("expected UnknownKeywordArgumentError")
        except UnknownKeywordArgumentError as e:
            self.assertIn("unknown", str(e))
            self.assertIn("foo", str(e))

    def test_duplicate_argument_message_includes_param_name(self):
        sig = FunctionSignature("foo", [Parameter("a")])
        try:
            bind_arguments(sig, [1], {"a": 2})
            self.fail("expected DuplicateArgumentError")
        except DuplicateArgumentError as e:
            self.assertIn("'a'", str(e))

    def test_missing_argument_message_includes_param_name(self):
        sig = FunctionSignature("foo", [Parameter("a"), Parameter("b")])
        try:
            bind_arguments(sig, [], {})
            self.fail("expected MissingArgumentError")
        except MissingArgumentError as e:
            self.assertIn("'a'", str(e))

    def test_argument_count_message_includes_counts(self):
        sig = FunctionSignature("foo", [Parameter("a")])
        try:
            bind_arguments(sig, [1, 2, 3], {})
            self.fail("expected ArgumentCountError")
        except ArgumentCountError as e:
            self.assertIn("1", str(e))
            self.assertIn("3", str(e))


class TestEdgeCases(unittest.TestCase):
    def test_no_params_no_args_succeeds(self):
        sig = FunctionSignature("noop", [])
        bound = bind_arguments(sig, [], {})
        self.assertEqual(bound, {})

    def test_no_params_raises_on_args(self):
        sig = FunctionSignature("noop", [])
        with self.assertRaises(ArgumentCountError):
            bind_arguments(sig, [1], {})

    def test_no_params_raises_on_kwargs(self):
        sig = FunctionSignature("noop", [])
        with self.assertRaises(UnknownKeywordArgumentError):
            bind_arguments(sig, [], {"x": 1})

    def test_default_none_with_has_default_true(self):
        # Distinguish "no default supplied" from "default is None".
        sig = FunctionSignature("foo", [
            Parameter("x", "int", default=None, has_default=True),
        ])
        bound = bind_arguments(sig, [], {})
        self.assertEqual(bound, {"x": None})

    def test_default_none_with_has_default_false(self):
        # `default=None`, `has_default=False` means "required".
        sig = FunctionSignature("foo", [
            Parameter("x", "int", default=None, has_default=False),
        ])
        with self.assertRaises(MissingArgumentError):
            bind_arguments(sig, [], {})


if __name__ == "__main__":
    unittest.main()
