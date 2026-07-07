"""
Tests for src/language_extensions/pattern_matching_guards.py — sol.md §3.1.
"""
import unittest

from src.language_extensions.pattern_matching_guards import (
    LiteralPattern,
    WildcardPattern,
    RangePattern,
    IsInstancePattern,
    ConstructorPattern,
    BindPattern,
    Guard,
    MatchCase,
    match_with_guards,
)


class TestLiteralPattern(unittest.TestCase):
    def test_matches_integer(self):
        p = LiteralPattern(5)
        self.assertEqual(p.try_match(5), {})

    def test_no_match_on_mismatch(self):
        p = LiteralPattern(5)
        self.assertIsNone(p.try_match(6))

    def test_matches_string(self):
        p = LiteralPattern("hello")
        self.assertEqual(p.try_match("hello"), {})

    def test_no_match_on_type_mismatch(self):
        p = LiteralPattern(5)
        self.assertIsNone(p.try_match("5"))

    def test_custom_eq_comparator(self):
        p = LiteralPattern(5, eq_cmp=lambda a, b: abs(a - b) < 0.001)
        self.assertEqual(p.try_match(5.0001), {})
        self.assertIsNone(p.try_match(6.0))


class TestWildcardPattern(unittest.TestCase):
    def test_matches_anything(self):
        p = WildcardPattern()
        for v in [None, 5, "x", [1, 2], {"k": "v"}]:
            self.assertEqual(p.try_match(v), {})


class TestRangePattern(unittest.TestCase):
    def test_inclusive_range_inside(self):
        p = RangePattern(1, 10, inclusive=True)
        self.assertEqual(p.try_match(5), {})
        self.assertEqual(p.try_match(1), {})
        self.assertEqual(p.try_match(10), {})

    def test_inclusive_range_outside(self):
        p = RangePattern(1, 10, inclusive=True)
        self.assertIsNone(p.try_match(0))
        self.assertIsNone(p.try_match(11))

    def test_exclusive_range(self):
        p = RangePattern(1, 10, inclusive=False)
        self.assertEqual(p.try_match(5), {})
        self.assertEqual(p.try_match(1), {})
        self.assertIsNone(p.try_match(10))

    def test_rejects_non_numeric(self):
        p = RangePattern(1, 10)
        self.assertIsNone(p.try_match("5"))


class TestIsInstancePattern(unittest.TestCase):
    def test_matches_int(self):
        p = IsInstancePattern("int")
        self.assertEqual(p.try_match(5), {})
        self.assertIsNone(p.try_match("5"))

    def test_matches_str(self):
        p = IsInstancePattern("str")
        self.assertEqual(p.try_match("hello"), {})
        self.assertIsNone(p.try_match(5))

    def test_matches_list(self):
        p = IsInstancePattern("list")
        self.assertEqual(p.try_match([1, 2, 3]), {})
        self.assertIsNone(p.try_match((1, 2, 3)))

    def test_custom_type_resolver(self):
        class Foo:
            pass
        p = IsInstancePattern("Foo")
        # default resolver doesn't know Foo
        self.assertIsNone(p.try_match(Foo()))
        # custom resolver provides the type
        resolver = lambda name: {"Foo": Foo}.get(name, type(None))
        self.assertEqual(p.try_match(Foo(), type_resolver=resolver), {})


class TestBindPattern(unittest.TestCase):
    def test_simple_bind_matches_anything(self):
        p = BindPattern("x")
        self.assertEqual(p.try_match(5), {"x": 5})
        self.assertEqual(p.try_match("hello"), {"x": "hello"})

    def test_bind_with_sub_pattern(self):
        p = BindPattern("x", LiteralPattern(5))
        self.assertEqual(p.try_match(5), {"x": 5})
        self.assertIsNone(p.try_match(6))


class TestConstructorPattern(unittest.TestCase):
    def test_matches_simple_adt_via_tuple(self):
        # Subject: ("Some", [5])
        p = ConstructorPattern("Some", [BindPattern("x")])
        result = p.try_match(("Some", [5]))
        self.assertEqual(result, {"x": 5})

    def test_no_match_on_different_constructor(self):
        p = ConstructorPattern("Some", [])
        self.assertIsNone(p.try_match(("None", [])))

    def test_no_match_on_wrong_arity(self):
        p = ConstructorPattern("Some", [BindPattern("x")])
        self.assertIsNone(p.try_match(("Some", [])))
        self.assertIsNone(p.try_match(("Some", [1, 2])))

    def test_nested_constructor_pattern(self):
        p = ConstructorPattern("Pair",
                                [BindPattern("a"), BindPattern("b")])
        self.assertEqual(p.try_match(("Pair", [1, 2])),
                          {"a": 1, "b": 2})

    def test_no_match_on_plain_value(self):
        p = ConstructorPattern("Some", [])
        self.assertIsNone(p.try_match(5))

    def test_adt_value_object(self):
        class ADTValue:
            def __init__(self, name, fields):
                self.constructor_name = name
                self.fields = fields
        p = ConstructorPattern("Some", [BindPattern("x")])
        result = p.try_match(ADTValue("Some", [42]))
        self.assertEqual(result, {"x": 42})


class TestGuard(unittest.TestCase):
    def test_guard_pass_returns_true(self):
        g = Guard(lambda b: b["x"] > 10)
        self.assertTrue(g.evaluate({"x": 20}))

    def test_guard_fail_returns_false(self):
        g = Guard(lambda b: b["x"] > 10)
        self.assertFalse(g.evaluate({"x": 5}))

    def test_guard_swallows_exceptions(self):
        g = Guard(lambda b: b["missing_key"] == 1)  # KeyError
        self.assertFalse(g.evaluate({}))


class TestMatchWithGuards(unittest.TestCase):
    def test_returns_body_of_first_matching_case(self):
        cases = [
            MatchCase(LiteralPattern(1), lambda b: "one"),
            MatchCase(LiteralPattern(2), lambda b: "two"),
            MatchCase(WildcardPattern(), lambda b: "other"),
        ]
        self.assertEqual(match_with_guards(1, cases), "one")
        self.assertEqual(match_with_guards(2, cases), "two")
        self.assertEqual(match_with_guards(99, cases), "other")

    def test_guard_filters_matches(self):
        cases = [
            # Match any int bound to x, but only when x > 10.
            MatchCase(
                BindPattern("x"),
                lambda b: f"big: {b['x']}",
                guard=Guard(lambda b: b["x"] > 10),
            ),
            # Fallback for everything that didn't pass the guard.
            MatchCase(WildcardPattern(), lambda b: "small"),
        ]
        self.assertEqual(match_with_guards(20, cases), "big: 20")
        self.assertEqual(match_with_guards(5, cases), "small")

    def test_default_returned_when_no_case_matches(self):
        cases = [MatchCase(LiteralPattern(1), lambda b: "one")]
        # 5 doesn't match → default returned
        sentinel = object()
        self.assertEqual(match_with_guards(5, cases, default=sentinel),
                            sentinel)

    def test_default_none(self):
        cases = [MatchCase(LiteralPattern(1), lambda b: "one")]
        self.assertIsNone(match_with_guards(99, cases))

    def test_first_match_wins(self):
        # Two cases with matching patterns; first wins.
        cases = [
            MatchCase(WildcardPattern(), lambda b: "first"),
            MatchCase(WildcardPattern(), lambda b: "second"),
        ]
        self.assertEqual(match_with_guards(5, cases), "first")

    def test_bindings_visible_in_body_and_guard(self):
        cases = [
            MatchCase(
                ConstructorPattern("Some", [BindPattern("x")]),
                lambda b: f"value={b['x']}",
                guard=Guard(lambda b: b["x"] % 2 == 0),
            ),
            MatchCase(WildcardPattern(), lambda b: "no match"),
        ]
        self.assertEqual(match_with_guards(("Some", [4]), cases), "value=4")
        self.assertEqual(match_with_guards(("Some", [5]), cases), "no match")

    def test_range_with_guard(self):
        cases = [
            MatchCase(
                RangePattern(0, 100),
                lambda b: "low",
                guard=Guard(lambda b: True),  # No-op guard
            ),
            MatchCase(RangePattern(0, 100), lambda b: "fallback-range"),
            MatchCase(WildcardPattern(), lambda b: "other"),
        ]
        self.assertEqual(match_with_guards(50, cases), "low")
        self.assertEqual(match_with_guards(-1, cases), "other")

    def test_body_invoked_only_for_matched_case(self):
        invocations = []
        cases = [
            MatchCase(LiteralPattern(1),
                       lambda b: invocations.append("one") or "one"),
            MatchCase(LiteralPattern(2),
                       lambda b: invocations.append("two") or "two"),
            MatchCase(WildcardPattern(),
                       lambda b: invocations.append("other") or "other"),
        ]
        match_with_guards(2, cases)
        # Only the second case's body runs.
        self.assertEqual(invocations, ["two"])

    def test_guarded_case_does_not_evaluate_body_if_guard_fails(self):
        invocations = []
        cases = [
            MatchCase(
                BindPattern("x"),
                lambda b: invocations.append("guarded") or "guarded",
                guard=Guard(lambda b: b["x"] < 0),
            ),
            MatchCase(WildcardPattern(),
                       lambda b: invocations.append("fallback") or "fallback"),
        ]
        match_with_guards(5, cases)
        # First case matched the pattern but guard failed; body not invoked.
        self.assertEqual(invocations, ["fallback"])

    def test_isinstance_pattern_pure(self):
        # The isinstance pattern has no binding, but the guard sees
        # an empty dict and can still evaluate successfully.
        invocations = []
        cases = [
            MatchCase(
                IsInstancePattern("int"),
                lambda b: invocations.append("int") or "int",
                guard=Guard(lambda b: True),
            ),
            MatchCase(WildcardPattern(),
                       lambda b: invocations.append("other") or "other"),
        ]
        self.assertEqual(match_with_guards(5, cases), "int")
        self.assertEqual(match_with_guards("hello", cases), "other")
        self.assertEqual(invocations, ["int", "other"])

    def test_guard_with_no_bindings_sees_empty_dict(self):
        # Pattern matches a literal but provides no binding.
        invocations = []
        cases = [
            MatchCase(
                LiteralPattern(5),
                lambda b: invocations.append(("ok", dict(b))) or "ok",
                guard=Guard(lambda b: True),
            ),
            MatchCase(WildcardPattern(),
                       lambda b: invocations.append(("else", dict(b))) or "else"),
        ]
        match_with_guards(5, cases)
        # The body should have been invoked with an empty dict.
        self.assertEqual(invocations, [("ok", {})])


class TestRealisticScenario(unittest.TestCase):
    """Mimic an ADT-style Option[T] match with guards.

    `enum Option { Some(int x), None }` — implement as `(name, [x])`.
    Match:
      case Some(x) if x > 0 → "positive"
      case Some(x) if x < 0 → "negative"
      case Some(0)          → "zero"
      case None             → "nothing"
    """

    def _classify(self, value):
        cases = [
            MatchCase(
                ConstructorPattern("Some", [BindPattern("x")]),
                lambda b: "positive",
                guard=Guard(lambda b: b["x"] > 0),
            ),
            MatchCase(
                ConstructorPattern("Some", [BindPattern("x")]),
                lambda b: "negative",
                guard=Guard(lambda b: b["x"] < 0),
            ),
            MatchCase(
                ConstructorPattern("Some", [LiteralPattern(0)]),
                lambda b: "zero",
            ),
            MatchCase(
                ConstructorPattern("None", []),
                lambda b: "nothing",
            ),
            MatchCase(WildcardPattern(), lambda b: "unmatched"),
        ]
        return match_with_guards(value, cases)

    def test_positive(self):
        self.assertEqual(self._classify(("Some", [5])), "positive")

    def test_negative(self):
        self.assertEqual(self._classify(("Some", [-3])), "negative")

    def test_zero(self):
        self.assertEqual(self._classify(("Some", [0])), "zero")

    def test_none(self):
        self.assertEqual(self._classify(("None", [])), "nothing")


if __name__ == "__main__":
    unittest.main()
