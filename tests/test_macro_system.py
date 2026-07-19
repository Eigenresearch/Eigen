"""
Tests for src/language_extensions/macro_system.py — sol.md §3.1.
"""
import unittest

from src.language_extensions.macro_system import (
    Macro,
    MacroTable,
    MacroContext,
    MacroExpander,
    MacroExpansionError,
    prelude_macros,
)


class _MockNode:
    """Minimal AST node mock: holds an args list and a name, no
    mutation issues."""
    def __init__(self, name, args=None, is_macro=False):
        self.name = name
        self.args = args or []
        self.is_macro = is_macro


def _macro_call(name, args=None):
    return _MockNode(name, args or [], is_macro=True)


class TestMacroContext(unittest.TestCase):
    def test_construction_defaults(self):
        ctx = MacroContext(scope={}, line=1, column=1)
        self.assertEqual(ctx.scope, {})
        self.assertEqual(ctx.line, 1)

    def test_next_temp_name_increments_counter(self):
        ctx = MacroContext(scope={}, line=1, column=1)
        a = ctx.next_temp_name("x")
        b = ctx.next_temp_name("y")
        self.assertIn("x", a)
        self.assertIn("y", b)
        self.assertNotEqual(a, b)

    def test_next_temp_name_unique_per_context(self):
        ctx = MacroContext(scope={}, line=1, column=1)
        names = {ctx.next_temp_name("tmp") for _ in range(100)}
        self.assertEqual(len(names), 100)

    def test_next_temp_name_format(self):
        ctx = MacroContext(scope={}, line=1, column=1)
        n = ctx.next_temp_name("foo")
        # Format: __macro_<counter>_foo
        self.assertTrue(n.startswith("__macro_"))
        self.assertTrue(n.endswith("_foo"))


class TestMacro(unittest.TestCase):
    def test_callable(self):
        def body(args, ctx):
            return ("expansion", args)
        m = Macro("foo", body)
        result = m([1, 2], MacroContext(scope={}, line=0, column=0))
        self.assertEqual(result, ("expansion", [1, 2]))

    def test_direct_callable_via_fn(self):
        m = Macro("foo", lambda a, c: a[0])
        self.assertEqual(m([42], None), 42)


class TestMacroTable(unittest.TestCase):
    def test_register_and_lookup(self):
        table = MacroTable()
        m = table.register("foo", lambda a, c: a)
        self.assertIs(table.lookup("foo"), m)
        self.assertIn("foo", table)

    def test_lookup_unknown_raises(self):
        table = MacroTable()
        with self.assertRaises(MacroExpansionError):
            table.lookup("nonexistent")

    def test_duplicate_register_raises(self):
        table = MacroTable()
        table.register("foo", lambda a, c: a)
        with self.assertRaises(MacroExpansionError):
            table.register("foo", lambda a, c: a)

    def test_expand_invokes_macro(self):
        table = MacroTable()
        table.register("inc", lambda a, c: a[0] + 1)
        result = table.expand("inc", [5],
                               MacroContext(scope={}, line=0, column=0))
        self.assertEqual(result, 6)

    def test_expand_wraps_exceptions(self):
        table = MacroTable()
        def bad_macro(a, c):
            raise RuntimeError("boom")
        table.register("bad", bad_macro)
        with self.assertRaises(MacroExpansionError) as e:
            table.expand("bad", [], MacroContext(scope={}, line=0, column=0))
        self.assertIn("boom", str(e.exception))

    def test_iter_yields_macros(self):
        table = MacroTable()
        table.register("a", lambda a, c: a)
        table.register("b", lambda a, c: a)
        names = sorted(m.name for m in table)
        self.assertEqual(names, ["a", "b"])

    def test_names_returns_all(self):
        table = MacroTable()
        table.register("a", lambda a, c: a)
        table.register("b", lambda a, c: a)
        self.assertEqual(sorted(table.names()), ["a", "b"])


class TestMacroExpander(unittest.TestCase):
    def test_expand_simple_macro_invocation(self):
        table = MacroTable()
        table.register("double", lambda a, c: _MockNode("num",
                                                            args=[a[0].args[0] * 2]))
        expander = MacroExpander(table)
        # `(double 5)` — args[0] is a literal node with args=[5]
        call = _macro_call("double", [_MockNode("num", args=[5])])
        result = expander.expand(call, MacroContext(scope={}, line=0, column=0))
        self.assertEqual(result.name, "num")
        self.assertEqual(result.args, [10])

    def test_walk_skips_unknown_macros(self):
        table = MacroTable()
        expander = MacroExpander(table)
        call = _macro_call("not_registered", [_MockNode("x")])
        result = expander.expand(call, MacroContext(scope={}, line=0, column=0))
        # Should return the original call unchanged.
        self.assertEqual(result.name, "not_registered")

    def test_recursive_expansion_passes_subargs(self):
        table = MacroTable()
        # `double(num(5))` — first `double` receives the result of
        # expanding `num`... but `num` isn't a macro, so it stays as-is.
        # `double` doubles the inner node's value.
        table.register("double", lambda a, c:
                        _MockNode("num", args=[a[0].args[0] * 2]))
        expander = MacroExpander(table)
        inner = _MockNode("num", args=[5])
        call = _macro_call("double", [inner])
        result = expander.expand(call, MacroContext(scope={}, line=0, column=0))
        self.assertEqual(result.args, [10])

    def test_macro_expands_to_macro_expands_to_value(self):
        table = MacroTable()
        # `wrap(x)` returns macro invocation `double(x)` which
        # expands to `num(2*x)`.
        table.register("wrap", lambda a, c:
                       _macro_call("double", a))
        table.register("double", lambda a, c:
                       _MockNode("num", args=[a[0].args[0] * 2]))
        expander = MacroExpander(table)
        call = _macro_call("wrap", [_MockNode("num", args=[5])])
        result = expander.expand(call, MacroContext(scope={}, line=0, column=0))
        # Two macro levels: `wrap(num(5))` → `double(num(5))` → `num(10)`
        self.assertEqual(result.name, "num")
        self.assertEqual(result.args, [10])

    def test_max_depth_loop_detects_recursion(self):
        table = MacroTable()
        # `recursive(x)` returns `recursive(x)` — infinite expansion.
        def recursor(a, c):
            return _macro_call("recursive", a)
        table.register("recursive", recursor)
        expander = MacroExpander(table, max_depth=5)
        call = _macro_call("recursive", [_MockNode("x")])
        with self.assertRaises(MacroExpansionError) as e:
            expander.expand(call)
        self.assertIn("max_depth", str(e.exception))

    def test_expand_does_not_modify_unrelated_nodes(self):
        table = MacroTable()
        table.register("noop", lambda a, c: a[0])
        expander = MacroExpander(table)
        node = _MockNode("other", args=[1, 2, 3])
        result = expander.expand(node, MacroContext(scope={}, line=0, column=0))
        self.assertIs(result, node)


class TestPreludeMacros(unittest.TestCase):
    def test_prelude_registers_identity(self):
        table = MacroTable()
        prelude_macros(table)
        self.assertIn("identity", table)
        result = table.expand("identity", [42],
                                MacroContext(scope={}, line=0, column=0))
        self.assertEqual(result, 42)

    def test_prelude_registers_quote(self):
        table = MacroTable()
        prelude_macros(table)
        self.assertIn("quote", table)
        node = _MockNode("body", args=[5])
        result = table.expand("quote", [node],
                                MacroContext(scope={}, line=0, column=0))
        self.assertIs(result, node)


class TestMacroContextHygiene(unittest.TestCase):
    def test_two_macros_share_context_counter(self):
        """If we reuse the same MacroContext, both macros see the
        same shared counter — useful for hygiene across a single
        expansion."""
        ctx = MacroContext(scope={}, line=0, column=0)
        MacroTable()
        n1 = ctx.next_temp_name("a")
        n2 = ctx.next_temp_name("a")
        self.assertNotEqual(n1, n2)

    def test_separate_contexts_have_separate_counters(self):
        ctx1 = MacroContext(scope={}, line=0, column=0)
        ctx2 = MacroContext(scope={}, line=0, column=0)
        n1 = ctx1.next_temp_name("x")
        n2 = ctx2.next_temp_name("x")
        # Both start from counter=1.
        self.assertEqual(n1, n2)


class TestExpanderWithNestedAST(unittest.TestCase):
    """Verify the expander walks into nested non-macro nodes and
    finds/replaces macro invocations inside them."""

    def test_macro_replaced_inside_args_of_regular_node(self):
        table = MacroTable()
        table.register("lit", lambda a, c: _MockNode("literal",
                                                          args=[a[0].args[0]]))
        expander = MacroExpander(table)
        # outer = ordinary_call(macro_call(lit, [42]))
        outer = _MockNode("call",
                            args=[_macro_call("lit", [_MockNode("value", args=[42])])])
        result = expander.expand(outer, MacroContext(scope={}, line=0, column=0))
        # outer.args[0] should be replaced with the expansion result.
        self.assertEqual(len(result.args), 1)
        # The expansion of `lit(value(42))` is `literal(42)`.
        expanded = result.args[0]
        self.assertEqual(expanded.name, "literal")
        self.assertEqual(expanded.args, [42])

    def test_macro_replaced_inside_body_list(self):
        table = MacroTable()
        table.register("lit", lambda a, c: _MockNode("literal",
                                                          args=[a[0].args[0]]))
        outer = _MockNode("block", args=None)
        outer.body = [_macro_call("lit", [_MockNode("v", args=[7])])]
        expander = MacroExpander(table)
        result = expander.expand(outer, MacroContext(scope={}, line=0, column=0))
        self.assertEqual(len(result.body), 1)
        self.assertEqual(result.body[0].name, "literal")
        self.assertEqual(result.body[0].args, [7])


class TestMacroErrorWrapping(unittest.TestCase):
    def test_expansion_error_for_unknown_macro_via_expander(self):
        table = MacroTable()
        expander = MacroExpander(table)
        # Unknown macro stays as-is (per spec).
        node = _macro_call("unknown", [_MockNode("x")])
        result = expander.expand(node)
        self.assertIs(result, node)
        self.assertEqual(result.name, "unknown")

    def test_expansion_error_for_failing_macro(self):
        table = MacroTable()
        def bad(a, c):
            raise ValueError("intentional failure")
        table.register("bad", bad)
        with self.assertRaises(MacroExpansionError) as e:
            table.expand("bad", [], MacroContext(scope={}, line=0, column=0))
        self.assertIn("intentional failure", str(e.exception))
        self.assertIn("bad", str(e.exception))


if __name__ == "__main__":
    unittest.main()
