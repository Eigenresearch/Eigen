"""
Tests for src/language_extensions/module_system.py — sol.md §3.1.
"""
import unittest

from src.language_extensions.module_system import (
    Module,
    ModuleVisibility,
    ModuleVisibilityError,
    ModuleLookupError,
    CircularReExportError,
    ModuleRegistry,
)


class TestModule(unittest.TestCase):
    def test_basic_define_get(self):
        m = Module("foo.bar")
        m.define("x", 42)
        sym = m.get("x")
        self.assertEqual(sym.name, "x")
        self.assertEqual(sym.value, 42)
        self.assertEqual(sym.visibility, ModuleVisibility.PUBLIC)

    def test_define_with_visibility(self):
        m = Module("foo")
        m.define("secret", "hush", visibility=ModuleVisibility.PRIVATE)
        sym = m.get("secret")
        self.assertEqual(sym.visibility, ModuleVisibility.PRIVATE)

    def test_get_unknown_raises(self):
        m = Module("foo")
        with self.assertRaises(ModuleLookupError):
            m.get("missing")

    def test_has(self):
        m = Module("foo")
        m.define("x", 5)
        self.assertTrue(m.has("x"))
        self.assertFalse(m.has("y"))

    def test_export_names_only_exported(self):
        m = Module("foo")
        m.define("pub", 1, visibility=ModuleVisibility.PUBLIC)
        m.define("priv", 2, visibility=ModuleVisibility.PRIVATE)
        m.define("exp", 3, visibility=ModuleVisibility.EXPORTED)
        names = m.export_names()
        self.assertIn("pub", names)
        self.assertIn("exp", names)
        self.assertNotIn("priv", names)

    def test_re_export_names_in_all_names(self):
        m = Module("foo")
        m.define("local", 1)
        m.re_exports = ["other.module.thing"]
        all_names = set(m.all_names())
        self.assertIn("local", all_names)
        self.assertIn("thing", all_names)

    def test_get_via_re_export(self):
        m = Module("foo")
        m.re_exports = ["bar.baz.thing"]
        sym = m.get("thing")
        self.assertEqual(sym.visibility, ModuleVisibility.REEXPORTED)
        self.assertEqual(sym.origin_module, "bar.baz")
        self.assertEqual(sym.name, "thing")

    def test_private_visible_to_self(self):
        m = Module("foo")
        m.define("secret", "v", visibility=ModuleVisibility.PRIVATE)
        sym = m.get("secret", requesting_module="foo")
        self.assertEqual(sym.value, "v")

    def test_private_hidden_from_other_modules(self):
        m = Module("foo")
        m.define("secret", "v", visibility=ModuleVisibility.PRIVATE)
        with self.assertRaises(ModuleVisibilityError):
            m.get("secret", requesting_module="other")


class TestModuleRegistry(unittest.TestCase):
    def test_register_and_lookup(self):
        reg = ModuleRegistry()
        m = Module("foo.bar")
        reg.register(m)
        self.assertIs(reg.lookup("foo.bar"), m)

    def test_lookup_unknown_raises(self):
        reg = ModuleRegistry()
        with self.assertRaises(ModuleLookupError):
            reg.lookup("missing")

    def test_duplicate_register_raises(self):
        reg = ModuleRegistry()
        reg.register(Module("foo"))
        with self.assertRaises(ModuleLookupError):
            reg.register(Module("foo"))

    def test_resolve_returns_symbol(self):
        reg = ModuleRegistry()
        m = Module("foo")
        m.define("x", 5)
        reg.register(m)
        sym = reg.resolve("foo", "x")
        self.assertEqual(sym.value, 5)

    def test_resolve_via_re_export(self):
        reg = ModuleRegistry()
        # source defines "thing"
        src = Module("bar.baz")
        src.define("thing", 99, visibility=ModuleVisibility.PUBLIC)
        reg.register(src)
        # foo re-exports bar.baz::thing
        foo = Module("foo")
        foo.re_exports = ["bar.baz.thing"]
        reg.register(foo)
        # When client asks foo::thing, it should resolve through bar.baz
        sym = reg.resolve("foo", "thing")
        self.assertEqual(sym.value, 99)

    def test_resolve_qualified(self):
        reg = ModuleRegistry()
        m = Module("foo.bar")
        m.define("baz", 42)
        reg.register(m)
        sym = reg.resolve_qualified("foo.bar::baz")
        self.assertEqual(sym.value, 42)

    def test_resolve_qualified_missing_separator(self):
        reg = ModuleRegistry()
        with self.assertRaises(ModuleLookupError):
            reg.resolve_qualified("foo.bar.baz")

    def test_resolve_qualified_with_nested_post(self):
        reg = ModuleRegistry()
        m = Module("foo.bar")
        m.define("baz", 123)
        reg.register(m)
        # `foo::bar.baz` — post-:: is itself dotted.
        sym = reg.resolve_qualified("foo::bar.baz")
        self.assertEqual(sym.value, 123)

    def test_visibility_enforced_across_modules(self):
        reg = ModuleRegistry()
        secret_mod = Module("secret")
        secret_mod.define("hidden", "v", visibility=ModuleVisibility.PRIVATE)
        reg.register(secret_mod)
        caller_mod = Module("caller")
        reg.register(caller_mod)
        with self.assertRaises(ModuleVisibilityError):
            reg.resolve("secret", "hidden", requesting_module="caller")

    def test_visibility_self_access_allowed(self):
        reg = ModuleRegistry()
        secret_mod = Module("secret")
        secret_mod.define("hidden", "v", visibility=ModuleVisibility.PRIVATE)
        reg.register(secret_mod)
        sym = reg.resolve("secret", "hidden", requesting_module="secret")
        self.assertEqual(sym.value, "v")

    def test_re_export_chain_returns_modules_traversed(self):
        reg = ModuleRegistry()
        # bar.baz defines "thing" directly (no chain)
        src = Module("bar.baz")
        src.define("thing", 99)
        reg.register(src)
        foo = Module("foo")
        foo.re_exports = ["bar.baz.thing"]
        reg.register(foo)
        chain = reg.re_export_chain("foo", "thing")
        self.assertEqual(chain, ["foo", "bar.baz"])

    def test_re_export_chain_detects_cycle(self):
        reg = ModuleRegistry()
        a = Module("a")
        a.re_exports = ["b.thing"]
        b = Module("b")
        b.re_exports = ["a.thing"]
        reg.register(a)
        reg.register(b)
        with self.assertRaises(CircularReExportError):
            reg.resolve("a", "thing")

    def test_resolve_handles_circular_chain(self):
        reg = ModuleRegistry()
        a = Module("a")
        b = Module("b")
        a.re_exports = ["b.thing"]
        b.re_exports = ["a.thing"]
        reg.register(a)
        reg.register(b)
        with self.assertRaises(CircularReExportError):
            reg.resolve("a", "thing")

    def test_iterate_registry(self):
        reg = ModuleRegistry()
        reg.register(Module("a"))
        reg.register(Module("b"))
        reg.register(Module("c"))
        names = sorted(m.path for m in reg)
        self.assertEqual(names, ["a", "b", "c"])

    def test_in_operator(self):
        reg = ModuleRegistry()
        reg.register(Module("foo"))
        self.assertIn("foo", reg)
        self.assertNotIn("bar", reg)

    def test_paths_returns_registered(self):
        reg = ModuleRegistry()
        reg.register(Module("foo"))
        reg.register(Module("bar.baz"))
        self.assertEqual(sorted(reg.paths()), ["bar.baz", "foo"])


class TestModuleHierarchicalPaths(unittest.TestCase):
    """Sanity-check dotted-path module registration. We don't have a
    true nested-children API since the registry stores flat paths,
    but the dotted form supports hierarchical naming conventions."""

    def test_dotted_path_lookup(self):
        reg = ModuleRegistry()
        # Register parent
        parent = Module("foo")
        parent.define("pub_val", 1, visibility=ModuleVisibility.PUBLIC)
        reg.register(parent)
        # Register child
        child = Module("foo.bar")
        child.define("sub_val", 2, visibility=ModuleVisibility.PUBLIC)
        reg.register(child)
        self.assertEqual(reg.resolve("foo", "pub_val").value, 1)
        self.assertEqual(reg.resolve("foo.bar", "sub_val").value, 2)


class TestRealisticExample(unittest.TestCase):
    """Simulate `import std.math.{sin, cos}` with private helpers
    being hidden."""

    def test_std_math_with_private_helpers(self):
        reg = ModuleRegistry()
        std = Module("std")
        math = Module("std.math")
        std.re_exports = ["std.math.sin", "std.math.cos"]
        # `sin` is a public function; `_norm` is private.
        math.define("sin", lambda x: x, visibility=ModuleVisibility.PUBLIC)
        math.define("cos", lambda x: x, visibility=ModuleVisibility.PUBLIC)
        math.define("_norm", lambda x: x,
                     visibility=ModuleVisibility.PRIVATE)
        reg.register(std)
        reg.register(math)

        # Importer code: `import std` — re-exports bring sin/cos through:
        self.assertEqual(reg.resolve("std", "sin").value(0.5), 0.5)
        self.assertEqual(reg.resolve("std.math", "sin").value(0.5), 0.5)

        # Private helper hidden from external code:
        with self.assertRaises(ModuleVisibilityError):
            reg.resolve("std.math", "_norm", requesting_module="user_code")

        # Same module accesses it fine:
        sym = reg.resolve("std.math", "_norm",
                            requesting_module="std.math")
        self.assertEqual(sym.value(0.7), 0.7)


if __name__ == "__main__":
    unittest.main()
