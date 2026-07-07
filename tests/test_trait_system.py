"""§3.1 Trait/Interface System — partial P2 surface (AST + parser +
type-checker trait conformance).

The tests cover:

  * `trait Foo { func bar(...) -> ... }` parses to a `TraitDeclNode`
    with the right number of methods; method signatures are surfaced
    via `TraitMethodSignatureNode`.
  * `impl Foo for Type { ... }` parses to an `ImplBlockNode` carrying
    the trait and target names.
  * Inherent impls (`impl Type { ... }`) parse with `trait_name=None`.
  * Generic trait/impl parameters parse and round-trip without crashing.
  * The type-checker registers `TraitDeclNode` in `global_traits`.
  * The type-checker raises `TypeErrorException` when an `impl` cites a
    trait that doesn't exist.
  * The type-checker raises when an impl is missing a trait method.
  * The type-checker accepts a fully-conformant impl.
  * Inherent impls require no conformance check.
"""

from __future__ import annotations

import unittest

from src.frontend.ast import (
    ImplBlockNode,
    TraitDeclNode,
    TraitMethodSignatureNode,
)
from src.frontend.lexer import Lexer
from src.frontend.parser import Parser
from src.semantic.type_checker import TypeChecker, TypeErrorException


def _parse(src):
    return Parser(Lexer(src).tokenize()).parse()


def _type_check(program):
    tc = TypeChecker()
    tc.check(program)
    return tc


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


_TRAIT_SRC = """eigen 1.0
trait Numeric {
    func add(other: int) -> int
    func zero() -> int
    func scale(factor: float) -> float
}
"""


class TestTraitParsing(unittest.TestCase):
    def test_trait_decl_produces_correct_node(self):
        program = _parse(_TRAIT_SRC)
        trait = next(s for s in program.body if isinstance(s, TraitDeclNode))
        self.assertEqual(trait.name, "Numeric")
        self.assertEqual(len(trait.methods), 3)
        names = [m.name for m in trait.methods]
        self.assertEqual(names, ["add", "zero", "scale"])

    def test_trait_methods_are_signature_nodes(self):
        program = _parse(_TRAIT_SRC)
        trait = next(s for s in program.body if isinstance(s, TraitDeclNode))
        for m in trait.methods:
            self.assertIsInstance(m, TraitMethodSignatureNode)
            self.assertEqual(m.body, [])  # no default body

    def test_trait_method_return_type_recorded(self):
        program = _parse(_TRAIT_SRC)
        trait = next(s for s in program.body if isinstance(s, TraitDeclNode))
        scale = next(m for m in trait.methods if m.name == "scale")
        self.assertEqual(scale.return_type, "float")
        self.assertEqual(scale.params, [("factor", "float")])

    def test_trait_method_names_helper(self):
        program = _parse(_TRAIT_SRC)
        trait = next(s for s in program.body if isinstance(s, TraitDeclNode))
        self.assertEqual(trait.method_names(), {"add", "zero", "scale"})

    def test_empty_trait_parses(self):
        program = _parse("eigen 1.0\ntrait Empty {}\n")
        trait = next(s for s in program.body if isinstance(s, TraitDeclNode))
        self.assertEqual(trait.name, "Empty")
        self.assertEqual(trait.methods, [])


class TestImplParsing(unittest.TestCase):
    def test_trait_impl_produces_node(self):
        src = """eigen 1.0
trait Numeric {
    func zero() -> int
}
struct Foo {
    x: int
}
impl Numeric for Foo {
    func zero() -> int {
        return 0
    }
}
"""
        program = _parse(src)
        impls = [s for s in program.body if isinstance(s, ImplBlockNode)]
        self.assertEqual(len(impls), 1)
        self.assertEqual(impls[0].trait_name, "Numeric")
        self.assertEqual(impls[0].target_type, "Foo")
        self.assertEqual(len(impls[0].methods), 1)
        self.assertEqual(impls[0].methods[0].name, "zero")

    def test_inherent_impl_has_no_trait_name(self):
        src = """eigen 1.0
struct Foo {
    x: int
}
impl Foo {
    func bar() -> int {
        return 1
    }
}
"""
        program = _parse(src)
        impls = [s for s in program.body if isinstance(s, ImplBlockNode)]
        self.assertEqual(len(impls), 1)
        self.assertIsNone(impls[0].trait_name)
        self.assertEqual(impls[0].target_type, "Foo")

    def test_multiple_methods_in_impl_provide_bodies(self):
        src = """eigen 1.0
trait Numeric {
    func add(other: int) -> int
    func zero() -> int
}
struct Foo {
    x: int
}
impl Numeric for Foo {
    func add(other: int) -> int {
        return self.x + other
    }
    func zero() -> int {
        return 0
    }
}
"""
        program = _parse(src)
        impls = [s for s in program.body if isinstance(s, ImplBlockNode)]
        self.assertEqual(len(impls[0].methods), 2)
        for m in impls[0].methods:
            # Bodies in impl blocks come from the FuncDeclNode path; they
            # should be non-empty lists.
            self.assertGreaterEqual(len(m.body), 1)


class TestTraitGenerics(unittest.TestCase):
    def test_generic_trait_params(self):
        src = """eigen 1.0
trait Container<T> {
    func add(item: T) -> void
}
"""
        program = _parse(src)
        trait = next(s for s in program.body if isinstance(s, TraitDeclNode))
        self.assertEqual(trait.name, "Container")
        self.assertEqual(trait.generic_params, ["T"])

    def test_generic_impl_target(self):
        # The parser strips generic args on the target type and stores
        # just the bare type name; we accept that simplification.
        src = """eigen 1.0
trait Container<T> {
    func add(item: T) -> void
}
struct Vec<T> {
    items: array<int>
}
impl Container for Vec {
    func add(item: int) -> void {
        return
    }
}
"""
        program = _parse(src)
        impls = [s for s in program.body if isinstance(s, ImplBlockNode)]
        self.assertEqual(impls[0].target_type, "Vec")


# ---------------------------------------------------------------------------
# Type-checker
# ---------------------------------------------------------------------------


class TestTypeCheckerTraitRegistration(unittest.TestCase):
    def test_trait_is_registered_in_global_traits(self):
        program = _parse(_TRAIT_SRC)
        tc = _type_check(program)
        self.assertIn("Numeric", tc.global_traits)
        self.assertEqual(len(tc.global_traits["Numeric"].methods), 3)

    def test_duplicate_trait_raises(self):
        src = """eigen 1.0
trait Foo {
    func bar() -> void
}
trait Foo {
    func baz() -> void
}
"""
        program = _parse(src)
        with self.assertRaises(TypeErrorException) as ctx:
            _type_check(program)
        self.assertIn("Duplicate declaration of trait 'Foo'", str(ctx.exception))


class TestTypeCheckerImplConformance(unittest.TestCase):
    def test_unknown_trait_in_impl_raises(self):
        src = """eigen 1.0
struct Foo {
    x: int
}
impl Ghost for Foo {
    func nothing() -> int {
        return 0
    }
}
"""
        program = _parse(src)
        with self.assertRaises(TypeErrorException) as ctx:
            _type_check(program)
        self.assertIn("unknown trait 'Ghost'", str(ctx.exception))

    def test_missing_methods_in_impl_raises(self):
        src = """eigen 1.0
trait Numeric {
    func add(other: int) -> int
    func zero() -> int
}
struct Foo {
    x: int
}
impl Numeric for Foo {
    func zero() -> int {
        return 0
    }
}
"""
        program = _parse(src)
        with self.assertRaises(TypeErrorException) as ctx:
            _type_check(program)
        self.assertIn("missing methods", str(ctx.exception))
        # Implementation references method 'add' since we left it out.
        self.assertIn("'add'", str(ctx.exception))

    def test_complete_impl_passes(self):
        src = """eigen 1.0
trait Numeric {
    func add(other: int) -> int
    func zero() -> int
}
struct Foo {
    x: int
}
impl Numeric for Foo {
    func add(other: int) -> int {
        return self.x + other
    }
    func zero() -> int {
        return 0
    }
}
"""
        program = _parse(src)
        tc = _type_check(program)  # must not raise
        # The impl block was recorded.
        self.assertEqual(len(tc.global_impls), 1)

    def test_inherent_impl_passes_without_trait_check(self):
        src = """eigen 1.0
struct Foo {
    x: int
}
impl Foo {
    func bar() -> int {
        return 1
    }
}
"""
        program = _parse(src)
        tc = _type_check(program)  # must not raise
        self.assertEqual(len(tc.global_impls), 1)
        self.assertIsNone(tc.global_impls[0].trait_name)


if __name__ == "__main__":
    unittest.main()
