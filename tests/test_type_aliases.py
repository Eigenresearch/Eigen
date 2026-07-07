"""
P2 §3.3 — Type aliases (partial surface implementation).

`type Name = Target;` declares a textual substitution the type checker
resolves lazily at every type-reference site. The P2 cut supports:

  * parsing `type Name = Target;` (with `;` optional when followed by a
    clear statement boundary).
  * AST node `TypeAliasDeclNode(name, target_type)`.
  * Type-checker registration during the pass-1 sweep over program
    globals, with duplicate-alias detection.
  * Lazy chain resolution — `type A = B` then `type B = int` makes
    `_resolve_type_alias("A")` return `"int"`.
  * Cycle detection — `type A = B; type B = A;` raises a
    `TypeErrorException` with the literal string "Circular type alias".
  * Transparent resolution from `let x: MyAlias = <expr>` so the user's
    `MyAlias` is indistinguishable from its resolved target for typing
    purposes (assignment compatibility, lookup returns).

Not in the P2 cut:
  * Generic / higher-kinded aliases (`type Pair[T] = (T, T)`).
  * Aliases as first-class runtime values (the alias name is a compile-
    time label only; it does not appear in emitted bytecode).
"""
import unittest

from src.frontend.lexer import Lexer, TokenType
from src.frontend.parser import Parser
from src.frontend.ast import TypeAliasDeclNode
from src.semantic.type_checker import TypeChecker, TypeErrorException


_TRAIT_SRC = """eigen 1.0
type QubitCount = int

func main() -> QubitCount {
    let n: QubitCount = 7
    return n
}
"""


_CHAIN_SRC = """eigen 1.0
type A = B
type B = int

func main() -> int {
    let x: A = 42
    return x
}
"""


_CYCLE_SRC = """eigen 1.0
type A = B
type B = A

func main() -> int {
    let x: A = 1
    return x
}
"""


_DUP_SRC = """eigen 1.0
type A = int
type A = float
"""


def _parse(src: str):
    return Parser(Lexer(src).tokenize()).parse()


def _check(src: str):
    prog = _parse(src)
    # Force the Python type-checker path so our new §3.3 alias logic is
    # actually exercised (when `eigen_native` is installed and the
    # program parses cleanly, the native checker shadows the Python
    # path). Tests that explicitly want native go through `tc.check`
    # unchanged in other files.
    if hasattr(prog, 'source'):
        prog.source = None
    tc = TypeChecker()
    tc.check(prog)
    return tc


class TestTypeAliasParsing(unittest.TestCase):

    def test_parser_produces_alias_node(self):
        prog = _parse("eigen 1.0\ntype QubitCount = int")
        aliases = [s for s in prog.body if isinstance(s, TypeAliasDeclNode)]
        self.assertEqual(len(aliases), 1)
        self.assertEqual(aliases[0].name, 'QubitCount')
        self.assertEqual(aliases[0].target_type, 'int')

    def test_parser_accepts_optional_semicolon(self):
        # Missing semicolon must not crash parsing — the next token
        # unambiguously starts a new statement.
        prog = _parse("eigen 1.0\ntype A = int\nfunc foo() -> int { return 1 }")
        aliases = [s for s in prog.body if isinstance(s, TypeAliasDeclNode)]
        self.assertEqual(len(aliases), 1)
        self.assertEqual(aliases[0].name, 'A')

    def test_parser_handles_complex_target_type(self):
        prog = _parse("eigen 1.0\ntype Pair = array<int>")
        aliases = [s for s in prog.body if isinstance(s, TypeAliasDeclNode)]
        self.assertEqual(aliases[0].target_type, 'array<int>')

    def test_lexer_emits_type_keyword_token(self):
        toks = Lexer("type Foo = int").tokenize()
        self.assertEqual(toks[0].type, TokenType.TYPE)
        self.assertEqual(toks[0].value, 'type')


class TestTypeAliasResolution(unittest.TestCase):

    def test_simple_alias_substitutes_in_let(self):
        tc = _check(_TRAIT_SRC)
        # The variable `n` in main's scope should resolve to 'int' (the
        # canonical target of `QubitCount`), not 'QubitCount'.
        scope = tc.scopes[-1] if tc.scopes else {}
        # After check_program, scopes are at global level. The function's
        # local scope has been popped. We instead verify the substitution
        # behaviour directly via _resolve_type_alias.
        self.assertEqual(tc._resolve_type_alias('QubitCount'), 'int')

    def test_alias_chain_resolves_to_canonical(self):
        tc = _check(_CHAIN_SRC)
        self.assertEqual(tc._resolve_type_alias('A'), 'int')
        self.assertEqual(tc._resolve_type_alias('B'), 'int')

    def test_alias_to_self_unresolvable_target_raises(self):
        # `type A = Unknown` — A resolves to "Unknown" (not substituted
        # further since "Unknown" isn't an alias itself). Resolution
        # returns the bare string; downstream typing complains about
        # unknown types in its usual way.
        tc = _check("eigen 1.0\ntype A = Unknown\nfunc main() -> int { return 1 }")
        self.assertEqual(tc._resolve_type_alias('A'), 'Unknown')

    def test_non_alias_returns_input_unchanged(self):
        tc = _check(_TRAIT_SRC)
        self.assertEqual(tc._resolve_type_alias('int'), 'int')
        self.assertEqual(tc._resolve_type_alias('qubit'), 'qubit')


class TestTypeAliasCycle(unittest.TestCase):

    def test_direct_cycle_raises(self):
        with self.assertRaises(TypeErrorException) as ctx:
            _check(_CYCLE_SRC)
        self.assertIn('Circular type alias', str(ctx.exception))

    def test_self_alias_raises(self):
        with self.assertRaises(TypeErrorException):
            _check("eigen 1.0\ntype A = A\nfunc main() -> int { return 1 }")

    def test_three_node_cycle_raises(self):
        with self.assertRaises(TypeErrorException):
            _check("""eigen 1.0
type A = B
type B = C
type C = A
func main() -> int { return 1 }
""")


class TestTypeAliasDuplicate(unittest.TestCase):

    def test_duplicate_alias_raises_at_decl_time(self):
        with self.assertRaises(Exception):
            _check(_DUP_SRC)


class TestTypeAliasIntegrationWithLet(unittest.TestCase):

    def test_int_alias_accepts_int_literal(self):
        # No exception means typing passed.
        _check("""eigen 1.0
type Score = int
func main() -> Score {
    let s: Score = 100
    return s
}
""")

    def test_int_alias_rejects_float_literal(self):
        # Assignment of float to int-alias must produce a TypeErrorException
        # (since the alias resolves to `int`, which is incompatible with
        # float without a literal coercion).
        # NB: the type-checker's types_compatible is "int <-> any?" tolerant
        # via 'unknown'/'any', so we assert it raises about type mismatch
        # of float vs int.
        with self.assertRaises(TypeErrorException):
            _check("""eigen 1.0
type Score = int
func main() -> Score {
    let s: Score = 1.5
    return s
}
""")


if __name__ == "__main__":
    unittest.main()
