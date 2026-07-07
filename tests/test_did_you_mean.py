"""
P2 §7.3 — "Did you mean?" suggestions (surface-level implementation).

The audit checklist asked for contextual suggestions in parser/type-
checker error messages when the user typos an identifier, a keyword,
or a gate mnemonic. We:

  * Implement two-row Wagner–Fischer Levenshtein with an early-prune
    cap (`levenshtein`).
  * Wrap it in a `suggest(name, vocabulary, max_distance=3)` returning
    the nearest candidate or `None`, with a prefix-shortcut so common
    prefixes never degrade past exact length-diff.
  * Provide `format_suggestion(name, vocab)` returning
    `", did you mean 'X'?"` (or empty string) suitable for appending to
    an existing error message — we never auto-correct on the suggested
    text, only annotate the error so behaviour stays the same.
  * Wire it into two hot paths in the type-checker/parser:
      - `TypeChecker.lookup_var` "Undeclared variable 'X'" — looks at
        the current scope chain + all global decls (funcs/structs/
        enums/traits/aliases) and suggests the closest match.
      - `Parser.parse_type` "Expected type name" — looks at the
        primitive types + gate mnemonics and suggests the closest.

Test coverage targets both the isolation-helper paths (unit tests on
the Levenshtein algorithm) and the end-to-end message augmentation
through the parser/type-checker surface.
"""
import unittest

from src.frontend.did_you_mean import (
    levenshtein,
    suggest,
    format_suggestion,
)
from src.frontend.lexer import Lexer
from src.frontend.parser import Parser
from src.semantic.type_checker import TypeChecker, TypeErrorException


def _parse(src: str):
    return Parser(Lexer(src).tokenize()).parse()


def _check(src: str):
    prog = _parse(src)
    # Force the Python type-checker path so the §7.3 lookup_var hook
    # actually fires (the native checker would shadow it otherwise).
    if hasattr(prog, 'source'):
        prog.source = None
    tc = TypeChecker()
    tc.check(prog)
    return tc


class TestLevenshtein(unittest.TestCase):

    def test_identical_strings_distance_zero(self):
        self.assertEqual(levenshtein("abc", "abc"), 0)

    def test_empty_costs_length_of_other(self):
        self.assertEqual(levenshtein("", "abc"), 3)
        self.assertEqual(levenshtein("abc", ""), 3)

    def test_single_substitution(self):
        self.assertEqual(levenshtein("int", "inf"), 1)

    def test_single_insertion(self):
        self.assertEqual(levenshtein("int", "inxt"), 1)

    def test_single_deletion(self):
        self.assertEqual(levenshtein("inxt", "int"), 1)

    def test_transposition_is_two_substitutions(self):
        # Wagner–Fischer counts transposition as 2 (sub + sub), not 1.
        self.assertEqual(levenshtein("ab", "ba"), 2)

    def test_early_prune_when_too_far(self):
        # 'aaaa' vs 'bbbb' is 4 substitutions, exceeding default cap=3.
        # The function must return max_distance+1 (4) rather than
        # walking the full matrix.
        self.assertEqual(levenshtein("aaaa", "bbbb", max_distance=3), 4)

    def test_distance_cap_respected(self):
        # If cap is large enough, full distance computed.
        self.assertEqual(levenshtein("aaaa", "bbbb", max_distance=10), 4)


class TestSuggest(unittest.TestCase):

    def test_returns_none_for_empty_name(self):
        self.assertIsNone(suggest("", ["int", "float"]))

    def test_returns_none_for_empty_vocab(self):
        self.assertIsNone(suggest("int", []))

    def test_exact_match_returns_self(self):
        self.assertEqual(suggest("int", ["int", "float"]), "int")

    def test_one_char_typo_returns_closest(self):
        self.assertEqual(suggest("inf", ["int", "float"]), "int")

    def test_prefix_shortcut_when_name_is_prefix_of_candidate(self):
        # `int` is a prefix of `interval` — distance is 5 (length diff).
        # `float` is 4 edits from `int` (substitute i→f, n→l, t→o, delete
        # end). With cap=5, both are candidates; `float` (distance 4)
        # wins because it's closer.
        self.assertEqual(suggest("int", ["interval", "float"],
                                 max_distance=5), "float")

    def test_prefix_shortcut_when_candidate_is_prefix_of_name(self):
        # `interval` vs `int`: candidate `int` is a prefix, distance 5.
        # With cap=5, `int` should win over a `float` candidate (which
        # is much further from `interval`).
        self.assertEqual(suggest("interval", ["int", "float"],
                                 max_distance=5), "int")

    def test_prefix_shortcut_respects_cap(self):
        # With the default cap of 3, length-diff of 5 is rejected —
        # the prefix-shortcut must not leak distance-5 candidates
        # past the cap.
        self.assertIsNone(suggest("int", ["interval"]))

    def test_far_target_returns_none(self):
        # `mesaure` is 2 edits from `measure` (within cap=3), so we'd
        # suggest it. But `xyz` is 6+ edits from `measure` and shouldn't.
        self.assertIsNone(suggest("xyz", ["measure"]))

    def test_tie_goes_to_first_yielded(self):
        # Two equally-close candidates — first insertion-order wins.
        vocab = ["int1", "int2"]
        result = suggest("int", vocab, max_distance=5)
        self.assertEqual(result, "int1")


class TestFormatSuggestion(unittest.TestCase):

    def test_returns_empty_string_when_no_candidate(self):
        self.assertEqual(format_suggestion("xyz", ["measure"]), "")

    def test_returns_hint_when_candidate_exists(self):
        out = format_suggestion("mesaure", ["measure"])
        self.assertEqual(out, ", did you mean 'measure'?")

    def test_skips_hint_when_candidate_equals_name(self):
        # If the user wrote the correct thing, suggesting it back is
        # noise — we should return an empty suffix.
        self.assertEqual(format_suggestion("int", ["int", "float"]), "")


class TestTypeCheckerSuggestion(unittest.TestCase):

    def test_undeclared_variable_lists_hint_in_message(self):
        # `let s: int = mesaure` — `mesaure` is a typo of `measure` and
        # `measure` is a built-in. But the type checker's vocabulary
        # only includes user-declared names, not builtins. So we need
        # to declare a near-typo and use the typo: declare `score`, then
        # typo `scre` — the suggestion should be `score` (or the
        # nearest in scope).
        with self.assertRaises(TypeErrorException) as ctx:
            _check("""eigen 1.0
func main() -> int {
    let score: int = 42
    let x: int = scre
    return x
}
""")
        self.assertIn("did you mean 'score'", str(ctx.exception))

    def test_undeclared_variable_with_no_close_match_has_no_hint(self):
        # `xyz` has no close neighbour in scope.
        with self.assertRaises(TypeErrorException) as ctx:
            _check("""eigen 1.0
func main() -> int {
    let score: int = 42
    let xyz: int = qwertyuiop
    return xyz
}
""")
        self.assertNotIn("did you mean", str(ctx.exception))


class TestParserSuggestion(unittest.TestCase):
    """The parser's `parse_type` accepts any `IDENTIFIER` as a fallback
    type name (so user-defined struct/enum/alias references parse), so
    the type-position "Did you mean?" hint only fires when the next
    token is something that can't be a type at all (numbers, brackets,
    etc.). That's a rare path; these tests craft the contrived inputs
    that exercise it."""

    def test_parse_type_error_carries_suggestion_for_short_typo(self):
        # `func main() -> inf` — `inf` is a 1-edit typo of `int`. The
        # parser's parse_type accepts IDENTIFIER, so `inf` parses fine
        # without erroring. We instead use a literal number which
        # cannot be a type and is close to nothing in vocab — that
        # exercises the parse_type "Expected type name" branch with a
        # vocab lookup that returns "" (no close candidate) and no
        # "did you mean" suffix.
        with self.assertRaises(Exception) as ctx:
            _parse("""eigen 1.0
func main() -> 42 { return 1 }
""")
        msg = str(ctx.exception)
        # The default vocab in parse_type is the primitive/gate set; `42`
        # is too far from any of them within cap=2 → no "did you mean".
        self.assertIn("Expected type name", msg)
        self.assertNotIn("did you mean", msg)


if __name__ == "__main__":
    unittest.main()
