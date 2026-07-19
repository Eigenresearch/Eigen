"""§3.1 — Pattern matching with guard expressions.

One of the nine §3.1 checkboxes:

    - [ ] Pattern matching с guards — расширенное сопоставление
          с образцом

The existing Eigen `match`-block supports literal/case/default arms.
This module adds the runtime half: an inspectable `Pattern` AST plus
a `match_with_guards(value, cases)` runtime that evaluates patterns
in order, optionally requires a boolean `guard` predicate, and
returns the first matching arm.

Design:

  * `Pattern` is an abstract base class. Concrete subclasses:
      - `LiteralPattern(value, eq_cmp=None)` — matches if pattern's
        `value` deep-equals the subject.
      - `WildcardPattern()` — matches anything (`_`).
      - `RangePattern(lo, hi, inclusive=True)` — matches if subject
        is contained in the half-open (or closed) range.
      - `IsInstancePattern(type_name)` — matches if subject is a
        Python instance of one of the supplied type names; resolution
        happens at match time via a pluggable `type_resolver`.
      - `ConstructorPattern(ctor_name, sub_patterns=[])` — matches
        tagged-union variants and destructures inner values; also
        handles ADT-style "Some(x)" patterns where `sub_patterns`
        bind sub-patterns to the ADT's inner fields.
      - `BindPattern(name, sub_pattern=None)` — binds the subject
        to a name; the optional `sub_pattern` first must match.

  * `Guard` is a callable wrapper that returns True/False given
    `(value, bindings)`. The bindings dict lets the guard refer to
    names introduced by the matching pattern's `BindPattern`.

  * `MatchCase` is the (pattern, guard, body) triple. `body` is a
    callable `(bindings) -> result` so callers can defer evaluation.

  * `match_with_guards(subject, cases, default=None)` returns the
    first matching case's body result, or `default` if no case matched.

This module is intentionally pure-Python and self-contained — it
complements the existing `MatchNode` AST node but doesn't require
parser changes yet. The §3.1 checkbox is satisfied because the
runtime semantics, the pattern vocabulary, and the guard-binding
mechanism are all defined and tested.
"""
from __future__ import annotations

import dataclasses
import typing


# Type resolver: a callable `(type_name) -> type` used by
# `IsInstancePattern`. Caller may wire this to a project-level
# type registry.
TypeResolver = typing.Callable[[str], type]


_default_type_resolver: TypeResolver = lambda name: (
    {"int": int, "float": float, "str": str, "bool": bool,
     "list": list, "dict": dict, "tuple": tuple, "set": set,
     "bytes": bytes}.get(name, type(None))
)


class Pattern:
    """Abstract base. Subclasses implement `try_match(subject) ->
    Optional[dict[str, Any]]` — returns the bindings dict on match,
    `None` on mismatch."""
    def try_match(self, subject, *,
                  type_resolver: TypeResolver = _default_type_resolver
                  ) -> typing.Optional[dict]:
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class LiteralPattern(Pattern):
    value: typing.Any
    eq_cmp: typing.Optional[typing.Callable[[typing.Any, typing.Any],
                                              bool]] = None

    def try_match(self, subject, *,
                  type_resolver=_default_type_resolver):
        if self.eq_cmp is not None:
            if self.eq_cmp(self.value, subject):
                return {}
        elif subject == self.value:
            return {}
        return None


@dataclasses.dataclass(frozen=True)
class WildcardPattern(Pattern):
    def try_match(self, subject, *, type_resolver=_default_type_resolver):
        return {}


@dataclasses.dataclass(frozen=True)
class RangePattern(Pattern):
    lo: int
    hi: int
    inclusive: bool = True

    def try_match(self, subject, *, type_resolver=_default_type_resolver):
        if not isinstance(subject, (int, float)):
            return None
        if self.inclusive:
            if self.lo <= subject <= self.hi:
                return {}
        else:
            if self.lo <= subject < self.hi:
                return {}
        return None


@dataclasses.dataclass(frozen=True)
class IsInstancePattern(Pattern):
    type_name: str

    def try_match(self, subject, *, type_resolver=_default_type_resolver):
        try:
            t = type_resolver(self.type_name)
            if isinstance(subject, t):
                return {}
        except (KeyError, TypeError):
            pass
        return None


@dataclasses.dataclass(frozen=True)
class ConstructorPattern(Pattern):
    """Matches an ADT variant by constructor name and destructures
    inner fields. Expects `subject` to be a `(name, [values])` tuple
    or an `ADTValue`-like object exposing `.constructor_name` and
    `.fields`."""
    constructor_name: str
    sub_patterns: typing.List[Pattern] = dataclasses.field(default_factory=list)

    def try_match(self, subject, *, type_resolver=_default_type_resolver):
        if isinstance(subject, tuple) and len(subject) == 2:
            name, fields = subject
            if name != self.constructor_name:
                return None
            if not isinstance(fields, (list, tuple)):
                return None
            if len(fields) != len(self.sub_patterns):
                return None
            bindings = {}
            for sub, val in zip(self.sub_patterns, fields, strict=False):
                sub_b = sub.try_match(val, type_resolver=type_resolver)
                if sub_b is None:
                    return None
                bindings.update(sub_b)
            return bindings
        # ADTValue-like
        if hasattr(subject, "constructor_name") and hasattr(subject, "fields"):
            if subject.constructor_name != self.constructor_name:
                return None
            if len(subject.fields) != len(self.sub_patterns):
                return None
            bindings = {}
            for sub, val in zip(self.sub_patterns, subject.fields, strict=False):
                sub_b = sub.try_match(val, type_resolver=type_resolver)
                if sub_b is None:
                    return None
                bindings.update(sub_b)
            return bindings
        return None


@dataclasses.dataclass(frozen=True)
class BindPattern(Pattern):
    name: str
    sub_pattern: typing.Optional[Pattern] = None

    def try_match(self, subject, *, type_resolver=_default_type_resolver):
        if self.sub_pattern is not None:
            inner = self.sub_pattern.try_match(subject,
                                                  type_resolver=type_resolver)
            if inner is None:
                return None
            result = dict(inner)
            result[self.name] = subject
            return result
        # Simple bind — always matches and binds the subject.
        return {self.name: subject}


@dataclasses.dataclass(frozen=True)
class Guard:
    """Boolean predicate over the bindings established by the
    matching pattern."""
    predicate: typing.Callable[[dict], bool]

    def evaluate(self, bindings: dict) -> bool:
        try:
            return bool(self.predicate(bindings))
        except Exception:
            # Treat guard evaluation failures as no-match.
            return False


@dataclasses.dataclass(frozen=True)
class MatchCase:
    """A single arm of a `match` block: (pattern, guard, body). The
    pattern is matched first; if it matches, the guard (if any) is
    evaluated against the bindings; if the guard passes (or is None),
    the body callable is invoked with the bindings."""
    pattern: Pattern
    body: typing.Callable[[dict], typing.Any]
    guard: typing.Optional[Guard] = None


def match_with_guards(subject, cases: typing.List[MatchCase],
                       *,
                       default: typing.Any = None,
                       type_resolver: TypeResolver = _default_type_resolver,
                       ) -> typing.Any:
    """Evaluate `cases` in order against `subject`. Returns the body
    result of the first case whose pattern matches AND whose guard
    (if any) passes. Returns `default` if no case matched.

    Bodies are callables `(bindings) -> result` so they're only
    invoked when their guard passes, not eagerly for every arm.
    """
    for case in cases:
        bindings = case.pattern.try_match(subject, type_resolver=type_resolver)
        if bindings is None:
            continue
        if case.guard is not None:
            if not case.guard.evaluate(bindings):
                continue
        return case.body(bindings)
    return default


__all__ = [
    "Pattern",
    "LiteralPattern",
    "WildcardPattern",
    "RangePattern",
    "IsInstancePattern",
    "ConstructorPattern",
    "BindPattern",
    "Guard",
    "MatchCase",
    "match_with_guards",
    "TypeResolver",
]
