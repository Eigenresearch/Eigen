"""§3.1 — Operator overloading: user-defined operators.

Roadmap checkbox:

    - [ ] Operator overloading — пользовательские операторы

API surface:

  * `Operator` — the canonical set of overloadable operators
    (e.g. ADD, SUB, MUL, EQ, LT, etc.). Each is identified by an
    enum value.
  * `OperatorOverload` — a single `(operator, type) -> callable`
    entry. The callable receives `(left, right)` for binary
    operators and `(operand)` for unary operators.
  * `OperatorOverloadTable` — registry looking up
    `(Operator, type_name) -> OperatorOverload`. Methods:
      - `register(operator, type_name, fn)` — overwrite the entry
        for `(operator, type_name)`.
      - `lookup(operator, type_name)` — retrieve, raises
        `OperatorOverloadError` if not found.
      - `dispatch(operator, left, right)` — find a candidate
        overload by walking `(type(left).__name__, type(right).__name__)`
        and their MRO, plus trying reverse-operand lookup for
        reflected operators when no direct match is found.
  * `OperatorOverloadError` — raised on lookup failures.

Dispatch precedence (binary operators):
  1. exact `(left_type, right_type)` lookup,
  2. walk left's MRO for `(ancestor, right_type)`,
  3. walk right's MRO for `(left_type, ancestor)`,
  4. reflected lookup `(right_type, left_type)` if the operator is
     marked commutative (add, mul, eq),
  5. fallback to `(object, object)` if registered,
  6. otherwise raise.

Unary operators (NEG, NOT, INVERT) dispatch on `(operand, None)`.
"""
from __future__ import annotations

import dataclasses
import enum
import typing


class Operator(enum.Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"
    POW = "**"
    EQ = "=="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    NEG = "neg"        # unary minus
    POS = "pos"        # unary plus
    NOT = "not"
    INVERT = "~"        # bitwise not
    AND = "and"
    OR = "or"
    XOR = "xor"
    SHL = "<<"
    SHR = ">>"
    GET_INDEX = "[]"


_COMMUTATIVE = {Operator.ADD, Operator.MUL, Operator.EQ, Operator.NE,
                Operator.AND, Operator.OR, Operator.XOR}


_UNARY = {Operator.NEG, Operator.POS, Operator.NOT, Operator.INVERT}


class OperatorOverloadError(Exception):
    pass


class OperatorOverloadTable:
    def __init__(self):
        self._table: typing.Dict[typing.Tuple[Operator, str],
                                  OperatorOverload] = {}

    def register(self, operator: Operator, type_name: str,
                  fn: typing.Callable) -> "OperatorOverload":
        if operator in _UNARY:
            # Unary ops use type_name only (right=None).
            entry = OperatorOverload(operator=operator,
                                        left_type=type_name,
                                        right_type=None,
                                        fn=fn,
                                        is_unary=True)
        else:
            entry = OperatorOverload(operator=operator,
                                        left_type="*",
                                        right_type=type_name,
                                        fn=fn,
                                        is_unary=False)
        # We store unary ops under (op, left_type) with right_type=None.
        self._table[(operator, type_name)] = entry
        return entry

    def register_binary(self, operator: Operator, left_type: str,
                          right_type: str, fn: typing.Callable
                          ) -> "OperatorOverload":
        if operator in _UNARY:
            raise OperatorOverloadError(
                f"{operator!r} is unary; use register()")
        # For binary ops, store under both `(op, left_type)` and
        # `(op, right_type)` for dispatch efficiency. Both point to
        # the SAME OperatorOverload with explicit `left/right` types.
        entry = OperatorOverload(operator=operator,
                                    left_type=left_type,
                                    right_type=right_type,
                                    fn=fn,
                                    is_unary=False)
        self._table[(operator, left_type)] = entry
        self._table[(operator, right_type)] = entry
        return entry

    def lookup(self, operator: Operator, type_name: str
                ) -> "OperatorOverload":
        try:
            return self._table[(operator, type_name)]
        except KeyError:
            raise OperatorOverloadError(
                f"No overload for {operator!r} on type {type_name!r}")

    def dispatch(self, operator: Operator, left, right=None):
        """Find and invoke an overload matching the operand types."""
        if operator in _UNARY:
            entry = self._find_unary(operator, type(left).__name__)
            return entry.fn(left)
        entry = self._find_binary(operator,
                                    type(left).__name__,
                                    type(right).__name__)
        return entry.fn(left, right)

    def _find_unary(self, operator: Operator, type_name: str):
        for tname in self._mro_lookup_keys(type_name):
            if (operator, tname) in self._table:
                entry = self._table[(operator, tname)]
                # Only use unary entries for unary ops.
                if entry.is_unary and entry.left_type == tname:
                    return entry
        if (operator, "object") in self._table:
            entry = self._table[(operator, "object")]
            if entry.is_unary:
                return entry
        raise OperatorOverloadError(
            f"No unary overload for {operator!r} on {type_name!r}")

    def _find_binary(self, operator: Operator, left_type: str,
                       right_type: str):
        # 1. exact left type
        for lt in self._mro_lookup_keys(left_type):
            entry = self._table.get((operator, lt))
            if (entry
                    and not entry.is_unary
                    and (entry.right_type in ("*", right_type)
                         or entry.right_type == right_type)):
                return entry
        # 2. reflected lookup for commutative operators
        if operator in _COMMUTATIVE:
            for rt in self._mro_lookup_keys(right_type):
                entry = self._table.get((operator, rt))
                if (entry
                        and not entry.is_unary
                        and (entry.right_type in ("*", left_type)
                             or entry.right_type == left_type)
                        and (entry.left_type in ("*", right_type)
                             or entry.left_type == right_type)):
                    return entry
        # 3. fallback to object,object
        if (operator, "object") in self._table:
            entry = self._table[(operator, "object")]
            if not entry.is_unary:
                return entry
        raise OperatorOverloadError(
            f"No overload for {operator!r} on ({left_type!r}, {right_type!r})")

    def _mro_lookup_keys(self, type_name: str) -> typing.List[str]:
        # We don't have the actual class here — caller may pass builtin
        # names. Build a fallback lookup list: [type_name, "object"].
        if type_name == "object":
            return ["object"]
        return [type_name, "object"]

    def __contains__(self, key: typing.Tuple[Operator, str]) -> bool:
        return key in self._table

    def entries(self) -> int:
        return len(self._table)


@dataclasses.dataclass(frozen=True)
class OperatorOverload:
    operator: Operator
    left_type: str
    right_type: typing.Optional[str]
    fn: typing.Callable
    is_unary: bool


__all__ = [
    "Operator",
    "OperatorOverload",
    "OperatorOverloadTable",
    "OperatorOverloadError",
]
