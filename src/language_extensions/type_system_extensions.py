"""§3.3 — Type System Extensions.

Roadmap checkboxes (6 of the 7 §3.3 items — type aliases were
completed in P2):

    - [ ] Protocol/Interface types — структурная типизация
    - [ ] Union types — `int | string`
    - [ ] Optional types — `T?` синтаксис
    - [ ] Result types — `Result<T, Error>`
    - [x] Type aliases — done in P2 (`type X = Y`)
    - [ ] Const generics — параметризация по значениям
    - [ ] Higher-kinded types — HKT

This module is an API-level envelope implementing:

  * `TypeRef` — a uniform handle for type references.
  * `Protocol` — structural record of method signatures. A
    `Protocol("Dessert", methods=["calories"])` declares the
    shape; `Protocol.check(object, registry)` returns True if the
    object has every required method.
  * `UnionType` — an unordered set of `TypeRef`s; `UnionType.contains(value)`
    returns True if the value's runtime type is in the union.
  * `OptionalType` — wraps an inner type; values are accepted if
    they match the inner type OR are `None`.
  * `ResultType` — wraps two inner types: `ok` and `err`. Pairs with
    the existing `StandardADTs.Result` runtime ADT.
  * `ConstGeneric` — a type parameter pinned at a constant value
    (e.g. the `5` in `Array<int, 5>`).
  * `HigherKindedType` — abstract HKT marker; tracks the `kind`
    signature (`* -> *` or `* -> * -> *`) and a "shape binder"

Tests verify each of these constructs' semantics. The module is
self-contained; the type-checker proper can consult this registry
in later phases.
"""
from __future__ import annotations

import dataclasses
import enum
import typing


# Forward-declared
class TypeKind(enum.Enum):
    PRIMITIVE = "primitive"
    PROTOCOL = "protocol"
    UNION = "union"
    OPTIONAL = "optional"
    RESULT = "result"
    GENERIC = "generic"
    CONST_GENERIC = "const_generic"
    HKT = "hkt"


@dataclasses.dataclass(frozen=True)
class TypeRef:
    """A uniform reference to a named type, possibly with
    generic-argument `TypeRef`s."""
    name: str
    type_args: typing.List["TypeRef"] = dataclasses.field(default_factory=list)
    kind: TypeKind = TypeKind.PRIMITIVE

    def __str__(self) -> str:
        if not self.type_args:
            return self.name
        return f"{self.name}<{', '.join(str(a) for a in self.type_args)}>"

    def __repr__(self) -> str:
        return f"TypeRef({self.kind.value}: {self})"

    def __eq__(self, other):
        if not isinstance(other, TypeRef):
            return False
        return (self.name == other.name
                and self.type_args == other.type_args
                and self.kind == other.kind)

    def __hash__(self):
        return hash((self.name, tuple(self.type_args), self.kind))


@dataclasses.dataclass(frozen=True)
class Protocol:
    """Structural type defined by method signatures.

    A value satisfies the protocol if it implements every method
    by name (duck-typed). Methods are tuples of
    `(name, parameter_types: list[str], return_type: str)`.
    """
    name: str
    methods: typing.List[typing.Tuple[str, typing.List[str], str]] = dataclasses.field(default_factory=list)

    def check(self, obj) -> bool:
        """Return True iff `obj` has every required method."""
        for (name, _params, _return) in self.methods:
            if not hasattr(obj, name):
                return False
            if not callable(getattr(obj, name)):
                return False
        return True

    def check_with_registry(self, obj, registry) -> bool:
        """For protocols whose methods are known statically, the
        caller may pass a `TypeRegistry` to verify signatures,
        not just names. In this envelope, we treat the static
        signature as a hint and only verify attribute presence."""
        return self.check(obj)


@dataclasses.dataclass(frozen=True)
class UnionType:
    """A `T1 | T2 | ...` union type."""
    members: typing.List[TypeRef]

    def __str__(self) -> str:
        return " | ".join(str(m) for m in self.members)

    def contains(self, value, *, type_resolver: typing.Optional[
            typing.Callable[[str], type]] = None) -> bool:
        """Return True iff the runtime type of `value` is among the
        union members. `type_resolver` maps type names to classes
        (defaults to Python builtins)."""
        # The None singleton is allowed iff at least one member is the
        # OptionalType or NoneType.
        for member in self.members:
            if value is None and member.name in ("None", "none", "Null"):
                return True
            tname = type(value).__name__
            if member.name == tname:
                return True
            if member.name == "Any":
                return True
            if type_resolver is not None and type_resolver(member.name) is type(value):
                return True
        return False


@dataclasses.dataclass(frozen=True)
class OptionalType:
    """A `T?` type — T or None."""
    inner: TypeRef

    def __str__(self) -> str:
        return f"{self.inner}?"

    def contains(self, value) -> bool:
        if value is None:
            return True
        return type(value).__name__ == self.inner.name


@dataclasses.dataclass(frozen=True)
class ResultType:
    """`Result<T, E>` — wraps `ok: T` and `err: E` type refs."""
    ok: TypeRef
    err: TypeRef

    def __str__(self) -> str:
        return f"Result<{self.ok}, {self.err}>"

    def __eq__(self, other):
        if not isinstance(other, ResultType):
            return False
        return self.ok == other.ok and self.err == other.err


@dataclasses.dataclass(frozen=True)
class ConstGeneric:
    """`Array<int, 5>` — the `5` is a const-generic argument."""
    name: str  # "Array"
    type_arg: TypeRef  # "int"
    const_value: typing.Any  # 5

    def __str__(self) -> str:
        return f"{self.name}<{self.type_arg}, {self.const_value}>"

    def __eq__(self, other):
        if not isinstance(other, ConstGeneric):
            return False
        return (self.name == other.name
                and self.type_arg == other.type_arg
                and self.const_value == other.const_value)


# ---- Higher-kinded types ----


@dataclasses.dataclass(frozen=True)
class HigherKindedType:
    """`F` where `F : * -> *` (kind signature)."""
    name: str
    arity: int  # how many `*` it takes
    applied_args: typing.List[TypeRef] = dataclasses.field(default_factory=list)

    @property
    def kind_signature(self) -> str:
        return " -> ".join(["*"] * self.arity + ["*"])

    def apply(self, type_arg: TypeRef) -> "HigherKindedType":
        """Apply a type argument to the HKT, returning a partially
        applied HKT."""
        if len(self.applied_args) >= self.arity:
            raise TypeError(
                f"HKT {self.name!r} of arity {self.arity} fully applied")
        return HigherKindedType(
            name=self.name,
            arity=self.arity,
            applied_args=self.applied_args + [type_arg],
        )

    def is_fully_applied(self) -> bool:
        """True when this HKT has all expected arguments; once fully
        applied, it behaves as a `*` (concrete type)."""
        return len(self.applied_args) == self.arity

    def __str__(self) -> str:
        if not self.applied_args:
            return self.name
        return f"{self.name}<{', '.join(str(a) for a in self.applied_args)}>"


__all__ = [
    "TypeKind",
    "TypeRef",
    "Protocol",
    "UnionType",
    "OptionalType",
    "ResultType",
    "ConstGeneric",
    "HigherKindedType",
]
