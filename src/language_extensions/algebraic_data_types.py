"""§3.1 — Algebraic data types (sum types, tagged unions).

Roadmap checkbox:

    - [ ] Algebraic data types — sum types, tagged unions

API surface:

  * `Variant` — a single constructor of an ADT, e.g.
    `Variant("Some", ["value"])`.
  * `AlgebraicDataType` — a tagged-union definition consisting of
    one or more `Variant`s, optionally parameterised by type
    parameters. e.g. `AlgebraicDataType("Option", ["T"],
    [Variant("Some", ["T"]), Variant("None", [])])`.
  * `ADTValue` — a runtime instance of an ADT variant; carries
    `constructor_name` + `fields` (positional list) + a back-pointer
    `adt`. The back-pointer lets `match` blocks verify the variant
    belongs to the expected ADT.
  * `ADTRegistry` — a project-level registry that stores
    `AlgebraicDataType` definitions by name; provides:
      - `define(adt)` — store, also indexes the variants
      - `lookup(name)` — retrieve
      - `constructor(name, args=...)` — invoke the named constructor
        of an ADT (the ADT is disambiguated by the registry, which
        assumes constructor names are unique within the registry;
        if there's ambiguity, the user supplies the ADT name
        explicitly).
  * Pre-defined ADTs:
      - `Option[T]` — `Some(T) | None`
      - `Result[T, E]` — `Ok(T) | Err(E)`
      - `Either[L, R]` — `Left(L) | Right(R)`
      - `List[T]` — `Cons(T, List[T]) | Nil`

This is a runtime library — pure Python. The parser/AST side can be
wired in later; for now, callers construct ADTs programmatically.
"""
from __future__ import annotations

import dataclasses
import typing


@dataclasses.dataclass(frozen=True)
class Variant:
    """A single constructor of an ADT. `name` is the constructor
    name (e.g. "Some", "Cons"), `field_types` is the list of inner
    field type names (length 0 for a unit variant)."""
    name: str
    field_types: typing.List[str] = dataclasses.field(default_factory=list)

    @property
    def arity(self) -> int:
        return len(self.field_types)


@dataclasses.dataclass(frozen=True)
class AlgebraicDataType:
    """A tagged-union of variants."""
    name: str
    type_params: typing.List[str] = dataclasses.field(default_factory=list)
    variants: typing.List[Variant] = dataclasses.field(default_factory=list)

    @property
    def variant_names(self) -> typing.List[str]:
        return [v.name for v in self.variants]

    def variant(self, name: str) -> Variant:
        for v in self.variants:
            if v.name == name:
                return v
        raise KeyError(f"ADT {self.name} has no variant named {name!r}")


@dataclasses.dataclass
class ADTValue:
    """A runtime instance of an ADT variant."""
    constructor_name: str
    fields: typing.List[typing.Any]
    adt: typing.Optional[AlgebraicDataType] = None

    def __eq__(self, other):
        if not isinstance(other, ADTValue):
            return False
        return (self.constructor_name == other.constructor_name
                and self.fields == other.fields)
        # NB: don't compare adt identity -- two values from the same
        # variant definitions should be equal regardless of which
        # specific ADT object produced them.

    def __hash__(self):
        return hash((self.constructor_name, tuple(self.fields)))

    def __repr__(self):
        if not self.fields:
            return f"{self.constructor_name}"
        inner = ", ".join(repr(f) for f in self.fields)
        return f"{self.constructor_name}({inner})"

    def match(self, handlers: typing.Dict[str, typing.Callable]):
        """Convenience: match this ADT value against named handlers.
        `handlers` maps constructor-name → callable receiving the
        unpacked fields (or just the fields list if the callable
        takes a single positional arg). The key `"_"` is the
        fallback handler. Raises if no handler matches.

        We use a dict (not ``**kwargs``) because ADT variant names
        can include Python keywords such as ``None`` and ``True``
        that are illegal as keyword-argument names.
        """
        if self.constructor_name in handlers:
            fn = handlers[self.constructor_name]
            return _invoke_handler(fn, self.fields)
        if "_" in handlers:
            fn = handlers["_"]
            return _invoke_handler(fn, self.fields)
        raise ValueError(
            f"Non-exhaustive match: {self.constructor_name!r} not in "
            f"{list(handlers.keys())}")


def _invoke_handler(fn, fields):
    """Invoke a handler callable. If the callable accepts *args, the
    fields are unpacked positionally; if it accepts exactly one
    positional arg, the entire fields list is passed in (so the
    handler can index/iterate)."""
    if not hasattr(fn, "__call__") and not callable(fn):
        # Treat as a static return value (rare but useful).
        return fn
    import inspect
    try:
        sig = inspect.signature(fn)
        n_pos = sum(1 for p in sig.parameters.values()
                    if p.kind in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.VAR_POSITIONAL,
                    ))
    except (TypeError, ValueError):
        n_pos = len(fields)
    if n_pos == 1 and len(fields) != 1:
        return fn(list(fields))
    if n_pos == 0 and len(fields) != 0:
        # Handler takes no args but fields are non-empty — pass list,
        # let the handler ignore it.
        return fn()
    return fn(*fields)


class ADTValueError(Exception):
    """Raised when an ADT constructor is misused."""


class ADTRegistry:
    """Project-level registry mapping ADT names to definitions
    and variant names to ADT definitions."""

    def __init__(self):
        self._adts: typing.Dict[str, AlgebraicDataType] = {}
        self._variant_to_adt: typing.Dict[str, str] = {}

    def define(self, adt: AlgebraicDataType) -> None:
        if adt.name in self._adts:
            raise ADTValueError(
                f"ADT {adt.name!r} already defined")
        self._adts[adt.name] = adt
        for v in adt.variants:
            if v.name in self._variant_to_adt:
                first = self._variant_to_adt[v.name]
                raise ADTValueError(
                    f"Variant {v.name!r} already defined by ADT "
                    f"{first!r}")
            self._variant_to_adt[v.name] = adt.name

    def lookup(self, name: str) -> AlgebraicDataType:
        try:
            return self._adts[name]
        except KeyError:
            raise ADTValueError(f"No ADT named {name!r} is registered")

    def adt_for_variant(self, variant_name: str) -> AlgebraicDataType:
        try:
            adt_name = self._variant_to_adt[variant_name]
        except KeyError:
            raise ADTValueError(
                f"No ADT variant named {variant_name!r} is registered")
        return self._adts[adt_name]

    def constructor(self, variant_name: str,
                     *args, adt_name: typing.Optional[str] = None) -> ADTValue:
        if adt_name is not None:
            adt = self.lookup(adt_name)
            variant = adt.variant(variant_name)
        else:
            # look up the variant in the global index
            adt = self.adt_for_variant(variant_name)
            variant = adt.variant(variant_name)
        if len(args) != variant.arity:
            raise ADTValueError(
                f"Variant {variant_name!r} expects {variant.arity} "
                f"arguments but {len(args)} were given")
        return ADTValue(constructor_name=variant_name,
                          fields=list(args), adt=adt)

    def __contains__(self, name: str) -> bool:
        return name in self._adts

    def __iter__(self):
        return iter(self._adts.values())

    def names(self) -> typing.List[str]:
        return list(self._adts.keys())


# ---- Pre-defined ADTs: Option, Result, Either, List ----


def make_option_adt() -> AlgebraicDataType:
    return AlgebraicDataType(
        name="Option",
        type_params=["T"],
        variants=[
            Variant("Some", ["T"]),
            Variant("None", []),
        ])

def make_result_adt() -> AlgebraicDataType:
    return AlgebraicDataType(
        name="Result",
        type_params=["T", "E"],
        variants=[
            Variant("Ok", ["T"]),
            Variant("Err", ["E"]),
        ])

def make_either_adt() -> AlgebraicDataType:
    return AlgebraicDataType(
        name="Either",
        type_params=["L", "R"],
        variants=[
            Variant("Left", ["L"]),
            Variant("Right", ["R"]),
        ])

def make_list_adt() -> AlgebraicDataType:
    return AlgebraicDataType(
        name="List",
        type_params=["T"],
        variants=[
            Variant("Cons", ["T", "List"]),
            Variant("Nil", []),
        ])


class StandardADTs:
    """Container with the standard pre-defined ADTs."""

    def __init__(self):
        self.registry = ADTRegistry()
        self.registry.define(make_option_adt())
        self.registry.define(make_result_adt())
        self.registry.define(make_either_adt())
        self.registry.define(make_list_adt())

    @property
    def option(self) -> AlgebraicDataType:
        return self.registry.lookup("Option")

    @property
    def result(self) -> AlgebraicDataType:
        return self.registry.lookup("Result")

    @property
    def either(self) -> AlgebraicDataType:
        return self.registry.lookup("Either")

    @property
    def list(self) -> AlgebraicDataType:
        return self.registry.lookup("List")


__all__ = [
    "Variant",
    "AlgebraicDataType",
    "ADTValue",
    "ADTValueError",
    "ADTRegistry",
    "StandardADTs",
    "make_option_adt",
    "make_result_adt",
    "make_either_adt",
    "make_list_adt",
]
