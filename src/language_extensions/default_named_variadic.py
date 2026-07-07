"""§3.1 — Default / Named / Variadic arguments.

See `src/language_extensions/__init__.py` for the full docstring.
"""
from __future__ import annotations

import dataclasses
import enum
import typing


class ParameterKind(enum.Enum):
    POSITIONAL_OR_NAMED = "positional_or_named"
    VARIADIC = "variadic"


class FunctionSignatureError(Exception):
    """Base class for signature-validation errors."""


class ArgumentCountError(FunctionSignatureError):
    """Too many or too few positional arguments supplied."""


class UnknownKeywordArgumentError(FunctionSignatureError):
    """A keyword argument that the function signature doesn't recognise."""


class DuplicateArgumentError(FunctionSignatureError):
    """The same parameter was supplied both positionally and by keyword."""


class MissingArgumentError(FunctionSignatureError):
    """A required parameter was not supplied."""


@dataclasses.dataclass(frozen=True)
class Parameter:
    """A single formal parameter of a function signature."""
    name: str
    type_hint: typing.Optional[str] = None
    default: typing.Optional[typing.Any] = None
    has_default: bool = False
    kind: ParameterKind = ParameterKind.POSITIONAL_OR_NAMED

    def __post_init__(self):
        if self.kind == ParameterKind.VARIADIC and self.has_default:
            raise FunctionSignatureError(
                f"Variadic parameter {self.name!r} cannot have a default value"
            )


@dataclasses.dataclass(frozen=True)
class FunctionSignature:
    """Signature for a single function with default/named/variadic
    parameter support."""
    name: str
    params: typing.List[Parameter]
    return_type: typing.Optional[str] = None

    @property
    def required_count(self) -> int:
        """Number of required (no-default, non-variadic) params."""
        return sum(1 for p in self.params
                   if not p.has_default and p.kind != ParameterKind.VARIADIC)

    @property
    def has_variadic(self) -> bool:
        return any(p.kind == ParameterKind.VARIADIC for p in self.params)

    @property
    def total_params(self) -> int:
        return len(self.params)

    def param_by_name(self, name: str) -> typing.Optional[Parameter]:
        for p in self.params:
            if p.name == name:
                return p
        return None

    def param_index(self, name: str) -> int:
        for i, p in enumerate(self.params):
            if p.name == name:
                return i
        raise UnknownKeywordArgumentError(
            f"Function {self.name!r} has no parameter named {name!r}")


def _bind_positional(sig: FunctionSignature, args: list) -> typing.Tuple[dict, list]:
    """Bind positional args to params.

    For signatures that have a variadic parameter, params that come
    BEFORE the variadic fill positional slots by ordinal, the
    variadic absorbs the surplus, and any params AFTER the variadic
    are keyword-only (not fillable positionally — like Python's
    `def f(a, *rest, b=10)`).

    For non-variadic signatures, all positional slots are filled by
    ordinal. Returns (bound_dict, leftover_args).
    """
    bound = {}
    leftover = []
    if not sig.has_variadic:
        # No variadic: every positional arg maps to the corresponding
        # parameter by ordinal.
        for i, p in enumerate(sig.params):
            if i < len(args):
                bound[p.name] = args[i]
        leftover = args[len(sig.params):]
        return bound, leftover

    # With variadic: walk up to (and through) the variadic, then stop.
    var_idx = next(i for i, p in enumerate(sig.params)
                   if p.kind == ParameterKind.VARIADIC)
    pre_var = sig.params[:var_idx]
    for i, p in enumerate(pre_var):
        if i < len(args):
            bound[p.name] = args[i]
    leftover = args[len(pre_var):]
    return bound, leftover


def _is_keyword_only(sig: FunctionSignature, name: str) -> bool:
    """Params AFTER the variadic are keyword-only — they are not
    fillable positionally."""
    if not sig.has_variadic:
        return False
    var_idx = next(i for i, p in enumerate(sig.params)
                   if p.kind == ParameterKind.VARIADIC)
    for i, p in enumerate(sig.params[var_idx + 1:], start=var_idx + 1):
        if p.name == name:
            return True
    return False


def _bind_keywords(sig: FunctionSignature, kwargs: dict,
                    bound: dict) -> typing.Tuple[dict, typing.Optional[str]]:
    for k, v in kwargs.items():
        if sig.param_by_name(k) is None:
            return bound, k
        if k in bound:
            raise DuplicateArgumentError(
                f"Argument {k!r} supplied both positionally and by keyword "
                f"to {sig.name!r}")
        bound[k] = v
    return bound, None


def _mark_keyword_only_missing(sig: FunctionSignature, bound: dict) -> None:
    """If there are keyword-only params (i.e. params after a
    variadic), they must be filled by keyword OR by their default."""
    if not sig.has_variadic:
        return
    var_idx = next(i for i, p in enumerate(sig.params)
                   if p.kind == ParameterKind.VARIADIC)
    for p in sig.params[var_idx + 1:]:
        if p.name not in bound and not p.has_default:
            raise MissingArgumentError(
                f"Function {sig.name!r} keyword-only parameter {p.name!r} "
                f"was not supplied")


def _apply_defaults(sig: FunctionSignature, bound: dict) -> dict:
    for p in sig.params:
        if p.kind == ParameterKind.VARIADIC:
            continue
        if p.name not in bound:
            if p.has_default:
                bound[p.name] = p.default
            else:
                raise MissingArgumentError(
                    f"Function {sig.name!r} is missing required parameter "
                    f"{p.name!r}")
    return bound


def bind_arguments(sig: FunctionSignature, args: list, kwargs: dict) -> dict:
    """Bind the supplied positional + keyword args against the
    signature, applying defaults and collecting variadic args.
    Returns a dict mapping parameter-name → bound-value (where the
    variadic parameter maps to a list).

    Raises:
      * UnknownKeywordArgumentError — kwargs has a name not in `sig`.
      * DuplicateArgumentError — same param given positionally + by kw.
      * MissingArgumentError — a required param is unsatisfied.
      * ArgumentCountError — too many positional args and the
        function is not variadic.

    Note: with a variadic parameter, params AFTER the variadic are
    keyword-only (cannot be filled positionally). Their absence
    triggers MissingArgumentError.
    """
    if not sig.has_variadic:
        if len(args) > len(sig.params):
            raise ArgumentCountError(
                f"Function {sig.name!r} takes {len(sig.params)} positional "
                f"arguments but {len(args)} were given"
            )
    else:
        # With variadic: surplus positionals are absorbed by the
        # variadic collector. Only pre-variadic positionals count
        # against the limit.
        var_idx = next(i for i, p in enumerate(sig.params)
                       if p.kind == ParameterKind.VARIADIC)
        if len(args) > var_idx:
            # OK — surplus goes to variadic. No error.
            pass
    bound, leftover = _bind_positional(sig, list(args))
    bound, leftover_kw = _bind_keywords(sig, dict(kwargs), bound)
    if leftover_kw is not None:
        raise UnknownKeywordArgumentError(
            f"Function {sig.name!r} got an unexpected keyword argument "
            f"{leftover_kw!r}")
    _mark_keyword_only_missing(sig, bound)
    bound = _apply_defaults(sig, bound)
    if sig.has_variadic:
        var_param = next(p for p in sig.params
                         if p.kind == ParameterKind.VARIADIC)
        bound[var_param.name] = list(leftover)
    return bound


def positional_list(sig: FunctionSignature, bound: dict) -> list:
    """Convert a `bind_arguments` result back to a positional list
    suitable for VM dispatch — useful when the VM only knows how
    to call functions positionally."""
    out = []
    for p in sig.params:
        if p.name in bound:
            out.append(bound[p.name])
        else:
            if p.kind == ParameterKind.VARIADIC:
                out.append([])
            elif p.has_default:
                out.append(p.default)
    return out


def validate_call(sig: FunctionSignature, args: list, kwargs: dict) -> list:
    """Validate a call site and return the canonical positional
    argument list. Raises on signature mismatch."""
    bound = bind_arguments(sig, list(args), dict(kwargs))
    return positional_list(sig, bound)


# ---- Convenience constructors ----


def parameter(name: str, type_hint: typing.Optional[str] = None,
              *,
              default: typing.Any = None, has_default: bool = False,
              variadic: bool = False) -> Parameter:
    """Shorthand for constructing a `Parameter`."""
    return Parameter(
        name=name,
        type_hint=type_hint,
        default=default,
        has_default=has_default,
        kind=ParameterKind.VARIADIC if variadic else ParameterKind.POSITIONAL_OR_NAMED,
    )


def signature(name: str, params: typing.List[Parameter],
              return_type: typing.Optional[str] = None) -> FunctionSignature:
    return FunctionSignature(name=name, params=list(params),
                                return_type=return_type)
