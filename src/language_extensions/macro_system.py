"""§3.1 — Macro System (AST-level metaprogramming).

Roadmap checkbox:

    - [ ] Macro system — метапрограммирование на уровне AST

API surface:

  * `MacroContext` — context passed to every macro body. Exposes:
    - the caller's lexical scope variables (dict, read-only
      by convention),
    - the invocation site's source line/column for diagnostics,
    - a `compile_time_table` (the symbol table of values known
      at compile time).
  * `Macro` — a callable wrapper `(args, ctx) -> ASTNode` invoked
    at compile-time. The macro receives the *already-parsed*
    expression arguments and returns an AST that replaces the
    macro invocation.
  * `MacroTable` — a registry mapping macro names to `Macro`
    instances. Methods: `register(name, macro)`, `lookup(name)`,
    `expand(name, args, ctx)`, `__contains__`, `__iter__`.
  * `MacroExpansionError` — raised when a macro fails or recurses
    beyond a configurable depth.
  * `MacroExpander` — a visitor that walks an AST top-down and
    replaces `MacroInvocationNode`-shaped nodes with their
    expanded result. Calls `MacroTable.expand` for each node it
    finds that begins with the user-defined prefix `macro_name`.
    Recursively re-expands the result up to
    `MacroExpander.max_depth` (default 32) to support macros
    that expand to other macro calls.

We don't define a hard-coded AST `MacroInvocationNode` here —
callers can supply an arbitrary object that responds to `.is_macro`
(True), `.name`, and `.args` (iterable of AST nodes). This keeps
the module decoupled from any specific AST class.

Hygiene
-------
This is a *hygienic-bounded* macro system: macro-generated names
carry a "synthetic" suffix (`__macro_<id>_<original_name>`) so they
don't clash with caller-side bindings. The `MacroContext`
`next_temp_name()` helper generates fresh names.

Limitations
-----------
The visitor walks public `attributes` (children) of the AST node.
For AST node classes with `__slots__`, the caller must expose
the children via `__dict__` or override the visit. (We only
walk `.children()` and `.args` attributes if present.)
"""
from __future__ import annotations

import dataclasses
import typing


class MacroExpansionError(Exception):
    """Raised when a macro fails or recurses too deeply."""


@dataclasses.dataclass
class MacroContext:
    """Context handed to every macro at expansion time."""
    scope: typing.Dict[str, typing.Any]
    line: int
    column: int
    source: typing.Optional[str] = None
    _temp_counter: int = dataclasses.field(default=0, repr=False)

    def next_temp_name(self, prefix: str = "tmp") -> str:
        self._temp_counter += 1
        return f"__macro_{self._temp_counter}_{prefix}"


class Macro:
    """A single AST-level macro. The wrapped callable receives the
    macro-argument list plus the `MacroContext` and returns the
    replacement AST node."""
    def __init__(self, name: str, fn: typing.Callable):
        self.name = name
        self.fn = fn

    def __call__(self, args, ctx: MacroContext):
        return self.fn(args, ctx)


class MacroTable:
    """Registry mapping macro names to `Macro` instances."""

    def __init__(self):
        self._macros: typing.Dict[str, Macro] = {}

    def register(self, name: str, fn: typing.Callable) -> Macro:
        if name in self._macros:
            raise MacroExpansionError(f"Macro {name!r} already registered")
        m = Macro(name, fn) if not isinstance(fn, Macro) else fn
        if m.name != name:
            # Allow direct Macro instances under a different name.
            m = Macro(name, m.fn)
        self._macros[name] = m
        return m

    def lookup(self, name: str) -> Macro:
        try:
            return self._macros[name]
        except KeyError:
            raise MacroExpansionError(f"No macro named {name!r}")

    def expand(self, name: str, args: list, ctx: MacroContext):
        """Invoke the macro named `name` with `args` and `ctx`.
        Returns whatever the macro returned (typically an AST node)."""
        m = self.lookup(name)
        try:
            return m.fn(args, ctx)
        except MacroExpansionError:
            raise
        except Exception as e:
            raise MacroExpansionError(
                f"Macro {name!r} raised {type(e).__name__}: {e}") from e

    def __contains__(self, name: str) -> bool:
        return name in self._macros

    def __iter__(self):
        return iter(self._macros.values())

    def names(self) -> typing.List[str]:
        return list(self._macros.keys())


def _is_macro_invocation(node) -> bool:
    return (node is not None
            and getattr(node, "is_macro", False) is True
            and hasattr(node, "name")
            and hasattr(node, "args"))


def _iter_children(node):
    """Yield child AST nodes of `node`. We look at common attributes
    first (`.children`, `.args`, `.body`) and fall back to iterating
    over public dict-like values if needed."""
    if node is None:
        return
    # `children()` method
    if hasattr(node, "children") and callable(node.children):
        try:
            for c in node.children() or []:
                yield c
        except Exception:
            pass
        return
    # `args` list (the macro invocation case)
    if hasattr(node, "args") and isinstance(node.args, (list, tuple)):
        for c in node.args:
            yield c
    if hasattr(node, "body") and isinstance(node.body, (list, tuple)):
        for c in node.body:
            yield c


class MacroExpander:
    """AST visitor that walks a tree top-down and expands every
    `MacroInvocationNode`-shaped node it finds, recursively.

    Recurse depth is bounded by `max_depth` to detect infinite
    expansion loops (e.g. a macro that expands to a call of itself).
    """

    def __init__(self, table: MacroTable, *,
                 max_depth: int = 32):
        self.table = table
        self.max_depth = max_depth

    def expand(self, node, ctx: typing.Optional[MacroContext] = None):
        """Returns the (possibly transformed) AST. May mutate the
        original tree in place if the macro mutates the returned
        AST; that's the macro author's responsibility to avoid."""
        if ctx is None:
            ctx = MacroContext(scope={}, line=0, column=0)
        return self._walk(node, ctx, depth=0)

    def _walk(self, node, ctx, depth):
        if depth > self.max_depth:
            raise MacroExpansionError(
                f"Macro expansion exceeded max_depth={self.max_depth}")
        if _is_macro_invocation(node):
            name = node.name
            if name not in self.table:
                # Not a registered macro — return node unchanged.
                # We still walk its args (in case a child is a macro).
                return self._walk_non_macro(node, ctx, depth)
            args = [self._walk(a, ctx, depth) for a in node.args]
            new_node = self.table.expand(name, args, ctx)
            # The macro's result may itself contain macro calls or
            # be a macro call — recurse.
            return self._walk(new_node, ctx, depth + 1)
        return self._walk_non_macro(node, ctx, depth)

    def _walk_non_macro(self, node, ctx, depth):
        """Walk an ordinary (non-macro) AST node, recursing through
        its children. Mutates the node in place if it's mutable."""
        if node is None:
            return None
        # Walk children of a non-macro node. We replace each child
        # if it's a macro invocation.
        for c in list(_iter_children(node)):
            if _is_macro_invocation(c):
                replacement = self._walk(c, ctx, depth + 1)
                # Replace the reference in the parent if possible.
                _replace_child_reference(node, c, replacement)
        return node


def _replace_child_reference(parent, old_child, new_child):
    """Best-effort replacement of `old_child` with `new_child` inside
    `parent`. This depends on the AST shape; we support attributes
    named `args`, `body`, `cond`, `then`, `else`, and generic
    attributes listed via `__dict__`.

    For opaque AST classes, we can't replace — but we also can't
    get into the shape where we'd want to replace.
    """
    for attr_name in ("args", "body"):
        if hasattr(parent, attr_name):
            seq = getattr(parent, attr_name)
            if isinstance(seq, list):
                for i, x in enumerate(seq):
                    if x is old_child:
                        seq[i] = new_child
                        return
    # Generic dict-based scan
    if hasattr(parent, "__dict__"):
        for k, v in vars(parent).items():
            if v is old_child:
                setattr(parent, k, new_child)
                return
            if isinstance(v, list):
                for i, x in enumerate(v):
                    if x is old_child:
                        v[i] = new_child
                        return


# ---- Pre-defined utility macros ----


def _identity_macro(args, ctx):
    """Useful for tests and as a placeholder."""
    if not args:
        return None
    return args[0]


def _quote_macro(args, ctx):
    """`quote(body)` — returns the body unchanged (a no-op marker
    macro useful for testing the expander)."""
    if not args:
        return None
    return args[0]


def prelude_macros(table: MacroTable) -> MacroTable:
    """Register a small set of standard library macros:
      - `identity(x)` — returns x unchanged
      - `quote(body)` — returns body unchanged
    """
    table.register("identity", _identity_macro)
    table.register("quote", _quote_macro)
    return table


__all__ = [
    "MacroExpansionError",
    "MacroContext",
    "Macro",
    "MacroTable",
    "MacroExpander",
    "prelude_macros",
]
