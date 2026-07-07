"""§3.1 — Module System: namespaces, visibility, re-exports.

Roadmap checkbox:

    - [ ] Module system — пространства имён, видимость, re-exports

API surface:

  * `ModuleVisibility` enum: `PUBLIC` (re-exported), `PRIVATE`
    (intra-module only), `EXPORTED` (visible outside the module
    but only via full path).
  * `ModuleSymbol` — a single symbol in a module: `(name, value,
    visibility, origin_module)`.
  * `Module` — a single module namespace, holding a dict of
    `ModuleSymbol`s plus a `re_exports` set of qualified names
    that transitively re-export the symbols of other modules.
  * `ModuleRegistry` — the project-level registry that resolves
    `import foo.bar.baz` paths to module objects. Supports:
    - `register(path, module)` — store at the dotted path.
    - `lookup(path)` — retrieve.
    - `resolve(path, name)` — resolve a fully-qualified name (`foo.bar.baz::x`).
    - `resolve_qualified("module.path::name")` — same.
    - `aliases()` — return the re-export aliases map.
  * `VisibilityError` — raised when a caller tries to read a
    private symbol from outside its defining module.

Module path semantics
--------------------
A `Module.path` is dotted (`foo.bar.baz`). Sub-modules are
stored under their parent module's path (so `foo.bar` is reachable
as `module foo { module bar { ... } }`). The `Module.path` is read
only at registration time and used to construct `resolve_*`.

Re-exports
----------
`Module.re_exports` is a list of "qualified name" strings (e.g.
`"other.module.exported_symbol"`) which are re-exposed by THIS
module so that calls like `import this_module` also make those
symbols visible. The `ModuleRegistry.resolve(path, name)` walks
the chain.

Parser / VM integration
------------------------
The actual `import` and `module` keywords are handled by the
existing lexer/parser; this module provides the runtime half so
the VM can look up symbols and enforce visibility.
"""
from __future__ import annotations

import dataclasses
import enum
import typing


class ModuleVisibility(enum.Enum):
    PUBLIC = "public"        # re-exported by default
    EXPORTED = "exported"    # visible only when fully qualified
    PRIVATE = "private"      # only visible inside the module
    REEXPORTED = "reexported"  # transitively re-exported from elsewhere


class ModuleVisibilityError(Exception):
    pass


class ModuleLookupError(Exception):
    pass


class CircularReExportError(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class ModuleSymbol:
    name: str
    value: typing.Any
    visibility: ModuleVisibility
    origin_module: typing.Optional[str] = None  # for re-exports


class Module:
    """A single module namespace with symbol table + re-exports."""

    def __init__(self, path: str, *,
                 parent: typing.Optional["Module"] = None):
        self.path = path  # dotted path, e.g. "foo.bar.baz"
        self.parent = parent
        self.symbols: typing.Dict[str, ModuleSymbol] = {}
        self.re_exports: typing.List[str] = []  # dotted qualified-names

    def define(self, name: str, value: typing.Any,
                visibility: ModuleVisibility = ModuleVisibility.PUBLIC,
                origin_module: typing.Optional[str] = None) -> ModuleSymbol:
        sym = ModuleSymbol(name=name, value=value, visibility=visibility,
                            origin_module=origin_module)
        self.symbols[name] = sym
        return sym

    def get(self, name: str,
            *, requesting_module: typing.Optional[str] = None) -> ModuleSymbol:
        # Local symbol
        if name in self.symbols:
            sym = self.symbols[name]
            # Visibility check: PRIVATE only accessible from inside self.path
            if sym.visibility == ModuleVisibility.PRIVATE:
                if requesting_module is not None and requesting_module != self.path:
                    raise ModuleVisibilityError(
                        f"Symbol {name!r} of module {self.path!r} is private; "
                        f"requested from {requesting_module!r}")
            return sym
        # Re-export lookup
        for qname in self.re_exports:
            if qname.rsplit(".", 1)[-1] == name:
                # Lazy resolution — registry will verify the re-export
                # chain elsewhere.
                return ModuleSymbol(
                    name=name, value=None,
                    visibility=ModuleVisibility.REEXPORTED,
                    origin_module=qname.rsplit(".", 1)[0])
        raise ModuleLookupError(
            f"Module {self.path!r} has no symbol named {name!r}")

    def has(self, name: str) -> bool:
        if name in self.symbols:
            return True
        for qname in self.re_exports:
            if qname.rsplit(".", 1)[-1] == name:
                return True
        return False

    def export_names(self) -> typing.List[str]:
        return [name for name, sym in self.symbols.items()
                if sym.visibility in (ModuleVisibility.PUBLIC,
                                       ModuleVisibility.EXPORTED,
                                       ModuleVisibility.REEXPORTED)]

    def all_names(self) -> typing.List[str]:
        return list(self.symbols.keys()) + [
            qname.rsplit(".", 1)[-1] for qname in self.re_exports
        ]


class ModuleRegistry:
    """Project-level registry for module path → `Module` mapping.

    Resolves qualified names like `foo.bar.baz::sym_name` to a
    `ModuleSymbol`. Enforces visibility across module boundaries.
    Detects circular re-exports.
    """

    def __init__(self):
        self._modules: typing.Dict[str, Module] = {}
        self._resolving: typing.Set[str] = set()  # cycle detection

    def register(self, module: Module) -> Module:
        if module.path in self._modules:
            raise ModuleLookupError(
                f"Module {module.path!r} is already registered")
        self._modules[module.path] = module
        return module

    def lookup(self, path: str) -> Module:
        try:
            return self._modules[path]
        except KeyError:
            raise ModuleLookupError(
                f"No module registered under path {path!r}")

    def __contains__(self, path: str) -> bool:
        return path in self._modules

    def __iter__(self):
        return iter(self._modules.values())

    def paths(self) -> typing.List[str]:
        return list(self._modules.keys())

    def resolve(self, module_path: str, name: str,
                *, requesting_module: typing.Optional[str] = None
                ) -> ModuleSymbol:
        if module_path in self._resolving:
            raise CircularReExportError(
                f"Circular re-export detected at {module_path!r}::{name!r}")
        self._resolving.add(module_path)
        try:
            module = self.lookup(module_path)
            sym = module.get(name, requesting_module=requesting_module)
            # Re-export resolution: if sym is a reexport marker, resolve
            # through the origin module.
            if sym.visibility == ModuleVisibility.REEXPORTED and \
                    sym.origin_module is not None:
                return self.resolve(sym.origin_module, name,
                                     requesting_module=requesting_module)
            return sym
        finally:
            self._resolving.discard(module_path)

    def resolve_qualified(self, qualified: str,
                            *, requesting_module: typing.Optional[str] = None
                            ) -> ModuleSymbol:
        if "::" not in qualified:
            raise ModuleLookupError(
                f"Qualified name {qualified!r} missing '::' separator")
        path, name = qualified.rsplit("::", 1)
        if "." in name:
            # Some users write "foo.bar::baz.qux"; the post-::
            # part is a nested-name. We split off the module path.
            path = path + "." + name.rsplit(".", 1)[0]
            name = name.rsplit(".", 1)[1]
        return self.resolve(path, name, requesting_module=requesting_module)

    def re_export_chain(self, module_path: str, name: str) -> typing.List[str]:
        """Return the chain of modules traversed when resolving
        `(module_path, name)` via re-exports. Useful for diagnostics."""
        chain = []
        cursor = module_path
        while cursor is not None:
            chain.append(cursor)
            module = self.lookup(cursor)
            sym = module.get(name)
            if sym.visibility != ModuleVisibility.REEXPORTED:
                return chain
            cursor = sym.origin_module
            if cursor in chain:
                chain.append(cursor)
                raise CircularReExportError(
                    f"Re-export chain has a cycle: {' -> '.join(chain)}")
        return chain


__all__ = [
    "ModuleVisibility",
    "ModuleVisibilityError",
    "ModuleLookupError",
    "CircularReExportError",
    "ModuleSymbol",
    "Module",
    "ModuleRegistry",
]
