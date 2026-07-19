"""§8.3 — Project scalability envelope.

Roadmap ("8.3 Project Scalability"):

    - [x] Workspace support — `eigen-workspace.toml` reader +
          `Workspace` class to bundle multiple member packages.
    - [x] Monorepo support — packages are linked together through the
          shared `Workspace`; cross-package dependencies are tracked
          separately from remote (PyPI) dependencies so a
          topological monorepo build is reproducible.
    - [x] Dependency graph visualization — render Graphviz DOT
          output of the inter-package edges.
    - [x] Build DAG — topological ordering of packages so a CI build
          can fan out across cores or machines.

This is an envelope module: we keep the implementations tractable
(no actual file IO side-effects beyond reading the workspace
manifest); the heavy lifting (caching, network fetch, native
compilation) is delegated to existing packager / build hooks.
"""
from __future__ import annotations

import collections
import dataclasses
import typing


# ---------------------------------------------------------------------------
# Manifest and Package dataclasses
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Package:
    """A single workspace member package."""
    name: str
    path: str = ""
    dependencies: typing.List[str] = dataclasses.field(default_factory=list)
    description: str = ""


@dataclasses.dataclass
class WorkspaceManifest:
    """Parsed content of an `eigen-workspace.toml` file."""
    name: str
    members: typing.List[str] = dataclasses.field(default_factory=list)
    shared_dependencies: typing.List[str] = \
        dataclasses.field(default_factory=list)
    description: str = ""

    @classmethod
    def from_dict(cls, d: typing.Mapping[str, typing.Any]) \
            -> "WorkspaceManifest":
        """Construct from a dict produced by a TOML parser.

        Accepts both the "flat" form (direct keys `name`, `members`,
        etc.) and the "section" form (which is what
        `packager.parse_toml` returns: a dict with a `[workspace]`
        section).
        """
        # Normalize: if "workspace" key exists, prefer that section.
        sect = d.get("workspace", d) if isinstance(d, dict) else d
        name = sect.get("name", "")
        members_raw = sect.get("members", [])
        deps_raw = sect.get("shared_dependencies",
                              sect.get("dependencies", []))
        return cls(
            name=name,
            members=_parse_list(members_raw),
            shared_dependencies=_parse_list(deps_raw),
            description=sect.get("description", "") or "",
        )


def _parse_list(value: typing.Any) -> typing.List[str]:
    """Coerce `value` into a list of strings.

    Handles:
      - already-list inputs;
      - a string like ``"a, b, c"`` (comma-separated);
      - a TOML-pretty string like ``[\"a\", "b"]`` (regex split).
    """
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for v in value:
            if isinstance(v, str):
                out.append(v.strip())
            else:
                out.append(str(v))
        return out
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return []
        # Strip surrounding brackets
        if v.startswith("[") and v.endswith("]"):
            v = v[1:-1]
        # Split on commas (with optional surrounding quotes)
        parts = [p.strip() for p in v.split(",") if p.strip()]
        out = []
        for p in parts:
            # Strip quotes if wrapped
            if (len(p) >= 2 and ((p[0] == '"' and p[-1] == '"')
                                  or (p[0] == "'" and p[-1] == "'"))):
                p = p[1:-1]
            if p:
                out.append(p)
        return out
    return [str(value)]


# ---------------------------------------------------------------------------
# TOML reader (small, dependency-free)
# ---------------------------------------------------------------------------

def _read_toml_simple(text: str) -> dict:
    """A *very* small TOML reader that understands:

    - ``[section]`` headers;
    - ``key = value`` lines;
    - string literals ``"..."`` / ``'...'`` (stripped);
    - inline lists ``[a, b, c]`` returned as Python lists;
    - inline tables ``{a = 1, b = 2}`` returned as dicts;
    - integers as int; floats as float; booleans as bool.

    Comments (lines starting with `#`) are skipped. Inline
    comments after values are also stripped (naive: does not
    handle `#` inside quoted strings).
    """
    out: typing.Dict[str, typing.Any] = {}
    section: typing.Optional[str] = None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.rstrip().endswith("]") and "=" not in line:
            section = line[1:-1].strip()
            out.setdefault(section, {})
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        # Strip trailing comment (naive: doesn't strip quotes-containing #'s)
        if "#" in val and not (val.startswith('"') or val.startswith("'")):
            val = val.split("#", 1)[0].strip()
        # Booleans
        if val == "true":
            parsed: typing.Any = True
        elif val == "false":
            parsed = False
        elif val.startswith("[") and val.endswith("]"):
            # Inline array: parse as list
            parsed = _parse_toml_array(val)
        elif val.startswith("{") and val.endswith("}"):
            # Inline table: parse as dict (minimal)
            parsed = _parse_toml_inline_table(val)
        else:
            try:
                parsed = int(val)
            except ValueError:
                try:
                    parsed = float(val)
                except ValueError:
                    # Strip quotes if both ends quote
                    if (len(val) >= 2 and ((val[0] == '"' and val[-1] == '"')
                                             or (val[0] == "'"
                                                    and val[-1] == "'"))):
                        parsed = val[1:-1]
                    else:
                        parsed = val
        if section:
            out[section][key] = parsed
        else:
            out[key] = parsed
    return out


def _parse_toml_array(val: str) -> list:
    """Parse a TOML inline array ``[a, b, "c", 1, true]`` into a list."""
    inner = val[1:-1].strip()
    if not inner:
        return []
    result = []
    # Split on commas, respecting quotes
    parts = []
    current = ""
    in_quote = False
    quote_char = ""
    for ch in inner:
        if not in_quote and ch in ('"', "'"):
            in_quote = True
            quote_char = ch
            current += ch
        elif in_quote and ch == quote_char:
            in_quote = False
            quote_char = ""
            current += ch
        elif not in_quote and ch == ",":
            parts.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current.strip())
    for p in parts:
        p = p.strip()
        if p == "true":
            result.append(True)
        elif p == "false":
            result.append(False)
        elif (len(p) >= 2 and ((p[0] == '"' and p[-1] == '"')
                                 or (p[0] == "'" and p[-1] == "'"))):
            result.append(p[1:-1])
        else:
            try:
                result.append(int(p))
            except ValueError:
                try:
                    result.append(float(p))
                except ValueError:
                    result.append(p)
    return result


def _parse_toml_inline_table(val: str) -> dict:
    """Parse a TOML inline table ``{a = 1, b = "x"}`` into a dict."""
    inner = val[1:-1].strip()
    if not inner:
        return {}
    result = {}
    parts = []
    current = ""
    in_quote = False
    quote_char = ""
    for ch in inner:
        if not in_quote and ch in ('"', "'"):
            in_quote = True
            quote_char = ch
            current += ch
        elif in_quote and ch == quote_char:
            in_quote = False
            quote_char = ""
            current += ch
        elif not in_quote and ch == ",":
            parts.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current.strip())
    for p in parts:
        if "=" in p:
            k, _, v = p.partition("=")
            k = k.strip()
            v = v.strip()
            if (len(v) >= 2 and ((v[0] == '"' and v[-1] == '"')
                                   or (v[0] == "'" and v[-1] == "'"))):
                v = v[1:-1]
            else:
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
            result[k] = v
    return result


def parse_workspace_toml(text: str) -> WorkspaceManifest:
    """Parse a TOML string into a `WorkspaceManifest`."""
    return WorkspaceManifest.from_dict(_read_toml_simple(text))


def parse_workspace_file(path: str) -> WorkspaceManifest:
    """Parse a workspace TOML file at `path`."""
    with open(path, "r", encoding="utf-8") as f:
        return parse_workspace_toml(f.read())


# ---------------------------------------------------------------------------
# Package graph
# ---------------------------------------------------------------------------

class CircularDependencyError(ValueError):
    """Raised when a cycle is detected in the package DAG."""


class PackageGraph:
    """Directed graph of `Package` nodes, keyed by `name`. Each
    edge ``(a, b)`` represents the dependency "package a depends
    on package b"."""

    def __init__(self):
        self._packages: typing.Dict[str, Package] = {}
        # Edges: deps_of[a] = {b : a depends on b}
        self._edges: typing.Dict[str, typing.Set[str]] = \
            collections.defaultdict(set)

    def add_package(self, pkg: Package) -> None:
        if not pkg.name:
            raise ValueError("packages must have a name")
        if pkg.name in self._packages:
            raise ValueError(
                f"package {pkg.name!r} already registered")
        self._packages[pkg.name] = pkg
        self._edges.setdefault(pkg.name, set())
        for dep in pkg.dependencies:
            self._edges[pkg.name].add(dep)

    def add_dependency(self, package_name: str, dep_name: str) -> None:
        if package_name not in self._packages:
            raise ValueError(
                f"unknown package {package_name!r}")
        self._edges[package_name].add(dep_name)

    def package(self, name: str) -> typing.Optional[Package]:
        return self._packages.get(name)

    def names(self) -> typing.List[str]:
        return sorted(self._packages.keys())

    def dependencies_of(self, name: str) -> typing.Set[str]:
        return set(self._edges.get(name, set()))

    # ------------------------------------------- algorithms

    def topological_order(self) -> typing.List[str]:
        """Kahn's algorithm — returns a deterministic topological
        ordering (alphabetical tie-break). Raises
        `CircularDependencyError` if a cycle exists."""
        in_degree: typing.Dict[str, int] = \
            {n: 0 for n in self._packages}
        # For an edge ``(p, d)`` meaning "p depends on d", d must
        # be processed BEFORE p — so ``in_degree[p]`` counts the
        # number of still-unsatisfied in-workspace dependencies.
        for p in self._packages:
            for d in self._edges.get(p, ()):
                if d in self._packages:
                    in_degree[p] += 1
                # External deps (`d` not a registered package) are
                # ignored for the workspace DAG.
        # Process nodes with in-degree 0, alphabetically first
        ready = sorted([n for n, deg in in_degree.items()
                          if deg == 0])
        order: typing.List[str] = []
        while ready:
            n = ready.pop(0)
            order.append(n)
            # Decrement in_degree of all packages that depended on n
            for p in self._packages:
                if n in self._edges.get(p, set()):
                    in_degree[p] -= 1
                    if in_degree[p] == 0:
                        # Insert alphabetically into ready
                        idx = 0
                        while (idx < len(ready)
                               and ready[idx] < p):
                            idx += 1
                        ready.insert(idx, p)
        if len(order) != len(self._packages):
            cycle_nodes = sorted(
                n for n in self._packages
                if n not in order)
            raise CircularDependencyError(
                "package graph contains a cycle; involved: "
                + ", ".join(cycle_nodes))
        return order

    def detect_cycles(self) -> typing.List[str]:
        """Return a sorted list of node names that participate in
        some cycle. Empty list if no cycles."""
        try:
            self.topological_order()
            return []
        except CircularDependencyError as e:
            # Parse out the involved names from the message
            msg = str(e)
            after = msg.split("involved: ", 1)
            if len(after) == 2:
                parts = after[1].strip()
                if parts:
                    return sorted(p.strip() for p in parts.split(","))
            return []

    def layers(self) -> typing.List[typing.List[str]]:
        """Return a list of layers; layer[k] is the set of
        packages that have all their dependencies in layers 0..k-1
        (so they can build in parallel across each layer)."""
        # All nodes whose all deps are visited can go in this layer
        layers: typing.List[typing.List[str]] = []
        # First layer = nodes with no dependencies inside the workspace
        in_degree: typing.Dict[str, int] = {
            n: 0 for n in self._packages}
        for p in self._packages:
            for d in self._edges.get(p, ()):
                if d in in_degree:
                    in_degree[p] += 1
        # Build layer by layer — Kahn
        remaining = dict(in_degree)
        while remaining:
            layer = sorted([n for n, deg in remaining.items()
                              if deg == 0])
            if not layer:
                # There's a cycle - bail out
                raise CircularDependencyError(
                    "cycle detected in package graph")
            layers.append(layer)
            for n in layer:
                del remaining[n]
            # Decrement in_degree for any pkg that depends on
            # something in this layer
            for p in remaining:
                for n in layer:
                    if n in self._edges.get(p, ()):
                        remaining[p] -= 1
        return layers


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------

class Workspace:
    """A multi-package monorepo workspace."""

    def __init__(self, manifest: WorkspaceManifest):
        self.manifest = manifest
        self.graph: PackageGraph = PackageGraph()

    def add_package(self, pkg: Package) -> None:
        self.graph.add_package(pkg)

    def add_dependency(self, package_name: str, dep_name: str) -> None:
        self.graph.add_dependency(package_name, dep_name)

    def topological_order(self) -> typing.List[str]:
        return self.graph.topological_order()

    def build_layers(self) -> typing.List[typing.List[str]]:
        return self.graph.layers()

    def package_count(self) -> int:
        return len(self.graph.names())


def workspace_from_manifest(manifest: WorkspaceManifest,
                              package_loader: typing.Optional[
                                  typing.Callable[[str],
                                                    typing.Optional[Package]]]
                              = None) -> Workspace:
    """Build a `Workspace` from a parsed `WorkspaceManifest` by
    loading each member via `package_loader`. If no loader is
    supplied, the member strings themselves are treated as package
    names with no dependencies."""
    ws = Workspace(manifest)
    for member in manifest.members:
        pkg = None
        if package_loader is not None:
            pkg = package_loader(member)
        if pkg is None:
            pkg = Package(name=member, path=member)
        ws.add_package(pkg)
    return ws


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def render_dot_dag(graph: PackageGraph) -> str:
    """Render the package DAG as a Graphviz DOT-format string."""
    lines = ["digraph packages {"]
    lines.append("  rankdir=LR;")
    # First declare all nodes (so disconnected nodes still appear)
    for name in sorted(graph._packages.keys()):
        lines.append(f'  "{name}";')
    # Then declare edges
    for src in sorted(graph._packages.keys()):
        for dst in sorted(graph.dependencies_of(src)):
            if dst in graph._packages:
                lines.append(f'  "{src}" -> "{dst}";')
            else:
                # External / unknown dep — render as dashed.
                lines.append(f'  "{src}" -> "{dst}" [style=dashed];')
    lines.append("}")
    return "\n".join(lines)


def render_ascii_layers(graph: PackageGraph,
                          *,
                          indent: int = 2) -> str:
    """Render the DAG as a layered ASCII view, where each line is a
    layer and the layer's packages are listed.
    """
    layers = graph.layers()
    out: typing.List[str] = []
    for idx, layer in enumerate(layers):
        out.append(f"Layer {idx}: {' '.join(layer)}")
    return "\n".join(out)


__all__ = [
    "Package",
    "WorkspaceManifest",
    "parse_workspace_toml",
    "parse_workspace_file",
    "PackageGraph",
    "Workspace",
    "workspace_from_manifest",
    "CircularDependencyError",
    "render_dot_dag",
    "render_ascii_layers",
]
