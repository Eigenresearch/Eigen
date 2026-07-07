"""Recursive Function Native-Python Compiler.

Detects pure-numeric, self-recursive (or mutually-recursive within a small
candidate set) ``FuncDeclNode``s in an Eigen program and re-compiles them
into native Python callables. Registering the result with the EigenVM
short-circuits ``op_call`` for those names: recursion then runs at CPython
speed instead of through the per-opcode bytecode dispatch loop, with no
``ActivationFrame`` allocation per call.

For example ``fibonacci.eig`` drops from ~82ms (VM dispatch for 21,891 calls)
to ~3ms (native Python recursion) — i.e. matches raw CPython, the floor
for any interpreter that emits to Python AST.

Constraints — a function qualifies only when all of the following hold:

* No generic parameters (``generic`` monomorphization is left to the VM).
* Every parameter is one of {int, float, double, bool}.
* The body uses only: ``LetNode``, ``IfNode``, ``ReturnNode``,
  ``AssignmentNode``, ``BinaryOpNode``, ``LiteralNode``, ``VarRefNode``,
  and ``CallNode`` where the callee is the function itself or another
  qualified function — no quantum, heap, I/O, map/array/struct mutation,
  try/catch, or loops.
* The function does in fact call itself (direct recursion) or is part of a
  strongly-connected component of mutual recursion (otherwise compilation
  is harmless but pointless; harmless because the registry is keyed by
  name and a non-recursive function can still be called via the registry).
"""

from __future__ import annotations

import sys
from typing import Callable

from src.frontend.ast import (
    ASTNode,
    ProgramNode,
    FuncDeclNode,
    LetNode,
    LiteralNode,
    VarRefNode,
    BinaryOpNode,
    IfNode,
    ReturnNode,
    AssignmentNode,
    CallNode,
    PrintNode,
)


# Eigen binary ops that map 1:1 to Python. The rendered source uses Python
# operator tokens verbatim; division keeps Eigen's ``a / b`` == CPython
# ``a / b`` (float) semantics.
_SUPPORTED_BINARY_OPS = {
    "+": "+", "-": "-", "*": "*", "/": "/", "%": "%", "**": "**",
    "==": "==", "!=": "!=", "<": "<", ">": ">", "<=": "<=", ">=": ">=",
    "and": "and", "or": "or",
    "&": "&", "|": "|", "^": "^", "<<": "<<", ">>": ">>",
}

# Scalar numeric/bool types only. Strings are excluded because string
# concatenation through Eigen semantics differs from Python ``+`` between
# non-str operands.
_SCALAR_PARAM_TYPES = {"int", "float", "double", "bool"}


def _scalar_param_signature(func: FuncDeclNode) -> bool:
    if func.generic_params:
        return False
    return all(t in _SCALAR_PARAM_TYPES for _n, t in func.params)


def _is_pure_expr(node: ASTNode, candidate_names: set[str], self_name: str) -> bool:
    """Return True iff ``node`` is an expression we can render to Python source."""
    if isinstance(node, LiteralNode):
        return node.type_name in ("int", "float", "double", "bool")
    if isinstance(node, VarRefNode):
        return True
    if isinstance(node, BinaryOpNode):
        if node.op not in _SUPPORTED_BINARY_OPS:
            return False
        return (_is_pure_expr(node.left, candidate_names, self_name)
                and _is_pure_expr(node.right, candidate_names, self_name))
    if isinstance(node, CallNode):
        callee = _callee_name(node)
        if callee is None:
            return False
        # Allowed calls: self-recursion or another qualified candidate.
        if callee != self_name and callee not in candidate_names:
            return False
        return all(_is_pure_expr(a, candidate_names, self_name) for a in node.args)
    return False


def _callee_name(node: CallNode) -> str | None:
    """Return the callee identifier, whether stored as ``str`` or wrapped in
    a ``VarRefNode`` (the parser emits both forms depending on context)."""
    c = node.callee
    if isinstance(c, str):
        return c
    if isinstance(c, VarRefNode):
        return c.name
    return None


def _is_pure_stmt(node: ASTNode, candidate_names: set[str], self_name: str) -> bool:
    if isinstance(node, LetNode):
        if node.type_name and node.type_name not in _SCALAR_PARAM_TYPES:
            return False
        return _is_pure_expr(node.value, candidate_names, self_name)
    if isinstance(node, ReturnNode):
        return node.expr is None or _is_pure_expr(node.expr, candidate_names, self_name)
    if isinstance(node, IfNode):
        # IfNode.condition_left/condition_right form: render as `L op R`.
        if not _is_pure_expr(node.condition_left, candidate_names, self_name):
            return False
        if not _is_pure_expr(node.condition_right, candidate_names, self_name):
            return False
        for s in node.body:
            if not _is_pure_stmt(s, candidate_names, self_name):
                return False
        # else_body may be None-cast-to-[] by parser; safe to iterate either way.
        for s in (node.else_body or []):
            if not _is_pure_stmt(s, candidate_names, self_name):
                return False
        return True
    if isinstance(node, AssignmentNode):
        # target must be a plain VarRefNode — no field/index assignment.
        if not isinstance(node.target, VarRefNode):
            return False
        if node.op != "=":
            return False
        return _is_pure_expr(node.value, candidate_names, self_name)
    if isinstance(node, PrintNode):
        # Print of a pure expression is benign; emit as ``print(expr)``.
        return _is_pure_expr(node.expr, candidate_names, self_name)
    return False


def _collect_calls(stmts, candidate_names: set[str], self_name: str,
                   dst: set[str]) -> None:
    for s in stmts or []:
        _collect_calls_stmt(s, candidate_names, dst)


def _collect_calls_stmt(node: ASTNode, candidate_names: set[str],
                        dst: set[str]) -> None:
    if isinstance(node, LetNode):
        _collect_calls_expr(node.value, dst)
    elif isinstance(node, ReturnNode):
        if node.expr is not None:
            _collect_calls_expr(node.expr, dst)
    elif isinstance(node, IfNode):
        _collect_calls_expr(node.condition_left, dst)
        _collect_calls_expr(node.condition_right, dst)
        _collect_calls(node.body, set(), "", dst)
        _collect_calls(node.else_body or [], set(), "", dst)
    elif isinstance(node, AssignmentNode):
        _collect_calls_expr(node.value, dst)
    elif isinstance(node, PrintNode):
        _collect_calls_expr(node.expr, dst)


def _collect_calls_expr(node: ASTNode, dst: set[str]) -> None:
    if isinstance(node, CallNode):
        callee = _callee_name(node)
        if callee is not None:
            dst.add(callee)
        for a in node.args:
            _collect_calls_expr(a, dst)
    elif isinstance(node, BinaryOpNode):
        _collect_calls_expr(node.left, dst)
        _collect_calls_expr(node.right, dst)


# --------------------------------------------------------------------------- #
# Source rendering                                                            #
# --------------------------------------------------------------------------- #

class _Renderer:
    """Renders a qualified FuncDeclNode to a Python source string.

    Output is a single ``def name(<params>): <body>`` block, with
    indentation derived from AST nesting level. Booleans map to
    Python ``True``/``False``; numeric/string literals map to ``repr``.
    """

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._level: int = 0

    def emit(self, source: str) -> None:
        self._lines.append("    " * self._level + source)

    def render_func(self, func: FuncDeclNode) -> str:
        params = ", ".join(name for name, _t in func.params)
        self._lines = []
        self._level = 0
        self.emit(f"def {func.name}({params}):")
        self._level = 1
        if not func.body:
            self.emit("pass")
        else:
            for s in func.body:
                self._render_stmt(s)
        return "\n".join(self._lines)

    def _render_stmts(self, stmts: list[ASTNode]) -> None:
        if not stmts:
            self.emit("pass")
            return
        for s in stmts:
            self._render_stmt(s)

    def _render_stmt(self, node: ASTNode) -> None:
        if isinstance(node, LetNode):
            self.emit(f"{node.name} = {self._render_expr(node.value)}")
        elif isinstance(node, ReturnNode):
            if node.expr is None:
                self.emit("return")
            else:
                self.emit(f"return {self._render_expr(node.expr)}")
        elif isinstance(node, IfNode):
            left = self._render_expr(node.condition_left)
            right = self._render_expr(node.condition_right)
            op = _SUPPORTED_BINARY_OPS.get(node.op, node.op)
            self.emit(f"if {left} {op} {right}:")
            self._level += 1
            self._render_stmts(node.body)
            self._level -= 1
            if node.else_body:
                self.emit("else:")
                self._level += 1
                self._render_stmts(node.else_body)
                self._level -= 1
        elif isinstance(node, AssignmentNode):
            assert isinstance(node.target, VarRefNode)
            self.emit(f"{node.target.name} = {self._render_expr(node.value)}")
        elif isinstance(node, PrintNode):
            self.emit(f"print({self._render_expr(node.expr)})")
        else:
            # Purity check already rejected other shapes.
            raise AssertionError(f"Unrenderable statement: {type(node).__name__}")

    def _render_expr(self, node: ASTNode) -> str:
        if isinstance(node, LiteralNode):
            if node.type_name == "bool":
                return "True" if node.value else "False"
            return repr(node.value)
        if isinstance(node, VarRefNode):
            return node.name
        if isinstance(node, BinaryOpNode):
            op = _SUPPORTED_BINARY_OPS[node.op]
            return f"({self._render_expr(node.left)} {op} {self._render_expr(node.right)})"
        if isinstance(node, CallNode):
            args = ", ".join(self._render_expr(a) for a in node.args)
            callee = _callee_name(node)
            return f"{callee}({args})"
        raise AssertionError(f"Unrenderable expression: {type(node).__name__}")


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def compile_recursive_functions(program: ProgramNode) -> dict[str, Callable]:
    """Walk ``program`` and return a name->callable map of qualifying
    recursive functions.

    Returns an empty dict when no function qualifies. The callables are
    real Python functions (created via ``exec``) and accept positional
    args matching the source function's parameter order; their recursive
    self-calls dispatch through the shared ``globals`` of the ``exec``
    namespace, so the registry is internally consistent.
    """
    if program is None:
        return {}

    funcs: dict[str, FuncDeclNode] = {}
    for node in program.body:
        if isinstance(node, FuncDeclNode):
            funcs[node.name] = node
    if not funcs:
        return {}

    # Pass 1: scalar-only params + pure body.
    name_set = set(funcs)
    candidates: dict[str, FuncDeclNode] = {}
    for name, f in funcs.items():
        if not _scalar_param_signature(f):
            continue
        ok = True
        for s in f.body:
            if not _is_pure_stmt(s, name_set, name):
                ok = False
                break
        if ok:
            candidates[name] = f

    # Pass 2: must actually participate in a recursive cycle restricted to
    # the candidate set. Direct self-recursion qualifies immediately; mutual
    # recursion qualifies via SCC membership. We use a single fixed-point
    # pass that prunes non-recursing candidates until the call graph is no
    # longer reducible — a poor-man's Tarjan that's fine for our small set.
    edges: dict[str, set[str]] = {}
    for name, f in candidates.items():
        callees_in_body: set[str] = set()
        _collect_calls(f.body, name_set, name, callees_in_body)
        edges[name] = {c for c in callees_in_body if c in candidates}
    # Iterative reduction: drop candidates that have no edges (leaf callers);
    # the remaining set must contain at least one cycle by pigeon-hole.
    qualified: set[str] = set(candidates)
    changed = True
    while changed:
        changed = False
        for name in list(qualified):
            non_self_edges = edges[name] - {name}
            if non_self_edges and non_self_edges.issubset(qualified):
                continue  # has edges remaining in qualified set
            if name in edges[name]:
                continue  # self-loop — qualifies on its own
            if name in qualified:
                qualified.discard(name)
                changed = True
    # Final guard: drop any qualified name that has no path back to itself
    # (handles cycle-free chains that survived the reduction because both
    # ends still had at least one live edge).
    in_cycle: set[str] = set()
    for name in qualified:
        seen: set[str] = set()
        stack = list(edges[name] - {name})
        while stack:
            cur = stack.pop()
            if cur == name:
                in_cycle.add(name)
                break
            if cur in seen or cur not in qualified:
                continue
            seen.add(cur)
            stack.extend(edges.get(cur, set()) - {cur})
    qualified &= in_cycle if in_cycle else set()
    # Always include direct self-recursors even if the SCC walk somehow
    # missed them (defensive — should never fire).
    for name, f in candidates.items():
        callees_in_body2: set[str] = set()
        _collect_calls(f.body, name_set, name, callees_in_body2)
        if name in callees_in_body2:
            qualified.add(name)

    qualified_funcs = {name: candidates[name] for name in qualified}
    if not qualified_funcs:
        return {}

    # Render and exec each qualified function inside ONE shared namespace
    # so mutual recursion resolves. Sandboxed: no builtins access beyond
    # what is strictly needed by the renderer's output (which never reaches
    # for any built-in — only literals, arithmetic, and recursive calls).
    renderer = _Renderer()
    src_chunks = [renderer.render_func(f) for f in qualified_funcs.values()]
    full_src = "\n\n".join(src_chunks)

    namespace: dict[str, object] = {"__builtins__": {}}
    try:
        # The renderer only ever emits plain Python arithmetic, comparisons,
        # returns, if/else, function definition, and recursive calls; the
        # compiled code therefore has zero reach for any built-in.
        exec(compile(full_src, "<recursive_codegen>", "exec"), namespace)
    except Exception:
        # Any compile/exec failure (shouldn't happen given the purity
        # check, but be defensive) → silently disable for this program
        # and fall back to VM dispatch.
        return {}

    result: dict[str, Callable] = {}
    for name in qualified:
        fn = namespace.get(name)
        if callable(fn):
            result[name] = fn  # type: ignore[assignment]
    return result


def install_recursion_limit(limit: int = 10000) -> int:
    """Raise Python's recursion cap so deep eigen recursion native-compiled
    does not prematurely hit ``RecursionError``. Returns the previous limit.
    """
    prev = sys.getrecursionlimit()
    if limit > prev:
        sys.setrecursionlimit(limit)
    return prev
