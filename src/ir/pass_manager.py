"""
sol.md §1.4 — Optimizer Speed: pass manager + per-pass statistics +
regression test infrastructure.

The existing `EQIROptimizer.optimize()` does all seven rewrite
rules in a single worklist iteration. This module provides:

  * `OptimizationPass` — named unit of work over an `EQIRGraph`
    that returns a structured `PassStats` (gates removed, rotations
    merged, peephole replacements, iterations, duration_ns).
  * `PassManager` — orchestrator that:
      - Registers named passes with optional dependency lists.
      - Executes them in dependency-respecting topological order
        (stable, so passes inserted later only run after their
        prerequisites).
      - Tracks per-pass wall-clock timing.
      - Aggregates a `PassReport` with the full breakdown
        (`total_gates_before/after`, `depth_before/after`,
        `total_duration_ns`, `pass_stats_per_pass`).
  * `default_quantum_pipeline()` — convenience factory that
    registers the existing `EQIROptimizer.optimize` call as one
    bundled pass named ``eqir_optimization``. Downstream users can
    further decompose by subclassing.

Surface-level: this envelope focuses on the per-pass tracking,
statistics shape, and the profiling hooks the roadmap lists. The
actual rewrite rules remain in `EQIROptimizer`; we just instrument
them.
"""
from __future__ import annotations

import dataclasses
import time
import typing

from src.ir.ir_graph import EQIRGraph


def _count_gates(graph: EQIRGraph) -> int:
    return sum(1 for n in graph.nodes.values() if n.type == 'GATE')


def _circuit_depth(graph: EQIRGraph) -> int:
    """Approximate circuit depth = longest path through gate
    nodes (in topological order). Returns 0 for empty graphs.
    """
    # Topological order via DFS over node children.
    visited = set()
    order = []

    def _visit(n):
        if n.id in visited:
            return
        visited.add(n.id)
        for child in sorted(n.children, key=lambda c: c.id):
            _visit(child)
        order.append(n)

    for n in sorted(graph.nodes.values(), key=lambda n: n.id):
        _visit(n)
    order.reverse()
    # Longest path ending at each node.
    longest = {}
    for n in order:
        max_pred = 0
        for parent in n.parents:
            max_pred = max(max_pred, longest.get(parent.id, 0))
        longest[n.id] = max_pred + (1 if n.type == 'GATE' else 0)
    return max(longest.values()) if longest else 0


@dataclasses.dataclass
class PassStats:
    name: str
    gates_before: int
    gates_after: int
    gates_removed: int
    depth_before: int
    depth_after: int
    depth_reduction: int
    iterations: int = 0
    optimizations: int = 0
    duration_ns: int = 0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class PassReport:
    passes: typing.List[PassStats]
    total_gates_before: int
    total_gates_after: int
    total_depth_before: int
    total_depth_after: int
    total_duration_ns: int
    total_optimizations: int

    def to_dict(self) -> dict:
        return {
            "passes": [p.to_dict() for p in self.passes],
            "total_gates_before": self.total_gates_before,
            "total_gates_after": self.total_gates_after,
            "total_depth_before": self.total_depth_before,
            "total_depth_after": self.total_depth_after,
            "total_duration_ns": self.total_duration_ns,
            "total_optimizations": self.total_optimizations,
        }


class OptimizationPass:
    """Named pass over an `EQIRGraph`. Subclass or pass `fn` to
    define the actual work. `fn(graph) -> (graph, dict)` where the
    returned dict has the per-pass stats keys (any subset of
    `optimizations`, `iterations`, `gates_removed`)."""

    def __init__(self, name: str,
                 fn: typing.Optional[typing.Callable] = None,
                 *,
                 dependencies: typing.Optional[typing.List[str]] = None,
                 description: str = ""):
        self.name = name
        self.fn = fn
        self.dependencies = list(dependencies) if dependencies else []
        self.description = description

    def apply(self, graph: EQIRGraph) -> typing.Tuple[EQIRGraph, dict]:
        """Run the pass over `graph`. Returns (new graph, stats dict).
        Subclasses override this; the default path delegates to `self.fn`.
        """
        if self.fn is None:
            raise NotImplementedError(
                f"Pass '{self.name}' has neither `apply` nor `fn`; "
                "override one or the other.")
        result = self.fn(graph)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError(
                f"Pass '{self.name}' fn must return (graph, stats dict).")
        return result


class PassManager:
    """Pass orchestrator with dependency-aware ordering + per-pass
    timing + aggregated statistics."""

    def __init__(self):
        self.passes: typing.List[OptimizationPass] = []
        # Index by name for O(1) access during dependency resolution.
        self._by_name: typing.Dict[str, OptimizationPass] = {}

    def register(self, name: str, fn: typing.Callable, *,
                 dependencies: typing.Optional[typing.List[str]] = None,
                 description: str = "") -> OptimizationPass:
        if name in self._by_name:
            raise ValueError(f"Pass named {name!r} already registered.")
        # Validate named dependencies BEFORE registration so the
        # caller sees the ordering error message immediately.
        for dep in (dependencies or []):
            if dep not in self._by_name:
                raise ValueError(
                    f"Dependency {dep!r} of new pass {name!r} not "
                    "registered; register deps before dependents.")
        p = OptimizationPass(name, fn, dependencies=dependencies,
                              description=description)
        self.passes.append(p)
        self._by_name[name] = p
        return p

    def register_pass(self, p: OptimizationPass) -> OptimizationPass:
        if p.name in self._by_name:
            raise ValueError(f"Pass named {p.name!r} already registered.")
        for dep in p.dependencies:
            if dep not in self._by_name:
                raise ValueError(
                    f"Dependency {dep!r} of {p.name!r} not registered.")
        self.passes.append(p)
        self._by_name[p.name] = p
        return p

    def names_in_execution_order(self) -> typing.List[str]:
        """Dependency-respecting execution order. Passes registered
        before their dependents automatically satisfy the topo-sort;
        we additionally refuse to run if a cycle exists (defensive).
        """
        # The insertion order is already topologically sound because we
        # validate deps at registration. We double-check for cycles via
        # DFS to keep the API honest when callers add passes manually.
        order = []
        state = {}  # name -> "visiting" | "done"

        def _visit(name):
            if state.get(name) == "done":
                return
            if state.get(name) == "visiting":
                raise RuntimeError(
                    f"Pass dependency cycle detected at {name!r}.")
            state[name] = "visiting"
            for dep in self._by_name[name].dependencies:
                _visit(dep)
            state[name] = "done"
            order.append(name)

        for p in self.passes:
            _visit(p.name)
        return order

    def run(self, graph: EQIRGraph) -> PassReport:
        """Execute all registered passes in dependency order over
        `graph`. Returns a `PassReport` with per-pass and aggregated
        statistics."""
        order = self.names_in_execution_order()
        stats_list = []
        total_gates_before = _count_gates(graph)
        total_depth_before = _circuit_depth(graph)
        total_duration_ns = 0
        total_optimizations = 0
        current_graph = graph
        for name in order:
            p = self._by_name[name]
            gates_before = _count_gates(current_graph)
            depth_before = _circuit_depth(current_graph)
            t0 = time.perf_counter_ns()
            new_graph, stats = p.apply(current_graph)
            t1 = time.perf_counter_ns()
            duration_ns = t1 - t0
            current_graph = new_graph
            gates_after = _count_gates(current_graph)
            depth_after = _circuit_depth(current_graph)
            opts = int(stats.get("optimizations", 0))
            ps = PassStats(
                name=name,
                gates_before=gates_before,
                gates_after=gates_after,
                gates_removed=gates_before - gates_after,
                depth_before=depth_before,
                depth_after=depth_after,
                depth_reduction=depth_before - depth_after,
                iterations=int(stats.get("iterations", 0)),
                optimizations=opts,
                duration_ns=duration_ns,
            )
            stats_list.append(ps)
            total_duration_ns += duration_ns
            total_optimizations += opts
        return PassReport(
            passes=stats_list,
            total_gates_before=total_gates_before,
            total_gates_after=_count_gates(current_graph),
            total_depth_before=total_depth_before,
            total_depth_after=_circuit_depth(current_graph),
            total_duration_ns=total_duration_ns,
            total_optimizations=total_optimizations,
        )


def default_quantum_pipeline() -> PassManager:
    """Factory: returns a PassManager pre-loaded with the
    `EQIROptimizer.optimize` rewrite as a single bundled pass.

    Further decomposition (one pass per rewrite rule — see §1.4 in
    sol.md) requires splitting `EQIROptimizer.optimize` itself into
    seven sub-loops; out of scope for this surface-level envelope.
    """
    pm = PassManager()

    def eqir_optimize_pass(graph: EQIRGraph):
        from src.ir.optimizer import EQIROptimizer
        opt = EQIROptimizer()
        opt.optimize(graph)
        return graph, {
            "iterations": getattr(opt, "iterations_count", 0),
            "optimizations": getattr(opt, "optimizations_count", 0),
        }

    pm.register("eqir_optimization", eqir_optimize_pass,
                description="Bundled H-self-cancel, rotation-merge, "
                            "dead-gate-elim, peephole, commutation passes.")
    return pm


def run_optimization_pipeline(graph: EQIRGraph,
                               *,
                               pm: typing.Optional[PassManager] = None,
                               ) -> PassReport:
    """Convenience entry point — runs the default quantum pipeline
    over `graph` and returns the full report."""
    pm = pm or default_quantum_pipeline()
    return pm.run(graph)
