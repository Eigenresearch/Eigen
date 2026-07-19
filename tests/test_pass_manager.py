"""
sol.md §1.4 — Optimizer pass manager tests.

Covers `src.ir.pass_manager`:
  * `PassManager.register`/`register_pass` with dependency validation.
  * Dependency-respecting execution order; cycle detection.
  * Per-pass stats shape: gates_before/after/removed, depth,
    iterations, optimizations, duration_ns.
  * Aggregated `PassReport` totals.
  * End-to-end run on a self-inverse cancellation example using
    `default_quantum_pipeline` against the existing `EQIROptimizer`
    — surfaces regression detection: a graph with `H . H` should
    end with 0 gates after the optimizer pass runs.
"""
from __future__ import annotations

import time
import unittest

from src.ir.ir_graph import EQIRGraph
from src.ir.pass_manager import (
    OptimizationPass,
    PassManager,
    PassReport,
    default_quantum_pipeline,
    run_optimization_pipeline,
)


def _build_hh_graph(qubit="q0") -> EQIRGraph:
    """Build a tiny circuit: H(q) . H(q). After self-inverse
    cancellation this should reduce to the empty graph (0 gates)."""
    g = EQIRGraph()
    g.add_operation("GATE", gate_name="H", args=[], targets=[qubit])
    g.add_operation("GATE", gate_name="H", args=[], targets=[qubit])
    return g


def _build_rxrx_graph(qubit="q0", angle1: float = 1.0,
                       angle2: float = 0.5) -> EQIRGraph:
    """Build RX(θ1) · RX(θ2) graph — after rotation-merge should
    produce 1 node with angle ≡ (θ1+θ2) mod 2π."""
    g = EQIRGraph()
    g.add_operation("GATE", gate_name="RX", args=[angle1], targets=[qubit])
    g.add_operation("GATE", gate_name="RX", args=[angle2], targets=[qubit])
    return g


class TestPassManagerCore(unittest.TestCase):

    def test_register_and_run_basic(self):
        pm = PassManager()
        called = []

        def pass_a(graph):
            called.append("a")
            return graph, {"optimizations": 1, "iterations": 5}

        pm.register("pass_a", pass_a, description="A")
        report = pm.run(EQIRGraph())
        self.assertIsInstance(report, PassReport)
        self.assertEqual(len(report.passes), 1)
        self.assertEqual(report.passes[0].name, "pass_a")
        self.assertEqual(report.passes[0].optimizations, 1)
        self.assertEqual(report.passes[0].iterations, 5)
        self.assertEqual(report.total_optimizations, 1)
        self.assertEqual(called, ["a"])

    def test_dependency_execution_order(self):
        pm = PassManager()
        call_order = []

        def make(name):
            def _pass(graph):
                call_order.append(name)
                return graph, {}
            return _pass

        pm.register("base", make("base"))
        pm.register("mid", make("mid"), dependencies=["base"])
        pm.register("leaf", make("leaf"), dependencies=["mid"])
        pm.run(EQIRGraph())
        self.assertEqual(call_order, ["base", "mid", "leaf"])

    def test_dependency_not_registered_raises(self):
        pm = PassManager()
        with self.assertRaises(ValueError):
            pm.register("a", lambda g: (g, {}), dependencies=["b"])

    def test_cycle_detection(self):
        pm = PassManager()
        # We add manually to inject a cycle (the validator should
        # reject dep-on-self at register time).
        pm.register("a", lambda g: (g, {}))
        # Manually inject a self-dependency.
        p = OptimizationPass("b", dependencies=["a"])
        pm.register_pass(p)
        # Now add a cycle a -> b -> a.
        pm._by_name["a"].dependencies.append("b")
        with self.assertRaises(RuntimeError):
            pm.run(EQIRGraph())

    def test_duplicate_registration_raises(self):
        pm = PassManager()
        pm.register("a", lambda g: (g, {}))
        with self.assertRaises(ValueError):
            pm.register("a", lambda g: (g, {}))

    def test_pass_stats_toplevel_keys(self):
        pm = PassManager()
        pm.register("p", lambda g: (g, {"optimizations": 7}))
        report = pm.run(EQIRGraph())
        d = report.to_dict()
        for k in ("passes", "total_gates_before", "total_gates_after",
                 "total_depth_before", "total_depth_after",
                 "total_duration_ns", "total_optimizations"):
            self.assertIn(k, d)

    def test_per_pass_duration_nonzero(self):
        pm = PassManager()
        def slow(graph):
            time.sleep(0.001)
            return graph, {}
        pm.register("slow", slow)
        report = pm.run(EQIRGraph())
        self.assertGreater(report.passes[0].duration_ns, 0)

    def test_register_pass_direct(self):
        pm = PassManager()
        p = OptimizationPass("z", lambda g: (g, {}))
        pm.register_pass(p)
        report = pm.run(EQIRGraph())
        self.assertEqual(report.passes[0].name, "z")

    def test_unknown_dependency_raises_on_register_pass(self):
        pm = PassManager()
        p = OptimizationPass("y", dependencies=["x"])
        with self.assertRaises(ValueError):
            pm.register_pass(p)


class TestPassManagerPipelineIntegration(unittest.TestCase):

    def test_default_pipeline_runs_eqir_optimization_pass(self):
        pm = default_quantum_pipeline()
        report = pm.run(EQIRGraph())
        self.assertEqual(len(report.passes), 1)
        self.assertEqual(report.passes[0].name, "eqir_optimization")
        self.assertIsInstance(report.passes[0].optimizations, int)
        self.assertIsInstance(report.passes[0].iterations, int)

    def test_run_optimization_pipeline_hh_collapses(self):
        # H(q) . H(q) → 0 gates after eqir_optimization.
        g = _build_hh_graph("q0")
        report = run_optimization_pipeline(g)
        self.assertEqual(report.total_gates_before, 2)
        self.assertEqual(report.total_gates_after, 0)
        self.assertGreater(report.passes[0].optimizations, 0)

    def test_run_optimization_pipeline_rxrx_merges(self):
        # RX(1) . RX(0.5) → 1 gate after eqir_optimization.
        g = _build_rxrx_graph("q0", 1.0, 0.5)
        report = run_optimization_pipeline(g)
        self.assertEqual(report.total_gates_before, 2)
        # At least one gate removed.
        self.assertGreaterEqual(report.passes[0].gates_removed, 1)
        self.assertGreaterEqual(report.passes[0].optimizations, 1)

    def test_run_optimization_pipeline_empty_graph_is_safe(self):
        report = run_optimization_pipeline(EQIRGraph())
        self.assertEqual(report.total_gates_before, 0)
        self.assertEqual(report.total_gates_after, 0)
        self.assertEqual(report.total_optimizations, 0)

    def test_run_optimization_pipeline_idempotent_second_run(self):
        # After first optimization, second run should not produce
        # additional optimizations — the graph is normalized.
        g = _build_hh_graph("q0")
        run_optimization_pipeline(g)
        report2 = run_optimization_pipeline(g)  # re-run on already-opt graph
        self.assertEqual(report2.total_optimizations, 0)
        self.assertEqual(report2.total_gates_after, 0)

    class TestPassManagerRegression(unittest.TestCase):
        """Regression tests — each captures a known good optimization
        that the eqir_optimization pass must always produce. These
        are the `регрессионные тесты оптимизатора` checkbox in the
        roadmap §1.4."""

        def test_self_inverse_h_cancel(self):
            g = _build_hh_graph("q0")
            report = run_optimization_pipeline(g)
            self.assertEqual(report.total_gates_after, 0)

        def test_rotation_merging(self):
            g = _build_rxrx_graph("q0", 1.5, 2.5)
            report = run_optimization_pipeline(g)
            # RX(1.5) · RX(2.5) → RX(4.0); 1 gate left.
            # Mod 2π for canonical representation.
            self.assertLessEqual(report.total_gates_after, 1)
            self.assertGreaterEqual(report.passes[0].optimizations, 1)

        def test_dead_gate_elimination(self):
            # RX(0) . H(q0) → H(q0) (dead RX(0) pruned).
            g = EQIRGraph()
            g.add_operation("GATE", gate_name="RX",
                            args=[0.0], targets=["q0"])
            g.add_operation("GATE", gate_name="H",
                            args=[], targets=["q0"])
            report = run_optimization_pipeline(g)
            self.assertEqual(report.total_gates_after, 1)

        def test_report_depth_reduction_for_hh(self):
            g = _build_hh_graph("q0")
            report = run_optimization_pipeline(g)
            self.assertGreaterEqual(report.total_depth_before, 2)
            self.assertEqual(report.total_depth_after, 0)


if __name__ == "__main__":
    unittest.main()
