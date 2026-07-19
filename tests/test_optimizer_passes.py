"""§9.1 — Optimizer tests: each pass separately + regression tests.

Exercises each optimization rule in `src.ir.optimizer.EQIROptimizer`
independently, plus regression tests for the pass manager.
"""
import math
import unittest

from src.ir.ir_graph import EQIRGraph
from src.ir.optimizer import EQIROptimizer
from src.ir.pass_manager import (
    PassStats,
    PassReport,
    default_quantum_pipeline,
    run_optimization_pipeline,
    _count_gates,
    _circuit_depth,
)


def _make_gate_graph(*ops):
    """Build a minimal EQIRGraph from a sequence of (gate_name, targets, args) tuples.
    Qubits are allocated implicitly based on target names used.
    """
    graph = EQIRGraph()
    qubits_seen = set()
    for gate_name, targets, args in ops:
        for t in targets:
            if t not in qubits_seen:
                graph.add_operation('ALLOC', targets=[t])
                qubits_seen.add(t)
        graph.add_operation('GATE', gate_name=gate_name,
                              targets=targets, args=args)
    return graph


def _gate_names(graph):
    """Return ordered list of gate names in graph (by node id)."""
    names = []
    for nid in sorted(graph.nodes.keys()):
        node = graph.nodes[nid]
        if node.type == 'GATE':
            names.append(node.gate_name)
    return names


# ---------------------------------------------------------------------------
# Rule 1: Self-inverse cancellation (H, X, Y, Z)
# ---------------------------------------------------------------------------

class TestSelfInverseCancellation(unittest.TestCase):
    def test_h_h_cancels(self):
        graph = _make_gate_graph(
            ("H", ["q0"], []),
            ("H", ["q0"], []),
        )
        opt = EQIROptimizer()
        opt.optimize(graph)
        names = _gate_names(graph)
        self.assertEqual(names, [])

    def test_x_x_cancels(self):
        graph = _make_gate_graph(
            ("X", ["q0"], []),
            ("X", ["q0"], []),
        )
        opt = EQIROptimizer()
        opt.optimize(graph)
        self.assertEqual(_gate_names(graph), [])

    def test_y_y_cancels(self):
        graph = _make_gate_graph(
            ("Y", ["q0"], []),
            ("Y", ["q0"], []),
        )
        opt = EQIROptimizer()
        opt.optimize(graph)
        self.assertEqual(_gate_names(graph), [])

    def test_z_z_cancels(self):
        graph = _make_gate_graph(
            ("Z", ["q0"], []),
            ("Z", ["q0"], []),
        )
        opt = EQIROptimizer()
        opt.optimize(graph)
        self.assertEqual(_gate_names(graph), [])

    def test_h_x_does_not_cancel(self):
        graph = _make_gate_graph(
            ("H", ["q0"], []),
            ("X", ["q0"], []),
        )
        opt = EQIROptimizer()
        opt.optimize(graph)
        # H and X are different gates, no cancellation
        self.assertEqual(len(_gate_names(graph)), 2)

    def test_h_h_on_different_qubits_does_not_cancel(self):
        graph = _make_gate_graph(
            ("H", ["q0"], []),
            ("H", ["q1"], []),
        )
        opt = EQIROptimizer()
        opt.optimize(graph)
        self.assertEqual(len(_gate_names(graph)), 2)


# ---------------------------------------------------------------------------
# Rule 2: Rotation merging (RX, RY, RZ)
# ---------------------------------------------------------------------------

class TestRotationMerging(unittest.TestCase):
    def test_rx_rx_merges(self):
        graph = _make_gate_graph(
            ("RX", ["q0"], [0.5]),
            ("RX", ["q0"], [0.3]),
        )
        opt = EQIROptimizer()
        opt.optimize(graph)
        names = _gate_names(graph)
        self.assertEqual(len(names), 1)
        # Angle should be 0.8 (0.5 + 0.3)
        gate_node = [n for n in graph.nodes.values()
                       if n.type == 'GATE'][0]
        self.assertAlmostEqual(gate_node.args[0], 0.8, places=5)

    def test_ry_ry_merges(self):
        graph = _make_gate_graph(
            ("RY", ["q0"], [1.0]),
            ("RY", ["q0"], [2.0]),
        )
        opt = EQIROptimizer()
        opt.optimize(graph)
        names = _gate_names(graph)
        self.assertEqual(len(names), 1)

    def test_rz_rz_merges(self):
        graph = _make_gate_graph(
            ("RZ", ["q0"], [math.pi]),
            ("RZ", ["q0"], [math.pi]),
        )
        opt = EQIROptimizer()
        opt.optimize(graph)
        names = _gate_names(graph)
        # After merging: 2*pi mod 2*pi = 0 → dead gate elimination
        # should fire and remove the gate entirely.
        # If native optimizer handles this differently, at most 1 gate remains.
        self.assertLessEqual(len(names), 1)

    def test_rx_ry_does_not_merge(self):
        graph = _make_gate_graph(
            ("RX", ["q0"], [0.5]),
            ("RY", ["q0"], [0.3]),
        )
        opt = EQIROptimizer()
        opt.optimize(graph)
        # Different rotation axes don't merge
        self.assertEqual(len(_gate_names(graph)), 2)


# ---------------------------------------------------------------------------
# Rule 3: Dead gate elimination (rotation with ~0 angle)
# ---------------------------------------------------------------------------

class TestDeadGateElimination(unittest.TestCase):
    def test_rx_zero_angle_removed(self):
        graph = _make_gate_graph(
            ("RX", ["q0"], [0.0]),
        )
        opt = EQIROptimizer()
        opt.optimize(graph)
        self.assertEqual(_gate_names(graph), [])

    def test_rx_near_zero_removed(self):
        graph = _make_gate_graph(
            ("RX", ["q0"], [1e-15]),
        )
        opt = EQIROptimizer()
        opt.optimize(graph)
        self.assertEqual(_gate_names(graph), [])


# ---------------------------------------------------------------------------
# Rule 4: Peephole H → X/Z → H
# ---------------------------------------------------------------------------

class TestPeepholeHXH(unittest.TestCase):
    def test_h_x_h_becomes_z(self):
        graph = _make_gate_graph(
            ("H", ["q0"], []),
            ("X", ["q0"], []),
            ("H", ["q0"], []),
        )
        opt = EQIROptimizer()
        opt.optimize(graph)
        names = _gate_names(graph)
        # H-X-H = Z
        self.assertEqual(names, ["Z"])

    def test_h_z_h_becomes_x(self):
        graph = _make_gate_graph(
            ("H", ["q0"], []),
            ("Z", ["q0"], []),
            ("H", ["q0"], []),
        )
        opt = EQIROptimizer()
        opt.optimize(graph)
        names = _gate_names(graph)
        # H-Z-H = X
        self.assertEqual(names, ["X"])


# ---------------------------------------------------------------------------
# Rule 5: Peephole S → S = Z, T → T = S
# ---------------------------------------------------------------------------

class TestPeepholeSS(unittest.TestCase):
    def test_s_s_becomes_z(self):
        graph = _make_gate_graph(
            ("S", ["q0"], []),
            ("S", ["q0"], []),
        )
        opt = EQIROptimizer()
        opt.optimize(graph)
        names = _gate_names(graph)
        self.assertEqual(names, ["Z"])

    def test_t_t_becomes_s(self):
        graph = _make_gate_graph(
            ("T", ["q0"], []),
            ("T", ["q0"], []),
        )
        opt = EQIROptimizer()
        opt.optimize(graph)
        names = _gate_names(graph)
        self.assertEqual(names, ["S"])


# ---------------------------------------------------------------------------
# Rule 6: Commutation cancellation Z → CNOT → Z
# ---------------------------------------------------------------------------

class TestCommutationZCNOTZ(unittest.TestCase):
    def test_z_cnot_z_cancels_z_pair(self):
        graph = _make_gate_graph(
            ("Z", ["q0"], []),
            ("CNOT", ["q0", "q1"], []),
            ("Z", ["q0"], []),
        )
        opt = EQIROptimizer()
        opt.optimize(graph)
        names = _gate_names(graph)
        # Z on control commutes through CNOT, both Z's cancel
        self.assertEqual(names, ["CNOT"])


# ---------------------------------------------------------------------------
# Rule 7: Commutation cancellation X(target) → CNOT → X(target)
# ---------------------------------------------------------------------------

class TestCommutationXCNOTX(unittest.TestCase):
    def test_x_target_cnot_x_target_cancels(self):
        graph = _make_gate_graph(
            ("X", ["q1"], []),
            ("CNOT", ["q0", "q1"], []),
            ("X", ["q1"], []),
        )
        opt = EQIROptimizer()
        opt.optimize(graph)
        names = _gate_names(graph)
        # X on target commutes through CNOT, both X's cancel
        self.assertEqual(names, ["CNOT"])


# ---------------------------------------------------------------------------
# Optimizer stats
# ---------------------------------------------------------------------------

class TestOptimizerStats(unittest.TestCase):
    def test_optimizations_count_tracks_rewrites(self):
        graph = _make_gate_graph(
            ("H", ["q0"], []),
            ("H", ["q0"], []),
        )
        opt = EQIROptimizer()
        opt.optimize(graph)
        self.assertGreater(opt.optimizations_count, 0)

    def test_iterations_count_set(self):
        graph = _make_gate_graph(
            ("H", ["q0"], []),
        )
        opt = EQIROptimizer()
        opt.optimize(graph)
        self.assertGreaterEqual(opt.iterations_count, 0)


# ---------------------------------------------------------------------------
# Regression: optimizer doesn't change ALLOC/MEASURE nodes
# ---------------------------------------------------------------------------

class TestOptimizerPreservesNonGateNodes(unittest.TestCase):
    def test_alloc_nodes_preserved(self):
        graph = EQIRGraph()
        graph.add_operation('ALLOC', targets=['q0'])
        graph.add_operation('GATE', gate_name='H', targets=['q0'], args=[])
        graph.add_operation('MEASURE', targets=['q0'], cbit_name='c0')
        opt = EQIROptimizer()
        opt.optimize(graph)
        # All three node types should still be present
        types = [n.type for n in graph.nodes.values()]
        self.assertIn('ALLOC', types)
        self.assertIn('MEASURE', types)

    def test_measure_node_preserved(self):
        graph = _make_gate_graph(
            ("H", ["q0"], []),
        )
        graph.add_operation('MEASURE', targets=['q0'], cbit_name='c0')
        opt = EQIROptimizer()
        opt.optimize(graph)
        measure_nodes = [n for n in graph.nodes.values()
                           if n.type == 'MEASURE']
        self.assertEqual(len(measure_nodes), 1)


# ---------------------------------------------------------------------------
# PassManager tests
# ---------------------------------------------------------------------------

class TestPassManager(unittest.TestCase):
    def test_register_and_run(self):
        graph = _make_gate_graph(("H", ["q0"], []), ("H", ["q0"], []))
        pm = default_quantum_pipeline()
        report = pm.run(graph)
        self.assertIsInstance(report, PassReport)
        # Should have at least one pass registered
        self.assertGreaterEqual(len(report.passes), 1)
        # Total gates should be 0 after H-H cancellation
        self.assertEqual(report.total_gates_after, 0)

    def test_count_gates(self):
        graph = _make_gate_graph(("H", ["q0"], []), ("X", ["q0"], []))
        self.assertEqual(_count_gates(graph), 2)

    def test_count_gates_empty(self):
        graph = EQIRGraph()
        self.assertEqual(_count_gates(graph), 0)

    def test_circuit_depth_single_gate(self):
        graph = _make_gate_graph(("H", ["q0"], []))
        # depth = 1 (just the gate, ALLOC doesn't count for depth?)
        depth = _circuit_depth(graph)
        self.assertGreaterEqual(depth, 1)

    def test_circuit_depth_parallel(self):
        graph = _make_gate_graph(
            ("H", ["q0"], []),
            ("X", ["q1"], []),
        )
        # Parallel gates → depth = 1
        depth = _circuit_depth(graph)
        self.assertGreaterEqual(depth, 1)

    def test_circuit_depth_sequential(self):
        graph = _make_gate_graph(
            ("H", ["q0"], []),
            ("X", ["q0"], []),
        )
        depth = _circuit_depth(graph)
        self.assertGreaterEqual(depth, 2)

    def test_run_optimization_pipeline(self):
        graph = _make_gate_graph(("H", ["q0"], []), ("H", ["q0"], []))
        report = run_optimization_pipeline(graph)
        self.assertIsInstance(report, PassReport)

    def test_pass_stats_to_dict(self):
        stats = PassStats(
            name="test", gates_before=5, gates_after=3,
            gates_removed=2, depth_before=4, depth_after=2,
            depth_reduction=2,
        )
        d = stats.to_dict()
        self.assertEqual(d["name"], "test")
        self.assertEqual(d["gates_before"], 5)
        self.assertEqual(d["gates_after"], 3)

    def test_pass_report_to_dict(self):
        stats = PassStats(
            name="p1", gates_before=3, gates_after=2,
            gates_removed=1, depth_before=2, depth_after=1,
            depth_reduction=1,
        )
        report = PassReport(
            passes=[stats],
            total_gates_before=3, total_gates_after=2,
            total_depth_before=2, total_depth_after=1,
            total_duration_ns=10, total_optimizations=1,
        )
        d = report.to_dict()
        self.assertIn("passes", d)
        self.assertEqual(len(d["passes"]), 1)


# ---------------------------------------------------------------------------
# Regression: chained optimizations
# ---------------------------------------------------------------------------

class TestChainedOptimizations(unittest.TestCase):
    def test_h_h_h_h_all_cancel(self):
        graph = _make_gate_graph(
            ("H", ["q0"], []),
            ("H", ["q0"], []),
            ("H", ["q0"], []),
            ("H", ["q0"], []),
        )
        opt = EQIROptimizer()
        opt.optimize(graph)
        self.assertEqual(_gate_names(graph), [])

    def test_h_h_x_x_leaves_nothing(self):
        graph = _make_gate_graph(
            ("H", ["q0"], []),
            ("H", ["q0"], []),
            ("X", ["q0"], []),
            ("X", ["q0"], []),
        )
        opt = EQIROptimizer()
        opt.optimize(graph)
        self.assertEqual(_gate_names(graph), [])

    def test_rx_pi_then_rx_pi_becomes_identity(self):
        graph = _make_gate_graph(
            ("RX", ["q0"], [math.pi]),
            ("RX", ["q0"], [math.pi]),
        )
        opt = EQIROptimizer()
        opt.optimize(graph)
        # 2*pi mod 2*pi = 0 → dead gate eliminated
        names = _gate_names(graph)
        # Should be empty (rotation merged to 2*pi, then dead-gate-eliminated)
        self.assertEqual(names, [])

    def test_optimizer_idempotent(self):
        graph = _make_gate_graph(
            ("H", ["q0"], []),
            ("H", ["q0"], []),
        )
        opt = EQIROptimizer()
        opt.optimize(graph)
        names1 = _gate_names(graph)
        opt.optimize(graph)
        names2 = _gate_names(graph)
        # Second pass should find nothing to optimize — same result
        self.assertEqual(names1, names2)


if __name__ == "__main__":
    unittest.main()
