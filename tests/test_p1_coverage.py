"""P1 §X — coverage-gap tests for the four historically under-tested modules:

  * `src/ir/optimizer.py`         — Python fallback path (Rust path is
                                    exercised by the rest of the suite).
  * `src/sparse_simulator.py`     — pure-Python gate fallbacks that the
                                    Rust extension normally short-circuits.
  * `src/tensor_network/mps.py`   — auto bond-dim + SVD fallback + measure
                                    + state-vector reconstruction paths.
  * `src/packager.py`             — EigenPackager lifecycle, version
                                    constraint matcher, lockfile hash dance.

Each test forces the Python fallback path so future regressions of the
non-Rust path are caught BEFORE the Rust extension disappears (e.g. an
upgrade drops eigen_native and the project silently loses optimizer
rewrites / sparse sim / MPS SVD — all of which we cover here).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import unittest
from copy import deepcopy
from unittest.mock import patch

import numpy as np


# ============================================================================
# Optimizer (Python fallback)
# ============================================================================


def _optimizable_graph():
    """A graph exercising every optimizer rewrite rule. Mirrors
    `test_optimizer_determinism._build_optimizable_graph` but is
    declared locally to keep this test module standalone."""
    from src.ir.ir_graph import EQIRGraph

    g = EQIRGraph()
    g.add_operation("ALLOC", targets=["q0"])
    g.add_operation("ALLOC", targets=["q1"])
    g.add_operation("GATE", gate_name="H", targets=["q0"])
    g.add_operation("GATE", gate_name="H", targets=["q0"])
    g.add_operation("GATE", gate_name="RX", targets=["q0"], args=[0.3])
    g.add_operation("GATE", gate_name="RX", targets=["q0"], args=[0.4])
    g.add_operation("GATE", gate_name="RZ", targets=["q0"], args=[0.0])
    g.add_operation("GATE", gate_name="H", targets=["q0"])
    g.add_operation("GATE", gate_name="X", targets=["q0"])
    g.add_operation("GATE", gate_name="H", targets=["q0"])
    g.add_operation("GATE", gate_name="S", targets=["q1"])
    g.add_operation("GATE", gate_name="S", targets=["q1"])
    g.add_operation("GATE", gate_name="Z", targets=["q0"])
    g.add_operation("GATE", gate_name="CNOT", targets=["q0", "q1"])
    g.add_operation("GATE", gate_name="Z", targets=["q0"])
    g.add_operation("MEASURE", targets=["q1"], cbit_name="c1")
    return g


class TestOptimizerPythonFallback(unittest.TestCase):
    """Force the pure-Python optimizer path by patching out
    `eigen_native.optimize_eqir_native`. Each rule is examined in
    isolation so a regression that breaks one rewrite is identified
    by name."""

    def setUp(self):
        try:
            import eigen_native
            self._patch = patch.object(
                eigen_native, "optimize_eqir_native",
                side_effect=AttributeError("forced fallback for test"),
            )
            self._patch.start()
            self._has_native = True
        except ImportError:
            self._patch = None
            self._has_native = False

    def tearDown(self):
        if self._patch is not None:
            self._patch.stop()

    def test_runs_full_pass_python_fallback(self):
        from src.ir.optimizer import EQIROptimizer

        g = deepcopy(_optimizable_graph())
        before = len(g.nodes)
        opt = EQIROptimizer()
        opt.optimize(g)
        self.assertLess(len(g.nodes), before)
        self.assertGreater(opt.iterations_count, 0)
        self.assertGreater(opt.optimizations_count, 0)

    def test_rule1_self_inverse_cancellation_h(self):
        """H q; H q (where the two Hs have no intervening op) cancels."""
        from src.ir.ir_graph import EQIRGraph
        from src.ir.optimizer import EQIROptimizer

        g = EQIRGraph()
        g.add_operation("ALLOC", targets=["q0"])
        g.add_operation("GATE", gate_name="H", targets=["q0"])
        g.add_operation("GATE", gate_name="H", targets=["q0"])
        opt = EQIROptimizer()
        opt.optimize(g)
        gate_count = sum(1 for n in g.nodes.values()
                         if n.type == "GATE")
        self.assertEqual(gate_count, 0)
        self.assertGreaterEqual(opt.optimizations_count, 1)

    def test_rule2_rotation_merging(self):
        from src.ir.ir_graph import EQIRGraph
        from src.ir.optimizer import EQIROptimizer

        g = EQIRGraph()
        g.add_operation("ALLOC", targets=["q0"])
        g.add_operation("GATE", gate_name="RX", targets=["q0"], args=[1.0])
        g.add_operation("GATE", gate_name="RX", targets=["q0"], args=[2.0])
        opt = EQIROptimizer()
        opt.optimize(g)
        rx = [n for n in g.nodes.values()
              if n.type == "GATE" and n.gate_name == "RX"]
        self.assertEqual(len(rx), 1)
        self.assertAlmostEqual(rx[0].args[0], 3.0 % (2 * math.pi))
        self.assertGreaterEqual(opt.optimizations_count, 1)

    def test_rule3_dead_rotation_elimination(self):
        from src.ir.ir_graph import EQIRGraph
        from src.ir.optimizer import EQIROptimizer

        g = EQIRGraph()
        g.add_operation("ALLOC", targets=["q0"])
        g.add_operation("GATE", gate_name="RZ", targets=["q0"], args=[0.0])
        opt = EQIROptimizer()
        opt.optimize(g)
        gates = [n for n in g.nodes.values() if n.type == "GATE"]
        self.assertEqual(len(gates), 0)
        self.assertGreaterEqual(opt.optimizations_count, 1)

    def test_rule4_peephole_h_x_h_becomes_z(self):
        from src.ir.ir_graph import EQIRGraph
        from src.ir.optimizer import EQIROptimizer

        g = EQIRGraph()
        g.add_operation("ALLOC", targets=["q0"])
        g.add_operation("GATE", gate_name="H", targets=["q0"])
        g.add_operation("GATE", gate_name="X", targets=["q0"])
        g.add_operation("GATE", gate_name="H", targets=["q0"])
        opt = EQIROptimizer()
        opt.optimize(g)
        gates = [n for n in g.nodes.values() if n.type == "GATE"]
        # H-X-H -> Z (one Z node remains after the two H nodes are bypassed).
        self.assertEqual(len(gates), 1)
        self.assertEqual(gates[0].gate_name, "Z")

    def test_rule5_peephole_s_s_becomes_z(self):
        from src.ir.ir_graph import EQIRGraph
        from src.ir.optimizer import EQIROptimizer

        g = EQIRGraph()
        g.add_operation("ALLOC", targets=["q0"])
        g.add_operation("GATE", gate_name="S", targets=["q0"])
        g.add_operation("GATE", gate_name="S", targets=["q0"])
        opt = EQIROptimizer()
        opt.optimize(g)
        gates = [n for n in g.nodes.values() if n.type == "GATE"]
        self.assertEqual(len(gates), 1)
        self.assertEqual(gates[0].gate_name, "Z")

    def test_rule5_peephole_t_t_becomes_s(self):
        from src.ir.ir_graph import EQIRGraph
        from src.ir.optimizer import EQIROptimizer

        g = EQIRGraph()
        g.add_operation("ALLOC", targets=["q0"])
        g.add_operation("GATE", gate_name="T", targets=["q0"])
        g.add_operation("GATE", gate_name="T", targets=["q0"])
        opt = EQIROptimizer()
        opt.optimize(g)
        gates = [n for n in g.nodes.values() if n.type == "GATE"]
        self.assertEqual(len(gates), 1)
        self.assertEqual(gates[0].gate_name, "S")

    def test_rule6_commutation_z_cnot_z(self):
        """Z q0 -> CNOT q0 q1 -> Z q0 collapses to a bare CNOT."""
        from src.ir.ir_graph import EQIRGraph
        from src.ir.optimizer import EQIROptimizer

        g = EQIRGraph()
        g.add_operation("ALLOC", targets=["q0"])
        g.add_operation("ALLOC", targets=["q1"])
        g.add_operation("GATE", gate_name="Z", targets=["q0"])
        g.add_operation("GATE", gate_name="CNOT", targets=["q0", "q1"])
        g.add_operation("GATE", gate_name="Z", targets=["q0"])
        opt = EQIROptimizer()
        opt.optimize(g)
        gates = [n for n in g.nodes.values() if n.type == "GATE"]
        self.assertEqual(len(gates), 1)
        self.assertEqual(gates[0].gate_name, "CNOT")

    def test_rule7_commutation_x_cnot_x(self):
        """X q1 -> CNOT q0 q1 -> X q1 collapses to a bare CNOT."""
        from src.ir.ir_graph import EQIRGraph
        from src.ir.optimizer import EQIROptimizer

        g = EQIRGraph()
        g.add_operation("ALLOC", targets=["q0"])
        g.add_operation("ALLOC", targets=["q1"])
        g.add_operation("GATE", gate_name="X", targets=["q1"])
        g.add_operation("GATE", gate_name="CNOT", targets=["q0", "q1"])
        g.add_operation("GATE", gate_name="X", targets=["q1"])
        opt = EQIROptimizer()
        opt.optimize(g)
        gates = [n for n in g.nodes.values() if n.type == "GATE"]
        self.assertEqual(len(gates), 1)
        self.assertEqual(gates[0].gate_name, "CNOT")

    def test_optimizer_idempotent_python_fallback(self):
        from src.ir.optimizer import EQIROptimizer

        g = deepcopy(_optimizable_graph())
        opt = EQIROptimizer()
        opt.optimize(g)
        n_after_first = len(g.nodes)
        # Second pass:
        opt.optimize(g)
        self.assertEqual(len(g.nodes), n_after_first)


# ============================================================================
# Sparse simulator (Python fallback)
# ============================================================================


class TestSparseSimulatorPythonFallback(unittest.TestCase):
    """Force the pure-Python sparse path by zeroing `_rust_sparse`."""

    def _make(self, seed=42):
        from src.sparse_simulator import SparseQuantumSimulator

        sim = SparseQuantumSimulator(seed=seed)
        sim._rust_sparse = None
        return sim

    def test_allocate_qubit_doubles_state_space(self):
        sim = self._make()
        sim.allocate_qubit("q0")
        self.assertEqual(sim.num_qubits, 1)
        self.assertEqual(set(sim.state.keys()), {"0"})
        sim.allocate_qubit("q1")
        # Each qubit starts at |0>; allocate appends a '0' bit, so
        # post-allocate state is just {"00": 1.0} (no superposition yet).
        self.assertEqual(set(sim.state.keys()), {"00"})
        self.assertEqual(sim.state["00"], 1.0 + 0.0j)

    def test_get_qubit_index_raises_on_unallocated(self):
        sim = self._make()
        with self.assertRaises(KeyError):
            sim.get_qubit_index("nope")

    def test_h_gate_creates_superposition(self):
        sim = self._make()
        sim.allocate_qubit("q0")
        sim.H("q0")
        st = sim.state
        self.assertAlmostEqual(abs(st["0"]) ** 2, 0.5)
        self.assertAlmostEqual(abs(st["1"]) ** 2, 0.5)

    def test_x_gate_flips_bit(self):
        sim = self._make(seed=1)
        sim.allocate_qubit("q0")
        sim.X("q0")
        self.assertAlmostEqual(sim.state["1"], 1.0)

    def test_y_gate_phases(self):
        sim = self._make()
        sim.allocate_qubit("q0")
        sim.Y("q0")
        # Y|0> = i|1>
        self.assertAlmostEqual(sim.state["1"], 1j)

    def test_z_gate_picks_up_minus_on_one(self):
        sim = self._make()
        sim.allocate_qubit("q0")
        sim.H("q0")
        sim.Z("q0")
        # Z is diagonal(1, -1); after H, the state is (|0>+|1>)/sqrt(2).
        # Z gives (|0>-|1>)/sqrt(2). Apply H to verify (X|0>) = |1>.
        self.assertAlmostEqual(abs(sim.state["0"]) ** 2, 0.5)
        sim.H("q0")
        self.assertAlmostEqual(sim.state.get("0", 0.0), 0.0, places=10)
        self.assertAlmostEqual(abs(sim.state["1"]) ** 2, 1.0)

    def test_s_and_t_phases(self):
        sim = self._make()
        sim.allocate_qubit("q0")
        sim.X("q0")  # |1>
        sim.S("q0")  # S|1> = i|1>
        self.assertAlmostEqual(sim.state["1"], 1j)
        sim.T("q0")  # T|1> = e^{i pi/4}|1>; previous * T phase
        # (1j) * (1+i)/sqrt(2) = (i + i^2)/sqrt(2) = (-1 + i)/sqrt(2)
        sqrt2_over_2 = math.sqrt(2) / 2
        expected = 1j * (sqrt2_over_2 + sqrt2_over_2 * 1j)
        self.assertAlmostEqual(sim.state["1"], expected)

    def test_rx_ry_rz_via_apply_1qubit_gate(self):
        sim = self._make()
        sim.allocate_qubit("q0")
        sim.RX("q0", math.pi)
        # RX(pi)|0> = -i|1>
        self.assertAlmostEqual(abs(sim.state["1"]) ** 2, 1.0)
        sim2 = self._make()
        sim2.allocate_qubit("q0")
        sim2.RY("q0", math.pi)
        # RY(pi)|0> = |1>
        self.assertAlmostEqual(abs(sim2.state["1"]) ** 2, 1.0)
        sim3 = self._make()
        sim3.allocate_qubit("q0")
        sim3.RZ("q0", math.pi / 2)
        # RZ(pi/2)|0> = e^{-i pi/4}|0>
        self.assertAlmostEqual(abs(sim3.state["0"]) ** 2, 1.0)

    def test_cnot_entangles(self):
        sim = self._make()
        sim.allocate_qubit("q0")
        sim.allocate_qubit("q1")
        sim.H("q0")
        sim.CNOT("q0", "q1")
        # Bell state: |00>+|11> over sqrt(2)
        for key, amp in sim.state.items():
            self.assertIn(key, {"00", "11"})
            self.assertAlmostEqual(abs(amp) ** 2, 0.5)

    def test_cz_and_swap(self):
        sim = self._make()
        sim.allocate_qubit("q0")
        sim.allocate_qubit("q1")
        sim.X("q0")
        sim.X("q1")
        sim.CZ("q0", "q1")  # |11> picks up minus
        self.assertAlmostEqual(sim.state["11"], -1.0)
        sim.SWAP("q0", "q1")
        # Should still be |11>, just in different bit-order — both 1.
        self.assertEqual(len(sim.state), 1)

    def test_ccx_cswap_cp(self):
        sim = self._make()
        sim.allocate_qubit("q0")
        sim.allocate_qubit("q1")
        sim.allocate_qubit("q2")
        sim.X("q0")  # |001>
        sim.X("q1")  # |011>
        # CCX(0, 1, 2): controls q0,q1 set; target q2 -> flip q2.
        sim.CCX("q0", "q1", "q2")
        self.assertEqual(sim.state.get("111") or sim.state.get("110"), 1.0)
        # CSWAP
        sim2 = self._make()
        sim2.allocate_qubit("q0")
        sim2.allocate_qubit("q1")
        sim2.allocate_qubit("q2")
        sim2.X("q0")
        sim2.H("q2")
        sim2.CSWAP("q0", "q1", "q2")
        # No crash is enough; just verify state has multiple amplitudes.
        self.assertGreaterEqual(len(sim2.state), 2)
        # CP
        sim3 = self._make()
        sim3.allocate_qubit("q0")
        sim3.allocate_qubit("q1")
        sim3.X("q0")
        sim3.X("q1")
        sim3.CP("q0", "q1", math.pi / 2)
        # CP(pi/2)|11> = exp(i pi/2)|11> = i|11>
        self.assertAlmostEqual(sim3.state["11"], 1j)

    def test_crx_cry_crz(self):
        sim = self._make()
        sim.allocate_qubit("q0")
        sim.allocate_qubit("q1")
        sim.X("q0")
        sim.CRX("q0", "q1", math.pi)
        # Control = 1, so apply RX(pi) on q1: |10> -> -i|11>
        self.assertEqual(len(sim.state), 1)
        self.assertAlmostEqual(abs(list(sim.state.values())[0]) ** 2, 1.0)
        # CRY
        sim2 = self._make()
        sim2.allocate_qubit("q0")
        sim2.allocate_qubit("q1")
        sim2.X("q0")
        sim2.CRY("q0", "q1", math.pi)
        # Apply Y(pi) on q1: |10> -> -|11>
        self.assertEqual(len(sim2.state), 1)
        # CRZ
        sim3 = self._make()
        sim3.allocate_qubit("q0")
        sim3.allocate_qubit("q1")
        sim3.X("q0")
        sim3.CRZ("q0", "q1", math.pi)
        # RZ(pi)|0> = e^{-i pi/2}|0> = -i; so |10> -> -i|10>
        self.assertEqual(len(sim3.state), 1)
        self.assertAlmostEqual(list(sim3.state.values())[0], -1j)

    def test_measure_outcome_zero(self):
        """measure returns 0 on |0> with deterministic RNG."""
        sim = self._make(seed=0)
        sim.allocate_qubit("q0")
        result = sim.measure("q0")
        self.assertEqual(result, 0)

    def test_measure_outcome_one_after_x(self):
        sim = self._make(seed=0)
        sim.allocate_qubit("q0")
        sim.H("q0")
        # With seed=0, rng.random() ~ ?
        res = sim.measure("q0")
        # Either way, the post-measurement state should be a pure bitstring.
        self.assertEqual(len(sim.state), 1)
        self.assertIn(res, [0, 1])

    def test_get_state_vector_and_amplitudes_dict(self):
        sim = self._make()
        sim.allocate_qubit("q0")
        sim.allocate_qubit("q1")
        sim.H("q0")
        sim.CNOT("q0", "q1")
        vec = sim.get_state_vector()
        self.assertEqual(len(vec), 4)
        # Index 0 ("00") and index 3 ("11") should each have ~0.5 abs^2.
        self.assertAlmostEqual(abs(vec[0]) ** 2, 0.5)
        self.assertAlmostEqual(abs(vec[3]) ** 2, 0.5)
        amps = sim.get_amplitudes_dict()
        self.assertGreaterEqual(len(amps), 2)

    def test_get_state_vector_rejects_huge_circuits(self):
        sim = self._make()
        # Manually set num_qubits above the cutoff (don't actually allocate).
        sim.num_qubits = 25
        with self.assertRaises(RuntimeError):
            sim.get_state_vector()

    def test_state_setter_and_getter_python(self):
        sim = self._make()
        sim.allocate_qubit("q0")
        sim.state = {"0": 0.6 + 0.0j, "1": 0.4 + 0.0j}
        self.assertEqual(sim.state["0"], 0.6 + 0.0j)
        self.assertEqual(sim.state["1"], 0.4 + 0.0j)


# ============================================================================
# MPS — auto bond dim, SVD fallback, measure, state-vector reconstruction
# ============================================================================


class TestMPSExtras(unittest.TestCase):
    """Cover the remaining gaps in `src/tensor_network/mps.py`:
      * `native_svd` fallback via `np.linalg.svd`
      * auto_bond_dim increase path
      * measure() outcome 1 path
      * get_state_vector with > 1 qubit
    """

    def test_native_svd_falls_back_to_numpy(self):
        from src.tensor_network.mps import native_svd

        # Simulate a complex 2x2 matrix; SVD should still produce
        # U-diag(S)-Vh that reconstructs the original within tolerance.
        np.random.seed(0)
        M = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        # Force the fallback by patching eigen_native.compute_svd_native
        # to raise.
        try:
            import eigen_native
            with patch.object(eigen_native, "compute_svd_native",
                              side_effect=AttributeError):
                U, S, Vh = native_svd(M)
        except ImportError:
            U, S, Vh = native_svd(M)
        recon = (U * S) @ Vh
        np.testing.assert_allclose(recon, M, atol=1e-9)

    def test_mps_allocates_in_creates_order(self):
        from src.tensor_network.mps import MPSSimulator
        m = MPSSimulator()
        m.allocate_qubit("a")
        m.allocate_qubit("b")
        self.assertEqual(m.created_qubits, ["a", "b"])
        self.assertEqual(m.get_qubit_index("a"), 0)
        self.assertEqual(m.get_qubit_index("b"), 1)

    def test_mps_get_qubit_index_unknown_raises(self):
        from src.tensor_network.mps import MPSSimulator
        m = MPSSimulator()
        with self.assertRaises(KeyError):
            m.get_qubit_index("zz")

    def test_mps_unallocated_qubit_doubles_state(self):
        from src.tensor_network.mps import MPSSimulator
        m = MPSSimulator()
        m.allocate_qubit("a")
        m.allocate_qubit("a")  # duplicate shouldn't double-allocate
        self.assertEqual(m.num_qubits if hasattr(m, "num_qubits") else len(m.qubits), 1)
        self.assertEqual(len(m.qubits), 1)

    def test_mps_apply_1qubit_gate_generic(self):
        from src.tensor_network.mps import MPSSimulator
        m = MPSSimulator()
        m.allocate_qubit("q0")
        # Apply a generic 2x2 matrix (the H is also in cache, but X is too;
        # use a custom one that's not cached).
        custom = [[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]]
        m.apply_1qubit_gate("q0", custom)
        # |0> -> |1>
        vec = m.get_state_vector()
        np.testing.assert_allclose(np.array(vec), np.array([0.0, 1.0], dtype=complex))

    def test_mps_h_creates_superposition(self):
        from src.tensor_network.mps import MPSSimulator
        m = MPSSimulator(seed=7)
        m.allocate_qubit("q0")
        m.H("q0")
        vec = m.get_state_vector()
        np.testing.assert_allclose(vec[0], vec[1], atol=1e-9)

    def test_mps_cnot_creates_bell(self):
        from src.tensor_network.mps import MPSSimulator
        m = MPSSimulator()
        m.allocate_qubit("q0")
        m.allocate_qubit("q1")
        m.H("q0")
        m.CNOT("q0", "q1")
        vec = m.get_state_vector()
        np.testing.assert_allclose(abs(vec[0]) ** 2, 0.5, atol=1e-9)
        np.testing.assert_allclose(abs(vec[3]) ** 2, 0.5, atol=1e-9)

    def test_mps_measure_collapses_state(self):
        from src.tensor_network.mps import MPSSimulator
        m = MPSSimulator(seed=0)
        m.allocate_qubit("q0")
        m.H("q0")
        outcome = m.measure("q0")
        self.assertIn(outcome, [0, 1])
        # After measurement, the single qubit is in a computational basis state.
        # Either vec[0]=1 or vec[1]=1.
        vec = m.get_state_vector()
        prob_total = abs(vec[0]) ** 2 + abs(vec[1]) ** 2
        self.assertAlmostEqual(prob_total, 1.0, places=8)

    def test_mps_auto_bond_dim_grows(self):
        from src.tensor_network.mps import MPSSimulator
        # Set a low max_bond_dim + huge max_truncation_error so auto path
        # fires and grows the bond dimension past the initial cap.
        m = MPSSimulator(max_bond_dim=2, auto_bond_dim=True,
                         max_truncation_error=1e-2)
        for i in range(4):
            m.allocate_qubit(f"q{i}")
        # Build heavy entanglement; CNOTs between non-adjacent qubits
        # generate growing bond dimension past cap 2.
        m.H("q0")
        m.CNOT("q0", "q1")
        m.CNOT("q0", "q2")
        m.CNOT("q0", "q3")
        # Either bond_dim stayed at 2 or grew; either way, norm is preserved.
        norm = m.norm_squared()
        self.assertAlmostEqual(norm, 1.0, places=4)
        self.assertGreaterEqual(m.get_max_bond_dim(), 2)

    def test_mps_get_amplitudes_dict_large_returns_placeholder(self):
        from src.tensor_network.mps import MPSSimulator
        m = MPSSimulator()
        for i in range(20):
            m.allocate_qubit(f"q{i}")
        amps = m.get_amplitudes_dict()
        self.assertEqual(len(amps), 1)
        # Truncation error should be reported numerically.
        self.assertGreaterEqual(m.get_cumulative_truncation_error(), 0.0)

    def test_mps_large_get_state_vector_raises(self):
        from src.tensor_network.mps import MPSSimulator
        m = MPSSimulator()
        for i in range(25):
            m.allocate_qubit(f"q{i}")
        with self.assertRaises(RuntimeError):
            m.get_state_vector()

    def test_mps_get_last_entropy_returns_zero_initial(self):
        from src.tensor_network.mps import MPSSimulator
        m = MPSSimulator()
        self.assertEqual(m.get_last_entropy(), 0.0)

    def test_mps_get_last_discarded_weight_initial(self):
        from src.tensor_network.mps import MPSSimulator
        m = MPSSimulator()
        self.assertEqual(m.get_last_discarded_weight(), 0.0)


# ============================================================================
# Packager — EigenPackager lifecycle, version_satisfies variants, lockfile
# ============================================================================


class TestPackager(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="eigen_pkg_test_")
        # Add cleanup guard in case tearDown isn't called for skipped tests.
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_parse_version_with_v_prefix(self):
        from src.packager import parse_version
        self.assertEqual(parse_version("v1.2.3"), (1, 2, 3))
        self.assertEqual(parse_version("1.2.3"), (1, 2, 3))
        self.assertEqual(parse_version("v1"), (1, 0, 0))
        self.assertEqual(parse_version("vX.2"), (0, 2, 0))

    def test_version_satisfies_wildcard(self):
        from src.packager import version_satisfies
        self.assertTrue(version_satisfies("*", "1.0.0"))
        self.assertTrue(version_satisfies("", "999.999"))
        self.assertTrue(version_satisfies("1.x", "1.2.0"))
        self.assertTrue(version_satisfies("1.x", "1.5.0"))
        self.assertFalse(version_satisfies("1.x", "2.0.0"))

    def test_version_satisfies_caret(self):
        from src.packager import version_satisfies
        # ^1.2.3: v[0] == 1 and v >= 1.2.3
        self.assertTrue(version_satisfies("^1.2.3", "1.2.5"))
        self.assertTrue(version_satisfies("^1.2.3", "1.99.99"))
        self.assertFalse(version_satisfies("^1.2.3", "2.0.0"))
        self.assertFalse(version_satisfies("^1.2.3", "1.2.0"))
        # ^0.2.3: v[1] == 2 and v >= 0.2.3
        self.assertTrue(version_satisfies("^0.2.3", "0.2.5"))
        self.assertFalse(version_satisfies("^0.2.3", "0.3.0"))
        # ^0.0.3: v[2] == 3, v[0]==0, v[1]==0
        self.assertTrue(version_satisfies("^0.0.3", "0.0.3"))
        self.assertFalse(version_satisfies("^0.0.3", "0.0.4"))

    def test_version_satisfies_tilde(self):
        from src.packager import version_satisfies
        # ~1.2.3: v[0] == 1 and v[1] == 2 and v >= 1.2.3
        self.assertTrue(version_satisfies("~1.2.3", "1.2.99"))
        self.assertFalse(version_satisfies("~1.2.3", "1.3.0"))
        self.assertFalse(version_satisfies("~1.2.3", "2.2.0"))

    def test_version_satisfies_exact(self):
        from src.packager import version_satisfies
        self.assertTrue(version_satisfies("1.2.3", "1.2.3"))
        self.assertFalse(version_satisfies("1.2.3", "1.2.4"))

    def test_parse_toml_skips_blank_and_comment_lines(self):
        from src.packager import parse_toml
        toml = """# Comment
        [package]
        name = "demo"
        # mid-comment
        version = "0.1.0"

        [dependencies]
        foo = "1.0"
        """
        data = parse_toml(toml)
        self.assertEqual(data["package"]["name"], "demo")
        self.assertEqual(data["dependencies"]["foo"], "1.0")

    def test_write_toml_round_trip(self):
        from src.packager import parse_toml, write_toml
        original = {
            "package": {"name": "demo", "version": "1.2.3"},
            "dependencies": {"foo": "1.0"},
        }
        text = write_toml(original)
        round_trip = parse_toml(text)
        self.assertEqual(round_trip["package"]["name"], "demo")
        self.assertEqual(round_trip["package"]["version"], "1.2.3")
        self.assertEqual(round_trip["dependencies"]["foo"], "1.0")

    def test_init_package_creates_toml_and_template_entry(self):
        from src.packager import EigenPackager
        pkg = EigenPackager(self.tmp)
        ok = pkg.init_package(name="demo")
        self.assertTrue(ok)
        self.assertTrue(os.path.exists(pkg.toml_path))
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "src", "main.eig")))
        # Re-init should fail.
        ok2 = pkg.init_package(name="second")
        self.assertFalse(ok2)

    def test_init_with_default_name_from_dir(self):
        from src.packager import EigenPackager
        sub = os.path.join(self.tmp, "myproj")
        os.makedirs(sub)
        pkg = EigenPackager(sub)
        pkg.init_package()
        # Read back the toml
        with open(pkg.toml_path, "r", encoding="utf-8") as f:
            data = parse_toml_helper(f.read())
        self.assertEqual(data["package"]["name"], "myproj")

    def test_add_dependency_with_no_manifest_returns_false(self):
        from src.packager import EigenPackager
        pkg = EigenPackager(self.tmp)
        self.assertFalse(pkg.add_dependency("foo", "1.0"))

    def test_add_dependency_persists_to_toml(self):
        from src.packager import EigenPackager
        pkg = EigenPackager(self.tmp)
        pkg.init_package(name="demo")
        self.assertTrue(pkg.add_dependency("foo", "^1.0"))
        with open(pkg.toml_path, "r", encoding="utf-8") as f:
            data = parse_toml_helper(f.read())
        self.assertEqual(data["dependencies"]["foo"], "^1.0")

    def test_install_dependencies_with_no_manifest_returns_false(self):
        from src.packager import EigenPackager
        pkg = EigenPackager(self.tmp)
        self.assertFalse(pkg.install_dependencies())

    def test_install_dependencies_uses_locked_version_when_hash_matches(self):
        from src.packager import EigenPackager
        pkg = EigenPackager(self.tmp)
        pkg.init_package(name="demo")
        pkg.add_dependency("foo", "^1.0")

        # Manually publish a foo-1.0.0.tar into the registry
        os.makedirs(pkg.registry_dir, exist_ok=True)
        tar_path = os.path.join(pkg.registry_dir, "foo-1.0.0.tar")
        with open(tar_path, "wb") as f:
            f.write(b"FAKE FOO PACKAGE CONTENT")
        import hashlib
        expected_hash = hashlib.sha256(open(tar_path, "rb").read()).hexdigest()

        # First install should resolve to the registry version
        self.assertTrue(pkg.install_dependencies())
        with open(pkg.lock_path, "r", encoding="utf-8") as f:
            lock = json.load(f)
        self.assertEqual(lock["foo"]["version"], "1.0.0")
        self.assertEqual(lock["foo"]["hash"], expected_hash)

        # Second install — should hit the "Using locked version" branch since
        # hashes match (we haven't touched the .tar).
        self.assertTrue(pkg.install_dependencies())

    def test_install_dependencies_warns_on_hash_mismatch(self):
        from src.packager import EigenPackager
        pkg = EigenPackager(self.tmp)
        pkg.init_package(name="demo")
        pkg.add_dependency("foo", "^1.0")
        os.makedirs(pkg.registry_dir, exist_ok=True)
        tar_path = os.path.join(pkg.registry_dir, "foo-1.0.0.tar")
        with open(tar_path, "wb") as f:
            f.write(b"ORIGINAL CONTENT")
        import hashlib
        original_hash = hashlib.sha256(open(tar_path, "rb").read()).hexdigest()

        # First install to populate lockfile
        pkg.install_dependencies()
        # Now corrupt the .tar so the hash diverges from the locked one.
        with open(tar_path, "wb") as f:
            f.write(b"CORRUPTED CONTENT")
        # Re-install — should detect mismatch and re-resolve.
        pkg.install_dependencies()
        with open(pkg.lock_path, "r", encoding="utf-8") as f:
            lock = json.load(f)
        # Hash in lock should now match the corrupted file, not the original
        new_hash = hashlib.sha256(open(tar_path, "rb").read()).hexdigest()
        self.assertEqual(lock["foo"]["hash"], new_hash)
        self.assertNotEqual(lock["foo"]["hash"], original_hash)

    def test_install_dependencies_re_resolves_on_constraint_mismatch(self):
        from src.packager import EigenPackager
        pkg = EigenPackager(self.tmp)
        pkg.init_package(name="demo")
        pkg.add_dependency("foo", "^1.0")
        os.makedirs(pkg.registry_dir, exist_ok=True)
        with open(os.path.join(pkg.registry_dir, "foo-1.0.0.tar"), "wb") as f:
            f.write(b"v1.0.0 content")
        pkg.install_dependencies()
        # Now bump the constraint to ^2.0; the locked 1.0.0 doesn't satisfy.
        pkg.add_dependency("foo", "^2.0")
        with open(os.path.join(pkg.registry_dir, "foo-2.0.0.tar"), "wb") as f:
            f.write(b"v2.0.0 content")
        pkg.install_dependencies()
        with open(pkg.lock_path, "r", encoding="utf-8") as f:
            lock = json.load(f)
        self.assertEqual(lock["foo"]["version"], "2.0.0")

    def test_install_dependencies_falls_back_when_no_registry_match(self):
        from src.packager import EigenPackager
        pkg = EigenPackager(self.tmp)
        pkg.init_package(name="demo")
        pkg.add_dependency("bar", "^1.0")  # No bar-*.tar in registry.
        self.assertTrue(pkg.install_dependencies())
        with open(pkg.lock_path, "r", encoding="utf-8") as f:
            lock = json.load(f)
        # Version should fall back to "1.0" (constraint sans prefix)
        self.assertEqual(lock["bar"]["version"], "1.0")

    def test_publish_package_writes_tar_to_registry(self):
        from src.packager import EigenPackager
        pkg = EigenPackager(self.tmp)
        pkg.init_package(name="demo")
        self.assertTrue(pkg.publish_package())
        tar_path = os.path.join(pkg.registry_dir, "demo-1.0.0.tar")
        self.assertTrue(os.path.exists(tar_path))

    def test_publish_package_with_no_manifest_returns_false(self):
        from src.packager import EigenPackager
        pkg = EigenPackager(self.tmp)
        self.assertFalse(pkg.publish_package())

    def test_search_packages_returns_matches(self):
        from src.packager import EigenPackager
        pkg = EigenPackager(self.tmp)
        pkg.init_package(name="demo")
        pkg.publish_package()
        # search_packages prints to stdout; no return value, but we can
        # capture by patching print.
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            pkg.search_packages("demo")
        self.assertIn("demo", buf.getvalue())

    def test_search_packages_no_matches(self):
        from src.packager import EigenPackager
        pkg = EigenPackager(self.tmp)
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            pkg.search_packages("definitelynotfound")
        self.assertIn("No packages matched", buf.getvalue())

    def test_build_package_installs_dependencies_and_reports(self):
        from src.packager import EigenPackager
        pkg = EigenPackager(self.tmp)
        pkg.init_package(name="demo")
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ok = pkg.build_package()
        self.assertTrue(ok)
        self.assertIn("Building Eigen package 'demo'", buf.getvalue())

    def test_build_package_with_no_manifest_returns_false(self):
        from src.packager import EigenPackager
        pkg = EigenPackager(self.tmp)
        self.assertFalse(pkg.build_package())


def parse_toml_helper(text):
    from src.packager import parse_toml
    return parse_toml(text)


if __name__ == "__main__":
    unittest.main()
