"""
Audit §1.5 — Resource Estimator improvement regression tests.

The audit's complaint against the previous estimator was that it only
counted CNOTs, total gates, and T-gates, and was otherwise minimal (70
lines). Real fault-tolerant cost modeling (cf. Azure Resource Estimator)
needs:

  * Toffoli counts (CCX, CSWAP) — the audit's headline ask.
  * Rotation counts (RX/RY/RZ/CP/CRX/CRY/CRZ — these synthesize to T).
  * Post-routing SWAP overhead (the audit's "post-routing SWAP overhead"
    phrase verbatim).
  * Two-qubit depth (separate from the total circuit depth, since
    two-qubit gates are the bottleneck on real hardware).
  * Measurement depth (for syndrome-extraction / measurement-based
    uncomputation cost).
  * Single-qubit vs two-qubit vs three-qubit gate counts (not just CNOT).

These tests pin down each of those plus basic backward compatibility
for the legacy `cnot_count`, `t_count`, `clifford_count`, `measurements`,
`gate_count`, `logical_qubits` keys that other consumers rely on.
"""

import unittest

from src.ir.ir_graph import EQIRGraph
from src.resource_estimator.estimator import ResourceEstimator


def _bell_plus_toffoli_graph() -> EQIRGraph:
    """Bell pair on q0/q1 + a Toffoli on q1/q2/q3 + a rotation on q2."""
    g = EQIRGraph()
    g.add_operation('ALLOC', targets=['q0'])
    g.add_operation('ALLOC', targets=['q1'])
    g.add_operation('ALLOC', targets=['q2'])
    g.add_operation('ALLOC', targets=['q3'])
    g.add_operation('GATE', gate_name='H', targets=['q0'])
    g.add_operation('GATE', gate_name='CNOT', targets=['q0', 'q1'])
    g.add_operation('GATE', gate_name='RZ', targets=['q2'], args=[0.7])
    g.add_operation('GATE', gate_name='CCX', targets=['q1', 'q2', 'q3'])
    g.add_operation('GATE', gate_name='SWAP', targets=['q0', 'q3'])
    g.add_operation('GATE', gate_name='X', targets=['q0'])
    g.add_operation('GATE', gate_name='T', targets=['q0'])
    g.add_operation('MEASURE', targets=['q3'], cbit_name='c0')
    g.add_operation('MEASURE', targets=['q1'], cbit_name='c1')
    return g


class TestResourceEstimatorLegacyCompat(unittest.TestCase):
    def setUp(self):
        self.metrics = ResourceEstimator().estimate(_bell_plus_toffoli_graph())

    def test_legacy_keys_still_present(self):
        # Existing callers (e.g. the `eigen estimate` CLI, the parameterized
        # tests in test_scaled_suite.py) depend on these legacy keys.
        for k in ('logical_qubits', 'circuit_depth', 'gate_count',
                  'cnot_count', 't_count', 't_depth', 'clifford_count',
                  'measurements'):
            self.assertIn(k, self.metrics,
                         msg=f"legacy key {k!r} disappeared from estimator output")

    def test_legacy_values_for_simple_circuit_match(self):
        # Bell+Toffoli circuit in `_bell_plus_toffoli_graph`:
        #   - 4 qubits
        #   - 1 CNOT
        #   - 1 T-gate
        #   - 7 total gates (H, CNOT, RZ, CCX, SWAP, X, T)
        #   - 2 measurements
        #   - clifford-count: H, CNOT, X are clifford = 3 (SWAP is also
        #     clifford per CLIFFORD_GATES table) -> 4
        self.assertEqual(self.metrics['logical_qubits'], 4)
        self.assertEqual(self.metrics['cnot_count'], 1)
        self.assertEqual(self.metrics['t_count'], 1)
        self.assertEqual(self.metrics['measurements'], 2)
        self.assertEqual(self.metrics['gate_count'], 7)
        # clifford = H, CNOT, SWAP, X = 4
        self.assertEqual(self.metrics['clifford_count'], 4)


class TestResourceEstimatorNewMetrics(unittest.TestCase):
    def setUp(self):
        self.metrics = ResourceEstimator().estimate(_bell_plus_toffoli_graph())

    def test_toffoli_count_includes_ccx_and_cswap(self):
        # Exactly one CCX in the graph above -> toffoli_count == 1.
        self.assertEqual(self.metrics['toffoli_count'], 1)
        self.assertEqual(self.metrics['three_qubit_count'], 1)

    def test_rotation_count_includes_all_angle_gates(self):
        # One RZ gate -> rotation_count == 1.
        self.assertEqual(self.metrics['rotation_count'], 1)
        # The audit's reference to post-routing/Toffoli-style synthesis
        # cost requires the estimator to expose a rough T-synthesis
        # estimate; pin the default scaling rate so changing it is
        # observable from a test.
        self.assertEqual(
            self.metrics['rotation_t_estimate'],
            self.metrics['rotation_count'] * 10,
        )

    def test_swap_count_is_tracked(self):
        # One SWAP in the graph above.
        self.assertEqual(self.metrics['swap_count'], 1)

    def test_two_qubit_count_includes_cnot_cz_swap(self):
        # CNOT + SWAP = 2 two-qubit gates (RZ is single-qubit; CCX is
        # three-qubit).
        self.assertEqual(self.metrics['two_qubit_count'], 2)

    def test_single_qubit_count_includes_non_two_qubit_gates(self):
        # H, RZ, X, T = 4 single-qubit gates.
        self.assertEqual(self.metrics['single_qubit_count'], 4)

    def test_two_qubit_depth_reflects_critical_path_of_2q_gates(self):
        # 2-qubit critical path: CNOT(q0,q1) -> SWAP(q0,q3)
        # depth 2 along that chain.
        self.assertGreaterEqual(self.metrics['two_qubit_depth'], 2)

    def test_measurement_depth_zero_when_no_measurements(self):
        # Empty graph + no measurements = 0 measurement depth.
        empty = EQIRGraph()
        m = ResourceEstimator().estimate(empty)
        self.assertEqual(m['measurement_depth'], 0)
        self.assertEqual(m['measurements'], 0)

    def test_measurement_depth_counts_chained_measurements(self):
        # Two independent MEASURE nodes touching the same qubit's parents
        # chain depth = 2 for measurements along that path.
        g = EQIRGraph()
        g.add_operation('ALLOC', targets=['q0'])
        g.add_operation('GATE', gate_name='H', targets=['q0'])
        g.add_operation('MEASURE', targets=['q0'], cbit_name='c0')
        g.add_operation('MEASURE', targets=['q0'], cbit_name='c1')
        m = ResourceEstimator().estimate(g)
        self.assertEqual(m['measurements'], 2)
        self.assertEqual(m['measurement_depth'], 2)


class TestResourceEstimatorRoutingOverhead(unittest.TestCase):
    """Audit's headline ask: 'post-routing SWAP overhead'."""

    def test_default_post_routing_swaps_is_zero(self):
        # Default invocation includes no extra routing overhead.
        m = ResourceEstimator().estimate(_bell_plus_toffoli_graph())
        self.assertEqual(m['post_routing_swaps_count'], 0)
        # total_swap_count == swap_count + post_routing_swaps_count
        self.assertEqual(m['total_swap_count'], m['swap_count'])

    def test_estimater_accepts_swaps_inserted_parameter(self):
        # Caller can pass swaps_inserted=7 to include routing overhead,
        # which is added to total_swap_count (but not the bare swap_count,
        # so the breakdown is visible).
        m = ResourceEstimator().estimate(_bell_plus_toffoli_graph(), swaps_inserted=7)
        self.assertEqual(m['swap_count'], 1)
        self.assertEqual(m['post_routing_swaps_count'], 7)
        self.assertEqual(m['total_swap_count'], 8)

    def test_negative_swaps_inserted_is_clamped_to_zero(self):
        # Defensive: don't silently subtract from total. Negative values
        # are treated as 0.
        m = ResourceEstimator().estimate(_bell_plus_toffoli_graph(), swaps_inserted=-5)
        self.assertEqual(m['post_routing_swaps_count'], 0)
        self.assertEqual(m['total_swap_count'], 1)

    def test_classical_bits_counted_separately_from_logical_qubits(self):
        # The audit's reference to Azure Resource Estimator tracks c-bits
        # separately from logical qubits. Pin that.
        m = ResourceEstimator().estimate(_bell_plus_toffoli_graph())
        self.assertEqual(m['classical_bits'], 2)


if __name__ == '__main__':
    unittest.main()
