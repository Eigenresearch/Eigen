"""Regression tests for the centralized automatic simulator selector
(``src/backend/sim_selector.py``), introduced in sol.md §5.1.

The selector replaces three near-duplicate ad-hoc decision sites in
``src/backend/vm.py``, ``src/runtime.py`` and ``src/commands/run.py``. Each
case below pins one branch of the §5.1 mermaid decision-diagram and gives
the audit/research reader a stable reference for future refactors.
"""

import unittest

from src.backend.sim_selector import (
    SimSelector,
    CircuitMetrics,
    select_from_counts,
    DEFAULT_SELECTOR,
)


class TestSimSelector(unittest.TestCase):

    def test_clifford_only_routes_to_stabilizer(self):
        # 50 qubits, all-Clifford, ~0 entanglement → stabilizer wins
        r = select_from_counts(n_qubits=50, n_2q_gates=0, n_gates=10,
                               is_all_clifford=True)
        self.assertEqual(r.chosen, "stabilizer")
        self.assertIn("Clifford", r.reason)

    def test_clifford_above_stabilizer_cap_falls_through(self):
        # Stabilizer has its own qubit cap (10_000 default).
        r = select_from_counts(n_qubits=99999, n_2q_gates=0,
                               is_all_clifford=True)
        self.assertNotEqual(r.chosen, "stabilizer")

    def test_two_qubit_clifford_go_stabilizer(self):
        # Tiny circuits that are pure-Clifford still go to stabilizer.
        r = select_from_counts(n_qubits=2, n_2q_gates=1, n_gates=2,
                               is_all_clifford=True)
        self.assertEqual(r.chosen, "stabilizer")

    def test_non_clifford_small_circuit_goes_dense(self):
        # 4 qubits, includes T → dense statevector (no noise, fits easily).
        r = select_from_counts(n_qubits=4, n_2q_gates=2, n_gates=6,
                               is_all_clifford=False)
        self.assertEqual(r.chosen, "dense")

    def test_noise_active_selects_density_matrix(self):
        # Stochastic noise requires a density matrix backend.
        r = select_from_counts(n_qubits=4, n_2q_gates=2,
                               is_all_clifford=False, noise_active=True)
        self.assertEqual(r.chosen, "density_matrix")

    def test_noise_with_too_many_qubits_falls_back_gracefully(self):
        # Density matrix overflows memory at 16 qubits (~64 GiB for DM but
        # only ~17 GiB budget by default). Selector should fire DM first then
        # degrade to mps/sparse via the fallback path.
        r = select_from_counts(n_qubits=16, n_2q_gates=2,
                               is_all_clifford=False, noise_active=True)
        self.assertIn(r.chosen, ("mps", "sparse"))
        self.assertTrue(r.fallback_used)
        self.assertEqual(r.fallback_from, "density_matrix")

    def test_low_entanglement_at_high_qubit_count_picks_mps(self):
        # 50 qubits, 50 gates, low 2-qubit gate count → MPS.
        r = select_from_counts(n_qubits=50, n_2q_gates=5,
                               is_all_clifford=False)
        self.assertEqual(r.chosen, "mps")

    def test_high_entanglement_above_dense_cap_picks_sparse(self):
        # 30 qubits, dense cap is 25, high entanglement → sparse fallback.
        r = select_from_counts(n_qubits=30, n_2q_gates=50,
                               is_all_clifford=False)
        self.assertEqual(r.chosen, "sparse")

    def test_user_hint_overrides_everything(self):
        # User explicitly asks for 'mps' even on a tiny Clifford circuit.
        r = select_from_counts(n_qubits=2, n_2q_gates=1,
                               is_all_clifford=True, user_hint="mps")
        self.assertEqual(r.chosen, "mps")
        self.assertEqual(r.user_hint, "mps")

    def test_invalid_hint_ignored(self):
        # An unrecognized hint silently degrades to normal auto behavior.
        r = select_from_counts(n_qubits=2, n_2q_gates=1,
                               is_all_clifford=True, user_hint="qiskit")
        self.assertEqual(r.chosen, "stabilizer")

    def test_zero_qubits_defaults_to_dense(self):
        # Defensive: zero-qubit circuit shouldn't crash any rule.
        r = select_from_counts(n_qubits=0, n_2q_gates=0)
        self.assertEqual(r.chosen, "dense")

    def test_selection_report_str_repr(self):
        # The audit/CLI ``print`` of SelectionReport shouldn't blow up.
        r = select_from_counts(n_qubits=50, n_2q_gates=2,
                               is_all_clifford=True, user_hint="stabilizer")
        s = str(r)
        self.assertIn("user-hint override", s)
        self.assertIn("stabilizer", s)

    def test_custom_subclass_can_override_rule(self):
        # A user / research variant can substitute a rule for a custom policy.
        class CustomSelector(SimSelector):
            def _rule_stabilizer(self, metrics):
                # Always returns dense for testing customisation.
                return None, ""

        sel = CustomSelector()
        r = sel.select(CircuitMetrics(n_qubits=4, is_all_clifford=True))
        self.assertNotEqual(r.chosen, "stabilizer")

    def test_metrics_carried_in_report(self):
        # The Metrics object should round-trip via the Selector's report
        # so downstream tools (audit / profile) can inspect the inputs.
        m = CircuitMetrics(n_qubits=10, n_2q_gates=3, n_gates=12,
                           is_all_clifford=True, noise_active=False)
        r = DEFAULT_SELECTOR.select(m)
        self.assertIs(r.metrics, m)
        self.assertEqual(r.metrics.n_qubits, 10)


if __name__ == "__main__":
    unittest.main()
