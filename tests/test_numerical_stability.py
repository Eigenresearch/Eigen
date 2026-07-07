"""
Tests for src/numerical_stability.py — sol.md §2.2.
"""
import math
import unittest

from src.numerical_stability import (
    NumericalStabilityReport,
    normalise,
    TruncationMetrics,
    TruncationAccumulator,
    EntanglementEntropyTracker,
    binary_von_neumann_entropy,
    ApproximationLog,
)


class TestNormalise(unittest.TestCase):
    def test_unit_norm_returns_same(self):
        sv = [1 / math.sqrt(2), 1 / math.sqrt(2)]
        out, scale, report = normalise(sv)
        self.assertAlmostEqual(scale, 1.0, places=10)
        self.assertFalse(report.had_warning)

    def test_normalises_drifted_vector(self):
        sv = [3.0, 4.0]  # norm = 5
        out, scale, report = normalise(sv)
        # scale = 1/5
        self.assertAlmostEqual(scale, 0.2)
        # Renormalised vector should have unit norm
        norm_sq = sum(abs(z) ** 2 for z in out)
        self.assertAlmostEqual(math.sqrt(norm_sq), 1.0, places=10)
        self.assertTrue(report.had_warning)

    def test_empty_vector_warns(self):
        out, scale, report = normalise([])
        self.assertEqual(out, [])
        self.assertEqual(scale, 0.0)
        self.assertTrue(report.had_warning)

    def test_zero_vector_warns(self):
        sv = [0.0, 0.0, 0.0]
        out, scale, report = normalise(sv)
        self.assertTrue(report.had_warning)
        self.assertEqual(scale, 0.0)

    def test_report_drift_estimate(self):
        sv = [2.0, 0.0]  # norm = 2, drift = 1
        _, _, report = normalise(sv)
        self.assertAlmostEqual(report.drift_estimate, 1.0)

    def test_compensated_sum_handles_cancellation(self):
        """The stable L2 norm via math.fsum should not suffer from
        catastrophic cancellation when the state vector has many
        small-magnitude amplitudes summing to ~1."""
        n = 100_000
        # Choose magnitude so that unit norm is preserved.
        # norm² = N * s² ; norm=1 requires s = 1/sqrt(N).
        small = 1.0 / math.sqrt(n)
        sv = [complex(small, 0) for _ in range(n)]
        out, scale, report = normalise(sv)
        # scale should be ~1 (norm ≈ 1 already).
        self.assertAlmostEqual(scale, 1.0, places=5)
        self.assertFalse(report.had_warning)


class TestTruncationAccumulator(unittest.TestCase):
    def test_records_truncation_metrics(self):
        acc = TruncationAccumulator()
        m = acc.record(step_index=0, bond_dimension=4,
                        truncation_error=1e-3, discarded_weight=1e-3)
        self.assertEqual(m.bond_dimension, 4)
        self.assertEqual(len(acc.records), 1)
        self.assertAlmostEqual(acc.cumulative_discarded_weight, 1e-3)

    def test_cumulative_discarded_weight(self):
        acc = TruncationAccumulator()
        acc.record(0, 4, 1e-3, 1e-3)
        acc.record(1, 8, 1e-4, 1e-4)
        self.assertAlmostEqual(acc.cumulative_discarded_weight, 1.1e-3)

    def test_per_step_errors(self):
        acc = TruncationAccumulator()
        acc.record(0, 4, 1e-3, 0)
        acc.record(1, 4, 1e-5, 0)
        self.assertEqual(acc.per_step_errors(), [1e-3, 1e-5])

    def test_cumulative_error(self):
        acc = TruncationAccumulator()
        acc.record(0, 4, 1.0, 0)
        acc.record(1, 4, 2.0, 0)
        self.assertAlmostEqual(acc.cumulative_error(), 3.0)

    def test_max_per_step_error_returns_zero_when_empty(self):
        acc = TruncationAccumulator()
        self.assertEqual(acc.max_per_step_error(), 0.0)

    def test_max_per_step_error(self):
        acc = TruncationAccumulator()
        acc.record(0, 4, 1.0, 0)
        acc.record(1, 4, 3.0, 0)
        acc.record(2, 4, 2.0, 0)
        self.assertEqual(acc.max_per_step_error(), 3.0)


class TestEntanglementEntropyTracker(unittest.TestCase):
    def test_record_and_max(self):
        t = EntanglementEntropyTracker()
        t.record(0, 0.5)
        t.record(1, 1.5)
        t.record(2, 0.25)
        self.assertEqual(t.max_entropy(), 1.5)

    def test_mean(self):
        t = EntanglementEntropyTracker()
        t.record(0, 1.0)
        t.record(1, 2.0)
        self.assertAlmostEqual(t.mean_entropy(), 1.5)

    def test_max_entropy_empty(self):
        t = EntanglementEntropyTracker()
        self.assertEqual(t.max_entropy(), 0.0)

    def test_mean_entropy_empty(self):
        t = EntanglementEntropyTracker()
        self.assertEqual(t.mean_entropy(), 0.0)

    def test_entropy_at_returns_recorded_value(self):
        t = EntanglementEntropyTracker()
        t.record(0, 0.5)
        t.record(2, 1.5)
        self.assertEqual(t.entropy_at(0), 0.5)
        self.assertEqual(t.entropy_at(2), 1.5)
        # No record for cut 1
        self.assertIsNone(t.entropy_at(1))

    def test_nan_entropy_skipped(self):
        t = EntanglementEntropyTracker()
        t.record(0, float("nan"))
        # NaN should have been skipped
        self.assertEqual(t.entropies, [])


class TestBinaryVonNeumannEntropy(unittest.TestCase):
    def test_uniform_distribution_yields_log2_n(self):
        # 4 equally likely outcomes: H = log2(4) = 2
        h = binary_von_neumann_entropy([0.25, 0.25, 0.25, 0.25])
        self.assertAlmostEqual(h, 2.0)

    def test_certainty_yields_zero_entropy(self):
        # One certain event: H = 0
        h = binary_von_neumann_entropy([1.0, 0.0, 0.0])
        self.assertEqual(h, 0.0)

    def test_zero_probabilities_skipped(self):
        # All-zero list (caller error): H = 0
        h = binary_von_neumann_entropy([0.0, 0.0, 0.0])
        self.assertEqual(h, 0.0)

    def test_two_outcome_uniform(self):
        # (0.5, 0.5): H = 1 bit
        h = binary_von_neumann_entropy([0.5, 0.5])
        self.assertAlmostEqual(h, 1.0)

    def test_greater_than_one_clamped(self):
        # p > 1 should be silently clamped in the formula
        h = binary_von_neumann_entropy([2.0, 0.0])
        # 2.0 → clamped to 1.0 → -1*log2(1) = 0
        self.assertEqual(h, 0.0)


class TestApproximationLog(unittest.TestCase):
    def test_log_appends_entry(self):
        log = ApproximationLog()
        log.log("MPS", "truncation", step=5, discarded_weight=1e-3)
        self.assertEqual(len(log.entries), 1)
        self.assertEqual(log.entries[0]["step"], 5)

    def test_by_method_filters(self):
        log = ApproximationLog()
        log.log("MPS", "truncation")
        log.log("Stabilizer", "decompose")
        log.log("MPS", "compression")
        mps_entries = log.by_method("MPS")
        self.assertEqual(len(mps_entries), 2)
        self.assertEqual(len(log.by_method("Stabilizer")), 1)

    def test_summary_counts_by_method(self):
        log = ApproximationLog()
        log.log("MPS", "truncation")
        log.log("MPS", "truncation")
        log.log("Stabilizer", "decompose")
        summary = log.summary()
        self.assertEqual(summary, {"MPS": 2, "Stabilizer": 1})


class TestRealisticScenario(unittest.TestCase):
    """Simulate a small MPS-style circuit and verify the truncation
    log + entropy tracker + cumulative error all report the
    expected values."""

    def test_small_circuit_accumulation(self):
        # Three truncation steps; check cumulative discarded weight.
        acc = TruncationAccumulator()
        for step in range(3):
            acc.record(step_index=step, bond_dimension=8,
                         truncation_error=1e-3 * (step + 1),
                         discarded_weight=1e-3 * (step + 1))
        self.assertEqual(len(acc.records), 3)
        # Cumulative weight = 1e-3 * (1+2+3) = 6e-3
        self.assertAlmostEqual(acc.cumulative_discarded_weight, 6e-3)
        # Cumulative error = 6e-3 too (same per-step)
        self.assertAlmostEqual(acc.cumulative_error(), 6e-3)

    def test_log_documents_approximation_use(self):
        log = ApproximationLog()
        # The caller documents what approximation was applied at
        # what step.
        log.log("MPS", "truncation", step=3, bond_dim=8,
                  discarded_weight=1e-4)
        log.log("Stabilizer", "decompose", gate_count=10)
        # Summaries
        self.assertEqual(log.summary()["MPS"], 1)
        self.assertEqual(log.summary()["Stabilizer"], 1)

    def test_normalise_after_truncation(self):
        """After a truncation step, the MPS state vector's norm
        drifts slightly; normalise() should bring it back to 1."""
        sv = [0.707, 0.707, 1e-10, 1e-10]  # drifted
        out, scale, report = normalise(sv)
        self.assertAlmostEqual(sum(abs(z) ** 2 for z in out), 1.0,
                                places=8)


class TestStableSumAvoidsCancellation(unittest.TestCase):
    """Verify that the L2 norm uses compensated summation, which is
    the §2.2 numerical-drift-avoidance promise."""

    def test_sum_of_small_values_close_to_unit(self):
        # 10^5 small amplitudes each of magnitude 1/sqrt(10^5) →
        # squared norm = 10^5 * (1/10^5) = 1.0.
        n = 100_000
        small = 1.0 / math.sqrt(n)
        sv = [complex(small, 0) for _ in range(n)]
        out, scale, report = normalise(sv)
        # scale should be ~1 (the input already has unit norm)
        self.assertAlmostEqual(scale, 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
