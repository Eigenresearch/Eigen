"""§2.3 — Verification Correctness tests.

Tests are organised by the four roadmap checkboxes:

  1. Canonical hash как fast-reject, НЕ как proof of equivalence.
  2. Rewrite-based верификация как fallback.
  3. Exact equivalence только где математически обосновано.
  4. Предупреждения в CLI output о границах верификации.
"""
import math
import unittest

from src.ir.ir_graph import EQIRGraph
from src.verification_correctness import (
    VerificationMode,
    VerificationReport,
    verify_equivalence,
    format_cli_banner,
)


def _make_identity_graph() -> EQIRGraph:
    """A graph that does nothing — equivalent only to a graph
    that performs no operations on the same set of qubits."""
    g = EQIRGraph()
    return g


def _make_h_graph(qname: str = "q0") -> EQIRGraph:
    """A graph that applies a single H gate."""
    g = EQIRGraph()
    g.add_operation("GATE", gate_name="H", args=[], targets=[qname])
    return g


def _make_x_graph(qname: str = "q0") -> EQIRGraph:
    """A graph that applies a single X gate."""
    g = EQIRGraph()
    g.add_operation("GATE", gate_name="X", args=[], targets=[qname])
    return g


def _make_hh_graph(qname: str = "q0") -> EQIRGraph:
    """Two consecutive H gates — equivalent to identity (for
    non-conditional, non-measurement circuits)."""
    g = EQIRGraph()
    g.add_operation("GATE", gate_name="H", args=[], targets=[qname])
    g.add_operation("GATE", gate_name="H", args=[], targets=[qname])
    return g


def _make_x_then_h_graph(qname: str = "q0") -> EQIRGraph:
    """X followed by H — NOT equivalent to identity."""
    g = EQIRGraph()
    g.add_operation("GATE", gate_name="X", args=[], targets=[qname])
    g.add_operation("GATE", gate_name="H", args=[], targets=[qname])
    return g


class TestVerificationMode(unittest.TestCase):
    """Sanity checks of the VerificationMode enum."""

    def test_modes_are_distinct(self):
        modes = {VerificationMode.FAST_REJECT, VerificationMode.REWRITE,
                 VerificationMode.EXACT, VerificationMode.AUTO}
        self.assertEqual(len(modes), 4)

    def test_mode_value_is_string(self):
        for m in VerificationMode:
            self.assertIsInstance(m.value, str)


class TestFastReject(unittest.TestCase):
    """§2.3 item 1: canonical hash is a fast-reject, NOT proof."""

    def test_different_graphs_hash_mismatch_returns_false(self):
        """Two structurally different graphs MUST have different
        canonical hashes; FAST_REJECT MUST report False."""
        g1 = _make_h_graph()
        g2 = _make_x_graph()
        report = verify_equivalence(g1, g2, mode=VerificationMode.FAST_REJECT)
        self.assertFalse(report.is_equivalent)
        self.assertEqual(report.mode_used, VerificationMode.FAST_REJECT)
        self.assertTrue(report.canonical_hash_1 != report.canonical_hash_2)
        self.assertTrue(any("definitely NOT equivalent" in w
                             for w in report.warnings))

    def test_same_graph_hash_match_is_inconclusive(self):
        """Two structurally identical graphs MUST have matching
        canonical hashes; FAST_REJECT MUST report None
        (inconclusive — NOT a proof of equivalence)."""
        g1 = _make_h_graph()
        g2 = _make_h_graph()
        report = verify_equivalence(g1, g2, mode=VerificationMode.FAST_REJECT)
        self.assertIsNone(report.is_equivalent)
        self.assertEqual(report.canonical_hash_1, report.canonical_hash_2)
        self.assertTrue(any("does NOT prove equivalence" in w
                             for w in report.warnings))

    def test_same_graph_object_returns_inconclusive(self):
        """A single graph compared against itself should be
        inconclusive under FAST_REJECT — the canonical hash matches
        but we cannot claim a proof without further work."""
        g = _make_h_graph()
        report = verify_equivalence(g, g, mode=VerificationMode.FAST_REJECT)
        self.assertIsNone(report.is_equivalent)
        self.assertEqual(report.canonical_hash_1, report.canonical_hash_2)

    def test_fast_reject_records_hash_values(self):
        """The FAST_REJECT report MUST surface the canonical
        hashes computed for both graphs."""
        g1 = _make_h_graph()
        g2 = _make_x_graph()
        report = verify_equivalence(g1, g2, mode=VerificationMode.FAST_REJECT)
        self.assertIsNotNone(report.canonical_hash_1)
        self.assertIsNotNone(report.canonical_hash_2)
        self.assertIsInstance(report.canonical_hash_1, str)
        self.assertIsInstance(report.canonical_hash_2, str)


class TestExactMode(unittest.TestCase):
    """§2.3 item 3: exact equivalence only where mathematically
    justified — for small circuits (≤8 qubits)."""

    def test_hh_is_equivalent_to_identity(self):
        """H·H = I, so two H gates should be equivalent to no
        gate at all (under exact unitary comparison)."""
        g1 = _make_hh_graph()
        g2 = _make_identity_graph()
        report = verify_equivalence(g1, g2, mode=VerificationMode.EXACT)
        self.assertTrue(report.is_equivalent)
        self.assertEqual(report.mode_used, VerificationMode.EXACT)
        self.assertTrue(any("mathematically sound" in n for n in report.notes))

    def test_xh_is_not_equivalent_to_identity(self):
        """X·H ≠ I, so the EXACT comparison MUST return False."""
        g1 = _make_x_then_h_graph()
        g2 = _make_identity_graph()
        report = verify_equivalence(g1, g2, mode=VerificationMode.EXACT)
        self.assertFalse(report.is_equivalent)

    def test_h_not_equivalent_to_x(self):
        """H ≠ X — these are different unitaries and EXACT mode
        MUST return False."""
        g1 = _make_h_graph()
        g2 = _make_x_graph()
        report = verify_equivalence(g1, g2, mode=VerificationMode.EXACT)
        self.assertFalse(report.is_equivalent)

    def test_too_many_qubits_skips_exact(self):
        """For a circuit with more than 8 qubits the EXACT mode
        MUST skip and warn."""

        # Build a 9-qubit circuit (each qubit gets one H gate).
        def _make_9q_h():
            g = EQIRGraph()
            for i in range(9):
                g.add_operation("GATE", gate_name="H", args=[],
                                  targets=[f"q{i}"])
            return g

        g1 = _make_9q_h()
        g2 = _make_9q_h()
        report = verify_equivalence(g1, g2, mode=VerificationMode.EXACT)
        # Either skipped (with a warning) or fell back.
        self.assertIn(VerificationMode.EXACT, report.skipped_modes)
        self.assertTrue(any("EXACT comparison skipped" in w
                             for w in report.warnings))


class TestRewriteMode(unittest.TestCase):
    """§2.3 item 2: rewrite-based verification as a fallback."""

    def test_rewrite_hh_vs_identity(self):
        """REWRITE mode MUST report the same equivalence verdict
        as EXACT mode for HH vs I."""
        g1 = _make_hh_graph()
        g2 = _make_identity_graph()
        report = verify_equivalence(g1, g2, mode=VerificationMode.REWRITE)
        self.assertTrue(report.is_equivalent)
        self.assertEqual(report.mode_used, VerificationMode.REWRITE)

    def test_rewrite_h_vs_x_not_equivalent(self):
        """REWRITE mode MUST report False for H vs X."""
        g1 = _make_h_graph()
        g2 = _make_x_graph()
        report = verify_equivalence(g1, g2, mode=VerificationMode.REWRITE)
        self.assertFalse(report.is_equivalent)


class TestAutoMode(unittest.TestCase):
    """AUTO must descend from fast-reject to exact when possible."""

    def test_auto_calls_fast_reject_first(self):
        g1 = _make_h_graph()
        g2 = _make_x_graph()
        report = verify_equivalence(g1, g2, mode=VerificationMode.AUTO)
        # Hash mismatch — FAST_REJECT decides.
        self.assertFalse(report.is_equivalent)
        self.assertEqual(report.mode_used, VerificationMode.FAST_REJECT)

    def test_auto_falls_through_to_exact_for_small_circuit(self):
        """For structurally-identical small circuits, AUTO should
        descend to EXACT and return a definitive answer."""
        g1 = _make_hh_graph()
        g2 = _make_identity_graph()
        report = verify_equivalence(g1, g2, mode=VerificationMode.AUTO)
        # In AUTO the canonical hashes will differ (HH ≠ I structurally)
        # so the EXACT step must run.
        self.assertTrue(report.is_equivalent)
        # mode_used should be EXACT (or possibly REWRITE if exact path
        # encountered trouble, but for this trivially-small case it
        # should be EXACT).
        self.assertEqual(report.mode_used, VerificationMode.EXACT)


class TestReportAndBanner(unittest.TestCase):
    """§2.3 item 4: warnings in CLI output about verification
    boundaries."""

    def test_report_has_warning_method(self):
        r = VerificationReport(is_equivalent=None,
                                mode_used=VerificationMode.FAST_REJECT,
                                warnings=["foo"])
        self.assertTrue(r.has_warning())

    def test_report_has_warning_method_false_on_empty(self):
        r = VerificationReport(is_equivalent=True,
                                mode_used=VerificationMode.EXACT)
        self.assertFalse(r.has_warning())

    def test_format_cli_banner_includes_warnings(self):
        r = VerificationReport(
            is_equivalent=None,
            mode_used=VerificationMode.FAST_REJECT,
            warnings=["hash match does NOT prove equivalence"],
            notes=["fast-reject only"],
        )
        banner = format_cli_banner(r)
        self.assertIn("Verification Report", banner)
        self.assertIn("INCONCLUSIVE", banner)
        self.assertIn("fast_reject", banner)
        self.assertIn("hash match", banner)

    def test_format_cli_banner_includes_equivalent_result(self):
        r = VerificationReport(
            is_equivalent=True,
            mode_used=VerificationMode.EXACT,
            notes=["mathematically sound"],
        )
        banner = format_cli_banner(r)
        self.assertIn("EQUIVALENT", banner)
        self.assertIn("exact", banner)

    def test_format_cli_banner_includes_not_equivalent(self):
        r = VerificationReport(
            is_equivalent=False,
            mode_used=VerificationMode.FAST_REJECT,
            warnings=["hash mismatch: definitely NOT equivalent"],
            canonical_hash_1="aaa",
            canonical_hash_2="bbb",
        )
        banner = format_cli_banner(r)
        self.assertIn("NOT EQUIVALENT", banner)
        self.assertIn("Hash 1: aaa", banner)
        self.assertIn("Hash 2: bbb", banner)

    def test_format_cli_banner_lists_skipped_modes(self):
        r = VerificationReport(
            is_equivalent=True,
            mode_used=VerificationMode.EXACT,
            skipped_modes=[VerificationMode.FAST_REJECT,
                           VerificationMode.REWRITE],
        )
        banner = format_cli_banner(r)
        self.assertIn("Skipped modes:", banner)
        self.assertIn("fast_reject", banner)
        self.assertIn("rewrite", banner)


class TestSafetyProperties(unittest.TestCase):
    """Document the safety contracts of each mode."""

    def test_fast_reject_cannot_claim_equivalent(self):
        """FAST_REJECT may only return False (a hash mismatch
        disproves equivalence) or None (a hash match is
        inconclusive). It MUST NOT return True."""
        g1 = _make_h_graph()
        g2 = _make_h_graph()
        report = verify_equivalence(g1, g2, mode=VerificationMode.FAST_REJECT)
        self.assertNotEqual(report.is_equivalent, True)

    def test_fast_reject_with_non_overlapping_qubits_is_inconclusive(self):
        """Two empty graphs have the same canonical hash (the empty
        hash) but FAST_REJECT cannot claim equivalence — there is
        nothing to verify, the hashes match, and we can't rule on
        qubit overlap mismatch."""
        g1 = _make_identity_graph()
        g2 = _make_identity_graph()
        report = verify_equivalence(g1, g2, mode=VerificationMode.FAST_REJECT)
        self.assertIsNone(report.is_equivalent)


if __name__ == "__main__":
    unittest.main()
